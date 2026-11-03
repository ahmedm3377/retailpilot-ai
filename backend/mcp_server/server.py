"""The read-only tools, over Streamable HTTP on loopback: six for agents, one for the trusted host.

- No resources, prompts, SQL or write paths. Customer, request and schema never come from tool arguments:
  the trusted host sends them in headers on every request, so a model can't set them.
- get_business_context is for the host's business checks. It returns the whole schema's state, so the host
  never offers it to a model.
- A shared bearer token (MCP_AUTH_TOKEN) identifies that host: whoever holds it can ask about any customer,
  so it belongs only to the host. DNS-rebinding protection refuses other Host headers.
- Arguments are re-validated strictly by middleware that runs before the SDK's own, lenient, parsing.
- Results are JSON built here from retail_domain rules, in the shape the host's agents expect.
"""
from __future__ import annotations

import dataclasses
import hmac
import json
import os
import re
from enum import Enum
from typing import Any, Callable, Mapping

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from retail_domain import policy

from mcp_server.store import StoreError, check_url, get_product, get_proposal, get_request, load_state

SERVER_NAME = "hp-retail-readonly-tools"
ID = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$"
SCOPE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
HEADER_PREFIX = "x-retail-scope-"
SCHEMA_PREFIXES = {"ui": ("retail_ui_", "retail_store"), "evaluation": ("retail_eval_",), "test": ("retail_t_",)}
DESCRIPTIONS = {
    "lookup_customer_orders": "List the customer's orders and lines. Arguments: none.",
    "get_order_line": "Details of one line. Arguments: order_id, line_id.",
    "check_return_eligibility": "Policy eligibility of one line. Arguments: order_id, line_id.",
    "get_exchange_options": "Other sizes of the line's product with eligibility and stock. Arguments: order_id, line_id.",
    "check_inventory": "Stock and price of one SKU. Arguments: sku.",
    "get_request_status": "Whether this request already completed. Arguments: none.",
    "get_business_context": "For the trusted host's business checks, never for a model: the store's state, this "
                            "request's status and one proposal. Arguments: proposal_id (empty for none).",
}


class ConfigError(RuntimeError):
    """The server can't start safely."""


class ScopeRejected(RuntimeError):
    """The request's scope headers are missing or point somewhere they must not."""


@dataclasses.dataclass(frozen=True)
class Config:
    token: str
    port: int
    database_url: str
    test_database_url: str

    @classmethod
    def from_env(cls, env: Mapping[str, str], port: int) -> "Config":
        token = env.get("MCP_AUTH_TOKEN", "")
        if not token:
            raise ConfigError("MCP_AUTH_TOKEN is not set")
        if not 0 < port < 65536:
            raise ConfigError("bad port")
        return cls(token, port, env.get("MCP_DATABASE_URL") or env.get("DATABASE_URL", ""),
                   env.get("MCP_TEST_DATABASE_URL") or env.get("RETAIL_TEST_DATABASE_URL", ""))


@dataclasses.dataclass(frozen=True)
class Scope:
    customer_id: str
    request_id: str
    schema: str
    url: str


def resolve_scope(headers: Mapping[str, str] | None, config: Config) -> Scope:
    """Validate the host's scope on every request, before any query."""
    values = {k.lower(): v for k, v in (headers or {}).items()}
    customer = values.get(f"{HEADER_PREFIX}customer", "")
    request = values.get(f"{HEADER_PREFIX}request", "")
    schema = values.get(f"{HEADER_PREFIX}schema", "")
    mode = values.get(f"{HEADER_PREFIX}mode", "")
    if mode not in SCHEMA_PREFIXES or not schema.startswith(SCHEMA_PREFIXES[mode]):
        raise ScopeRejected("schema not allowed for this mode")
    if not SCOPE_ID.fullmatch(customer) or not SCOPE_ID.fullmatch(request):
        raise ScopeRejected("customer or request missing or malformed")
    url = config.database_url if mode == "ui" else config.test_database_url
    return Scope(customer, request, schema, check_url(url, test_database=mode != "ui"))


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class NoArgs(_Strict):
    pass


class LineArgs(_Strict):
    order_id: str = Field(pattern=ID)
    line_id: str = Field(pattern=ID)


class SkuArgs(_Strict):
    sku: str = Field(pattern=ID)


class ContextArgs(_Strict):
    proposal_id: str = Field(pattern=r"^$|" + ID)


ARG_MODELS: dict[str, type[_Strict]] = {
    "lookup_customer_orders": NoArgs, "get_order_line": LineArgs, "check_return_eligibility": LineArgs,
    "get_exchange_options": LineArgs, "check_inventory": SkuArgs, "get_request_status": NoArgs,
    "get_business_context": ContextArgs,
}


class StrictArguments:
    """Server middleware: refuse tools/call whose raw arguments do not match the strict model."""

    async def __call__(self, ctx: Any, call_next: Any) -> Any:
        if ctx.method == "tools/call":
            params = ctx.params or {}
            model = ARG_MODELS.get(params.get("name"))
            if model is not None:
                try:
                    model.model_validate(params.get("arguments") or {})
                except ValidationError:
                    from mcp.types import CallToolResult, TextContent
                    return CallToolResult(content=[TextContent(type="text", text="INVALID_ARGUMENTS")], is_error=True)
        return await call_next(ctx)


def _json(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


def _line_view(state: Any, order_id: str, line_id: str) -> dict[str, Any]:
    order = next(o for o in state.orders if o.order_id == order_id)
    line = next(ln for ln in order.lines if ln.line_id == line_id)
    product = next((p for p in state.products if p.sku == line.sku), None)
    return {"order_id": order_id, "line_id": line.line_id, "sku": line.sku,
            "product_id": product.product_id if product else None, "size": product.size if product else None,
            "paid_cents": line.paid_cents, "unopened": line.unopened, "final_sale": line.final_sale,
            "returned": line.returned}


def lookup_customer_orders(scope: Scope) -> Any:
    state = load_state(scope.url, scope.schema)
    return [{"order_id": o.order_id, "status": o.status,
             "delivered_on": o.delivered_on.isoformat() if o.delivered_on else None,
             "lines": [_line_view(state, o.order_id, ln.line_id) for ln in o.lines]}
            for o in state.orders if o.customer_id == scope.customer_id]


def get_order_line(scope: Scope, order_id: str, line_id: str) -> Any:
    state = load_state(scope.url, scope.schema)
    found = policy.resolve_order_line(state, scope.customer_id, order_id, line_id)
    return {"found": found.ok, "reason_code": found.reason_code.value,
            "line": _line_view(state, found.order_id, found.line_id) if found.ok else None,
            "candidate_line_ids": list(found.candidate_line_ids)}


def check_return_eligibility(scope: Scope, order_id: str, line_id: str) -> Any:
    return _json(policy.check_return_eligibility(load_state(scope.url, scope.schema), scope.customer_id, order_id,
                                                 line_id))


def get_exchange_options(scope: Scope, order_id: str, line_id: str) -> Any:
    return _json(policy.list_exchange_options(load_state(scope.url, scope.schema), scope.customer_id, order_id,
                                              line_id))


def check_inventory(scope: Scope, sku: str) -> Any:
    row = get_product(scope.url, scope.schema, sku)
    if row is None:
        return {"sku": sku, "found": False, "product_id": None, "size": None, "price_cents": None, "stock": None}
    return {"sku": sku, "found": True, "product_id": row["product_id"], "size": row["size"],
            "price_cents": row["price_cents"], "stock": row["stock"]}


def get_request_status(scope: Scope) -> Any:
    row = get_request(scope.url, scope.schema, scope.request_id)
    if row is None:
        return {"state": "not_found", "request_id": scope.request_id, "result": None}
    if row["customer_id"] != scope.customer_id:  # another customer's request: no details
        return {"state": "conflict", "request_id": scope.request_id, "result": None}
    return {"state": "completed", "request_id": scope.request_id, "result": row["result"]}


def get_business_context(scope: Scope, proposal_id: str) -> Any:
    state = load_state(scope.url, scope.schema)
    proposal = get_proposal(scope.url, scope.schema, proposal_id) if proposal_id else None
    if proposal is not None and proposal["customer_id"] != scope.customer_id:
        proposal = None
    return {"state": state.model_dump(mode="json"), "request_status": get_request_status(scope), "proposal": proposal}


def build_server(config: Config) -> MCPServer:
    server = MCPServer(SERVER_NAME, log_level="ERROR", middleware=[StrictArguments()])

    def run(name: str, ctx: Context, fn: Callable[[Scope], Any]) -> dict[str, Any]:
        try:
            result = fn(resolve_scope(ctx.headers, config))
        except (ScopeRejected, StoreError):
            _mark(name, "scope_rejected")
            raise RuntimeError("SCOPE_REJECTED") from None
        except Exception:  # noqa: BLE001 - sanitized: no internals cross the boundary
            _mark(name, "failed")
            raise RuntimeError("TOOL_FAILED") from None
        _mark(name, "ok")
        return {"result": result}

    def tool(name: str):
        return server.tool(name=name, description=DESCRIPTIONS[name], structured_output=True)

    @tool("lookup_customer_orders")
    def _lookup(ctx: Context) -> dict[str, Any]:
        return run("lookup_customer_orders", ctx, lookup_customer_orders)

    @tool("get_order_line")
    def _line(order_id: str, line_id: str, ctx: Context) -> dict[str, Any]:
        return run("get_order_line", ctx, lambda s: get_order_line(s, order_id, line_id))

    @tool("check_return_eligibility")
    def _eligibility(order_id: str, line_id: str, ctx: Context) -> dict[str, Any]:
        return run("check_return_eligibility", ctx, lambda s: check_return_eligibility(s, order_id, line_id))

    @tool("get_exchange_options")
    def _options(order_id: str, line_id: str, ctx: Context) -> dict[str, Any]:
        return run("get_exchange_options", ctx, lambda s: get_exchange_options(s, order_id, line_id))

    @tool("check_inventory")
    def _inventory(sku: str, ctx: Context) -> dict[str, Any]:
        return run("check_inventory", ctx, lambda s: check_inventory(s, sku))

    @tool("get_request_status")
    def _status(ctx: Context) -> dict[str, Any]:
        return run("get_request_status", ctx, get_request_status)

    @tool("get_business_context")
    def _context(proposal_id: str, ctx: Context) -> dict[str, Any]:
        return run("get_business_context", ctx, lambda s: get_business_context(s, proposal_id))

    return server


class BearerToken:
    """ASGI guard: only the host, which holds the token, can reach the MCP endpoint."""

    def __init__(self, app: Any, token: str) -> None:
        self.app, self.expected = app, f"Bearer {token}".encode()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            supplied = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(supplied, self.expected):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"text/plain")]})
                await send({"type": "http.response.body", "body": b"unauthorized"})
                return
        await self.app(scope, receive, send)


def serve_http(server: Any, port: int, token: str) -> None:
    """Serve on loopback only; also used by the test-only rogue servers."""
    import uvicorn
    from mcp.server.transport_security import TransportSecuritySettings

    security = TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=[f"127.0.0.1:{port}"],
                                         allowed_origins=[])
    app = server.streamable_http_app(json_response=True, transport_security=security)
    uvicorn.run(BearerToken(app, token), host="127.0.0.1", port=port, log_level="error", access_log=False)


def _mark(tool: str, outcome: str) -> None:
    """Allowlisted attributes on the SDK's server span (exported only when telemetry is on)."""
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        span.set_attribute("retail.tool.name", tool)
        span.set_attribute("retail.tool.transport", "mcp_http")
        span.set_attribute("retail.tool.outcome", outcome)
    except Exception:  # noqa: BLE001 - telemetry never affects a tool result
        pass


class _JsonlSpanExporter:
    """TEST-ONLY exporter (MCP_SPAN_FILE): span name, ids and allowlisted attributes, one per line."""

    def __init__(self, path: str) -> None:
        self.path = path

    def export(self, spans: Any) -> Any:
        from opentelemetry.sdk.trace.export import SpanExportResult
        with open(self.path, "a", encoding="utf-8") as handle:
            for s in spans:
                handle.write(json.dumps({
                    "name": s.name, "trace_id": format(s.context.trace_id, "032x"),
                    "span_id": format(s.context.span_id, "016x"),
                    "parent_span_id": format(s.parent.span_id, "016x") if s.parent else None,
                    "attributes": {k: v for k, v in dict(s.attributes or {}).items()
                                   if k.startswith(("retail.tool.", "mcp.method", "gen_ai.tool"))},
                    "error": s.status.status_code.name == "ERROR"}) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 0) -> bool:
        return True


def _otlp_endpoint(env: Mapping[str, str]) -> str:
    from urllib.parse import urlparse
    url = env.get("RETAIL_OTEL_ENDPOINT", "http://127.0.0.1:14318").rstrip("/")
    parsed = urlparse(url)
    # 4317/4318 belong to another project's collector on this machine.
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") \
            or parsed.port in (None, 4317, 4318):
        raise ConfigError("RETAIL_OTEL_ENDPOINT must be this project's loopback collector")
    return url


def start_telemetry(env: Mapping[str, str]) -> Callable[[], None]:
    """Optional tracing of the SDK's server spans, which join the host's trace. Returns how to flush it.

    This process imports no other instrumented library, so setting the global provider is safe."""
    span_file = env.get("MCP_SPAN_FILE")
    enabled = env.get("RETAIL_OTEL_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
    if not span_file and not enabled:
        return lambda: None
    try:
        os.environ["OTEL_SDK_DISABLED"] = "false"
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
        name = env.get("OTEL_SERVICE_NAME", "hp-retail-agentic-ai")
        provider = TracerProvider(resource=Resource.create({"service.name": f"{name}-mcp"}))
        if span_file:
            provider.add_span_processor(SimpleSpanProcessor(_JsonlSpanExporter(span_file)))
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=f"{_otlp_endpoint(env)}/v1/traces", timeout=2)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        return provider.shutdown
    except Exception:  # noqa: BLE001 - telemetry must never stop the server
        return lambda: None
