"""The only tools an agent can reach: read-only tools of the standalone MCP server, by role.

Each agent is offered only its role's list. Customer and request identity are bound here from trusted
state; the model can only supply order/line/SKU arguments, which are type- and allowlist-checked.
Proposal creation, commits, faults, resets and SQL are not reachable.

Every call goes to that server, which must be running; there is no in-process fallback, and a
transport failure is a failed call. The server writes nothing, so the audit row is written here.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Protocol

from app.agents.schemas import READ_ONLY_TOOLS
from app.observability import telemetry as tel
from app.services.operations import RetailOperations

# Arguments each tool needs from the model (identity is never one of them).
TOOL_ARGS: dict[str, tuple[str, ...]] = {
    "lookup_customer_orders": (),
    "get_order_line": ("order_id", "line_id"),
    "check_return_eligibility": ("order_id", "line_id"),
    "get_exchange_options": ("order_id", "line_id"),
    "check_inventory": ("sku",),
    "get_request_status": (),
}

TOOL_DESCRIPTIONS = {
    "lookup_customer_orders": "List the customer's orders and lines. Arguments: none.",
    "get_order_line": "Details of one line. Arguments: order_id, line_id.",
    "check_return_eligibility": "Policy eligibility of one line. Arguments: order_id, line_id.",
    "get_exchange_options": "Other sizes of the line's product with eligibility and stock. Arguments: order_id, line_id.",
    "check_inventory": "Stock and price of one SKU. Arguments: sku.",
    "get_request_status": "Whether this request already completed. Arguments: none.",
}


# What each agent needs and no more. The server's host-only tool is in none of these.
ROLE_TOOLS: dict[str, tuple[str, ...]] = {
    "customer_resolution_agent": ("lookup_customer_orders",),
    "retail_operations_agent": ("check_return_eligibility", "get_exchange_options", "check_inventory",
                                "get_order_line", "get_request_status"),
}


class ToolNotAllowed(ValueError):
    """The requested tool or arguments are outside the allowlist."""


class ToolFailed(ToolNotAllowed):
    """The tool transport did not return a valid result (never treated as a success)."""


class ReadTransport(Protocol):
    def call(self, name: str, arguments: dict[str, str]) -> Any: ...


def call_read_tool(ops: RetailOperations, transport: Callable[[], ReadTransport], customer_id: str, request_id: str,
                   name: str, arguments: dict[str, str]) -> Any:
    """One call to the standalone server, audited here because that server writes nothing."""
    from app.mcp.client import McpToolError
    try:
        result = transport().call(name, arguments)
    except McpToolError as error:
        ops.record_failed_tool(name, arguments, error.args[0])
        raise ToolFailed(f"{name} unavailable ({error.args[0]})") from None
    ops.record_remote_tool(name, customer_id, request_id, arguments, result)
    return result


class ReadOnlyToolRegistry:
    def __init__(self, ops: RetailOperations, authenticated_customer_id: str, request_id: str,
                 transport: Callable[[], ReadTransport], role: str | None = None) -> None:
        """`transport`: factory for the scoped MCP client (resolved lazily on the first call).
        `role`: limits the tools to that agent's list."""
        self._ops = ops
        self._customer = authenticated_customer_id
        self._request = request_id
        self._transport = transport
        self._tools = ROLE_TOOLS[role] if role else tuple(TOOL_ARGS)
        assert set(READ_ONLY_TOOLS) == set(TOOL_ARGS)

    def describe(self) -> str:
        return "\n".join(f"- {name}: {TOOL_DESCRIPTIONS[name]}" for name in self._tools)

    def arguments_for(self, action: Any) -> dict[str, str]:
        if action.tool not in self._tools:
            raise ToolNotAllowed(f"tool {action.tool!r} is not allowed")
        args = {}
        for name in TOOL_ARGS[action.tool]:
            value = getattr(action, name)
            if not value:
                raise ToolNotAllowed(f"{action.tool} requires {name}")
            args[name] = value
        return args

    def call(self, action: Any) -> tuple[dict[str, str], Any]:
        """Run the requested read-only tool; returns (arguments used, JSON-safe result)."""
        name = action.tool if action.tool in TOOL_ARGS else "other"
        with tel.span(f"execute_tool {name}", {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name,
                                               "retail.tool.name": name, "retail.tool.transport": "mcp_http"}) as span:
            start = time.perf_counter()
            try:
                args = self.arguments_for(action)
            except ToolNotAllowed as error:
                tel.set_attributes(span, {"retail.tool.outcome": "refused"})
                tel.record_tool(action.tool, "refused")
                self._ops.record_refused_tool(str(action.tool), {"order_id": action.order_id,
                                                                 "line_id": action.line_id}, str(error))
                raise
            try:
                result = call_read_tool(self._ops, self._transport, self._customer, self._request, action.tool, args)
            except ToolFailed:
                tel.set_attributes(span, {"retail.tool.outcome": "failed"})
                tel.mark_error(span, "McpToolError")
                tel.record_tool(action.tool, "failed", time.perf_counter() - start)
                raise
            tel.set_attributes(span, {"retail.tool.outcome": "ok"})
            tel.record_tool(action.tool, "ok", time.perf_counter() - start)
            tel.capture(span, input=args, output=result)
            return args, result
