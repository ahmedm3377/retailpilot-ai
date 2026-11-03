"""McpReadTools: the trusted host's connection to the standalone MCP server, for one scope.

The server runs on its own (`python -m mcp_server`). MCP_SERVER_URL says where, and MCP_AUTH_TOKEN
identifies this host to it. The runtime is synchronous (LangGraph, AgentRuntime, FastAPI `def`
endpoints), so the async MCP Client lives on ONE owned anyio blocking portal (a dedicated event-loop
thread) per scope. Lifetime: opened lazily on the first tool call, reused for the scope's turns,
closed by AgentRuntime.close() or when the scope (request) changes. There is no fallback to
in-process tools: any connection, protocol, timeout or validation failure raises McpToolError, which
the registry records as a failed tool call.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any
from urllib.parse import urlparse

from app.agents.tool_registry import TOOL_ARGS
from app.mcp.scope import ToolScope
from app.observability import telemetry as tel

DEFAULT_URL = "http://127.0.0.1:8765/mcp"
# On the server but never in an agent's registry: only the host's business checks call it.
HOST_TOOLS: dict[str, tuple[str, ...]] = {"get_business_context": ("proposal_id",)}
ALL_TOOLS = {**TOOL_ARGS, **HOST_TOOLS}


class McpToolError(RuntimeError):
    """The MCP transport or server did not return a valid result (sanitized code in args[0])."""


def server_url() -> str:
    return os.environ.get("MCP_SERVER_URL", DEFAULT_URL)


class McpReadTools:
    def __init__(self, scope: ToolScope, *, startup_timeout_s: float = 15.0, call_timeout_s: float = 30.0,
                 url: str | None = None, token: str | None = None) -> None:
        self.scope = scope.validate()
        self.url = url or server_url()
        self.token = token if token is not None else os.environ.get("MCP_AUTH_TOKEN", "")
        self.startup_timeout_s = startup_timeout_s
        self.call_timeout_s = call_timeout_s
        self.startup_s: float | None = None
        self._portal_cm: Any = None
        self._portal: Any = None
        self._future: Any = None
        self._client: Any = None
        self._stop: Any = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._closed = False

    # -- lifecycle -------------------------------------------------------------------------------
    async def _lifetime(self) -> None:
        import anyio
        from mcp import Client
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
        self._stop = anyio.Event()
        headers = {"Authorization": f"Bearer {self.token}", **self.scope.headers()}
        async with create_mcp_http_client(headers=headers) as http:
            async with Client(streamable_http_client(self.url, http_client=http)) as client:
                listed = await client.list_tools()
                names = {t.name: set((t.input_schema or {}).get("properties", {})) for t in listed.tools}
                if names != {k: set(v) for k, v in ALL_TOOLS.items()}:
                    raise McpToolError("MCP_TOOL_LIST_MISMATCH")
                self._client = client
                self._ready.set()
                await self._stop.wait()

    def start(self) -> "McpReadTools":
        with self._lock:
            if self._client is not None:
                return self
            if self._closed:
                raise McpToolError("MCP_CLOSED")
            if not self.token or urlparse(self.url).hostname not in ("127.0.0.1", "localhost", "::1"):
                raise McpToolError("MCP_NOT_CONFIGURED")  # no token, or not this machine's server
            from anyio.from_thread import start_blocking_portal
            started = time.perf_counter()
            self._portal_cm = start_blocking_portal()
            self._portal = self._portal_cm.__enter__()
            self._future = self._portal.start_task_soon(self._lifetime)
            deadline = started + self.startup_timeout_s
            while not self._ready.wait(0.05):
                if self._future.done() or time.perf_counter() > deadline:
                    error = "MCP_STARTUP_TIMEOUT" if not self._future.done() else "MCP_STARTUP_FAILED"
                    self._teardown()
                    raise McpToolError(error)
            self.startup_s = time.perf_counter() - started
            tel.record_mcp_startup(self.startup_s)
            return self

    def _teardown(self) -> None:
        if self._portal is not None:
            if self._stop is not None:
                try:
                    self._portal.call(self._stop.set)
                except Exception:  # noqa: BLE001 - loop may already be gone
                    pass
            if self._future is not None:
                try:
                    self._future.result(timeout=15)
                except Exception:  # noqa: BLE001 - includes startup errors and cancellation
                    self._future.cancel()
            try:
                self._portal_cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        self._portal = self._portal_cm = self._future = self._client = self._stop = None
        self._ready.clear()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._teardown()

    # -- calls -----------------------------------------------------------------------------------
    def call(self, name: str, arguments: dict[str, str]) -> Any:
        """Run one allowlisted tool; returns the JSON-safe result or raises McpToolError."""
        if name not in ALL_TOOLS or set(arguments) != set(ALL_TOOLS[name]):
            raise McpToolError("MCP_ARGUMENTS_REJECTED")  # validated again by the server
        self.start()
        from opentelemetry.propagate import inject
        carrier: dict[str, str] = {}
        inject(carrier)  # the caller's execute_tool span, captured in THIS thread for the portal thread
        try:
            result = self._portal.call(self._call, name, dict(arguments), carrier)
        except McpToolError:
            raise
        except Exception as error:  # noqa: BLE001 - transport/timeout/crash: never a plausible success
            raise McpToolError(f"MCP_CALL_FAILED:{type(error).__name__}") from None
        if result.is_error:
            raise McpToolError("MCP_TOOL_ERROR")
        payload = result.structured_content
        if not isinstance(payload, dict) or set(payload) != {"result"}:
            raise McpToolError("MCP_RESULT_INVALID")
        return payload["result"]

    async def _call(self, name: str, arguments: dict[str, str], carrier: dict[str, str]) -> Any:
        if self._client is None:
            raise McpToolError("MCP_NOT_STARTED")
        return await self._client.call_tool(name, arguments, read_timeout_seconds=self.call_timeout_s,
                                            meta=carrier or None)


def check_server() -> None:
    """Raise McpToolError unless the server accepts this host's token and offers exactly the expected tools.

    The app can't run without it, so entry points call this at startup. No tool runs, so the scope is unused."""
    tools = McpReadTools(ToolScope("retail_store", "startup-check", "startup-check", "ui"))
    try:
        tools.start()
    finally:
        tools.close()


def unavailable_message(error: McpToolError) -> str:
    return (f"The MCP server is not available at {server_url()} ({error.args[0]}). Start it first, in its own "
            "terminal: uv run python -m mcp_server. Both sides read the same MCP_AUTH_TOKEN from .env.")
