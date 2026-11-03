"""The standalone MCP server's security: token, loopback, scope on every request, isolation, failures.

Uses real server processes. tests/mcp_rogue_server.py is a TEST-ONLY misbehaving server.
"""
from __future__ import annotations

import uuid

import httpx
import psutil
import psycopg
import pytest

from app.db.fixture_state import RetailStore
from app.mcp.client import McpReadTools, McpToolError
from app.mcp.scope import ScopeError, ToolScope, mode_for_schema
from mcp_server.server import Config, ScopeRejected, resolve_scope
from mcp_server.store import StoreError
from tests.conftest import make_line, make_state
from tests.mcp_support import running_server

pytestmark = pytest.mark.postgres
APP_URL = "postgresql://retail:retail@127.0.0.1:55432/retail"
TEST_URL = "postgresql://retail:retail@127.0.0.1:55432/retail_test"
CONFIG = Config("t", 1, APP_URL, TEST_URL)


# ---- host-side scope ---------------------------------------------------------------------------
@pytest.mark.parametrize("schema,mode", [("retail_eval_abc", "ui"), ("retail_ui_abc", "evaluation"), ("public", "test"),
                                         ("retail_t_abc", "admin"), ("retail_t_A;DROP", "test")])
def test_scope_rejects_wrong_schema_or_mode(schema, mode):
    with pytest.raises(ScopeError):
        ToolScope(schema, "C100", "REQ-1", mode).validate()


@pytest.mark.parametrize("customer,request_id", [("", "REQ-1"), ("C100", ""), ("C1 OR 1=1", "REQ-1"),
                                                 ("C100", "x" * 200)])
def test_scope_rejects_missing_or_malformed_identity(customer, request_id):
    with pytest.raises(ScopeError):
        ToolScope("retail_t_abc", customer, request_id, "test").validate()


@pytest.mark.parametrize("schema", ["retail_store", "retail_ui_abc"])
def test_the_live_store_is_a_valid_ui_scope(schema):
    """Regression: `retail_store` was rejected, so MCP mode failed on every real sign-in."""
    assert mode_for_schema(schema) == "ui"
    assert ToolScope(schema, "C100", "REQ-1", "ui").validate().schema == schema
    assert resolve_scope(ToolScope(schema, "C100", "REQ-1", "ui").headers(), CONFIG).url == APP_URL


# ---- server-side scope, checked on every request -------------------------------------------------
def _headers(**over):
    base = {"x-retail-scope-schema": "retail_t_abc", "x-retail-scope-customer": "C100",
            "x-retail-scope-request": "REQ-1", "x-retail-scope-mode": "test"}
    return {**base, **over}


@pytest.mark.parametrize("over", [{"x-retail-scope-mode": "admin"}, {"x-retail-scope-schema": "public"},
                                  {"x-retail-scope-schema": "retail_ui_abc"}, {"x-retail-scope-customer": ""},
                                  {"x-retail-scope-customer": "C1 OR 1=1"}, {"x-retail-scope-request": ""}])
def test_server_rejects_a_bad_scope(over):
    with pytest.raises(ScopeRejected):
        resolve_scope(_headers(**over), CONFIG)


def test_server_rejects_missing_scope_headers():
    with pytest.raises(ScopeRejected):
        resolve_scope(None, CONFIG)


def test_ui_and_test_modes_use_the_right_database():
    assert resolve_scope(_headers(), CONFIG).url == TEST_URL
    mixed_up = Config("t", 1, TEST_URL, APP_URL)  # misconfigured: ui on the test DB and tests on the app DB
    with pytest.raises(StoreError):
        resolve_scope(_headers(), mixed_up)


def _raw(url, headers=None, host=None):
    """One request straight at the running server, as another local process could send it."""
    send = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", **(headers or {})}
    if host:
        send["Host"] = host
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    return httpx.post(url, json=body, headers=send, timeout=5, trust_env=False)


def test_server_refuses_requests_without_its_token(mcp_server):
    url, token = mcp_server
    assert _raw(url).status_code == 401
    assert _raw(url, {"Authorization": "Bearer wrong"}).status_code == 401
    assert _raw(url, {"Authorization": f"Bearer {token}"}).status_code != 401


def test_server_refuses_a_foreign_host_header(mcp_server):
    """DNS rebinding: a web page reaching loopback through its own hostname."""
    url, token = mcp_server
    assert _raw(url, {"Authorization": f"Bearer {token}"}, host="evil.example.com").status_code in (400, 403, 421)


def test_wrong_token_is_a_failed_call_not_a_fallback(pg_store, load, mcp_server):
    load("public_01_refund_boundary")
    tools = McpReadTools(ToolScope(pg_store.schema, "C100", "REQ-1", "test"), url=mcp_server[0], token="wrong")
    with pytest.raises(McpToolError, match="MCP_STARTUP_FAILED"):
        tools.call("get_request_status", {})
    tools.close()


def test_client_needs_a_token_and_a_local_url(pg_store, load, mcp_server):
    load("public_01_refund_boundary")
    scope = ToolScope(pg_store.schema, "C100", "REQ-1", "test")
    for tools in (McpReadTools(scope, url=mcp_server[0], token=""),
                  McpReadTools(scope, url="http://tools.example.com/mcp", token="t")):
        with pytest.raises(McpToolError, match="MCP_NOT_CONFIGURED"):
            tools.call("get_request_status", {})


def test_server_listens_on_loopback_only(pg_url):
    with running_server(pg_url) as (url, _token, process):
        family = [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]
        listening = [c.laddr for p in family for c in p.net_connections() if c.status == psutil.CONN_LISTEN]
        assert listening and all(addr.ip == "127.0.0.1" for addr in listening)


def test_token_never_appears_on_the_command_line(pg_url):
    with running_server(pg_url) as (_url, token, process):
        assert token not in " ".join(psutil.Process(process.pid).cmdline())


def test_an_unmarked_schema_is_refused(pg_url, mcp_server):
    schema = f"retail_t_unmarked_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(pg_url, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
    url, token = mcp_server
    tools = McpReadTools(ToolScope(schema, "C100", "REQ-1", "test"), url=url, token=token)
    try:
        with pytest.raises(McpToolError, match="MCP_TOOL_ERROR"):
            tools.call("get_request_status", {})
    finally:
        tools.close()
        with psycopg.connect(pg_url, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA "{schema}"')


# ---- isolation ---------------------------------------------------------------------------------
def test_customer_a_cannot_read_customer_b_even_with_b_ids(pg_store, pg_url, mcp_server):
    pg_store.reset_from_initial_state(make_state(customer_id="CUST-A", order_id="ORD-A"))
    other = RetailStore(pg_url, f"retail_t_{uuid.uuid4().hex[:10]}", require_test_db=True)
    other.create_schema()
    other.reset_from_initial_state(make_state(customer_id="CUST-B", order_id="ORD-B",
                                              lines=[make_line("LN-B", "WIDGET-B")]))
    url, token = mcp_server
    tools = McpReadTools(ToolScope(pg_store.schema, "CUST-A", "REQ-A", "test"), url=url, token=token)
    try:
        assert tools.call("lookup_customer_orders", {})[0]["order_id"] == "ORD-A"
        foreign = tools.call("get_order_line", {"order_id": "ORD-B", "line_id": "LN-B"})
        assert foreign["found"] is False and foreign["line"] is None
    finally:
        tools.close()
        other.drop_schema()


def test_one_server_keeps_two_customers_apart_at_once(pg_store, mcp_server):
    state = make_state(customer_id="CUST-A", order_id="ORD-A")
    second = state.orders[0].model_copy(update={"order_id": "ORD-B", "customer_id": "CUST-B"})
    pg_store.reset_from_initial_state(state.model_copy(update={"orders": [state.orders[0], second]}))
    url, token = mcp_server
    a = McpReadTools(ToolScope(pg_store.schema, "CUST-A", "REQ-A", "test"), url=url, token=token)
    b = McpReadTools(ToolScope(pg_store.schema, "CUST-B", "REQ-B", "test"), url=url, token=token)
    try:
        for _ in range(3):
            assert [o["order_id"] for o in a.call("lookup_customer_orders", {})] == ["ORD-A"]
            assert [o["order_id"] for o in b.call("lookup_customer_orders", {})] == ["ORD-B"]
    finally:
        a.close()
        b.close()


# ---- failures: rogue servers, crash, timeout; never a fallback -----------------------------------
@pytest.fixture
def rogue(pg_url, pg_store):
    servers = []

    def start(mode: str, **kw) -> McpReadTools:
        manager = running_server(pg_url, module="tests.mcp_rogue_server", extra_env={"RETAIL_ROGUE_MODE": mode})
        url, token, _ = manager.__enter__()
        servers.append(manager)
        return McpReadTools(ToolScope(pg_store.schema, "C100", "REQ-1", "test"), url=url, token=token, **kw)

    yield start
    for manager in servers:
        manager.__exit__(None, None, None)


def test_server_advertising_an_extra_tool_is_refused(load, rogue):
    load("public_01_refund_boundary")
    tools = rogue("extra_tool", startup_timeout_s=30)
    with pytest.raises(McpToolError, match="MCP_STARTUP_FAILED"):
        tools.call("get_request_status", {})
    tools.close()


def test_crash_mid_call_is_a_failure_not_a_result(load, rogue):
    load("public_01_refund_boundary")
    tools = rogue("crash")
    with pytest.raises(McpToolError):
        tools.call("check_inventory", {"sku": "TEE-L"})
    tools.close()


def test_timeout_is_a_failure(load, rogue):
    load("public_01_refund_boundary")
    tools = rogue("slow", call_timeout_s=2)
    with pytest.raises(McpToolError, match="MCP_CALL_FAILED"):
        tools.call("check_inventory", {"sku": "TEE-L"})
    tools.close()


def test_server_that_is_not_running_is_a_failure(pg_store, load):
    load("public_01_refund_boundary")
    tools = McpReadTools(ToolScope(pg_store.schema, "C100", "REQ-1", "test"), url="http://127.0.0.1:9/mcp", token="t",
                         startup_timeout_s=10)
    with pytest.raises(McpToolError, match="MCP_STARTUP"):
        tools.call("get_request_status", {})
    tools.close()


def test_rogue_result_text_is_returned_as_data_only(load, rogue):
    """A tool result containing instructions is data; it cannot add tools (the client allowlist is fixed)."""
    load("public_01_refund_boundary")
    tools = rogue("result_injection")  # any mode other than crash/slow returns the text
    try:
        result = tools.call("check_inventory", {"sku": "TEE-L"})
        assert "ignore previous instructions" in result  # returned as plain data for the guard to check
        with pytest.raises(McpToolError):
            tools.call("commit_confirmed_proposal", {})
    finally:
        tools.close()
