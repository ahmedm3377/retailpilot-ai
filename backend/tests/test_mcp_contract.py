"""The standalone MCP server's contract: exactly six read-only tools, strict arguments, parity with direct.

Every test talks to a REAL `python -m mcp_server` process over HTTP, against the test PostgreSQL schema.
No LLM is involved.
"""
from __future__ import annotations

import ast
import subprocess
import sys

import anyio
import pytest

from app.agents.schemas import OperationsAction
from app.agents.tool_registry import TOOL_ARGS, TOOL_DESCRIPTIONS, ReadOnlyToolRegistry
from app.db.schema import SCHEMA_MARKER
from app.mcp.client import HOST_TOOLS, McpReadTools, McpToolError
from app.mcp.scope import ToolScope
from app.services.audit import jsonable
from app.services.operations import RetailOperations
from retail_domain import policy
from mcp_server import server as standalone
from mcp_server import store as standalone_store
from tests.conftest import count
from tests.mcp_support import BACKEND

pytestmark = pytest.mark.postgres
FORBIDDEN = ["commit_confirmed_proposal", "create_proposal", "cancel_proposal", "reconcile_request",
             "reset_from_initial_state", "raw_sql", "unknown_tool"]


def scope_for(store, customer="C100", request="REQ-1") -> ToolScope:
    return ToolScope(store.schema, customer, request, "test")


def over_http(url, token, store, fn):
    from mcp import Client
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

    async def run():
        headers = {"Authorization": f"Bearer {token}", **scope_for(store).headers()}
        async with create_mcp_http_client(headers=headers) as http:
            async with Client(streamable_http_client(url, http_client=http)) as client:
                return await fn(client)
    return anyio.run(run)


# ---- independence ------------------------------------------------------------------------------
def test_server_code_imports_nothing_from_app():
    for package in ("mcp_server", "retail_domain"):
        for path in (BACKEND / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                assert not [n for n in names if n == "app" or n.startswith("app.")], path


def test_running_server_loads_no_app_module():
    code = ("import sys, mcp_server.server, mcp_server.store, mcp_server.__main__; "
            "print([m for m in sys.modules if m == 'app' or m.startswith('app.')])")
    out = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "[]", out.stderr[-500:]


def test_server_agrees_with_the_app_on_shared_facts():
    assert standalone_store.SCHEMA_MARKER == SCHEMA_MARKER
    agents_only = {k: v for k, v in standalone.DESCRIPTIONS.items() if k not in HOST_TOOLS}
    assert agents_only == TOOL_DESCRIPTIONS
    assert {k: tuple(m.model_fields) for k, m in standalone.ARG_MODELS.items()} == {**TOOL_ARGS, **HOST_TOOLS}


def test_server_connections_cannot_write(pg_store, pg_url, load):
    import psycopg
    load("public_01_refund_boundary")
    with standalone_store._connect(pg_url, pg_store.schema) as conn:
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            conn.execute("UPDATE products SET stock = 99")


def test_server_will_not_start_without_a_token(pg_url):
    env = {"RETAIL_SKIP_DOTENV": "1", "PATH": __import__("os").environ["PATH"],
           "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}
    out = subprocess.run([sys.executable, "-m", "mcp_server"], cwd=BACKEND, env=env, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 3 and "MCP_AUTH_TOKEN" in out.stderr


# ---- the tools ---------------------------------------------------------------------------------
def test_tools_list_is_exactly_the_six_agent_tools_and_the_host_read(pg_store, load, mcp_server):
    load("public_01_refund_boundary")
    listed = over_http(*mcp_server, pg_store, lambda c: c.list_tools())
    tools = {t.name: t for t in listed.tools}
    expected = {**TOOL_ARGS, **HOST_TOOLS}
    assert set(tools) == set(expected)
    for name, tool in tools.items():
        assert set(tool.input_schema.get("properties", {})) == set(expected[name])
        assert set(tool.input_schema.get("required", [])) == set(expected[name])
        assert tool.description == standalone.DESCRIPTIONS[name]


def test_the_host_read_returns_state_status_and_only_this_customers_proposal(pg_store, load):
    from app.domain.proposals import Proposal
    load("public_01_refund_boundary")
    operations = RetailOperations(pg_store, actor="test")
    proposal = operations.create_proposal("C100", "REQ-1", "O100", "L1", "refund").proposal
    scope = standalone.Scope("C100", "REQ-1", pg_store.schema, pg_store.url)
    found = standalone.get_business_context(scope, proposal.proposal_id)
    assert found["state"]["orders"][0]["order_id"] == "O100" and found["request_status"]["state"] == "not_found"
    assert Proposal.from_row(found["proposal"]) == proposal
    assert standalone.get_business_context(scope, "")["proposal"] is None
    other = standalone.Scope("C200", "REQ-1", pg_store.schema, pg_store.url)
    assert standalone.get_business_context(other, proposal.proposal_id)["proposal"] is None


@pytest.mark.parametrize("value", ["PROP'; DROP TABLE proposals; --", "x" * 65, 7])
def test_the_host_read_refuses_bad_arguments(value):
    with pytest.raises(ValueError):
        standalone.ContextArgs.model_validate({"proposal_id": value})


@pytest.mark.parametrize("name", FORBIDDEN)
def test_write_and_unknown_tools_are_rejected(pg_store, load, mcp_server, name):
    load("public_01_refund_boundary")
    result = over_http(*mcp_server, pg_store, lambda c: c.call_tool(name, {"proposal_id": "PROP-1", "sql": "DELETE"}))
    assert result.is_error and count(pg_store, "refunds") == 0


@pytest.mark.parametrize("args", [
    {"order_id": "O100", "line_id": "L1", "customer_id": "C999"},          # scope override attempt
    {"order_id": "O100", "line_id": "L1", "schema": "retail_ui_other"},    # schema override attempt
    {"order_id": "O100" * 30, "line_id": "L1"},                            # oversized
    {"order_id": "../../etc/passwd", "line_id": "L1"},                      # path-like
    {"order_id": "O100'; DROP TABLE refunds; --", "line_id": "L1"},         # SQL-like
    {"order_id": 100, "line_id": "L1"},                                     # wrong type
    {"order_id": "O100"},                                                   # missing
])
def test_strict_arguments_fail_closed(pg_store, load, mcp_server, args):
    load("public_01_refund_boundary")
    result = over_http(*mcp_server, pg_store, lambda c: c.call_tool("check_return_eligibility", args))
    assert result.is_error and "C999" not in str(result.content)


def test_the_server_writes_nothing_even_when_called(pg_store, load, mcp_server):
    load("public_01_refund_boundary")
    before = count(pg_store, "audit_events")
    result = over_http(*mcp_server, pg_store,
                       lambda c: c.call_tool("check_return_eligibility", {"order_id": "O100", "line_id": "L1"}))
    assert not result.is_error and result.structured_content["result"]["eligible"] is True
    assert count(pg_store, "audit_events") == before


# ---- through the registry: results from the server, audit rows from the host ----------------------
CALLS = [("lookup_customer_orders", {}), ("get_order_line", {"order_id": "O100", "line_id": "L1"}),
         ("check_return_eligibility", {"order_id": "O100", "line_id": "L1"}),
         ("get_exchange_options", {"order_id": "O100", "line_id": "L1"}), ("check_inventory", {"sku": "TEE-L"}),
         ("check_inventory", {"sku": "NO-SUCH"}), ("get_request_status", {}),
         ("get_order_line", {"order_id": "O999", "line_id": "L1"}),  # not found
         ("check_return_eligibility", {"order_id": "O100", "line_id": "L9"})]


@pytest.mark.parametrize("case_id", ["public_02_exchange", "public_03_expired", "public_04_clarify"])
def test_results_follow_the_policy_and_the_host_audits_each_call(pg_store, load, mcp_server, case_id):
    load(case_id)
    url, token = mcp_server
    ops = RetailOperations(pg_store, actor="retail_operations_agent")
    tools = McpReadTools(scope_for(pg_store), url=url, token=token)
    registry = ReadOnlyToolRegistry(ops, "C100", "REQ-1", lambda: tools)
    try:
        got = [registry.call(OperationsAction(action="call_tool", tool=n, **a))[1] for n, a in CALLS]
    finally:
        tools.close()
    state = pg_store.load_state()
    assert got[2] == jsonable(policy.check_return_eligibility(state, "C100", "O100", "L1"))
    assert got[3] == jsonable(list(policy.list_exchange_options(state, "C100", "O100", "L1")))
    assert (got[5]["found"], got[6]["state"], got[7]["found"]) == (False, "not_found", False)
    rows = pg_store.audit_log()
    assert [r["tool"] for r in rows] == [n for n, _ in CALLS]
    assert {r["actor"] for r in rows} == {"retail_operations_agent"}
    assert rows[5]["outcome"] == "SKU_NOT_FOUND" and rows[6]["outcome"] == "not_found"


def test_status_of_a_completed_request_comes_from_the_server(pg_store, load, mcp_server):
    from app.services.transactions import commit_confirmed_proposal
    load("public_01_refund_boundary")
    url, token = mcp_server
    ops = RetailOperations(pg_store, actor="retail_operations_agent")
    proposal = ops.create_proposal("C100", "REQ-1", "O100", "L1", "refund").proposal
    commit_confirmed_proposal(pg_store, "C100", "REQ-1", proposal.proposal_id)
    tools = McpReadTools(scope_for(pg_store), url=url, token=token)
    try:
        status = tools.call("get_request_status", {})
    finally:
        tools.close()
    assert status["state"] == "completed" and status["result"]["refund_amount_cents"] == 3000


def test_closed_handles_never_reopen_and_report_connect_time(pg_store, load, mcp_server):
    load("public_01_refund_boundary")
    url, token = mcp_server
    tools = McpReadTools(scope_for(pg_store), url=url, token=token).start()
    assert tools.startup_s > 0
    tools.close()
    with pytest.raises(McpToolError):
        tools.call("get_request_status", {})


def test_client_refuses_unknown_tool_and_extra_arguments_before_sending(pg_store, load):
    load("public_01_refund_boundary")
    tools = McpReadTools(scope_for(pg_store), url="http://127.0.0.1:9/mcp", token="t")
    for name, args in (("commit_confirmed_proposal", {}), ("check_inventory", {"sku": "X", "extra": "y"})):
        with pytest.raises(McpToolError, match="MCP_ARGUMENTS_REJECTED"):
            tools.call(name, args)
    tools.close()
