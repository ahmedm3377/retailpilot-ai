"""Trace linkage: host execute_tool span -> span in the standalone server, same trace.

Host spans go to an in-memory exporter; the server writes its spans to a TEST-ONLY JSON-lines
file (MCP_SPAN_FILE). The W3C traceparent travels in the tools/call request `_meta`.
"""
from __future__ import annotations

import json
import time

import pytest

from app.agents.schemas import OperationsAction
from app.agents.tool_registry import ReadOnlyToolRegistry
from app.mcp.client import McpReadTools
from app.mcp.scope import ToolScope
from app.observability import telemetry as tel
from app.observability.config import TelemetrySettings
from app.services.operations import RetailOperations
from tests.mcp_support import running_server

pytestmark = pytest.mark.postgres


@pytest.fixture
def otel():
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    spans, reader = InMemorySpanExporter(), InMemoryMetricReader()
    tel.shutdown()
    assert tel.init(TelemetrySettings(enabled=True), span_exporter=spans, metric_reader=reader)
    yield spans, reader
    tel.shutdown()


def child_spans(path, wanted: int, timeout: float = 20.0) -> list[dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
            if len([r for r in rows if r["attributes"].get("retail.tool.name")]) >= wanted:
                return rows
        time.sleep(0.2)
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_server_span_joins_the_parent_trace(pg_store, pg_url, load, otel, tmp_path):
    load("public_02_exchange")
    spans, reader = otel
    span_file = tmp_path / "server_spans.jsonl"
    with running_server(pg_url, extra_env={"MCP_SPAN_FILE": str(span_file)}) as (url, token, _process):
        tools = McpReadTools(ToolScope(pg_store.schema, "C100", "REQ-1", "test"), url=url, token=token)
        registry = ReadOnlyToolRegistry(RetailOperations(pg_store), "C100", "REQ-1", lambda: tools)
        try:
            registry.call(OperationsAction(action="call_tool", tool="check_return_eligibility", order_id="O100",
                                           line_id="L1"))
            registry.call(OperationsAction(action="call_tool", tool="check_inventory", sku="TEE-L"))
        finally:
            tools.close()
        server_rows = child_spans(span_file, 2)  # it exports span by span
    parent = {s.name: s for s in spans.get_finished_spans()}
    rows = [r for r in server_rows if r["attributes"].get("retail.tool.name")]
    assert len(rows) == 2, rows
    for row in rows:
        tool = row["attributes"]["retail.tool.name"]
        p = parent[f"execute_tool {tool}"]
        assert p.attributes["retail.tool.transport"] == "mcp_http"
        assert row["trace_id"] == format(p.context.trace_id, "032x")       # same trace
        assert row["parent_span_id"] == format(p.context.span_id, "016x")  # child of execute_tool
        assert row["attributes"]["retail.tool.transport"] == "mcp_http"
        assert not row["error"]
    text = json.dumps(rows)
    for secret in ("O100", "C100", "REQ-1", "retail:retail"):
        assert secret not in text  # allowlisted attributes only


