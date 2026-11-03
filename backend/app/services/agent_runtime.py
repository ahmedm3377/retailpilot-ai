"""Shared entry point for the UI and the HP JSON adapter: one guarded agent turn.

The caller is trusted: it supplies the authenticated customer, request, session,
1-based turn number, the per-case FaultInjector and, for a confirmation turn,
the proposal ID it resolved (the adapter resolves confirm_proposal "last").

Graph checkpoints live in their own PostgreSQL schema (`retail_checkpoints` by
default); they hold conversation state only, never money or stock. The thread
is keyed by customer, session and request, all from trusted inputs.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import sql
from psycopg.rows import dict_row

from app.agents.graph import GraphDeps, audit_new_events, build_graph, model_call_event
from app.agents.model_client import ChatModel
from app.agents.schemas import TurnResult
from app.agents.settings import AgentSettings
from app.agents.state import TurnContext
from app.db.connection import check_schema, check_url
from app.db.fixture_state import RetailStore
from app.db.schema import SCHEMA_MARKER
from app.guardrails.service import GuardrailService
from app.mcp.client import McpReadTools, McpToolError
from app.mcp.scope import ScopeError, ToolScope, mode_for_schema
from app.observability import telemetry as tel
from app.services.faults import FaultInjector

RECURSION_LIMIT = 25


def open_checkpointer(url: str, schema: str) -> tuple[psycopg.Connection, PostgresSaver]:
    """Long-lived connection pinned to the checkpoint schema; tables created by .setup()."""
    check_url(url)
    check_schema(schema)
    conn = psycopg.connect(url, autocommit=True, prepare_threshold=0, row_factory=dict_row,
                           options=f"-c search_path={schema}")
    conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
    conn.execute(sql.SQL("COMMENT ON SCHEMA {} IS {}").format(sql.Identifier(schema), sql.Literal(SCHEMA_MARKER)))
    saver = PostgresSaver(conn)
    saver.setup()
    return conn, saver


CHECKPOINT_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")


def delete_session_checkpoints(url: str, schema: str, customer_id: str, session_id: str) -> None:
    """Remove one session's conversations from a schema that other sessions share."""
    check_url(url)
    check_schema(schema)
    prefix = AgentRuntime.thread_id(customer_id, session_id, "")
    with psycopg.connect(url, autocommit=True) as conn, conn.transaction():
        for table in CHECKPOINT_TABLES:
            name = sql.Identifier(schema, table)
            if conn.execute("SELECT to_regclass(%s)", (f"{schema}.{table}",)).fetchone()[0] is not None:
                conn.execute(sql.SQL("DELETE FROM {} WHERE left(thread_id, %s) = %s").format(name), (len(prefix), prefix))


class AgentRuntime:
    """Not safe for concurrent turns on one instance; turns are serialized by a lock."""

    def __init__(self, store: RetailStore, model: ChatModel, guards: GuardrailService, settings: AgentSettings,
                 checkpoint_url: str, checkpoint_schema: str | None = None, *,
                 ask_item_condition: bool = False) -> None:
        self.store = store
        self.model = model
        self.guards = guards
        self.settings = settings
        self._conn, self.checkpointer = open_checkpointer(checkpoint_url,
                                                          checkpoint_schema or settings.checkpoint_schema)
        self._mcp: Any = None  # McpReadTools for the current (schema, customer, request) scope
        self._deps = GraphDeps(store, model, guards, settings, read_tools=self.read_tools,
                               ask_item_condition=ask_item_condition)
        self.graph = build_graph(self._deps).compile(checkpointer=self.checkpointer)
        self._lock = threading.Lock()

    def read_tools(self, customer_id: str, request_id: str) -> Any:
        """The connection to the standalone MCP server for this runtime's schema and the trusted
        customer/request. A new scope (e.g. a new request in the UI) closes the old connection and opens
        a new one. Scope errors surface as McpToolError (a failed tool call)."""
        try:
            scope = ToolScope(self.store.schema, customer_id, request_id, mode_for_schema(self.store.schema)).validate()
        except ScopeError:
            raise McpToolError("MCP_SCOPE_INVALID") from None
        if self._mcp is None or self._mcp.scope.key() != scope.key():
            if self._mcp is not None:
                self._mcp.close()
            self._mcp = McpReadTools(scope)
        return self._mcp

    def close(self) -> None:
        if self._mcp is not None:
            self._mcp.close()
            self._mcp = None
        self._conn.close()

    @staticmethod
    def thread_id(customer_id: str, session_id: str, request_id: str) -> str:
        return f"{customer_id}:{session_id}:{request_id}"

    def handle_customer_turn(self, *, session_id: str, authenticated_customer_id: str, request_id: str,
                             message: str, turn_number: int, confirmed_proposal_id: str | None = None,
                             confirmation_requested: bool = False,
                             injector: FaultInjector | None = None) -> TurnResult:
        turn_kind = "confirm" if (confirmation_requested or confirmed_proposal_id) else "chat"
        started = time.perf_counter()
        with tel.span("retail.customer_turn", {"retail.turn.kind": turn_kind,
                                                "retail.turn.number": turn_number}) as root:
            result = self._handle(session_id=session_id, authenticated_customer_id=authenticated_customer_id,
                                  request_id=request_id, message=message, turn_number=turn_number,
                                  confirmed_proposal_id=confirmed_proposal_id,
                                  confirmation_requested=confirmation_requested, injector=injector, root=root)
        tel.record_turn(time.perf_counter() - started, turn_kind, result.decision)
        tel.flush_turn()
        return result

    def _handle(self, *, session_id: str, authenticated_customer_id: str, request_id: str, message: str,
                turn_number: int, confirmed_proposal_id: str | None, confirmation_requested: bool,
                injector: FaultInjector | None, root: Any) -> TurnResult:
        with self._lock:
            ctx = TurnContext(message=message, injector=injector, otel_context=tel.current_context())
            if injector is not None:
                # Faults are applied at the business-tool boundary, before the turn runs.
                ctx.internal["faults_applied"] = injector.apply_before_turn(self.store, turn_number)
            self._deps.ctx = ctx
            config = {"configurable": {"thread_id": self.thread_id(authenticated_customer_id, session_id,
                                                                    request_id)},
                      "recursion_limit": RECURSION_LIMIT}
            try:
                state = self.graph.invoke({"customer_id": authenticated_customer_id, "request_id": request_id,
                                           "session_id": session_id, "turn_number": turn_number,
                                           "confirmed_proposal_id": confirmed_proposal_id,
                                           "confirmation_requested": confirmation_requested}, config)
            finally:
                self._deps.ctx = None
            result = TurnResult(
                decision=state["decision"], response=state["response"], proposal=state.get("proposal"),
                agent_events=[model_call_event(c) for c in ctx.model_calls],
                tool_events=ctx.tool_events, guardrail_events=ctx.guardrail_events,
                internal_status={**ctx.internal, "turn_number": turn_number})
            audit_new_events(self.store, ctx)
            self._annotate(root, result, ctx, request_id, confirmed_proposal_id)
            return result

    @staticmethod
    def _annotate(root: Any, result: TurnResult, ctx: TurnContext, request_id: str,
                  confirmed_proposal_id: str | None) -> None:
        """Bounded turn attributes and, when content capture is on, the masked message and the reply."""
        if root is None:
            return
        proposal_id = (result.proposal or {}).get("proposal_id") or confirmed_proposal_id
        attrs: dict[str, Any] = {
            "retail.decision": result.decision,
            "retail.response.source": result.internal_status.get("response_source", "unknown"),
            "retail.llm.calls": len(result.agent_events),
            "retail.tool.calls": sum(1 for e in result.tool_events if e.get("agent") == "retail_operations_agent"),
            "retail.timeout.injected": "post_commit_timeout" in result.internal_status,
            "retail.proposal_ref": tel.proposal_ref(proposal_id, request_id)}
        tel.set_attributes(root, attrs)
        # The masked text from the input guard; text the guard blocked is never captured.
        tel.capture(root, query=ctx.safe_message if ctx.internal.get("input_allowed") else "[blocked by the input guard]",
                    answer=result.response)
        trace_id = tel.trace_id_hex(root)
        if trace_id:
            result.internal_status["trace_id"] = trace_id

    def resolve_last_proposal(self, customer_id: str, session_id: str, request_id: str) -> str | None:
        """Most recent non-null proposal ID in this conversation (helper for trusted callers)."""
        snapshot = self.graph.get_state({"configurable": {"thread_id": self.thread_id(customer_id, session_id,
                                                                                      request_id)}})
        return (snapshot.values or {}).get("last_proposal_id")


def describe_configuration(model: Any, guards: GuardrailService) -> dict[str, Any]:
    info: dict[str, Any] = {"guardrails": guards.describe()}
    ready = getattr(model, "ready", None)
    info["model"] = ready() if callable(ready) else {"model": getattr(model, "model", "fake")}
    return info
