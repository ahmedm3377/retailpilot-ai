"""DemoService: persistent, isolated demo sessions over the shared AgentRuntime.

A sign-in session works on the persistent store and keeps its conversation in the shared
`retail_ui_checkpoints` schema. A test-automation session = one allowlisted fixture case copied
into its own `retail_ui_<id>` business schema plus a `retail_ui_cp_<id>` checkpoint schema, a trusted
customer and request, a turn counter and one FaultInjector (rebuilt from the
audit log after a restart). Mutations take the session's row lock (fail fast
if busy), replay stored responses for a repeated client_action_id, then wait
for the single process-wide turn slot: the local model and the guardrail
classifiers are used by one turn at a time.
"""
from __future__ import annotations

import hashlib
import secrets
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import psycopg
from psycopg.types.json import Jsonb
from pydantic import TypeAdapter

from app.agents.model_client import ChatModel
from app.agents.schemas import TurnResult
from app.agents.settings import AgentSettings
from app.api import views
from app.api.registry import Registry, SessionBusy
from app.api.schemas import (
    CaseOptionDTO, CommittedResultDTO, HistoryItemDTO, RequestStatusDTO, SessionDTO, TurnResponseDTO,
)
from app.db import repositories as repo
from app.db.demo_store import ensure_store, find_customer, store_schema
from app.db.fixture_state import RetailStore
from retail_domain.models import Case, Fault, StockChangeFault
from app.evaluation.serialization import audit_entries
from app.fixtures.loader import load_fixtures
from app.guardrails.service import GuardrailService
from app.services.agent_runtime import AgentRuntime, delete_session_checkpoints
from app.services.faults import FaultInjector
from app.services.recovery import reconcile_request
from app.services.transactions import cancel_proposal

# Fixed text for the Confirm button's turn. It must pass the input guardrail
# (tested): wording like "proposal ... confirmed" is flagged as a forged approval.
CONFIRM_MESSAGE = "Confirm button pressed."
UI_ACTOR = "ui_controller"


class ApiProblem(Exception):
    """A transport-level failure with a stable error code (not a business decision)."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


@dataclass
class Resources:
    model: ChatModel
    guards: GuardrailService
    settings: AgentSettings


LIVE_STORE = "live_store"  # scenario_id of a session on the persistent store
UPLOADED_CASE = "uploaded_case"  # scenario_id of a session on a case from the person's file
MAX_UPLOADED_CASE_CHARS = 1_000_000
SHARED_CHECKPOINTS = "retail_ui_checkpoints"  # conversations of every session on that store


def release_session_data(url: str, session: dict[str, Any], store: str, shared_checkpoints: str,
                         *, require_test_db: bool = False) -> None:
    """Remove what belongs to this session only. The persistent store and the shared checkpoint schema
    stay; RetailStore also refuses schemas without the app marker."""
    for schema in (session["business_schema"], session["checkpoint_schema"]):
        if schema == shared_checkpoints:
            delete_session_checkpoints(url, schema, session["customer_id"], session["session_id"])
        elif schema != store:
            RetailStore(url, schema, require_test_db=require_test_db).drop_schema()


def check_uploaded_case(case: Case) -> None:
    """Refuse what the store can't hold or the faults can't apply, with a reason the person can act on."""
    if len(case.model_dump_json()) > MAX_UPLOADED_CASE_CHARS:
        raise ApiProblem(400, "invalid_case", "That case file is too large.")
    products = {p.sku for p in case.initial_state.products}
    for order in case.initial_state.orders:
        for line in order.lines:
            if line.sku not in products:
                raise ApiProblem(400, "invalid_case", f"Order {order.order_id}, line {line.line_id}: "
                                                      f"SKU {line.sku} is not in the case's products.")
    for fault in case.faults:
        if isinstance(fault, StockChangeFault) and (
                fault.sku not in products or fault.before_turn < 1 or fault.new_stock < 0):
            raise ApiProblem(400, "invalid_case", f"The stock_change fault for {fault.sku} needs a SKU from the "
                                                  "case's products, before_turn of 1 or more and new_stock of 0 or more.")


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class DemoService:
    def __init__(self, database_url: str, fixtures_path: Path, resources: Callable[[], Resources | None], *,
                 registry_schema: str = "retail_ui_registry", require_test_db: bool = False,
                 max_runtimes: int = 4, turn_wait_s: float = 5.0, store_schema_name: str | None = None,
                 ask_item_condition: bool = True) -> None:
        self.url = database_url
        self.ask_item_condition = ask_item_condition
        self.require_test_db = require_test_db
        # The MCP server serves retail_ui_* only from the real database, so test sessions use its test schemas.
        self.session_prefix = "retail_t_ui" if require_test_db else "retail_ui"
        self.registry = Registry(database_url, registry_schema, require_test_db=require_test_db)
        self.registry.migrate()
        root, self.fixtures_sha256 = load_fixtures(fixtures_path)
        # Fixture cases serve only test automation, which asks for one by scenario_id. Sign-in uses the store.
        self.cases: dict[str, Case] = {c.case_id: c for c in root.cases}
        self.live_store = RetailStore(database_url, store_schema_name or store_schema(),
                                      require_test_db=require_test_db)
        ensure_store(self.live_store)
        self.live_checkpoints = f"{self.live_store.schema}_cp" if require_test_db else SHARED_CHECKPOINTS
        self._resources = resources
        self._runtimes: OrderedDict[str, AgentRuntime] = OrderedDict()
        self._injectors: dict[str, FaultInjector] = {}
        self._cache_lock = threading.Lock()
        self._in_use: set[str] = set()
        self._turn_slot = threading.Semaphore(1)
        self.max_runtimes = max_runtimes
        self.turn_wait_s = turn_wait_s

    # --- helpers -------------------------------------------------------------------------------

    def resources(self) -> Resources:
        res = self._resources()
        if res is None:
            raise ApiProblem(503, "dependency_unavailable", "The local model and safety checks are still loading.")
        return res

    def store(self, session: dict[str, Any]) -> RetailStore:
        return RetailStore(self.url, session["business_schema"], require_test_db=self.require_test_db)


    def authenticate(self, token: str | None) -> dict[str, Any]:
        if not token or len(token) > 200:
            raise ApiProblem(401, "no_session", "Start a demo session first.")
        session = self.registry.by_token(token_hash(token))
        if session is None:
            raise ApiProblem(401, "no_session", "This demo session is unknown or has ended.")
        return session

    def _runtime(self, session: dict[str, Any]) -> AgentRuntime:
        """Cached runtime (one PostgreSQL connection each); LRU-evicted and rebuilt from persisted state.

        Must be called OUTSIDE any registry transaction: building a runtime runs the
        checkpointer's setup (CREATE INDEX CONCURRENTLY), which waits for every open
        transaction in the database, including one held by the caller.
        """
        sid = session["session_id"]
        with self._cache_lock:
            runtime = self._runtimes.pop(sid, None)
            if runtime is None:
                res = self.resources()
                runtime = AgentRuntime(self.store(session), res.model, res.guards, res.settings, self.url,
                                       session["checkpoint_schema"], ask_item_condition=self.ask_item_condition)
            self._runtimes[sid] = runtime
            for old_sid in [k for k in self._runtimes if k not in self._in_use and k != sid]:
                if len(self._runtimes) <= self.max_runtimes:
                    break
                self._runtimes.pop(old_sid).close()
            return runtime

    def _acquire(self, session: dict[str, Any]) -> AgentRuntime:
        """Runtime + injector ready before the registry transaction; protected from eviction until released."""
        self.resources()  # 503 while models load or after they failed, even if a runtime is cached
        runtime = self._runtime(session)
        self._injector(session)
        with self._cache_lock:
            self._in_use.add(session["session_id"])
        return runtime

    def _release(self, session: dict[str, Any]) -> None:
        with self._cache_lock:
            self._in_use.discard(session["session_id"])

    def _injector(self, session: dict[str, Any]) -> FaultInjector:
        sid = session["session_id"]
        with self._cache_lock:
            if sid not in self._injectors:
                # After a restart, rebuild one-shot fault state from the audit log: never re-apply.
                self._injectors[sid] = FaultInjector.restore(self._faults(session), self.store(session).audit_log())
            return self._injectors[sid]

    def _faults(self, session: dict[str, Any]) -> list[Fault]:
        if session["case_faults"] is not None:
            return TypeAdapter(list[Fault]).validate_python(session["case_faults"])
        case = self.cases.get(session["scenario_id"])
        return case.faults if case else []

    def close(self) -> None:
        with self._cache_lock:
            for runtime in self._runtimes.values():
                runtime.close()
            self._runtimes.clear()

    # --- sessions ----------------------------------------------------------------------------

    def case_options(self) -> list[CaseOptionDTO]:
        return [CaseOptionDTO(case_id=c.case_id, customer_id=c.customer_id) for c in self.cases.values()]

    def create_session(self, customer_id: str, scenario_id: str | None = None,
                       uploaded_case: Case | None = None) -> tuple[str, SessionDTO]:
        """Demo sign-in by customer ID (no password: a stand-in for a real login). The session is bound
        to that customer for its whole life; nothing in the chat can change it.

        The customer is looked up in the persistent store, and the session works on that store.
        scenario_id or uploaded_case instead gives an isolated copy of that one case, in its own schema."""
        if scenario_id is not None and uploaded_case is not None:
            raise ApiProblem(400, "invalid_case", "Choose a listed case or upload one, not both.")
        suffix = uuid.uuid4().hex[:12]
        case = uploaded_case
        checkpoint_schema = f"{self.session_prefix}_cp_{suffix}"
        if uploaded_case is not None:
            check_uploaded_case(uploaded_case)
        elif scenario_id is not None:
            case = self.cases.get(scenario_id)
            if case is None:
                raise ApiProblem(400, "unknown_scenario", "Choose one of the listed scenarios.")
        if case is None:
            found, business_schema = find_customer(self.live_store, customer_id), self.live_store.schema
            checkpoint_schema = self.live_checkpoints
        else:
            found = next((k for k in {case.customer_id, *(o.customer_id for o in case.initial_state.orders)}
                          if k.casefold() == customer_id.casefold()), None)
            business_schema = f"{self.session_prefix}_{suffix}"
        if found is None:
            raise ApiProblem(404, "unknown_customer", "We could not find that customer ID. Check it and try again.")
        token = secrets.token_urlsafe(32)
        session = {"session_id": f"S-{suffix}", "token_sha256": token_hash(token),
                   "scenario_id": UPLOADED_CASE if uploaded_case else case.case_id if case else LIVE_STORE,
                   "customer_id": found, "active_request_id": f"REQ-{uuid.uuid4().hex[:12]}",
                   "business_schema": business_schema, "checkpoint_schema": checkpoint_schema,
                   "case_faults": (Jsonb([f.model_dump(mode="json") for f in uploaded_case.faults])
                                   if uploaded_case else None)}
        if case:
            store = RetailStore(self.url, business_schema, require_test_db=self.require_test_db)
            store.create_schema()
            try:
                store.reset_from_initial_state(case.initial_state)  # deep copy; one case per session
            except (ValueError, psycopg.IntegrityError, psycopg.DataError) as error:
                store.drop_schema()
                if uploaded_case is None:
                    raise
                detail = str(error) if isinstance(error, ValueError) else (
                    "it has a duplicate ID or a negative price, stock or amount")
                raise ApiProblem(400, "invalid_case", f"That case can't be loaded: {detail}.") from error
        self.registry.create(**session)
        with self._cache_lock:
            self._injectors[session["session_id"]] = FaultInjector(case.faults if case else [])  # fresh: none applied
        row = self.registry.by_id(session["session_id"])
        if self._resources() is not None:
            self._runtime(row)  # build the checkpointer now, outside any registry transaction
        return token, self.session_view(row)

    def session_view(self, session: dict[str, Any]) -> SessionDTO:
        store = self.store(session)
        return SessionDTO(
            session_id=session["session_id"], scenario_id=session["scenario_id"],
            customer_id=session["customer_id"], request_id=session["active_request_id"],
            turn_number=session["turn_number"], status=session["status"],
            latest_proposal=views.proposal_dto(views.latest_proposal_row(store, session["active_request_id"])),
            orders=views.orders_dto(store.load_state(), session["customer_id"]),
            history=[HistoryItemDTO(**{k: h[k] for k in HistoryItemDTO.model_fields})
                     for h in self.registry.history(session["session_id"])])

    def end_session(self, session: dict[str, Any], session_id: str) -> None:
        if session_id != session["session_id"]:
            raise ApiProblem(404, "not_found", "Session not found.")
        with self._cache_lock:
            runtime = self._runtimes.pop(session_id, None)
            self._injectors.pop(session_id, None)
        if runtime is not None:
            runtime.close()
        self.registry.mark_ended(session_id)
        self.drop_schemas(session)

    def drop_schemas(self, session: dict[str, Any]) -> None:
        release_session_data(self.url, session, self.live_store.schema, self.live_checkpoints,
                             require_test_db=self.require_test_db)

    # --- status and audit ------------------------------------------------------------------

    def request_status(self, session: dict[str, Any]) -> RequestStatusDTO:
        store = self.store(session)
        request_id = session["active_request_id"]
        status = reconcile_request(store, session["customer_id"], request_id)
        latest = views.proposal_dto(views.latest_proposal_row(store, request_id))
        if status.state == "completed":
            r = status.result
            return RequestStatusDTO(request_id=request_id, state="completed", latest_proposal=latest,
                                    result=CommittedResultDTO(remedy=r.remedy, order_id=r.order_id,
                                                              line_id=r.line_id,
                                                              refund_amount_cents=r.refund_amount_cents,
                                                              original_sku=r.original_sku,
                                                              replacement_sku=r.replacement_sku))
        state = ("pending_confirmation" if latest and latest.status == "pending"
                 else "proposal_invalidated" if latest else "none")
        return RequestStatusDTO(request_id=request_id, state=state, result=None, latest_proposal=latest)

    def audit(self, session: dict[str, Any]) -> list[views.AuditEventDTO]:
        """Only what this session did. Sign-in sessions share the store's audit log with every other
        session, so its rows are kept only inside this session's recorded event ranges."""
        bounds = [(h["turn_number"], h["first_event_id"], h["last_event_id"])
                  for h in self.registry.history(session["session_id"]) if h["first_event_id"] is not None]
        own = [row for row in self.store(session).audit_log()
               if any(lo < row["event_id"] <= hi for _, lo, hi in bounds)
               and (row["arguments"] or {}).get("customer_id", session["customer_id"]) == session["customer_id"]]
        return [views.audit_event_dto(e) for e in audit_entries(own, bounds)]

    # --- mutations -------------------------------------------------------------------------

    def _mutate(self, session: dict[str, Any], client_action_id: str,
                work: Callable[[Any, dict[str, Any]], tuple[int, Any]]) -> tuple[int, Any]:
        """Session row lock -> idempotency check -> work -> store response, in one registry transaction."""
        with self.registry.connect() as conn, conn.transaction():
            try:
                locked = self.registry.lock(conn, session["session_id"])
            except SessionBusy as error:
                raise ApiProblem(409, "turn_in_progress",
                                 "Another action for this session is still running. Check the status.") from error
            stored = self.registry.stored_action(conn, session["session_id"], client_action_id)
            if stored is not None:
                return stored
            status_code, body = work(conn, locked)
            self.registry.store_action(conn, session["session_id"], client_action_id, status_code, body)
            return status_code, body

    def _run_turn(self, conn: Any, session: dict[str, Any], runtime: AgentRuntime, *, kind: str, message: str,
                  confirmed_proposal_id: str | None) -> TurnResponseDTO:
        if not self._turn_slot.acquire(timeout=self.turn_wait_s):
            raise ApiProblem(503, "busy", "The local model is busy with another turn. Try again in a few seconds.")
        try:
            store = self.store(session)
            injector = self._injector(session)
            turn_number = session["turn_number"] + 1
            before = _max_event_id(store)
            start = time.perf_counter()
            result: TurnResult = runtime.handle_customer_turn(
                session_id=session["session_id"], authenticated_customer_id=session["customer_id"],
                request_id=session["active_request_id"], message=message, turn_number=turn_number,
                confirmed_proposal_id=confirmed_proposal_id, confirmation_requested=kind == "confirm",
                injector=injector)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            after = _max_event_id(store)
        finally:
            self._turn_slot.release()
        blocked = any(e.get("outcome") == "blocked" and e.get("gate") == "input" for e in result.guardrail_events)
        self.registry.update(conn, session["session_id"], turn_number=turn_number)
        self.registry.add_history(conn, session["session_id"], kind=kind, turn_number=turn_number,
                                  request_id=session["active_request_id"],
                                  message=None if kind == "confirm" else
                                  ("[message blocked by a safety check]" if blocked else message),
                                  decision=result.decision, response=result.response, first_event_id=before,
                                  last_event_id=after, elapsed_ms=elapsed_ms)
        tokens_p = [e["prompt_tokens"] for e in result.agent_events]
        tokens_c = [e["completion_tokens"] for e in result.agent_events]
        status = self.request_status({**session, "turn_number": turn_number})
        commit = result.internal_status.get("commit") or {}
        return TurnResponseDTO(
            decision=result.decision, response=result.response,
            proposal=views.proposal_dto(views.latest_proposal_row(store, session["active_request_id"]))
            if result.proposal else None,
            request_id=session["active_request_id"], request_status=status.state, turn_number=turn_number,
            elapsed_ms=elapsed_ms, llm_calls=len(result.agent_events),
            prompt_tokens=None if not tokens_p or None in tokens_p else sum(tokens_p),
            completion_tokens=None if not tokens_c or None in tokens_c else sum(tokens_c),
            guardrail_blocked=blocked, replayed=bool(commit.get("replayed")),
            timeout_recovered="post_commit_timeout" in result.internal_status,
            correlation_id=f"{session['session_id']}-t{turn_number}")

    def send_message(self, session: dict[str, Any], message: str, client_action_id: str) -> tuple[int, Any]:
        runtime = self._acquire(session)

        def work(conn: Any, locked: dict[str, Any]) -> tuple[int, Any]:
            locked = self._next_request_if_completed(conn, locked)
            dto = self._run_turn(conn, locked, runtime, kind="message", message=message, confirmed_proposal_id=None)
            return 200, dto.model_dump(mode="json")
        try:
            return self._mutate(session, client_action_id, work)
        finally:
            self._release(session)

    def confirm(self, session: dict[str, Any], proposal_id: str, client_action_id: str) -> tuple[int, Any]:
        runtime = self._acquire(session)

        def work(conn: Any, locked: dict[str, Any]) -> tuple[int, Any]:
            with self.store(locked).connect() as bconn:
                row = repo.get_proposal(bconn, proposal_id)
            if (row is None or row["customer_id"] != locked["customer_id"]
                    or row["request_id"] != locked["active_request_id"]):
                raise ApiProblem(409, "proposal_mismatch", "That proposal does not belong to your current request.")
            if row["status"] == "invalidated":
                raise ApiProblem(409, "proposal_not_pending", "That proposal is no longer valid. Ask for a new one.")
            # pending -> commit; completed -> the runtime returns the stored original result (replay).
            dto = self._run_turn(conn, locked, runtime, kind="confirm", message=CONFIRM_MESSAGE,
                                 confirmed_proposal_id=proposal_id)
            return 200, dto.model_dump(mode="json")
        try:
            return self._mutate(session, client_action_id, work)
        finally:
            self._release(session)

    def cancel(self, session: dict[str, Any], proposal_id: str, client_action_id: str) -> tuple[int, Any]:
        def work(conn: Any, locked: dict[str, Any]) -> tuple[int, Any]:
            store = self.store(locked)
            before = _max_event_id(store)
            result = cancel_proposal(store, locked["customer_id"], locked["active_request_id"],
                                     proposal_id, actor=UI_ACTOR)
            if result.status != "cancelled":
                raise ApiProblem(409, "proposal_not_pending", "That proposal can no longer be cancelled.")
            self.registry.add_history(conn, locked["session_id"], kind="cancel",
                                      request_id=locked["active_request_id"], first_event_id=before,
                                      last_event_id=_max_event_id(store))
            return 200, self.request_status(locked).model_dump(mode="json")
        return self._mutate(session, client_action_id, work)

    def _start_request(self, conn: Any, session: dict[str, Any]) -> dict[str, Any]:
        request_id = f"REQ-{uuid.uuid4().hex[:12]}"
        self.registry.update(conn, session["session_id"], active_request_id=request_id)
        self.registry.add_history(conn, session["session_id"], kind="new_request", request_id=request_id)
        return {**session, "active_request_id": request_id}

    def _next_request_if_completed(self, conn: Any, session: dict[str, Any]) -> dict[str, Any]:
        """One request allows one transaction. In the chat, a message after it completed is the customer's
        next request, so it gets a fresh request ID (Confirm on the old proposal still replays)."""
        status = reconcile_request(self.store(session), session["customer_id"], session["active_request_id"])
        return self._start_request(conn, session) if status.state == "completed" else session


def _max_event_id(store: RetailStore) -> int:
    with store.connect() as conn:
        return conn.execute("SELECT coalesce(max(event_id), 0) AS n FROM audit_events").fetchone()["n"]
