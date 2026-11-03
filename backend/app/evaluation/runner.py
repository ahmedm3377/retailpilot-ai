"""Run contract cases through the real AgentRuntime, one isolated case at a time.

The runner only does fixture setup, trusted confirmation resolution, turn
scheduling, measurement and output mapping. Every decision comes from the same
AgentRuntime (two LLM agents, guardrails, trusted services) the UI will use.

Per case run: a fresh `retail_eval_<id>` business schema and a fresh
`retail_eval_cp_<id>` checkpoint schema in the test database, one AgentRuntime,
one FaultInjector and one session; all dropped afterwards. The model and the
guardrail validators stay loaded across cases (warm).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import psycopg

from app.agents.model_client import ChatModel, ModelClient, ModelUnavailableError
from app.agents.schemas import CustomerMessage, json_schema
from app.agents.settings import AgentSettings
from app.db.connection import check_url, test_database_url
from app.db.fixture_state import RetailStore, StateNotLoadedError
from retail_domain.models import Case, FixtureRoot
from app.evaluation import environment
from app.evaluation.metrics import GPU_DEVICE_SCOPE, RAM_SCOPE, VRAM_SCOPE, ResourceSampler, Timer
from app.evaluation.serialization import audit_entries, turn_output
from app.guardrails.service import GuardrailService
from app.observability import telemetry as tel
from app.mcp.client import McpToolError, check_server, unavailable_message
from app.services.agent_runtime import AgentRuntime
from app.services.faults import FaultInjector


class InfrastructureError(RuntimeError):
    """The run cannot produce trustworthy results (database, model or validators unavailable)."""


@dataclass
class EvalContext:
    model: ChatModel
    guards: GuardrailService
    settings: AgentSettings
    database_url: str
    model_info: dict[str, Any]
    cold_start_ms: float | None = None
    notes: list[str] = field(default_factory=list)


def postgres_version(url: str) -> str:
    try:
        with psycopg.connect(url, connect_timeout=5) as conn:
            return conn.execute("SHOW server_version").fetchone()[0]
    except psycopg.Error as error:
        raise InfrastructureError(f"PostgreSQL unreachable: {type(error).__name__}") from error


def prepare(settings: AgentSettings | None = None, database_url: str | None = None) -> EvalContext:
    """Load validators, check the model server and warm the model; the time taken is the cold start."""
    settings = settings or AgentSettings.from_env()
    tel.init()  # no-op unless RETAIL_OTEL_ENABLED; once per process
    url = database_url or test_database_url()
    check_url(url, require_test_db=True)
    postgres_version(url)
    try:
        check_server()
    except McpToolError as error:
        raise InfrastructureError(unavailable_message(error)) from error
    with Timer() as timer:
        guards = GuardrailService(settings)
        if guards.unavailable:
            raise InfrastructureError(f"guardrail validators unavailable: {sorted(guards.unavailable)}")
        model = ModelClient(settings)
        try:
            info = model.ready()
            # One tiny call so the first case is not charged with loading the model into memory.
            model.chat_json("warmup", [{"role": "user", "content": "Reply with the message: ready"}],
                            json_schema(CustomerMessage))
        except ModelUnavailableError as error:
            raise InfrastructureError(str(error)) from error
    notes = ["cold_start_ms = validator load + model server check + one warm-up call, which includes loading the "
             "model if the engine had not already; cases run warm"]
    return EvalContext(model, guards, settings, url, info, round(timer.elapsed_ms, 1), notes)


def _max_event_id(store: RetailStore) -> int:
    with store.connect() as conn:
        return conn.execute("SELECT coalesce(max(event_id), 0) AS n FROM audit_events").fetchone()["n"]


def _sum_or_none(values: list[int | None]) -> int | None:
    return None if not values or any(v is None for v in values) else sum(values)


FAILED_TURN = {"decision": "error", "proposal": None,
               "response": "The system could not process this request. Please check its status before trying again."}


def _failed_result(case: Case, turns: list[dict[str, Any]], audit: list[dict[str, Any]],
                   final_state: dict[str, Any], elapsed_ms: float, error: str) -> dict[str, Any]:
    """A contract result for a run that broke: finished turns as they were, the rest marked as errors.

    final_state is what the store last reported, so a commit made before the failure is not hidden.
    """
    return {"case_id": case.case_id, "turns": [*turns, *([FAILED_TURN] * (len(case.turns) - len(turns)))],
            "action_audit": audit, "final_state": final_state, "run_failed": True, "error": error,
            "metrics": {"elapsed_ms": round(elapsed_ms, 1), "peak_ram_mb": None, "peak_vram_mb": None,
                        "notes": "memory was not reported because the run failed"}}


def failed_document(root: FixtureRoot, input_sha256: str, error: str) -> dict[str, Any]:
    """The results document when nothing could run (for example, the model or database is down)."""
    settings = AgentSettings.from_env()
    results = [_failed_result(case, [], [], case.initial_state.model_dump(mode="json"), 0.0, error)
               for case in root.cases]
    return {"contract_version": 1, "input_sha256": input_sha256, "hardware": environment.hardware(),
            "configuration": {"status": "unavailable", "error": error, "engine": settings.engine or "unavailable",
                              "model": settings.model or "unavailable"},
            "results": results, "run": {"cases": len(results), "notes": [error]}}


def run_case(ctx: EvalContext, case: Case, *, keep_schemas: bool = False,
             include_failed: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one case from its initial state; returns (contract result, raw diagnostics record).

    A failure raises, so evidence runs stop, unless include_failed asks for a failed result instead.
    """
    run_id = uuid.uuid4().hex[:10]
    store = RetailStore(ctx.database_url, f"retail_eval_{run_id}", require_test_db=True)
    checkpoint_schema = f"retail_eval_cp_{run_id}"
    raw: dict[str, Any] = {"case_id": case.case_id, "run_id": run_id,
                           "schemas": [store.schema, checkpoint_schema], "turns": []}
    runtime = None
    turns: list[dict[str, Any]] = []
    bounds: list[tuple[int, int, int]] = []
    started = time.perf_counter()
    try:
        store.create_schema()
        store.reset_from_initial_state(case.initial_state)
        runtime = AgentRuntime(store, ctx.model, ctx.guards, ctx.settings, ctx.database_url, checkpoint_schema)
        injector = FaultInjector(case.faults)
        last_proposal_id: str | None = None
        turn_ms, agent_events = [], []
        with ResourceSampler(ctx.settings.server_process_names()) as sampler:
            with Timer() as timer:
                for number, turn in enumerate(case.turns, start=1):
                    # Contract: confirm_proposal "last" = latest non-null proposal from an earlier turn.
                    confirming = turn.confirm_proposal is not None
                    resolved = last_proposal_id if turn.confirm_proposal == "last" else None
                    before = _max_event_id(store)
                    start = time.perf_counter_ns()
                    result = runtime.handle_customer_turn(
                        session_id=f"eval-{run_id}", authenticated_customer_id=case.customer_id,
                        request_id=case.request_id, message=turn.message, turn_number=number,
                        confirmed_proposal_id=resolved if confirming else None,
                        confirmation_requested=confirming, injector=injector)
                    turn_ms.append(round((time.perf_counter_ns() - start) / 1e6, 1))
                    if result.proposal is not None:
                        last_proposal_id = result.proposal["proposal_id"]
                    bounds.append((number, before, _max_event_id(store)))
                    turns.append(turn_output(result))
                    agent_events += result.agent_events
                    raw["turns"].append({"turn": number, "decision": result.decision,
                                         "confirmed_proposal_id": resolved if confirming else None,
                                         "internal_status": result.internal_status})
        final_state = store.export_final_state()
        audit = audit_entries(store.audit_log(), bounds)
    except Exception as error:  # noqa: BLE001 - reported, never hidden
        message = f"{case.case_id}: {type(error).__name__}: {error}"
        if include_failed:
            try:
                state, audit = store.export_final_state(), audit_entries(store.audit_log(), bounds)
            except Exception:  # noqa: BLE001 - the store itself is gone
                state, audit = case.initial_state.model_dump(mode="json"), []
            raw.update(decisions=[t["decision"] for t in turns], error=message,
                       metrics={"elapsed_ms": round((time.perf_counter() - started) * 1000, 1)})
            return _failed_result(case, turns, audit, state, raw["metrics"]["elapsed_ms"], message), raw
        if isinstance(error, (psycopg.Error, StateNotLoadedError)):
            raise InfrastructureError(f"{case.case_id}: database failure ({type(error).__name__}: {error})") from error
        raise
    finally:
        if runtime is not None:
            runtime.close()
        if not keep_schemas:
            _drop_quietly(raw, store, RetailStore(ctx.database_url, checkpoint_schema, require_test_db=True))

    peaks = sampler.peaks
    llm_ms = [e["latency_ms"] for e in agent_events]
    completion = _sum_or_none([e["completion_tokens"] for e in agent_events])
    metrics = {
        "elapsed_ms": round(timer.elapsed_ms, 1),
        "peak_ram_mb": peaks.peak_ram_mb,
        "peak_vram_mb": peaks.peak_vram_mb,
        "turn_ms": turn_ms,
        "llm_calls": len(agent_events),
        "llm_calls_failed": sum(1 for e in agent_events if not e["ok"]),
        "llm_wall_ms": round(sum(llm_ms), 1),
        "prompt_tokens": _sum_or_none([e["prompt_tokens"] for e in agent_events]),
        "completion_tokens": completion,
        "completion_tokens_per_llm_second": (round(completion / (sum(llm_ms) / 1000), 1)
                                             if completion is not None and sum(llm_ms) > 0 else None),
        "peak_python_rss_mb": peaks.peak_python_rss_mb,
        "peak_model_server_rss_mb": peaks.peak_model_server_rss_mb,
        "peak_mcp_server_rss_mb": peaks.peak_mcp_server_rss_mb,
        "peak_gpu_device_used_mb": peaks.peak_gpu_device_used_mb,
        "resource_samples": peaks.samples,
        "metric_scope": {"elapsed_ms": "first turn start to last turn end (excludes schema setup/teardown)",
                         "peak_ram_mb": RAM_SCOPE, "peak_vram_mb": VRAM_SCOPE,
                         "peak_gpu_device_used_mb": GPU_DEVICE_SCOPE,
                         "completion_tokens_per_llm_second": "engine-reported completion tokens / summed LLM call "
                                                             "wall time (includes request overhead)"},
        "notes": peaks.notes,
    }
    raw.update(decisions=[t["decision"] for t in turns], metrics=metrics)
    result = {"case_id": case.case_id, "turns": turns, "action_audit": audit, "final_state": final_state,
              "metrics": metrics}
    return result, raw


def _drop_quietly(raw: dict[str, Any], *stores: RetailStore) -> None:
    for store in stores:
        try:
            store.drop_schema()
        except Exception as error:  # noqa: BLE001 - cleanup must not hide the original error
            raw.setdefault("cleanup_errors", []).append(f"{store.schema}: {type(error).__name__}")


def run_fixture(ctx: EvalContext, root: FixtureRoot, input_sha256: str, *, keep_schemas: bool = False,
                include_failed: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """All cases, sequentially, in input order; returns (results document, raw records)."""
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results, raws = [], []
    for case in root.cases:
        result, raw = run_case(ctx, case, keep_schemas=keep_schemas, include_failed=include_failed)
        results.append(result)
        raws.append(raw)
    document = {
        "contract_version": 1,
        "input_sha256": input_sha256,
        "hardware": environment.hardware(),
        "configuration": environment.configuration(ctx.model_info, ctx.settings, ctx.guards.describe(),
                                                   ctx.database_url, postgres_version(ctx.database_url)),
        "results": results,
        "run": {"started_at": started, "cold_start_ms": ctx.cold_start_ms, "cases": len(results),
                "notes": ctx.notes + (["schemas kept: " + ", ".join(s for r in raws for s in r["schemas"])]
                                      if keep_schemas else [])},
    }
    return document, raws
