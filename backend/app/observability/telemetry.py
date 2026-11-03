"""Telemetry facade: private OTel providers, safe spans, bounded metrics.

Design:
- Our TracerProvider/MeterProvider are PRIVATE (never set as the global OTel
  providers). Libraries that use the global API (Guardrails, LiteLLM,
  LangChain) keep no-op providers, so their internal spans, which could carry
  raw validator input, never reach our exporter. Context still propagates.
- Everything is best effort: when telemetry is off every call is a no-op, and
  a telemetry error is swallowed, never raised into business code.
- Spans never record exception messages or stack traces (only error.type).
- Metric labels pass through fixed allowlists; unknown values become "other".
- Spans carry business content only through capture(), under retail.content.*, and only when
  RETAIL_OTEL_CAPTURE_CONTENT is on. Customer text is captured only after the input guard has masked it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from app.observability.config import TelemetrySettings

log = logging.getLogger("retail.telemetry")

AGENT_ROLES = {"customer_resolution_agent": "customer_resolution", "retail_operations_agent": "retail_operations",
               "customer_resolution": "customer_resolution", "retail_operations": "retail_operations"}
TOOLS = {"get_business_context", "lookup_customer_orders", "get_order_line", "check_return_eligibility", "get_exchange_options",
         "check_inventory", "get_request_status"}
DECISIONS = {"propose", "clarify", "deny", "completed", "error"}
TURN_KINDS = {"chat", "confirm"}
GATES = {"input", "tool_context", "agent_action", "output"}
VALIDATORS = {"sanitization", "TaskInjection", "DetectJailbreak", "ToxicLanguage", "DetectPII", "consistency",
              "size_limit", "Schema[CustomerStep]", "Schema[CustomerHandoff]", "Schema[OperationsAction]",
              "Schema[CustomerMessage]"}
GUARD_OUTCOMES = {"passed", "sanitized", "flagged", "blocked", "validation_error", "validator_unavailable"}
REMEDIES = {"refund", "exchange", "none"}
BUSINESS_OUTCOMES = {"committed", "denied", "replayed", "reconciled", "conflict", "error"}
OK_OUTCOMES = {"ok", "error", "refused", "output_rejected", "failed"}
TRANSPORTS = {"mcp_http"}
STEPS = {"load_context", "guard_input", "customer_agent", "check_handoff", "operations_agent", "create_proposal",
         "commit", "present"}
COMPONENTS = {"app", "model_server", "mcp_server"}

BUCKETS = {
    "retail.turn.duration": [0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 60],
    "retail.llm.call.duration": [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 20],
    "retail.llm.tokens": [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560],
    "retail.llm.output_token_rate": [5, 10, 20, 30, 40, 50, 60, 80, 100],
    "retail.guardrail.duration": [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],
    "retail.mcp.startup.duration": [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60],
    "retail.step.duration": [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 30],
    "retail.tool.duration": [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30],
}


def label(value: Any, allowed: set[str]) -> str:
    return value if isinstance(value, str) and value in allowed else "other"


def role(value: str | None) -> str:
    return AGENT_ROLES.get(value or "", "other")


class _Telemetry:
    def __init__(self, settings: TelemetrySettings, span_exporter: Any = None, metric_reader: Any = None) -> None:
        # The SDK honours OTEL_SDK_DISABLED; apply_offline_defaults() only sets it when telemetry is off.
        os.environ["OTEL_SDK_DISABLED"] = "false"
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

        self.settings = settings
        # A random per-process instance ID keeps each process's cumulative counters in their own series.
        self.instance_id = uuid.uuid4().hex[:12]
        resource = Resource.create({"service.name": settings.service_name, "service.namespace": "hp-retail",
                                    "service.instance.id": self.instance_id,
                                    "deployment.environment": "local-demo"})
        self.tracer_provider = TracerProvider(resource=resource)
        if span_exporter is None:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            span_exporter = OTLPSpanExporter(endpoint=f"{settings.endpoint}/v1/traces",
                                             timeout=settings.export_timeout_s)
            # Sent once per turn (flush_turn), so that a trace reaches Tempo whole: its search can miss spans of
            # a trace that arrived in a second batch. The long delay only covers anything left over.
            self.tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter, max_queue_size=2048,
                                                                       schedule_delay_millis=60000,
                                                                       export_timeout_millis=2000))
        else:
            self.tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        self.tracer = self.tracer_provider.get_tracer("retail-desk")
        if metric_reader is None:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            metric_reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=f"{settings.endpoint}/v1/metrics", timeout=settings.export_timeout_s),
                export_interval_millis=settings.export_interval_ms, export_timeout_millis=2000)
        views = [View(instrument_name=name, aggregation=ExplicitBucketHistogramAggregation(bounds))
                 for name, bounds in BUCKETS.items()]
        self.meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader], views=views)
        meter = self.meter_provider.get_meter("retail-desk")
        self.turn_duration = meter.create_histogram("retail.turn.duration", unit="s",
                                                    description="Customer turn wall time (chat or confirmation)")
        self.llm_duration = meter.create_histogram("retail.llm.call.duration", unit="s",
                                                   description="One LiteLLM call to the model server")
        self.llm_tokens = meter.create_histogram("retail.llm.tokens", unit="{token}",
                                                 description="Tokens per LLM call as reported by the engine via LiteLLM")
        self.llm_rate = meter.create_histogram("retail.llm.output_token_rate", unit="{token}/s",
                                               description="Output tokens / call wall time (effective_call)")
        self.tool_calls = meter.create_counter("retail.tool.calls", description="Read-only tool executions")
        self.tool_duration = meter.create_histogram("retail.tool.duration", unit="s",
                                                    description="One read-only tool call to the MCP server")
        self.step_duration = meter.create_histogram("retail.step.duration", unit="s",
                                                    description="One step of a customer turn")
        self._system_gauges(meter)
        self.guard_checks = meter.create_counter("retail.guardrail.checks", description="Guardrail verdicts")
        self.guard_duration = meter.create_histogram("retail.guardrail.duration", unit="s",
                                                     description="Guardrail validator time")
        self.business = meter.create_counter("retail.business.outcomes",
                                             description="Verified business outcomes of confirmation turns")
        self.mcp_startup = meter.create_histogram("retail.mcp.startup.duration", unit="s",
                                                  description="Connect + initialize + tools/list against the MCP server")
        self.salt = os.environ.get("RETAIL_OTEL_REF_SALT") or secrets.token_hex(16)

    def _system_gauges(self, meter: Any) -> None:
        from opentelemetry.metrics import Observation

        from app.observability.system import SystemReadings
        readings = SystemReadings()

        def value(key: str) -> Any:
            def callback(_options: Any) -> list[Any]:
                try:
                    reading = readings.read().values.get(key)
                except Exception:  # noqa: BLE001 - a missing reading stays missing
                    return []
                return [] if reading is None else [Observation(reading)]
            return callback

        def rss(_options: Any) -> list[Any]:
            try:
                return [Observation(mb, {"component": label(c, COMPONENTS)}) for c, mb in readings.read().rss.items()]
            except Exception:  # noqa: BLE001
                return []

        meter.create_observable_gauge("retail.system.cpu.percent", [value("cpu")], description="Whole-machine CPU use")
        meter.create_observable_gauge("retail.system.memory.percent", [value("memory")],
                                      description="Whole-machine RAM use")
        meter.create_observable_gauge("retail.gpu.utilization.percent", [value("gpu_utilization")],
                                      description="Whole-GPU utilization from nvidia-smi")
        meter.create_observable_gauge("retail.gpu.memory.mb", [value("gpu_memory")],
                                      description="Whole-GPU memory in use from nvidia-smi")
        meter.create_observable_gauge("retail.process.rss.mb", [rss], description="RAM of each part of the system")

    def shutdown(self) -> None:
        for provider in (self.tracer_provider, self.meter_provider):
            try:
                provider.force_flush(timeout_millis=3000)
                provider.shutdown()
            except Exception:  # noqa: BLE001 - best effort
                log.warning("telemetry shutdown failed", exc_info=False)


_STATE: _Telemetry | None = None


def init(settings: TelemetrySettings | None = None, *, span_exporter: Any = None,
         metric_reader: Any = None) -> bool:
    """Initialize once per process when enabled; returns whether telemetry is active."""
    global _STATE
    if _STATE is not None:
        return True
    settings = settings or TelemetrySettings.from_env()
    if not settings.enabled:
        return False
    try:
        _STATE = _Telemetry(settings, span_exporter, metric_reader)
    except Exception:  # noqa: BLE001 - telemetry must never stop the app
        log.warning("telemetry init failed; continuing without telemetry", exc_info=False)
        _STATE = None
    return _STATE is not None


def shutdown() -> None:
    global _STATE
    state, _STATE = _STATE, None
    if state is not None:
        state.shutdown()


def active() -> bool:
    return _STATE is not None


def settings() -> TelemetrySettings | None:
    return _STATE.settings if _STATE else None


def instance_id() -> str | None:
    return _STATE.instance_id if _STATE else None


def _attrs(attributes: dict[str, Any] | None) -> dict[str, Any]:
    return {k: v for k, v in (attributes or {}).items() if isinstance(v, (str, bool, int, float))}


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None, *, server: bool = False) -> Iterator[Any]:
    """A child of the current span (or a root). Yields None when telemetry is off."""
    state = _STATE
    if state is None:
        yield None
        return
    try:
        from opentelemetry.trace import SpanKind
        manager = state.tracer.start_as_current_span(
            name, kind=SpanKind.SERVER if server else SpanKind.INTERNAL, attributes=_attrs(attributes),
            record_exception=False, set_status_on_exception=False)
        current = manager.__enter__()
    except Exception:  # noqa: BLE001
        yield None
        return
    try:
        yield current
    except BaseException as error:
        _mark_error(current, type(error).__name__)
        manager.__exit__(None, None, None)
        raise
    else:
        manager.__exit__(None, None, None)


def _mark_error(current: Any, error_type: str) -> None:
    try:
        from opentelemetry.trace import Status, StatusCode
        current.set_status(Status(StatusCode.ERROR))
        current.set_attribute("error.type", error_type)
    except Exception:  # noqa: BLE001
        pass


def set_attributes(current: Any, attributes: dict[str, Any]) -> None:
    if current is None:
        return
    try:
        current.set_attributes(_attrs(attributes))
    except Exception:  # noqa: BLE001
        pass


def mark_error(current: Any, error_type: str) -> None:
    if current is not None:
        _mark_error(current, error_type)


def add_event(current: Any, name: str, attributes: dict[str, Any]) -> None:
    if current is None:
        return
    try:
        current.add_event(name, _attrs(attributes))
    except Exception:  # noqa: BLE001
        pass


def timed_child(name: str, start_ns: int, end_ns: int, attributes: dict[str, Any]) -> None:
    """A finished child span with explicit times (used to place guardrail validator steps)."""
    state = _STATE
    if state is None:
        return
    try:
        child = state.tracer.start_span(name, start_time=start_ns, attributes=_attrs(attributes))
        if attributes.get("retail.guardrail.outcome") in ("blocked", "validator_unavailable"):
            _mark_error(child, str(attributes.get("retail.guardrail.outcome")))
        child.end(end_time=end_ns)
    except Exception:  # noqa: BLE001
        pass


def capture(current: Any, **content: Any) -> None:
    """Attach what a step received and produced, when content capture is on. Never raises."""
    if current is None or _STATE is None or not _STATE.settings.capture_content:
        return
    try:
        limit = _STATE.settings.max_content_chars
        for name, value in content.items():
            text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
            current.set_attribute(f"retail.content.{name}", text[:limit])
    except Exception:  # noqa: BLE001
        pass


def flush_turn() -> None:
    """Send a finished turn's spans now, in one batch. Best effort, like every export."""
    if _STATE is None:
        return
    try:
        _STATE.tracer_provider.force_flush(timeout_millis=2000)
    except Exception:  # noqa: BLE001
        pass


def trace_id_hex(current: Any) -> str | None:
    try:
        ctx = current.get_span_context()
        return format(ctx.trace_id, "032x") if ctx.is_valid else None
    except Exception:  # noqa: BLE001
        return None


def proposal_ref(proposal_id: str | None, request_id: str | None) -> str | None:
    """Salted hash that correlates a proposal turn with its confirmation without exposing IDs."""
    if _STATE is None or not proposal_id:
        return None
    return hashlib.sha256(f"{_STATE.salt}:{request_id}:{proposal_id}".encode()).hexdigest()[:16]


# --- metric recorders (all bounded, all best effort) -----------------------------------------------

def _safe(fn):
    def wrapper(*args, **kwargs):
        if _STATE is None:
            return None
        try:
            return fn(_STATE, *args, **kwargs)
        except Exception:  # noqa: BLE001
            return None
    return wrapper


@_safe
def record_turn(state: _Telemetry, seconds: float, turn_kind: str, decision: str) -> None:
    state.turn_duration.record(seconds, {"turn_kind": label(turn_kind, TURN_KINDS),
                                         "decision": label(decision, DECISIONS)})


@_safe
def record_llm_call(state: _Telemetry, agent: str, seconds: float, ok: bool, prompt_tokens: int | None,
                    completion_tokens: int | None) -> None:
    agent_role = role(agent)
    state.llm_duration.record(seconds, {"agent_role": agent_role, "outcome": "ok" if ok else "error"})
    if prompt_tokens is not None:
        state.llm_tokens.record(prompt_tokens, {"agent_role": agent_role, "token_type": "input"})
    if completion_tokens is not None:
        state.llm_tokens.record(completion_tokens, {"agent_role": agent_role, "token_type": "output"})
        if ok and seconds > 0:
            state.llm_rate.record(completion_tokens / seconds, {"agent_role": agent_role,
                                                                 "method": "effective_call"})


@_safe
def record_tool(state: _Telemetry, tool: str | None, outcome: str, seconds: float | None = None,
                transport: str = "mcp_http") -> None:
    attrs = {"tool_name": label(tool, TOOLS), "outcome": label(outcome, OK_OUTCOMES),
             "transport": label(transport, TRANSPORTS)}
    state.tool_calls.add(1, attrs)
    if seconds is not None:
        state.tool_duration.record(seconds, attrs)


@_safe
def record_step(state: _Telemetry, step: str, seconds: float, outcome: str) -> None:
    state.step_duration.record(seconds, {"step": label(step, STEPS), "outcome": label(outcome, OK_OUTCOMES)})


@_safe
def record_mcp_startup(state: _Telemetry, seconds: float) -> None:
    state.mcp_startup.record(seconds)


@_safe
def record_guard(state: _Telemetry, gate: str, validator: str, outcome: str, seconds: float) -> None:
    attrs = {"gate": label(gate, GATES), "validator": label(validator, VALIDATORS),
             "outcome": label(outcome, GUARD_OUTCOMES)}
    state.guard_checks.add(1, attrs)
    state.guard_duration.record(seconds, attrs)


@_safe
def record_business(state: _Telemetry, remedy: str | None, outcome: str) -> None:
    state.business.add(1, {"remedy": label(remedy or "none", REMEDIES), "outcome": label(outcome, BUSINESS_OUTCOMES)})


def now_ns() -> int:
    return time.time_ns()


def current_context() -> Any:
    if _STATE is None:
        return None
    try:
        from opentelemetry import context
        return context.get_current()
    except Exception:  # noqa: BLE001
        return None


def current_span() -> Any:
    if _STATE is None:
        return None
    try:
        from opentelemetry import trace
        return trace.get_current_span()
    except Exception:  # noqa: BLE001
        return None


@contextmanager
def use_context(ctx: Any) -> Iterator[None]:
    """Run with an explicitly captured OTel context (graph nodes may run outside the caller's context)."""
    if _STATE is None or ctx is None:
        yield
        return
    from opentelemetry import context
    token = context.attach(ctx)
    try:
        yield
    finally:
        context.detach(token)
