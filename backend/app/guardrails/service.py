"""GuardrailService: the four guardrail gates around the agents.

1. Customer input: sanitize, task injection, DetectJailbreak, PII masking
   before the LLM sees it; abusive tone is noted, not refused.
2. Tool context: string values of read-only tool results are checked for
   injection before they enter model context; output is size-capped.
3. Agent action: every model output goes through Guard.for_pydantic + Pydantic
   (extra="forbid"). Schema success is never authorization.
4. Customer output: sanitize, PII mask, ToxicLanguage, and a consistency check
   against trusted facts; any failure falls back to deterministic wording.

All ML validators run locally (use_local=True, Guardrails metrics off). With
fail_closed, a required validator that cannot load blocks input and forces the
deterministic output fallback. Events never contain full untrusted text.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel, ValidationError

from app.agents.settings import AgentSettings, apply_offline_defaults
from app.guardrails.sanitization import safe_preview, sanitize_text

apply_offline_defaults()
from guardrails import Guard  # noqa: E402
from guardrails.classes.rc import RC  # noqa: E402
from guardrails.settings import settings as guardrails_settings  # noqa: E402
from guardrails.utils.hub_telemetry_utils import HubTelemetry  # noqa: E402

from app.guardrails.injection import TaskInjection  # noqa: E402
from app.observability import telemetry as tel  # noqa: E402


def disable_guardrails_telemetry() -> None:
    """Guardrails sends hub telemetry (OTLP to a hosted endpoint) unless ~/.guardrailsrc opts out.

    Its singleton may already exist from import time, so switch the settings off,
    shut down any existing exporter, and drop the singleton so later instances
    are created disabled. OTEL_SDK_DISABLED (set in apply_offline_defaults when app telemetry is off) is a
    second layer; the offline test blocks sockets to catch any leak.
    """
    guardrails_settings.rc.enable_metrics = False
    guardrails_settings.rc.use_remote_inferencing = False
    guardrails_settings.use_server = False
    guardrails_settings.disable_tracing = True
    existing = HubTelemetry._instance
    if existing is not None:
        existing._enabled = False
        provider = getattr(existing, "_tracer_provider", None)
        if provider is not None:
            provider.shutdown()
        HubTelemetry._instance = None


def _load_rc_without_metrics(self: Guard) -> None:
    # Every Guard() re-reads ~/.guardrailsrc (enable_metrics defaults to True) and would re-enable hub
    # telemetry. Keep the opt-out sticky; this matters once OTEL_SDK_DISABLED is false for our own telemetry.
    rc = RC.load()
    rc.enable_metrics = False
    rc.use_remote_inferencing = False
    guardrails_settings.rc = rc


Guard._load_rc = _load_rc_without_metrics
disable_guardrails_telemetry()

Outcome = Literal["passed", "sanitized", "flagged", "blocked", "validation_error", "validator_unavailable"]
PII_ENTITIES = ["EMAIL_ADDRESS", "PHONE_NUMBER"]
MAX_TOOL_CONTEXT_CHARS = 4000


@dataclass(frozen=True)
class GuardEvent:
    gate: str
    validator: str
    outcome: Outcome
    reason_code: str
    severity: Literal["info", "warning", "high"]
    elapsed_ms: float
    detail: str | None = None  # short safe preview or category list, never full untrusted text

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InputGuardResult:
    allowed: bool
    text: str  # sanitized, PII-masked text for the LLM
    reason_code: str
    events: list[GuardEvent] = field(default_factory=list)


@dataclass
class SafeToolContext:
    allowed: bool
    payload: Any
    events: list[GuardEvent] = field(default_factory=list)


@dataclass
class ValidatedAction:
    ok: bool
    value: BaseModel | None
    reason_code: str
    error: str | None = None
    events: list[GuardEvent] = field(default_factory=list)


@dataclass
class OutputGuardResult:
    text: str
    used_fallback: bool
    events: list[GuardEvent] = field(default_factory=list)


def _traced(gate: str):
    """Span per gate call, with one child per validator step placed by its measured elapsed_ms.

    Only gate, validator, outcome and reason code are recorded: never the checked text or detail.
    """
    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not tel.active():
                return fn(self, *args, **kwargs)
            start = tel.now_ns()
            with tel.span(f"retail.guardrail.{gate}", {"retail.guardrail.gate": gate}) as span:
                result = fn(self, *args, **kwargs)
                cursor = start
                worst = "passed"
                for event in result.events:
                    duration = int(max(event.elapsed_ms, 0.0) * 1e6)
                    tel.timed_child("retail.guardrail.validate", cursor, cursor + duration, {
                        "retail.guardrail.gate": event.gate, "retail.guardrail.validator": event.validator,
                        "retail.guardrail.outcome": event.outcome, "retail.guardrail.reason_code": event.reason_code})
                    tel.record_guard(event.gate, event.validator, event.outcome, event.elapsed_ms / 1000)
                    cursor += duration
                    if event.outcome in ("blocked", "validator_unavailable", "validation_error"):
                        worst = event.outcome
                    elif event.outcome in ("sanitized", "flagged") and worst == "passed":
                        worst = event.outcome
                tel.set_attributes(span, {"retail.guardrail.outcome": worst})
                return result
        return wrapper
    return decorate


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _local_pii_validator() -> Any:
    """DetectPII with Presidio on spaCy en_core_web_sm (installed; never downloaded at runtime).

    DetectPII's own constructor builds a default AnalyzerEngine that wants
    en_core_web_lg and downloads it if missing, so engines are injected instead.
    EMAIL_ADDRESS and PHONE_NUMBER are pattern recognizers; the spaCy model
    size does not affect them.
    """
    from guardrails_ai.detect_pii import DetectPII
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    validator = DetectPII(pii_entities=PII_ENTITIES, on_fail="fix", use_local=False)
    nlp = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}).create_engine()
    validator.pii_analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
    validator.pii_anonymizer = AnonymizerEngine()
    validator.use_local = True
    return validator


class GuardrailService:
    def __init__(self, settings: AgentSettings, *, jailbreak: Guard | None = None,
                 toxicity: Guard | None = None, pii: Guard | None = None, load_models: bool = True) -> None:
        self.settings = settings
        self.unavailable: dict[str, str] = {}
        self.injection = Guard().use(TaskInjection(on_fail="noop"))
        self.jailbreak, self.toxicity, self.pii = jailbreak, toxicity, pii
        if settings.guardrails_enabled and load_models:
            self.jailbreak = self.jailbreak or self._load("DetectJailbreak", self._jailbreak_guard)
            self.toxicity = self.toxicity or self._load("ToxicLanguage", self._toxicity_guard)
            self.pii = self.pii or self._load("DetectPII", lambda: Guard().use(_local_pii_validator()))
        self.schema_guards: dict[type[BaseModel], Guard] = {}

    def _load(self, name: str, factory: Callable[[], Guard]) -> Guard | None:
        try:
            return factory()
        except Exception as error:  # noqa: BLE001 - recorded; fail-closed handling below
            self.unavailable[name] = f"{type(error).__name__}: {str(error)[:200]}"
            return None

    def _jailbreak_guard(self) -> Guard:
        from guardrails_ai.detect_jailbreak import DetectJailbreak
        return Guard().use(DetectJailbreak(threshold=self.settings.jailbreak_threshold, use_local=True,
                                           on_fail="noop"))

    def _toxicity_guard(self) -> Guard:
        from guardrails_ai.toxic_language import ToxicLanguage
        return Guard().use(ToxicLanguage(threshold=self.settings.toxicity_threshold, validation_method="full",
                                         use_local=True, on_fail="noop"))

    def describe(self) -> dict[str, Any]:
        s = self.settings
        return {"enabled": s.guardrails_enabled, "fail_closed": s.fail_closed,
                "validators": {
                    "TaskInjection": "custom regex validator (retail/task_injection)",
                    "DetectJailbreak": None if not self.jailbreak else f"local, threshold={s.jailbreak_threshold}",
                    "ToxicLanguage": None if not self.toxicity else
                    f"local Detoxify unbiased-small, threshold={s.toxicity_threshold}, full",
                    "DetectPII": None if not self.pii else f"local Presidio en_core_web_sm, {PII_ENTITIES}, fix",
                    "Schema": "Guard.for_pydantic + Pydantic extra=forbid"},
                "unavailable": dict(self.unavailable)}

    # --- helpers ----------------------------------------------------------------

    @staticmethod
    def _run(guard: Guard, text: str) -> tuple[bool, str, str | None]:
        outcome = guard.validate(text)
        reasons = [s.failure_reason for s in (outcome.validation_summaries or []) if s.failure_reason]
        validated = outcome.validated_output if isinstance(outcome.validated_output, str) else text
        return bool(outcome.validation_passed), validated, "; ".join(reasons) or None

    def _required_missing(self, name: str, gate: str) -> GuardEvent | None:
        if not self.settings.guardrails_enabled or name not in self.unavailable:
            return None
        return GuardEvent(gate, name, "validator_unavailable", "VALIDATOR_UNAVAILABLE",
                          "high" if self.settings.fail_closed else "warning", 0.0, self.unavailable[name][:120])

    def mask_pii(self, text: str, gate: str) -> tuple[str, list[GuardEvent]]:
        if self.pii is None:
            missing = self._required_missing("DetectPII", gate)
            return text, [missing] if missing else []
        start = time.perf_counter()
        # on_fail="fix" reports validation_passed=True after fixing, so compare the text.
        _, fixed, _ = self._run(self.pii, text)
        if fixed == text:
            return text, [GuardEvent(gate, "DetectPII", "passed", "NO_PII", "info", _ms(start))]
        return fixed, [GuardEvent(gate, "DetectPII", "sanitized", "PII_MASKED", "info", _ms(start))]

    # --- gate 1: customer input -------------------------------------------------------

    @_traced("input")
    def validate_and_sanitize_customer_input(self, raw_message: Any) -> InputGuardResult:
        gate = "input"
        events: list[GuardEvent] = []
        start = time.perf_counter()
        clean = sanitize_text(raw_message, self.settings.max_input_chars)
        if clean.reason_code:
            events.append(GuardEvent(gate, "sanitization", "blocked", clean.reason_code, "warning", _ms(start)))
            return InputGuardResult(False, "", clean.reason_code, events)
        events.append(GuardEvent(gate, "sanitization", "sanitized" if clean.removed_chars else "passed",
                                 "CHARS_REMOVED" if clean.removed_chars else "CLEAN", "info", _ms(start),
                                 f"removed={clean.removed_chars}"))
        text = clean.text

        start = time.perf_counter()
        passed, _, reason = self._run(self.injection, text)
        if not passed:
            events.append(GuardEvent(gate, "TaskInjection", "blocked", "PROMPT_INJECTION", "high", _ms(start),
                                     reason))
            return InputGuardResult(False, "", "PROMPT_INJECTION", events)
        events.append(GuardEvent(gate, "TaskInjection", "passed", "NO_MATCH", "info", _ms(start)))

        if self.jailbreak is None:
            missing = self._required_missing("DetectJailbreak", gate)
            if missing:
                events.append(missing)
                if self.settings.fail_closed:
                    return InputGuardResult(False, "", "VALIDATOR_UNAVAILABLE", events)
        else:
            start = time.perf_counter()
            passed, _, _ = self._run(self.jailbreak, text)
            if not passed:
                # The validator's failure message quotes the prompt; never log it.
                events.append(GuardEvent(gate, "DetectJailbreak", "blocked", "JAILBREAK", "high", _ms(start),
                                         f"score above threshold {self.settings.jailbreak_threshold}"))
                return InputGuardResult(False, "", "JAILBREAK", events)
            events.append(GuardEvent(gate, "DetectJailbreak", "passed", "BELOW_THRESHOLD", "info", _ms(start)))

        if self.toxicity is not None:
            # An upset customer may still be owed a refund: note it, never refuse on tone alone.
            start = time.perf_counter()
            passed, _, _ = self._run(self.toxicity, text)
            events.append(GuardEvent(gate, "ToxicLanguage", "passed" if passed else "flagged",
                                     "CLEAN" if passed else "ABUSIVE_TONE_NOTED",
                                     "info" if passed else "warning", _ms(start)))

        masked, pii_events = self.mask_pii(text, gate)
        events += pii_events
        if any(e.outcome == "validator_unavailable" for e in pii_events) and self.settings.fail_closed:
            return InputGuardResult(False, "", "VALIDATOR_UNAVAILABLE", events)
        return InputGuardResult(True, masked, "ALLOWED", events)

    # --- gate 2: tool output entering model context -------------------------------------

    @_traced("tool_context")
    def validate_tool_context(self, tool: str, result: Any) -> SafeToolContext:
        gate = "tool_context"
        start = time.perf_counter()
        hits: set[str] = set()

        def scan(value: Any) -> None:
            if isinstance(value, dict):
                for v in value.values():
                    scan(v)
            elif isinstance(value, list):
                for v in value:
                    scan(v)
            elif isinstance(value, str):
                passed, _, reason = self._run(self.injection, value)
                if not passed:
                    hits.add(reason or "injection")

        scan(result)
        if hits:
            return SafeToolContext(False, {"error": "TOOL_OUTPUT_REJECTED"},
                                   [GuardEvent(gate, "TaskInjection", "blocked", "TOOL_OUTPUT_INJECTION", "high",
                                               _ms(start), f"{tool}: {sorted(hits)[0][:100]}")])
        text = json.dumps(result, ensure_ascii=False)
        if len(text) > MAX_TOOL_CONTEXT_CHARS:
            return SafeToolContext(True, {"truncated": text[:MAX_TOOL_CONTEXT_CHARS]},
                                   [GuardEvent(gate, "size_limit", "sanitized", "TRUNCATED", "warning", _ms(start),
                                               tool)])
        return SafeToolContext(True, result, [GuardEvent(gate, "TaskInjection", "passed", "NO_MATCH", "info",
                                                         _ms(start), tool)])

    # --- gate 3: model output schema ----------------------------------------------------------

    @_traced("agent_action")
    def validate_agent_action(self, raw_model_output: str, schema_type: type[BaseModel]) -> ValidatedAction:
        gate = "agent_action"
        start = time.perf_counter()
        name = schema_type.__name__
        if not isinstance(raw_model_output, str) or len(raw_model_output) > 8000:
            event = GuardEvent(gate, f"Schema[{name}]", "validation_error", "BAD_OUTPUT", "warning", _ms(start))
            return ValidatedAction(False, None, "BAD_OUTPUT", "output missing or too long", [event])
        guard = self.schema_guards.get(schema_type)
        if guard is None:
            guard = self.schema_guards[schema_type] = Guard.for_pydantic(output_class=schema_type)
        try:
            outcome = guard.parse(raw_model_output)
            if not outcome.validation_passed or not isinstance(outcome.validated_output, dict):
                raise ValueError(str(outcome.error or "output is not a JSON object")[:200])
            # Guard.parse drops unknown keys, so also validate the raw JSON strictly (extra="forbid").
            value = schema_type.model_validate_json(raw_model_output)
            if value.model_dump(mode="json") != schema_type.model_validate(outcome.validated_output).model_dump(
                    mode="json"):
                raise ValueError("schema guard and strict parse disagree")
        except (ValueError, ValidationError, TypeError) as error:
            message = str(error).splitlines()[0][:200]
            event = GuardEvent(gate, f"Schema[{name}]", "validation_error", "SCHEMA_INVALID", "warning",
                               _ms(start), message)
            return ValidatedAction(False, None, "SCHEMA_INVALID", message, [event])
        return ValidatedAction(True, value, "VALID",
                               events=[GuardEvent(gate, f"Schema[{name}]", "passed", "VALID", "info", _ms(start))])

    # --- gate 4: customer-facing output ----------------------------------------------------

    @_traced("output")
    def validate_customer_output(self, raw_response: str, trusted_facts: dict[str, Any],
                                 fallback: str) -> OutputGuardResult:
        """Return safe text: the model's wording if every check passes, otherwise `fallback`.

        `fallback` is deterministic wording built from trusted status only. The
        decision, proposal and database state are never changed here.
        """
        gate = "output"
        events: list[GuardEvent] = []
        start = time.perf_counter()
        clean = sanitize_text(raw_response, 1200)
        if clean.reason_code:
            events.append(GuardEvent(gate, "sanitization", "blocked", clean.reason_code, "warning", _ms(start)))
            return OutputGuardResult(fallback, True, events)
        text = clean.text

        start = time.perf_counter()
        problem = consistency_problem(text, trusted_facts)
        if problem:
            events.append(GuardEvent(gate, "consistency", "blocked", "UNSUPPORTED_CLAIM", "warning", _ms(start),
                                     problem))
            return OutputGuardResult(fallback, True, events)
        events.append(GuardEvent(gate, "consistency", "passed", "CONSISTENT", "info", _ms(start)))

        if self.toxicity is None:
            missing = self._required_missing("ToxicLanguage", gate)
            if missing:
                events.append(missing)
                if self.settings.fail_closed:
                    return OutputGuardResult(fallback, True, events)
        else:
            start = time.perf_counter()
            passed, _, _ = self._run(self.toxicity, text)
            if not passed:
                events.append(GuardEvent(gate, "ToxicLanguage", "blocked", "TOXIC_OUTPUT", "high", _ms(start)))
                return OutputGuardResult(fallback, True, events)
            events.append(GuardEvent(gate, "ToxicLanguage", "passed", "CLEAN", "info", _ms(start)))

        masked, pii_events = self.mask_pii(text, gate)
        events += pii_events
        if any(e.outcome == "validator_unavailable" for e in pii_events) and self.settings.fail_closed:
            return OutputGuardResult(fallback, True, events)
        return OutputGuardResult(masked, False, events)


_SUCCESS_CLAIM = re.compile(r"\b(has|have) been (refunded|exchanged|processed|completed|approved)\b|"
                            r"\b(refund|exchange) (is|was) (complete|completed|processed|done)\b|"
                            r"\b(i|we) (have|'ve) (refunded|processed|completed|exchanged|approved)\b", re.IGNORECASE)
_MONEY = re.compile(r"\$\s?(\d{1,6}(?:,\d{3})*(?:\.\d{1,2})?)")
_PROPOSAL_ID = re.compile(r"\bPROP-\d+\b")


def consistency_problem(text: str, facts: dict[str, Any]) -> str | None:
    """Claims in `text` that trusted facts do not support, or None.

    Checks: success claims only after a completed commit; every dollar amount
    equals the trusted refund amount; proposal IDs only as given.
    """
    if facts.get("decision") != "completed" and _SUCCESS_CLAIM.search(text):
        return "claims completion before a committed transaction"
    allowed_cents = {facts.get("refund_amount_cents")} - {None}
    for amount in _MONEY.findall(text):
        cents = round(float(amount.replace(",", "")) * 100)
        if cents not in allowed_cents:
            return f"mentions ${amount}, not a trusted amount"
    allowed_ids = {facts.get("proposal_id")} - {None}
    for proposal_id in _PROPOSAL_ID.findall(text):
        if proposal_id not in allowed_ids:
            return f"mentions unknown proposal {proposal_id}"
    return None


def preview(text: str) -> str:
    return safe_preview(text)
