"""Graph state (checkpointed) and per-turn context (not checkpointed).

Checkpointed state is minimal: last handoff, decision, response,
proposal and a short history. Raw customer text, fault injectors and event
traces live only in the per-turn TurnContext.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from app.agents.model_client import ModelCall
from app.services.faults import FaultInjector


class TurnState(TypedDict, total=False):
    # trusted inputs (set by the runtime each turn)
    customer_id: str
    request_id: str
    session_id: str
    turn_number: int
    confirmed_proposal_id: str | None
    confirmation_requested: bool
    # per turn
    route: str
    handoff: dict[str, Any] | None
    facts: dict[str, Any]
    decision: str
    response: str
    proposal: dict[str, Any] | None
    # carried across turns
    last_handoff: dict[str, Any] | None
    last_question: str | None
    last_asked: str | None  # what the open question asked: "order", "line", "size" or "condition"
    remaining_items: list[dict[str, Any]]  # other items the customer also asked for: one transaction per request
    last_proposal_id: str | None
    history: list[dict[str, Any]]


@dataclass
class TurnContext:
    message: str
    safe_message: str = ""  # sanitized, PII-masked; deliberately not checkpointed
    injector: FaultInjector | None = None
    model_calls: list[ModelCall] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    guardrail_events: list[dict[str, Any]] = field(default_factory=list)
    internal: dict[str, Any] = field(default_factory=dict)
    otel_context: Any = None  # the turn span's context, attached inside every graph node
    audited_calls: int = 0  # how many model calls and guardrail events are already in the audit log
    audited_guards: int = 0
