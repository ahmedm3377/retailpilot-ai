"""API request/response DTOs. Requests forbid extra fields, so clients cannot pass identity or amounts."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from retail_domain.models import Case

ActionId = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- requests -------------------------------------------------------------------------------

class CreateSession(Strict):
    """Demo sign-in: the customer ID the person types, looked up in the persistent store. With scenario_id
    (one fixture case) or uploaded_case (a case from the person's file), the session instead works on an
    isolated copy of that case. Neither ever touches the persistent store."""
    customer_id: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")
    scenario_id: str | None = Field(default=None, min_length=1, max_length=100)
    uploaded_case: Case | None = None


class TurnRequest(Strict):
    message: str = Field(min_length=1, max_length=2000)
    client_action_id: str = ActionId


class ActionRequest(Strict):
    client_action_id: str = ActionId


class EndRequest(Strict):
    pass


# --- responses ------------------------------------------------------------------------------

Decision = Literal["propose", "clarify", "deny", "completed", "error"]
RequestState = Literal["completed", "pending_confirmation", "proposal_invalidated", "none"]


class ProposalDTO(BaseModel):
    proposal_id: str
    remedy: Literal["refund", "exchange"]
    order_id: str
    line_id: str
    original_sku: str
    replacement_sku: str | None
    amount_cents: int
    status: Literal["pending", "completed", "invalidated"]


class OrderLineDTO(BaseModel):
    line_id: str
    sku: str
    product_id: str | None
    size: str | None
    paid_cents: int
    returned: bool
    outcome: Literal["refund", "exchange"] | None = None
    refund_cents: int | None = None
    replacement_sku: str | None = None


class OrderDTO(BaseModel):
    order_id: str
    status: str
    delivered_on: str | None
    lines: list[OrderLineDTO]


class HistoryItemDTO(BaseModel):
    seq: int
    kind: Literal["message", "confirm", "cancel", "new_request"]
    turn_number: int | None
    request_id: str
    message: str | None
    decision: Decision | None
    response: str | None


class SessionDTO(BaseModel):
    session_id: str
    scenario_id: str
    customer_id: str
    request_id: str
    turn_number: int
    status: Literal["active", "ended"]
    latest_proposal: ProposalDTO | None
    orders: list[OrderDTO]
    history: list[HistoryItemDTO]


class CaseOptionDTO(BaseModel):
    case_id: str
    customer_id: str


class CreatedSessionDTO(BaseModel):
    token: str
    session: SessionDTO


class CommittedResultDTO(BaseModel):
    remedy: Literal["refund", "exchange"] | None
    order_id: str | None
    line_id: str | None
    refund_amount_cents: int | None
    original_sku: str | None
    replacement_sku: str | None


class RequestStatusDTO(BaseModel):
    request_id: str
    state: RequestState
    result: CommittedResultDTO | None
    latest_proposal: ProposalDTO | None


class TurnResponseDTO(BaseModel):
    decision: Decision
    response: str
    proposal: ProposalDTO | None
    request_id: str
    request_status: RequestState
    turn_number: int
    elapsed_ms: float
    llm_calls: int
    prompt_tokens: int | None
    completion_tokens: int | None
    guardrail_blocked: bool
    replayed: bool
    timeout_recovered: bool
    correlation_id: str


class AuditEventDTO(BaseModel):
    event_id: int
    turn: int | None
    actor: str
    actor_label: str
    kind: Literal["llm", "tool", "guardrail", "handoff", "proposal", "commit", "fault", "other"]
    summary: str
    outcome: str
    severity: Literal["info", "warning", "error"]


class ReadyDTO(BaseModel):
    state: Literal["loading", "ready", "degraded"]
    postgres: dict
    model_server: dict
    guardrails: dict
