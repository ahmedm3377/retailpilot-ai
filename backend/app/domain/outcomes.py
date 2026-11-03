"""Typed results and error codes for the transactional business layer."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from retail_domain.policy import ReasonCode


class WorkflowCode(str, Enum):
    """Outcomes of the proposal/approval workflow, beyond the policy's reasons."""

    COMMITTED = "COMMITTED"
    UNSUPPORTED_REMEDY = "UNSUPPORTED_REMEDY"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    PROPOSAL_NOT_FOUND = "PROPOSAL_NOT_FOUND"
    PROPOSAL_MISMATCH = "PROPOSAL_MISMATCH"
    PROPOSAL_NOT_PENDING = "PROPOSAL_NOT_PENDING"
    REQUEST_CONFLICT = "REQUEST_CONFLICT"
    REQUEST_COMPLETED = "REQUEST_COMPLETED"
    AMOUNT_CHANGED = "AMOUNT_CHANGED"


Code = ReasonCode | WorkflowCode


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class CommitResult(Frozen):
    """What a trusted commit returned. The idempotency record stores the completed one."""

    status: Literal["completed", "rejected", "conflict"]
    reason_code: Code
    request_id: str
    proposal_id: str | None = None
    customer_id: str | None = None
    order_id: str | None = None
    line_id: str | None = None
    remedy: Literal["refund", "exchange"] | None = None
    refund_amount_cents: int | None = None
    original_sku: str | None = None
    replacement_sku: str | None = None
    record_ids: dict[str, str] = {}
    # True when this is the stored result of an earlier commit, returned again.
    replayed: bool = False


class CancelResult(Frozen):
    """Outcome of cancelling a pending proposal (workflow only; never a business effect)."""

    status: Literal["cancelled", "rejected"]
    reason_code: Code
    proposal_id: str


class RequestStatus(Frozen):
    """Persisted status of a request, read on a fresh connection."""

    state: Literal["completed", "not_found", "conflict"]
    request_id: str
    result: CommitResult | None = None


class PostCommitTimeout(Exception):
    """Simulated timeout raised after a durable commit, before the result is delivered.

    The business effects are committed; recover by reconciling the request,
    never by committing again.
    """

    def __init__(self, request_id: str, proposal_id: str | None) -> None:
        super().__init__(f"timeout after commit for request {request_id}")
        self.request_id = request_id
        self.proposal_id = proposal_id
