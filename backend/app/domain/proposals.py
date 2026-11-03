"""Immutable proposal DTOs. A proposal binds every value a commit will use."""
from __future__ import annotations

from typing import Any, Literal

from app.domain.outcomes import Code, CommitResult, Frozen


class Proposal(Frozen):
    proposal_id: str
    request_id: str
    customer_id: str
    order_id: str
    line_id: str
    remedy: Literal["refund", "exchange"]
    # Refund: exactly the line's paid_cents. Exchange: always 0 (no price differences).
    amount_cents: int
    paid_cents: int
    original_sku: str
    replacement_sku: str | None
    status: Literal["pending", "completed", "invalidated"]

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Proposal":
        return cls.model_validate({k: row[k] for k in cls.model_fields})


class ProposalOutcome(Frozen):
    """Result of create_proposal. Only kind='proposed' carries an executable proposal."""

    kind: Literal["proposed", "clarify", "deny", "already_completed", "conflict"]
    reason_code: Code
    request_id: str
    proposal: Proposal | None = None
    existing_result: CommitResult | None = None
    candidate_line_ids: tuple[str, ...] = ()
