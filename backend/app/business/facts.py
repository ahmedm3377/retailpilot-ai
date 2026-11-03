"""Trusted facts built from service results. They are what a reply may state; wording lives with the agents."""
from __future__ import annotations

from typing import Any


def money(cents: int | None) -> str | None:
    return None if cents is None else f"${cents // 100}.{cents % 100:02d}"


def facts_from_proposal_outcome(outcome: Any) -> dict[str, Any]:
    p = outcome.proposal
    facts: dict[str, Any] = {"reason_code": outcome.reason_code.value,
                             "candidate_line_ids": list(outcome.candidate_line_ids)}
    if p is not None:
        facts.update(proposal_id=p.proposal_id, remedy=p.remedy, order_id=p.order_id, line_id=p.line_id,
                     original_sku=p.original_sku, replacement_sku=p.replacement_sku,
                     refund_amount_cents=p.amount_cents if p.remedy == "refund" else None)
    elif outcome.existing_result is not None:
        facts.update(facts_from_commit(outcome.existing_result))
        # Keep REQUEST_COMPLETED (not the stored COMMITTED): the reply must say the request was already done.
        facts.update(reason_code=outcome.reason_code.value, already_completed=True)
    facts["amount_text"] = money(facts.get("refund_amount_cents"))
    return facts


def facts_already_completed(result: Any) -> dict[str, Any]:
    """Facts for a chat turn on a request that already committed: report it, change nothing."""
    return {**facts_from_commit(result), "reason_code": "REQUEST_COMPLETED", "already_completed": True}


def facts_from_commit(result: Any) -> dict[str, Any]:
    return {"reason_code": result.reason_code.value, "proposal_id": result.proposal_id, "remedy": result.remedy,
            "order_id": result.order_id, "line_id": result.line_id,
            "refund_amount_cents": result.refund_amount_cents, "amount_text": money(result.refund_amount_cents),
            "original_sku": result.original_sku, "replacement_sku": result.replacement_sku,
            "replayed": result.replayed}
