"""Trusted-only commit of an explicitly confirmed proposal.

`commit_confirmed_proposal` must be called by the trusted controller (UI or
fixture adapter) with a proposal ID the customer confirmed through a structured
action. It must never be exposed as an LLM tool.

All effects (stock, returned flag, return, quarantine, refund/exchange,
proposal status, idempotency, audit) commit in one PostgreSQL transaction.
Locks, in this order: advisory lock on request_id, proposal row, order line
row, replacement product row. Database constraints back up every check.
"""
from __future__ import annotations

from typing import Any

import psycopg
from psycopg import errors

from app.db import repositories as repo
from app.db.fixture_state import RetailStore, state_from_rows
from retail_domain import policy
from app.domain.outcomes import CancelResult, Code, CommitResult, WorkflowCode
from app.services.audit import jsonable
from app.services.faults import FaultInjector
from app.services.recovery import reconcile_request, stored_result

TRUSTED_ACTOR = "trusted_controller"


class _Reject(Exception):
    def __init__(self, code: Code, *, invalidate: bool = False) -> None:
        super().__init__(code.value)
        self.code = code
        # Mark the proposal invalidated (workflow metadata only) after rollback.
        self.invalidate = invalidate


def commit_confirmed_proposal(store: RetailStore, authenticated_customer_id: str, request_id: str,
                              confirmed_proposal_id: str | None, *,
                              injector: FaultInjector | None = None,
                              actor: str = TRUSTED_ACTOR) -> CommitResult:
    """Execute a confirmed proposal exactly once.

    A replay with the same request, customer and proposal returns the stored
    original result (replayed=True) with no new effects. May raise
    PostCommitTimeout after a durable commit when the injector says so.
    """
    arguments = {"customer_id": authenticated_customer_id, "request_id": request_id,
                 "proposal_id": confirmed_proposal_id}
    if not confirmed_proposal_id:
        return _reject(store, actor, arguments, _Reject(WorkflowCode.CONFIRMATION_REQUIRED))

    try:
        with store.connect() as conn, conn.transaction():
            result = _commit_locked(conn, actor, arguments, authenticated_customer_id, request_id,
                                    confirmed_proposal_id)
    except _Reject as rejection:
        return _reject(store, actor, arguments, rejection)
    except errors.UniqueViolation:
        # A concurrent commit won the race; report what is persisted, never redo.
        status = reconcile_request(store, authenticated_customer_id, request_id)
        if status.state == "completed" and status.result.proposal_id == confirmed_proposal_id:
            result = status.result.model_copy(update={"replayed": True})
        else:
            return _reject(store, actor, arguments, _Reject(policy.ReasonCode.ALREADY_RETURNED))

    if result.replayed:
        with store.connect() as conn:
            repo.append_audit(conn, actor=actor, tool="commit_confirmed_proposal", arguments=arguments,
                              outcome="replayed", result=jsonable(result))
        return result
    if injector is not None:
        injector.after_commit(store, result)
    return result


def _commit_locked(conn: psycopg.Connection, actor: str, arguments: dict[str, Any], customer_id: str,
                   request_id: str, proposal_id: str) -> CommitResult:
    conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (request_id,))

    done = repo.get_idempotency(conn, request_id)
    if done is not None:
        if done["customer_id"] == customer_id and done["proposal_id"] == proposal_id:
            return stored_result(done).model_copy(update={"replayed": True})
        raise _Reject(WorkflowCode.REQUEST_CONFLICT)

    proposal = repo.get_proposal(conn, proposal_id, for_update=True)
    if proposal is None:
        raise _Reject(WorkflowCode.PROPOSAL_NOT_FOUND)
    if proposal["customer_id"] != customer_id or proposal["request_id"] != request_id:
        raise _Reject(WorkflowCode.PROPOSAL_MISMATCH)
    if proposal["status"] != "pending":
        raise _Reject(WorkflowCode.PROPOSAL_NOT_PENDING)

    order_id, line_id, remedy = proposal["order_id"], proposal["line_id"], proposal["remedy"]
    replacement_sku = proposal["replacement_sku"]
    line = repo.get_line_for_update(conn, order_id, line_id)
    replacement = repo.get_product(conn, replacement_sku, for_update=True) if replacement_sku else None
    if line["sku"] != proposal["original_sku"] or line["paid_cents"] != proposal["paid_cents"]:
        raise _Reject(WorkflowCode.PROPOSAL_MISMATCH, invalidate=True)

    # Revalidate with the policy on the locked rows.
    original = repo.get_product(conn, line["sku"])
    products = [p for p in (original, replacement) if p is not None]
    state = state_from_rows(repo.get_meta(conn), products, [repo.get_order(conn, order_id)], [line])
    if remedy == "refund":
        check = policy.assess_refund(state, customer_id, order_id, line_id)
        if check.eligible and check.refund_amount_cents != proposal["amount_cents"]:
            raise _Reject(WorkflowCode.AMOUNT_CHANGED, invalidate=True)
    else:
        check = policy.assess_exchange(state, customer_id, order_id, line_id, replacement_sku)
    if not check.eligible:
        raise _Reject(check.reason_code, invalidate=True)

    effects: list[dict[str, Any]] = []
    if remedy == "exchange":
        new_stock = repo.decrement_stock(conn, replacement_sku)
        if new_stock is None:
            raise _Reject(policy.ReasonCode.REPLACEMENT_OUT_OF_STOCK, invalidate=True)
        effects.append({"type": "stock_decrement", "sku": replacement_sku, "new_stock": new_stock})

    refs = {"request_id": request_id, "proposal_id": proposal_id, "customer_id": customer_id,
            "order_id": order_id, "line_id": line_id}
    return_id = repo.insert_return(conn, **refs, sku=line["sku"], remedy=remedy)
    if not repo.mark_line_returned(conn, order_id, line_id):
        raise _Reject(policy.ReasonCode.ALREADY_RETURNED, invalidate=True)
    quarantine_id = repo.insert_quarantine(conn, **refs, return_id=return_id, sku=line["sku"])
    record_ids = {"return_id": return_id, "quarantine_id": quarantine_id}
    effects += [{"type": "line_returned", "order_id": order_id, "line_id": line_id},
                {"type": "quarantined", "sku": line["sku"], "quantity": 1}]
    if remedy == "refund":
        record_ids["refund_id"] = repo.insert_refund(conn, **refs, return_id=return_id,
                                                     amount_cents=proposal["amount_cents"],
                                                     currency=state.currency)
        effects.append({"type": "refund", "amount_cents": proposal["amount_cents"]})
    else:
        record_ids["exchange_id"] = repo.insert_exchange(conn, **refs, return_id=return_id,
                                                         original_sku=line["sku"],
                                                         replacement_sku=replacement_sku)
        effects.append({"type": "exchange", "original_sku": line["sku"], "replacement_sku": replacement_sku})
    repo.set_proposal_status(conn, proposal_id, "completed")

    result = CommitResult(
        status="completed", reason_code=WorkflowCode.COMMITTED, **refs, remedy=remedy,
        refund_amount_cents=proposal["amount_cents"] if remedy == "refund" else None,
        original_sku=line["sku"], replacement_sku=replacement_sku, record_ids=record_ids)
    repo.insert_idempotency(conn, request_id, customer_id, proposal_id, result.model_dump(mode="json"))
    repo.append_audit(conn, actor=actor, tool="commit_confirmed_proposal", arguments=arguments,
                      outcome="committed", result=jsonable(result), effects=effects)
    return result


def _reject(store: RetailStore, actor: str, arguments: dict[str, Any], rejection: _Reject) -> CommitResult:
    """Record a refusal. Business state was rolled back; only workflow metadata is written."""
    status = "conflict" if rejection.code is WorkflowCode.REQUEST_CONFLICT else "rejected"
    result = CommitResult(status=status, reason_code=rejection.code, request_id=arguments["request_id"],
                          proposal_id=arguments["proposal_id"])
    with store.connect() as conn, conn.transaction():
        if rejection.invalidate:
            proposal = repo.get_proposal(conn, arguments["proposal_id"], for_update=True)
            if proposal is not None and proposal["status"] == "pending":
                repo.set_proposal_status(conn, arguments["proposal_id"], "invalidated")
        repo.append_audit(conn, actor=actor, tool="commit_confirmed_proposal", arguments=arguments,
                          outcome=status, result=jsonable(result), error=rejection.code.value)
    return result


def cancel_proposal(store: RetailStore, authenticated_customer_id: str, request_id: str, proposal_id: str, *,
                    actor: str = TRUSTED_ACTOR) -> CancelResult:
    """Trusted-only: mark a pending proposal of this customer/request `invalidated`.

    No business effect. A later confirmation of it returns PROPOSAL_NOT_PENDING.
    Never exposed as an LLM tool.
    """
    arguments = {"customer_id": authenticated_customer_id, "request_id": request_id, "proposal_id": proposal_id}
    with store.connect() as conn, conn.transaction():
        proposal = repo.get_proposal(conn, proposal_id, for_update=True)
        if proposal is None:
            code: Code = WorkflowCode.PROPOSAL_NOT_FOUND
        elif proposal["customer_id"] != authenticated_customer_id or proposal["request_id"] != request_id:
            code = WorkflowCode.PROPOSAL_MISMATCH
        elif proposal["status"] != "pending":
            code = WorkflowCode.PROPOSAL_NOT_PENDING
        else:
            repo.set_proposal_status(conn, proposal_id, "invalidated")
            result = CancelResult(status="cancelled", reason_code=WorkflowCode.PROPOSAL_NOT_PENDING,
                                  proposal_id=proposal_id)
            repo.append_audit(conn, actor=actor, tool="cancel_proposal", arguments=arguments, outcome="cancelled",
                              result=jsonable(result))
            return result
        result = CancelResult(status="rejected", reason_code=code, proposal_id=proposal_id)
        repo.append_audit(conn, actor=actor, tool="cancel_proposal", arguments=arguments, outcome="rejected",
                          result=jsonable(result), error=code.value)
        return result
