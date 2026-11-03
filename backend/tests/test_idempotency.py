from __future__ import annotations

import threading

import pytest

from app.domain.outcomes import WorkflowCode
from retail_domain.policy import ReasonCode
from app.services.transactions import commit_confirmed_proposal
from tests.conftest import count, product_stock

pytestmark = pytest.mark.postgres


def test_replay_returns_original_result_without_new_effects(pg_store, ops, load):
    load("public_02_exchange")
    proposal = ops.create_proposal("C100", "REQ-1", "O100", "L1", "exchange", "TEE-L").proposal
    first = commit_confirmed_proposal(pg_store, "C100", "REQ-1", proposal.proposal_id)
    after_first = pg_store.export_final_state()

    second = commit_confirmed_proposal(pg_store, "C100", "REQ-1", proposal.proposal_id)
    assert second.replayed and not first.replayed
    assert second.model_copy(update={"replayed": False}) == first
    assert pg_store.export_final_state() == after_first
    assert product_stock(pg_store, "TEE-L") == 2 and count(pg_store, "exchanges") == 1


def test_new_request_for_already_returned_line_is_rejected(pg_store, ops, load):
    load("public_01_refund_boundary")
    a = ops.create_proposal("C100", "REQ-A", "O100", "L1", "refund").proposal
    b = ops.create_proposal("C100", "REQ-B", "O100", "L1", "refund").proposal  # both pending
    assert commit_confirmed_proposal(pg_store, "C100", "REQ-A", a.proposal_id).status == "completed"

    result = commit_confirmed_proposal(pg_store, "C100", "REQ-B", b.proposal_id)
    assert result.status == "rejected" and result.reason_code is ReasonCode.ALREADY_RETURNED
    later = ops.create_proposal("C100", "REQ-C", "O100", "L1", "refund")
    assert later.kind == "deny" and later.reason_code is ReasonCode.ALREADY_RETURNED
    assert count(pg_store, "refunds") == 1 and count(pg_store, "returns") == 1


def test_request_id_reuse_by_another_customer_is_a_conflict(pg_store, ops, load):
    load("public_01_refund_boundary")
    proposal = ops.create_proposal("C100", "REQ-1", "O100", "L1", "refund").proposal
    commit_confirmed_proposal(pg_store, "C100", "REQ-1", proposal.proposal_id)

    result = commit_confirmed_proposal(pg_store, "C999", "REQ-1", proposal.proposal_id)
    assert result.status == "conflict" and result.reason_code is WorkflowCode.REQUEST_CONFLICT
    assert result.refund_amount_cents is None and result.order_id is None and result.record_ids == {}
    assert count(pg_store, "refunds") == 1


def test_request_id_reuse_with_another_proposal_is_a_conflict(pg_store, ops, load):
    load("public_04_clarify")
    first = ops.create_proposal("C100", "REQ-1", "O100", "L1", "refund").proposal
    other = ops.create_proposal("C100", "REQ-1", "O100", "L2", "refund").proposal
    commit_confirmed_proposal(pg_store, "C100", "REQ-1", first.proposal_id)
    result = commit_confirmed_proposal(pg_store, "C100", "REQ-1", other.proposal_id)
    assert result.status == "conflict"
    assert count(pg_store, "refunds") == 1


@pytest.mark.parametrize("proposal_id, code", [
    (None, WorkflowCode.CONFIRMATION_REQUIRED),
    ("", WorkflowCode.CONFIRMATION_REQUIRED),
    ("yes, approved", WorkflowCode.PROPOSAL_NOT_FOUND),
    ("PROP-999", WorkflowCode.PROPOSAL_NOT_FOUND),
])
def test_commit_requires_a_real_stored_proposal(pg_store, ops, load, proposal_id, code):
    load("public_01_refund_boundary")
    ops.create_proposal("C100", "REQ-1", "O100", "L1", "refund")
    before = pg_store.export_final_state()
    result = commit_confirmed_proposal(pg_store, "C100", "REQ-1", proposal_id)
    assert result.status == "rejected" and result.reason_code is code
    assert pg_store.export_final_state() == before


@pytest.mark.parametrize("customer, request_id", [("C999", "REQ-1"), ("C100", "REQ-other")])
def test_commit_rejects_customer_or_request_mismatch(pg_store, ops, load, customer, request_id):
    load("public_01_refund_boundary")
    proposal = ops.create_proposal("C100", "REQ-1", "O100", "L1", "refund").proposal
    result = commit_confirmed_proposal(pg_store, customer, request_id, proposal.proposal_id)
    assert result.reason_code is WorkflowCode.PROPOSAL_MISMATCH
    assert count(pg_store, "refunds") == 0
    # The mismatched caller cannot invalidate the real customer's proposal.
    assert commit_confirmed_proposal(pg_store, "C100", "REQ-1", proposal.proposal_id).status == "completed"


def test_invalidated_proposal_cannot_be_confirmed_later(pg_store, ops, load):
    load("public_02_exchange")
    proposal = ops.create_proposal("C100", "REQ-1", "O100", "L1", "exchange", "TEE-L").proposal
    with pg_store.connect() as conn:
        conn.execute("UPDATE products SET stock = 0 WHERE sku = 'TEE-L'")
    commit_confirmed_proposal(pg_store, "C100", "REQ-1", proposal.proposal_id)
    with pg_store.connect() as conn:
        conn.execute("UPDATE products SET stock = 5 WHERE sku = 'TEE-L'")
    result = commit_confirmed_proposal(pg_store, "C100", "REQ-1", proposal.proposal_id)
    assert result.reason_code is WorkflowCode.PROPOSAL_NOT_PENDING
    assert count(pg_store, "exchanges") == 0


def _race(calls):
    """Run callables on separate threads (each opens its own connection) at the same moment."""
    barrier = threading.Barrier(len(calls))
    results, errors = [None] * len(calls), []

    def run(i, fn):
        try:
            barrier.wait()
            results[i] = fn()
        except Exception as error:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(error)

    threads = [threading.Thread(target=run, args=(i, fn)) for i, fn in enumerate(calls)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    return results


@pytest.mark.parametrize("attempts", [2, 6])
def test_concurrent_duplicate_approvals_commit_once(pg_store, ops, load, attempts):
    load("public_02_exchange")
    pid = ops.create_proposal("C100", "REQ-1", "O100", "L1", "exchange", "TEE-L").proposal.proposal_id
    results = _race([lambda: commit_confirmed_proposal(pg_store, "C100", "REQ-1", pid)] * attempts)

    assert all(r.status == "completed" for r in results)
    assert sum(not r.replayed for r in results) == 1
    assert count(pg_store, "exchanges") == 1 and product_stock(pg_store, "TEE-L") == 2


def test_concurrent_requests_for_the_same_line_commit_once(pg_store, ops, load):
    load("public_01_refund_boundary")
    a = ops.create_proposal("C100", "REQ-A", "O100", "L1", "refund").proposal.proposal_id
    b = ops.create_proposal("C100", "REQ-B", "O100", "L1", "refund").proposal.proposal_id
    results = _race([lambda: commit_confirmed_proposal(pg_store, "C100", "REQ-A", a),
                     lambda: commit_confirmed_proposal(pg_store, "C100", "REQ-B", b)])

    assert sorted(r.status for r in results) == ["completed", "rejected"]
    loser = next(r for r in results if r.status == "rejected")
    assert loser.reason_code is ReasonCode.ALREADY_RETURNED
    assert count(pg_store, "refunds") == 1 and count(pg_store, "returns") == 1
