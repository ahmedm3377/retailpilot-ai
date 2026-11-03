"""cancel_proposal and FaultInjector.restore (real PostgreSQL)."""
from __future__ import annotations

import pytest

from app.domain.outcomes import WorkflowCode
from app.services.faults import FaultInjector
from app.services.transactions import cancel_proposal, commit_confirmed_proposal
from tests.conftest import count

pytestmark = pytest.mark.postgres


def test_cancel_invalidates_without_business_effects(pg_store, ops, load):
    load("public_01_refund_boundary")
    proposal = ops.create_proposal("C100", "REQ-1", "O100", "L1", "refund").proposal
    before = pg_store.export_final_state()
    result = cancel_proposal(pg_store, "C100", "REQ-1", proposal.proposal_id)
    assert result.status == "cancelled"
    assert pg_store.export_final_state() == before
    commit = commit_confirmed_proposal(pg_store, "C100", "REQ-1", proposal.proposal_id)
    assert commit.status == "rejected" and commit.reason_code is WorkflowCode.PROPOSAL_NOT_PENDING
    assert count(pg_store, "refunds") == 0
    assert [e["tool"] for e in pg_store.audit_log()][-2:] == ["cancel_proposal", "commit_confirmed_proposal"]


@pytest.mark.parametrize("customer, request_id, code", [
    ("C999", "REQ-1", WorkflowCode.PROPOSAL_MISMATCH),
    ("C100", "REQ-other", WorkflowCode.PROPOSAL_MISMATCH),
])
def test_cancel_refuses_other_bindings(pg_store, ops, load, customer, request_id, code):
    load("public_01_refund_boundary")
    proposal = ops.create_proposal("C100", "REQ-1", "O100", "L1", "refund").proposal
    result = cancel_proposal(pg_store, customer, request_id, proposal.proposal_id)
    assert result.status == "rejected" and result.reason_code is code
    assert commit_confirmed_proposal(pg_store, "C100", "REQ-1", proposal.proposal_id).status == "completed"


def test_restored_injector_never_reapplies_stock_change(pg_store, load, public_cases):
    case = public_cases["public_05_stale_stock"]
    load(case.case_id)
    FaultInjector(case.faults).apply_before_turn(pg_store, 2)
    with pg_store.connect() as conn:
        conn.execute("UPDATE products SET stock = 5 WHERE sku = 'TEE-L'")
    restored = FaultInjector.restore(case.faults, pg_store.audit_log())
    assert restored.apply_before_turn(pg_store, 2) == []
    assert next(p["stock"] for p in pg_store.export_final_state()["products"] if p["sku"] == "TEE-L") == 5


