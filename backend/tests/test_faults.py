from __future__ import annotations

import pytest

from retail_domain.models import StockChangeFault, TimeoutAfterCommitFault
from app.domain.outcomes import PostCommitTimeout
from retail_domain.policy import ReasonCode
from app.services.faults import FaultError, FaultInjector
from app.services.recovery import reconcile_request
from app.services.transactions import commit_confirmed_proposal
from tests.conftest import count, line_returned, product_stock

pytestmark = pytest.mark.postgres


def test_public_05_stock_change_before_approval(pg_store, ops, load, public_cases):
    case = public_cases["public_05_stale_stock"]
    load(case.case_id)
    injector = FaultInjector(case.faults)

    assert injector.apply_before_turn(pg_store, 1) == []
    proposal = ops.create_proposal("C100", case.request_id, "O100", "L1", "exchange", "TEE-L").proposal
    events = injector.apply_before_turn(pg_store, 2)
    assert events == [{"type": "stock_set", "sku": "TEE-L", "old_stock": 3, "new_stock": 0}]

    result = commit_confirmed_proposal(pg_store, "C100", case.request_id, proposal.proposal_id,
                                       injector=injector)
    assert result.status == "rejected" and result.reason_code is ReasonCode.REPLACEMENT_OUT_OF_STOCK
    state = pg_store.export_final_state()
    assert product_stock(pg_store, "TEE-L") == 0          # external change persists
    assert product_stock(pg_store, "TEE-M") == 2
    assert not line_returned(pg_store, "O100", "L1")
    assert state["returns"] == state["exchanges"] == state["quarantine"] == [] and state["idempotency"] == {}
    external = [e for e in pg_store.audit_log() if e["actor"] == "external"]
    assert [(e["tool"], e["outcome"]) for e in external] == [("stock_change", "applied")]


def test_stock_change_applies_once_per_run(pg_store, load):
    load("public_05_stale_stock")
    injector = FaultInjector([StockChangeFault(type="stock_change", before_turn=2, sku="TEE-L", new_stock=7)])
    assert len(injector.apply_before_turn(pg_store, 2)) == 1
    with pg_store.connect() as conn:
        conn.execute("UPDATE products SET stock = 1 WHERE sku = 'TEE-L'")
    assert injector.apply_before_turn(pg_store, 2) == []
    assert product_stock(pg_store, "TEE-L") == 1


@pytest.mark.parametrize("sku, new_stock", [("NOPE", 1), ("TEE-L", -1)])
def test_invalid_stock_change_is_a_visible_error(pg_store, load, sku, new_stock):
    load("public_05_stale_stock")
    injector = FaultInjector([StockChangeFault(type="stock_change", before_turn=1, sku=sku, new_stock=new_stock)])
    before = pg_store.export_final_state()
    with pytest.raises(FaultError):
        injector.apply_before_turn(pg_store, 1)
    assert pg_store.export_final_state() == before


def test_public_06_timeout_after_commit_then_reconcile_and_replay(pg_store, ops, load, public_cases):
    case = public_cases["public_06_timeout_replay"]
    load(case.case_id)
    injector = FaultInjector(case.faults)
    proposal = ops.create_proposal("C100", case.request_id, "O100", "L1", "refund").proposal

    with pytest.raises(PostCommitTimeout) as raised:
        commit_confirmed_proposal(pg_store, "C100", case.request_id, proposal.proposal_id, injector=injector)
    assert raised.value.request_id == case.request_id

    # Durable despite the timeout.
    status = reconcile_request(pg_store, "C100", case.request_id)
    assert status.state == "completed"
    assert status.result.refund_amount_cents == 3000 and status.result.proposal_id == proposal.proposal_id
    assert count(pg_store, "refunds") == 1 and line_returned(pg_store, "O100", "L1")

    replay = commit_confirmed_proposal(pg_store, "C100", case.request_id, proposal.proposal_id,
                                       injector=injector)
    assert replay.replayed and replay.model_copy(update={"replayed": False}) == status.result
    assert count(pg_store, "refunds") == 1 and count(pg_store, "quarantine") == 1
    tools = [(e["tool"], e["outcome"]) for e in pg_store.audit_log() if e["tool"] != "create_proposal"]
    assert tools == [("commit_confirmed_proposal", "committed"), ("timeout_after_commit", "raised"),
                     ("commit_confirmed_proposal", "replayed")]


def test_rejected_commit_does_not_consume_timeout(pg_store, ops, load):
    load("public_01_refund_boundary")
    injector = FaultInjector([TimeoutAfterCommitFault(type="timeout_after_commit")])
    proposal = ops.create_proposal("C100", "REQ-1", "O100", "L1", "refund").proposal

    assert commit_confirmed_proposal(pg_store, "C100", "REQ-1", None, injector=injector).status == "rejected"
    assert commit_confirmed_proposal(pg_store, "C100", "REQ-1", "PROP-404", injector=injector).status == "rejected"
    assert injector.timeout_pending
    with pytest.raises(PostCommitTimeout):
        commit_confirmed_proposal(pg_store, "C100", "REQ-1", proposal.proposal_id, injector=injector)
    assert not injector.timeout_pending


def test_reconcile_before_any_commit_reports_not_found(pg_store, load):
    load("public_01_refund_boundary")
    assert reconcile_request(pg_store, "C100", "REQ-nothing").state == "not_found"
