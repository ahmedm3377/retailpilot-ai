from __future__ import annotations

from datetime import date

import pytest

from app.domain.outcomes import WorkflowCode
from retail_domain.policy import ReasonCode
from app.services.transactions import commit_confirmed_proposal
from tests.conftest import count, line_returned, make_line, make_state, product_stock

pytestmark = pytest.mark.postgres


def test_public_01_refund_at_30_day_boundary(pg_store, ops, load):
    load("public_01_refund_boundary")
    proposal = ops.create_proposal("C100", "REQ-public_01", "O100", "L1", "refund").proposal
    result = commit_confirmed_proposal(pg_store, "C100", "REQ-public_01", proposal.proposal_id)

    assert result.status == "completed" and result.reason_code is WorkflowCode.COMMITTED
    assert result.refund_amount_cents == 3000 and not result.replayed
    state = pg_store.export_final_state()
    assert [(r["amount_cents"], r["currency"], r["proposal_id"]) for r in state["refunds"]] == [
        (3000, "USD", proposal.proposal_id)]
    assert [(q["sku"], q["quantity"]) for q in state["quarantine"]] == [("TEE-M", 1)]
    assert [(r["order_id"], r["line_id"], r["remedy"]) for r in state["returns"]] == [("O100", "L1", "refund")]
    assert state["exchanges"] == []
    assert line_returned(pg_store, "O100", "L1")
    assert product_stock(pg_store, "TEE-M") == 2  # quarantine never adds sellable stock
    assert state["idempotency"]["REQ-public_01"]["status"] == "completed"


def test_public_03_expired_denied_without_mutation(pg_store, ops, load):
    load("public_03_expired")
    before = pg_store.export_final_state()
    outcome = ops.create_proposal("C100", "REQ-3", "O100", "L1", "refund")
    assert outcome.kind == "deny" and outcome.reason_code is ReasonCode.DELIVERY_DATE_OUT_OF_RANGE
    assert pg_store.export_final_state() == before


def test_public_04_clarify_then_refund_selected_line(pg_store, ops, load):
    load("public_04_clarify")
    assert ops.create_proposal("C100", "REQ-4", "O100", None, "refund").kind == "clarify"
    proposal = ops.create_proposal("C100", "REQ-4", "O100", "L1", "refund").proposal
    result = commit_confirmed_proposal(pg_store, "C100", "REQ-4", proposal.proposal_id)
    assert result.status == "completed" and result.line_id == "L1" and result.refund_amount_cents == 3000
    assert not line_returned(pg_store, "O100", "L2")


@pytest.mark.parametrize("age_days, ok", [(0, True), (30, True), (31, False), (-1, False)])
def test_window_boundaries_through_the_database(pg_store, ops, load, age_days, ok):
    load(make_state(as_of=date(2029, 12, 31), age_days=age_days))
    outcome = ops.create_proposal("CUST-9", "REQ-1", "ORD-77", "LN-1", "refund")
    assert (outcome.kind == "proposed") is ok
    if ok:
        result = commit_confirmed_proposal(pg_store, "CUST-9", "REQ-1", outcome.proposal.proposal_id)
        assert result.refund_amount_cents == 1234


@pytest.mark.parametrize("sql, code", [
    ("UPDATE order_lines SET unopened = FALSE", ReasonCode.ITEM_OPENED),
    ("UPDATE order_lines SET final_sale = TRUE", ReasonCode.FINAL_SALE),
    ("UPDATE orders SET status = 'cancelled'", ReasonCode.NOT_DELIVERED),
    ("UPDATE fixture_meta SET as_of = as_of + 60", ReasonCode.DELIVERY_DATE_OUT_OF_RANGE),
    ("UPDATE orders SET customer_id = 'SOMEONE-ELSE'", ReasonCode.CUSTOMER_MISMATCH),
])
def test_commit_revalidates_current_state(pg_store, ops, load, sql, code):
    load(make_state())
    proposal = ops.create_proposal("CUST-9", "REQ-1", "ORD-77", "LN-1", "refund").proposal
    with pg_store.connect() as conn:
        conn.execute(sql)
    before = pg_store.export_final_state()

    result = commit_confirmed_proposal(pg_store, "CUST-9", "REQ-1", proposal.proposal_id)
    assert result.status == "rejected" and result.reason_code is code
    assert pg_store.export_final_state() == before
    assert count(pg_store, "refunds") == 0


def test_refund_never_uses_catalog_price(pg_store, ops, load):
    products = [{"sku": "HAT-1", "product_id": "HAT", "size": "1", "price_cents": 99999, "stock": 0}]
    load(make_state(products=products, lines=[make_line(sku="HAT-1", paid_cents=7)]))
    proposal = ops.create_proposal("CUST-9", "REQ-1", "ORD-77", "LN-1", "refund").proposal
    result = commit_confirmed_proposal(pg_store, "CUST-9", "REQ-1", proposal.proposal_id)
    assert result.refund_amount_cents == 7
    assert product_stock(pg_store, "HAT-1") == 0
