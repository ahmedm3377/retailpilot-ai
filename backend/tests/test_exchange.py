from __future__ import annotations

import pytest

from retail_domain.policy import ReasonCode
from app.services.transactions import commit_confirmed_proposal
from tests.conftest import count, line_returned, make_state, product_stock

pytestmark = pytest.mark.postgres


def test_public_02_exchange(pg_store, ops, load):
    load("public_02_exchange")
    proposal = ops.create_proposal("C100", "REQ-2", "O100", "L1", "exchange", "TEE-L").proposal
    result = commit_confirmed_proposal(pg_store, "C100", "REQ-2", proposal.proposal_id)

    assert result.status == "completed" and result.remedy == "exchange"
    assert (result.original_sku, result.replacement_sku) == ("TEE-M", "TEE-L")
    assert result.refund_amount_cents is None
    state = pg_store.export_final_state()
    assert product_stock(pg_store, "TEE-L") == 2      # 3 - 1, exactly once
    assert product_stock(pg_store, "TEE-M") == 2      # original not restocked
    assert [(e["original_sku"], e["replacement_sku"]) for e in state["exchanges"]] == [("TEE-M", "TEE-L")]
    assert [q["sku"] for q in state["quarantine"]] == ["TEE-M"]
    assert state["refunds"] == []
    assert line_returned(pg_store, "O100", "L1")


@pytest.mark.parametrize("replacement", ["WIDGET-A", "GADGET-B", "WIDGET-C", "WIDGET-D", "NOPE"])
def test_invalid_replacements_never_commit(pg_store, ops, load, replacement):
    load(make_state())
    before = pg_store.export_final_state()
    outcome = ops.create_proposal("CUST-9", "REQ-1", "ORD-77", "LN-1", "exchange", replacement)
    assert outcome.kind == "deny"
    assert pg_store.export_final_state() == before


def test_stock_reaching_zero_before_commit_rejects_with_no_effects(pg_store, ops, load):
    load(make_state())
    proposal = ops.create_proposal("CUST-9", "REQ-1", "ORD-77", "LN-1", "exchange", "WIDGET-B").proposal
    with pg_store.connect() as conn:
        conn.execute("UPDATE products SET stock = 0 WHERE sku = 'WIDGET-B'")
    before = pg_store.export_final_state()

    result = commit_confirmed_proposal(pg_store, "CUST-9", "REQ-1", proposal.proposal_id)
    assert result.status == "rejected" and result.reason_code is ReasonCode.REPLACEMENT_OUT_OF_STOCK
    assert pg_store.export_final_state() == before
    with pg_store.connect() as conn:
        status = conn.execute("SELECT status FROM proposals").fetchone()["status"]
    assert status == "invalidated"


def test_catalog_price_change_before_commit_rejects(pg_store, ops, load):
    load(make_state())
    proposal = ops.create_proposal("CUST-9", "REQ-1", "ORD-77", "LN-1", "exchange", "WIDGET-B").proposal
    with pg_store.connect() as conn:
        conn.execute("UPDATE products SET price_cents = price_cents + 1 WHERE sku = 'WIDGET-B'")
    result = commit_confirmed_proposal(pg_store, "CUST-9", "REQ-1", proposal.proposal_id)
    assert result.reason_code is ReasonCode.INVALID_REPLACEMENT
    assert count(pg_store, "exchanges") == 0 and product_stock(pg_store, "WIDGET-B") == 7


def test_exchange_takes_the_last_unit(pg_store, ops, load):
    state = make_state()
    state.products[1].stock = 1
    load(state)
    proposal = ops.create_proposal("CUST-9", "REQ-1", "ORD-77", "LN-1", "exchange", "WIDGET-B").proposal
    assert commit_confirmed_proposal(pg_store, "CUST-9", "REQ-1", proposal.proposal_id).status == "completed"
    assert product_stock(pg_store, "WIDGET-B") == 0
