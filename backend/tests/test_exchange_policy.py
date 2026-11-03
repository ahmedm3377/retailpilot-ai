from __future__ import annotations

import pytest

from retail_domain.policy import ReasonCode, assess_exchange, list_exchange_options
from tests.conftest import make_line, make_state

R = ReasonCode


def test_public_tee_m_to_tee_l_exchange(public_cases):
    state = public_cases["public_02_exchange"].initial_state.model_copy(deep=True)
    result = assess_exchange(state, "C100", "O100", "L1", "TEE-L")
    assert result.eligible and result.reason_code is R.ELIGIBLE
    assert result.remedy == "exchange"
    assert (result.original_sku, result.replacement_sku) == ("TEE-M", "TEE-L")
    assert result.price_cents == 4000
    assert result.replacement_stock > 0


def test_public_stale_stock_scenario_rejects_after_stock_hits_zero(public_cases):
    state = public_cases["public_05_stale_stock"].initial_state.model_copy(deep=True)
    assert assess_exchange(state, "C100", "O100", "L1", "TEE-L").eligible
    next(p for p in state.products if p.sku == "TEE-L").stock = 0
    assert assess_exchange(state, "C100", "O100", "L1", "TEE-L").reason_code is R.REPLACEMENT_OUT_OF_STOCK


@pytest.mark.parametrize("replacement_sku, code", [
    ("WIDGET-B", R.ELIGIBLE),
    ("WIDGET-A", R.INVALID_REPLACEMENT),       # same SKU / same size
    ("GADGET-B", R.INVALID_REPLACEMENT),       # other product, same price
    ("WIDGET-C", R.INVALID_REPLACEMENT),       # unequal catalog price
    ("WIDGET-D", R.REPLACEMENT_OUT_OF_STOCK),  # zero stock
    ("WIDGET-Z", R.SKU_NOT_FOUND),
    (None, R.CLARIFICATION_REQUIRED),
    ("", R.CLARIFICATION_REQUIRED),
])
def test_exchange_rules(replacement_sku, code):
    result = assess_exchange(make_state(), "CUST-9", "ORD-77", "LN-1", replacement_sku)
    assert result.reason_code is code
    assert result.eligible is (code is R.ELIGIBLE)
    assert (result.price_cents is not None) is (code is R.ELIGIBLE)


def test_same_size_different_sku_is_rejected():
    products = [
        {"sku": "P-M-1", "product_id": "P", "size": "M", "price_cents": 100, "stock": 1},
        {"sku": "P-M-2", "product_id": "P", "size": "M", "price_cents": 100, "stock": 5},
    ]
    state = make_state(products=products, lines=[make_line(sku="P-M-1")])
    assert assess_exchange(state, "CUST-9", "ORD-77", "LN-1", "P-M-2").reason_code is R.INVALID_REPLACEMENT


def test_price_compares_catalog_prices_not_paid_amount():
    # Paid a discounted 1234 on a 2500 catalog item; replacement at 2500 is still the same price.
    result = assess_exchange(make_state(lines=[make_line(paid_cents=1234)]),
                             "CUST-9", "ORD-77", "LN-1", "WIDGET-B")
    assert result.eligible and result.price_cents == 2500


def test_negative_stock_is_out_of_stock():
    state = make_state()
    state.products[1].stock = -3
    assert assess_exchange(state, "CUST-9", "ORD-77", "LN-1", "WIDGET-B").reason_code is R.REPLACEMENT_OUT_OF_STOCK


def test_original_sku_missing_from_catalog():
    state = make_state(lines=[make_line(sku="GHOST")])
    assert assess_exchange(state, "CUST-9", "ORD-77", "LN-1", "WIDGET-B").reason_code is R.SKU_NOT_FOUND


@pytest.mark.parametrize("state_kwargs, ids, code", [
    ({"age_days": 31}, ("CUST-9", "ORD-77", "LN-1"), R.DELIVERY_DATE_OUT_OF_RANGE),
    ({"lines": [make_line(unopened=False)]}, ("CUST-9", "ORD-77", "LN-1"), R.ITEM_OPENED),
    ({"lines": [make_line(final_sale=True)]}, ("CUST-9", "ORD-77", "LN-1"), R.FINAL_SALE),
    ({"lines": [make_line(returned=True)]}, ("CUST-9", "ORD-77", "LN-1"), R.ALREADY_RETURNED),
    ({"status": "shipped"}, ("CUST-9", "ORD-77", "LN-1"), R.NOT_DELIVERED),
    ({}, ("OTHER", "ORD-77", "LN-1"), R.CUSTOMER_MISMATCH),
    ({"lines": [make_line("A"), make_line("B")]}, ("CUST-9", "ORD-77", None), R.CLARIFICATION_REQUIRED),
])
def test_ineligible_original_blocks_exchange(state_kwargs, ids, code):
    result = assess_exchange(make_state(**state_kwargs), *ids, "WIDGET-B")
    assert not result.eligible and result.reason_code is code


def test_list_exchange_options_assesses_every_other_size():
    options = list_exchange_options(make_state(), "CUST-9", "ORD-77", "LN-1")
    assert {o.replacement_sku: o.reason_code for o in options} == {
        "WIDGET-B": R.ELIGIBLE,
        "WIDGET-C": R.INVALID_REPLACEMENT,
        "WIDGET-D": R.REPLACEMENT_OUT_OF_STOCK,
    }


def test_list_exchange_options_empty_for_ineligible_line():
    assert list_exchange_options(make_state(age_days=99), "CUST-9", "ORD-77", "LN-1") == ()


@pytest.mark.parametrize("replacement_sku", ["WIDGET-B", "WIDGET-C", "WIDGET-D", "GADGET-B", "NOPE", None])
def test_exchange_policy_never_mutates_state(replacement_sku):
    state = make_state()
    before = state.model_dump_json()
    assess_exchange(state, "CUST-9", "ORD-77", "LN-1", replacement_sku)
    list_exchange_options(state, "CUST-9", "ORD-77", "LN-1")
    assert state.model_dump_json() == before
