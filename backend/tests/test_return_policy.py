from __future__ import annotations

from datetime import date

import pytest

from retail_domain.policy import (
    ReasonCode,
    assess_refund,
    check_return_eligibility,
    resolve_order_line,
)
from tests.conftest import make_line, make_state

R = ReasonCode


@pytest.mark.parametrize("age_days, code", [
    (0, R.ELIGIBLE),
    (1, R.ELIGIBLE),
    (30, R.ELIGIBLE),
    (31, R.DELIVERY_DATE_OUT_OF_RANGE),
    (400, R.DELIVERY_DATE_OUT_OF_RANGE),
    (-1, R.DELIVERY_DATE_OUT_OF_RANGE),
])
def test_return_window_is_0_to_30_days_inclusive(age_days, code):
    result = check_return_eligibility(make_state(age_days=age_days), "CUST-9", "ORD-77", "LN-1")
    assert result.reason_code is code
    assert result.eligible is (code is R.ELIGIBLE)
    assert result.delivery_age_days == age_days


def test_window_uses_state_as_of_across_month_and_leap_day():
    # 2032-03-01 minus 30 days is 2032-01-31 in a leap year.
    state = make_state(as_of=date(2032, 3, 1), age_days=30)
    assert state.orders[0].delivered_on == date(2032, 1, 31)
    assert check_return_eligibility(state, "CUST-9", "ORD-77", "LN-1").eligible


def test_public_boundary_and_expired_cases(public_cases):
    on_boundary = public_cases["public_01_refund_boundary"].initial_state.model_copy(deep=True)
    expired = public_cases["public_03_expired"].initial_state.model_copy(deep=True)
    assert check_return_eligibility(on_boundary, "C100", "O100", "L1").delivery_age_days == 30
    assert assess_refund(on_boundary, "C100", "O100", "L1").eligible
    result = assess_refund(expired, "C100", "O100", "L1")
    assert result.reason_code is R.DELIVERY_DATE_OUT_OF_RANGE
    assert result.refund_amount_cents is None


def test_public_refund_pays_paid_cents_not_catalog_price(public_cases):
    state = public_cases["public_01_refund_boundary"].initial_state.model_copy(deep=True)
    catalog = {p.sku: p.price_cents for p in state.products}
    result = assess_refund(state, "C100", "O100", "L1")
    assert result.remedy == "refund"
    assert result.refund_amount_cents == 3000
    assert catalog["TEE-M"] == 4000


def test_refund_amount_is_line_paid_cents_for_synthetic_values():
    state = make_state(lines=[make_line(paid_cents=1)])
    assert assess_refund(state, "CUST-9", "ORD-77", "LN-1").refund_amount_cents == 1


@pytest.mark.parametrize("customer_id, order_id, line_id, code", [
    ("SOMEONE-ELSE", "ORD-77", "LN-1", R.CUSTOMER_MISMATCH),
    ("CUST-9", "ORD-404", "LN-1", R.ORDER_NOT_FOUND),
    ("CUST-9", "ORD-77", "LN-404", R.LINE_NOT_FOUND),
    ("CUST-9", None, "LN-1", R.CLARIFICATION_REQUIRED),
    ("CUST-9", "", "LN-1", R.CLARIFICATION_REQUIRED),
])
def test_lookup_failures(customer_id, order_id, line_id, code):
    state = make_state()
    assert resolve_order_line(state, customer_id, order_id, line_id).reason_code is code
    refund = assess_refund(state, customer_id, order_id, line_id)
    assert not refund.eligible and refund.reason_code is code
    assert refund.refund_amount_cents is None


def test_customer_mismatch_does_not_leak_line_details():
    result = check_return_eligibility(make_state(), "SOMEONE-ELSE", "ORD-77", "LN-1")
    assert result.sku is None and result.paid_cents is None and result.candidate_line_ids == ()


def test_missing_line_id_on_multi_line_order_requires_clarification(public_cases):
    state = public_cases["public_04_clarify"].initial_state.model_copy(deep=True)
    result = resolve_order_line(state, "C100", "O100", None)
    assert result.reason_code is R.CLARIFICATION_REQUIRED
    assert result.candidate_line_ids == ("L1", "L2")


def test_missing_line_id_on_single_line_order_resolves():
    result = resolve_order_line(make_state(), "CUST-9", "ORD-77", None)
    assert result.ok and result.line_id == "LN-1"


def test_duplicate_line_ids_require_clarification():
    state = make_state(lines=[make_line("DUP"), make_line("DUP", paid_cents=99)])
    assert resolve_order_line(state, "CUST-9", "ORD-77", "DUP").reason_code is R.CLARIFICATION_REQUIRED


@pytest.mark.parametrize("status", ["shipped", "processing", "cancelled", "Delivered"])
def test_only_delivered_orders_qualify(status):
    result = check_return_eligibility(make_state(status=status), "CUST-9", "ORD-77", "LN-1")
    assert result.reason_code is R.NOT_DELIVERED


def test_delivered_without_date_is_not_eligible():
    state = make_state()
    state.orders[0].delivered_on = None
    assert check_return_eligibility(state, "CUST-9", "ORD-77", "LN-1").reason_code is R.NOT_DELIVERED


@pytest.mark.parametrize("overrides, code", [
    ({"unopened": False}, R.ITEM_OPENED),
    ({"final_sale": True}, R.FINAL_SALE),
    ({"returned": True}, R.ALREADY_RETURNED),
])
def test_unit_condition_rules(overrides, code):
    state = make_state(lines=[make_line(**overrides)])
    result = assess_refund(state, "CUST-9", "ORD-77", "LN-1")
    assert not result.eligible and result.reason_code is code


def test_each_line_is_judged_independently():
    state = make_state(lines=[make_line("A", returned=True), make_line("B")])
    assert assess_refund(state, "CUST-9", "ORD-77", "A").reason_code is R.ALREADY_RETURNED
    assert assess_refund(state, "CUST-9", "ORD-77", "B").eligible


@pytest.mark.parametrize("call", [
    lambda s: resolve_order_line(s, "CUST-9", "ORD-77", "LN-1"),
    lambda s: check_return_eligibility(s, "CUST-9", "ORD-77", "LN-1"),
    lambda s: assess_refund(s, "CUST-9", "ORD-77", "LN-1"),
    lambda s: assess_refund(s, "CUST-9", "ORD-77", "LN-404"),
    lambda s: assess_refund(s, "SOMEONE-ELSE", "ORD-77", "LN-1"),
])
def test_return_policy_never_mutates_state(call):
    state = make_state()
    before = state.model_dump_json()
    call(state)
    assert state.model_dump_json() == before


def test_public_states_unchanged_by_refund_assessment(public_cases):
    for case in public_cases.values():
        state = case.initial_state
        before = state.model_dump_json()
        for order in state.orders:
            for line in order.lines:
                assess_refund(state, case.customer_id, order.order_id, line.line_id)
        assert state.model_dump_json() == before
