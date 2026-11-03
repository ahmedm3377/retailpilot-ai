"""What the customer explicitly NAMED must match the resolved target.

Found by the evaluation suite (var_14, adv_04, adv_05): the customer agent only sees the
customer's own orders, so a named foreign order became a proposal on the customer's own line,
and a named cross-product SKU became a same-product size change. Graph tests use the SCRIPTED
FakeModel (not real inference) to reproduce exactly that model behaviour.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.business.request_checks import WRONG_NUMBER, check_named_references
from app.agents.schemas import CustomerHandoff
from retail_domain.models import State
from tests.conftest import count, make_line, make_state
from tests.fakes import CUSTOMER, OPS, FakeModel, exchange_turn, handoff, refund_turn, say


def two_customers(lines=None) -> State:
    state = make_state(lines=lines or [make_line("LN-1", "WIDGET-A"), make_line("LN-2", "WIDGET-B")])
    data = state.model_dump(mode="json")
    data["orders"].append({"order_id": "ORD-88", "customer_id": "CUST-5", "status": "delivered",
                           "delivered_on": (date(2031, 3, 15) - timedelta(days=3)).isoformat(),
                           "lines": [make_line("LN-9", "GADGET-B")]})
    return State.model_validate(data)


def resolved(intent="refund", order_id="ORD-77", line_id="LN-1", size=None) -> CustomerHandoff:
    return CustomerHandoff.model_validate(handoff(intent, order_id, line_id, requested_size=size))


@pytest.mark.parametrize("message", ["Refund line LN-9 of order ORD-88.", "refund ord-88 please",
                                     "Refund order ORD-88 and ORD-77"])
def test_named_foreign_order_asks_like_an_unknown_order(message):
    facts, sku = check_named_references(message, resolved(), "CUST-9", two_customers())
    assert facts["question"].startswith(WRONG_NUMBER) and "ORD-88" not in facts["question"] and sku is None


@pytest.mark.parametrize("message", ["Refund line LN-1 of order ORD-77.", "Refund my widget please.",
                                     "Refund the item from ORD-777 maybe"])  # ORD-777 is not a real ID
def test_matching_or_unnamed_references_pass(message):
    assert check_named_references(message, resolved(), "CUST-9", two_customers()) == (None, None)


def test_named_line_that_differs_from_the_resolved_line_clarifies():
    facts, _ = check_named_references("Refund LN-2 of ORD-77", resolved(line_id="LN-1"), "CUST-9", two_customers())
    assert facts["candidate_line_ids"] == ["LN-1", "LN-2"]


def test_named_replacement_sku_is_used_and_two_are_ambiguous():
    state = two_customers()
    assert check_named_references("Exchange LN-1 of ORD-77 for GADGET-B", resolved("exchange"), "CUST-9",
                                  state) == (None, "GADGET-B")
    assert check_named_references("swap WIDGET-A for widget-c", resolved("exchange"), "CUST-9",
                                  state) == (None, "WIDGET-C")  # the original SKU itself is not a replacement
    facts, sku = check_named_references("Exchange for WIDGET-B or WIDGET-C", resolved("exchange"), "CUST-9", state)
    assert "replacement" in facts["question"] and sku is None


# --- through the real graph + PostgreSQL, with the substituting model behaviour scripted ----------


def turn(runtime, message, n, confirmed=None, customer="CUST-9"):
    return runtime.handle_customer_turn(session_id="s1", authenticated_customer_id=customer, request_id="REQ-9",
                                        message=message, turn_number=n, confirmed_proposal_id=confirmed,
                                        confirmation_requested=confirmed is not None)


@pytest.mark.postgres
def test_substituted_own_order_becomes_a_clarification(pg_store, load, make_runtime):
    load(two_customers(lines=[make_line("LN-1", "WIDGET-A")]))
    # The model only sees ORD-77 and hands it off although the customer named ORD-88.
    model = FakeModel().add(CUSTOMER, handoff("refund", "ORD-77", "LN-1"))
    result = turn(make_runtime(model), "Refund line LN-9 of order ORD-88 to my card.", 1)
    assert result.decision == "clarify" and result.proposal is None
    assert model.calls_for(OPS) == 0  # no handoff to operations
    assert result.internal_status["named_reference_mismatch"] is True
    assert count(pg_store, "refunds") == 0


@pytest.mark.postgres
def test_cross_product_sku_is_denied_by_policy_not_swapped(pg_store, load, make_runtime):
    load(two_customers(lines=[make_line("LN-1", "WIDGET-A")]))
    # The model turns "GADGET-B" into a same-product size B exchange (WIDGET-B).
    model = exchange_turn(FakeModel(), "B", "WIDGET-B", order_id="ORD-77", line_id="LN-1")
    result = turn(make_runtime(model), "Exchange line LN-1 of ORD-77 for GADGET-B.", 1)
    assert result.decision == "deny" and result.proposal is None
    assert result.internal_status["replacement_source"] == "customer_named"
    assert count(pg_store, "exchanges") == 0


@pytest.mark.postgres
def test_named_same_product_sku_still_proposes(pg_store, load, make_runtime):
    load(two_customers(lines=[make_line("LN-1", "WIDGET-A")]))
    model = exchange_turn(FakeModel(), "B", "WIDGET-B", order_id="ORD-77", line_id="LN-1")
    result = turn(make_runtime(model), "Exchange LN-1 of ORD-77 for WIDGET-B please.", 1)
    assert result.decision == "propose" and result.proposal["replacement_sku"] == "WIDGET-B"


@pytest.mark.postgres
def test_public_style_request_without_foreign_ids_is_unchanged(pg_store, load, make_runtime):
    load("public_01_refund_boundary")
    result = turn(make_runtime(refund_turn(FakeModel())), "Refund line L1 of my order O100. It is unopened.", 1,
                  customer="C100")
    assert result.decision == "propose" and result.proposal["amount_cents"] == 3000
