"""Multi-turn flows through the graph and checkpointer, with a SCRIPTED fake model."""
from __future__ import annotations

import pytest

from tests.conftest import count, make_line, make_state, product_stock
from tests.fakes import CUSTOMER, FakeModel, exchange_turn, handoff, refund_turn, say

pytestmark = pytest.mark.postgres


def run(runtime, message, n, confirmed=None, customer="C100", request="REQ-1", session="s1"):
    return runtime.handle_customer_turn(session_id=session, authenticated_customer_id=customer,
                                        request_id=request, message=message, turn_number=n,
                                        confirmed_proposal_id=confirmed)


def test_clarify_then_propose_then_complete(pg_store, load, make_runtime):
    load("public_04_clarify")
    model = FakeModel().add(CUSTOMER, handoff("refund", "O100", None, needs_clarification=True,
                                              question="Which item?"), say("Which item: L1 or L2?"))
    refund_turn(model).add(CUSTOMER, say("Your refund of $30.00 is complete."))
    runtime = make_runtime(model)

    first = run(runtime, "I want to return an item from O100.", 1)
    assert first.decision == "clarify" and first.proposal is None
    second = run(runtime, "I mean line L1, the size M tee. Refund it please.", 2)
    assert second.decision == "propose"
    # The second handoff call saw the earlier question via the checkpointed state.
    customer_calls = [m for role, m in model.seen if role == CUSTOMER]
    assert "question_asked" in customer_calls[2][1]["content"]
    third = run(runtime, "Yes, confirm the proposed refund.", 3, confirmed=second.proposal["proposal_id"])
    assert third.decision == "completed" and count(pg_store, "refunds") == 1


def test_expired_request_is_denied(pg_store, load, make_runtime):
    load("public_03_expired")
    model = refund_turn(FakeModel(), explanation="Sorry, this item is outside the 30-day window.")
    result = run(make_runtime(model), "Please refund line L1 of order O100.", 1)
    assert result.decision == "deny" and result.proposal is None and count(pg_store, "returns") == 0


def test_exchange_flow_decrements_stock_once(pg_store, load, make_runtime):
    load("public_02_exchange")
    model = exchange_turn(FakeModel(), "L", "TEE-L").add(CUSTOMER, say("Your exchange is complete."))
    runtime = make_runtime(model)
    proposal = run(runtime, "Exchange line L1 in order O100 for size L.", 1).proposal
    assert proposal["replacement_sku"] == "TEE-L"
    assert run(runtime, "Yes", 2, confirmed=proposal["proposal_id"]).decision == "completed"
    assert product_stock(pg_store, "TEE-L") == 2 and count(pg_store, "exchanges") == 1


def test_agent_sku_outside_options_is_not_used(pg_store, load, make_runtime):
    load("public_02_exchange")
    model = exchange_turn(FakeModel(), "L", "JACKET-L")  # agent picks another product
    result = run(make_runtime(model), "Exchange line L1 in order O100 for size L.", 1)
    assert result.decision == "propose" and result.proposal["replacement_sku"] == "TEE-L"
    assert result.internal_status["replacement_source"] == "requested_size"


def test_exchange_without_size_asks_which_size(pg_store, load, make_runtime):
    load("public_02_exchange")
    model = (FakeModel().add(CUSTOMER, handoff("exchange"), say("Which size would you like instead?"))
             .add("retail_operations_agent", {"action": "recommend", "remedy": "exchange", "order_id": "O100",
                                              "line_id": "L1"}))
    result = run(make_runtime(model), "Exchange line L1 of O100 please.", 1)
    assert result.decision == "clarify" and count(pg_store, "proposals") == 0


def test_variant_ids_dates_and_amounts(pg_store, load, make_runtime):
    load(make_state(customer_id="CX-42", order_id="ORD-9", lines=[make_line("LN-3", paid_cents=777)]))
    model = refund_turn(FakeModel(), "ORD-9", "LN-3").add(CUSTOMER, say("Done."))
    runtime = make_runtime(model)
    proposal = run(runtime, "refund ORD-9 LN-3", 1, customer="CX-42").proposal
    assert proposal["amount_cents"] == 777
    assert run(runtime, "ok", 2, confirmed=proposal["proposal_id"], customer="CX-42").decision == "completed"
