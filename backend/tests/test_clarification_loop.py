"""The two agents' handoff loop: customer -> operations -> customer (clarify) -> updated request -> operations.

The LLM is the SCRIPTED FakeModel (not inference); real-model behaviour is in the real_model tests.
"""
from __future__ import annotations

import pytest

from app.business.request_checks import item_condition_in, merge_pending_request
from app.agents.messages import ITEM_CONDITION_QUESTION, ITEM_OPENED_APOLOGY, UNOPENED_QUESTION
from app.agents.schemas import CustomerHandoff
from app.api.views import audit_event_dto
from tests.conftest import count, make_line, make_state, product_stock
from tests.fakes import CUSTOMER, OPS, FakeModel, exchange_turn, handoff, recommend, say, tool

ASK = {"action": "ask_customer", "reason": "requested_size is missing"}


def turn(runtime, message, n, confirmed=None):
    return runtime.handle_customer_turn(session_id="s1", authenticated_customer_id="C100", request_id="REQ-1",
                                        message=message, turn_number=n, confirmed_proposal_id=confirmed,
                                        confirmation_requested=confirmed is not None)


def handoffs(store):
    return [e for e in store.audit_log() if e["tool"] == "agent_handoff"]


def script_missing_size(model, ops_reply):
    return (model.add(CUSTOMER, handoff("exchange", "O100", "L1"), say("Which size would you like?"))
            .add(OPS, tool("check_return_eligibility", order_id="O100", line_id="L1"),
                 tool("get_exchange_options", order_id="O100", line_id="L1"), ops_reply))


pg = pytest.mark.postgres


@pg
def test_operations_hands_back_then_the_answer_updates_the_request(pg_store, load, make_runtime):
    load("public_02_exchange")
    model = script_missing_size(FakeModel(), ASK)
    # The answer only states the size: intent, order and line must come from the kept request.
    model.add(CUSTOMER, handoff("unknown", None, None, requested_size="L"), say("Please confirm the proposal."))
    model.add(OPS, tool("check_return_eligibility", order_id="O100", line_id="L1"),
              tool("get_exchange_options", order_id="O100", line_id="L1"),
              recommend("exchange", replacement_sku="TEE-L"))
    runtime = make_runtime(model)

    asked = turn(runtime, "I want to exchange my tee", 1)
    assert asked.decision == "clarify" and asked.proposal is None and count(pg_store, "proposals") == 0
    rows = handoffs(pg_store)
    assert [(r["arguments"]["from"], r["arguments"]["to"]) for r in rows] == [(CUSTOMER, OPS), (OPS, CUSTOMER)]
    assert rows[1]["outcome"] == "clarification_needed" and rows[1]["arguments"]["missing"] == "requested_size"
    assert rows[1]["arguments"]["options"] == ["S", "L"]

    answered = turn(runtime, "size L please", 2)
    assert answered.decision == "propose" and answered.proposal["replacement_sku"] == "TEE-L"
    last = handoffs(pg_store)[-1]["arguments"]
    assert (last["from"], last["to"], last["remedy"], last["order_id"], last["line_id"], last["requested_size"]) == (
        CUSTOMER, OPS, "exchange", "O100", "L1", "L")
    assert product_stock(pg_store, "TEE-L") == 3  # nothing executed before Confirm


@pg
def test_a_size_guessed_by_the_agent_is_not_used(pg_store, load, make_runtime):
    load("public_02_exchange")
    runtime = make_runtime(script_missing_size(FakeModel(), recommend("exchange", replacement_sku="TEE-L")))
    result = turn(runtime, "I want to exchange my tee", 1)
    assert result.decision == "clarify" and count(pg_store, "proposals") == 0


@pg
def test_ask_customer_is_ignored_when_nothing_is_missing(pg_store, load, make_runtime):
    load("public_01_refund_boundary")
    model = FakeModel().add(CUSTOMER, handoff("refund", "O100", "L1"), say("Please confirm the proposal."))
    model.add(OPS, tool("check_return_eligibility", order_id="O100", line_id="L1"), ASK)
    result = turn(make_runtime(model), "Refund line L1 of my order O100.", 1)
    assert result.decision == "propose" and result.internal_status["operations_asked_customer"] is True


@pg
def test_an_ineligible_exchange_is_denied_not_asked_about(pg_store, load, make_runtime):
    load("public_03_expired")
    model = script_missing_size(FakeModel(), ASK)
    result = turn(make_runtime(model), "I want to exchange my tee", 1)
    assert result.decision == "deny" and [r["arguments"]["to"] for r in handoffs(pg_store)] == [OPS]


@pg
def test_customer_agent_clarification_keeps_the_intent_for_the_next_turn(pg_store, load, make_runtime):
    load("public_04_clarify")
    model = FakeModel().add(CUSTOMER, handoff("refund", "O100", None), say("Which line, L1 or L2?"),
                            handoff("unknown", None, "L1"), say("Please confirm the proposal."))
    model.add(OPS, tool("check_return_eligibility", order_id="O100", line_id="L1"), recommend("refund"))
    runtime = make_runtime(model)
    assert turn(runtime, "I want to return an item from O100.", 1).decision == "clarify"
    assert handoffs(pg_store) == []  # the Operations Agent gets the request only once it is complete
    result = turn(runtime, "the first one", 2)
    assert result.decision == "propose" and result.proposal["line_id"] == "L1"


@pg
def test_a_complete_request_still_goes_straight_through(pg_store, load, make_runtime):
    load("public_02_exchange")
    result = turn(make_runtime(exchange_turn(FakeModel(), "L", "TEE-L")), "Exchange L1 of O100 for size L.", 1)
    assert result.decision == "propose" and len(handoffs(pg_store)) == 1


CHECK_REFUND = tool("check_return_eligibility", order_id="O100", line_id="L1")


def line_unopened(store):
    return store.export_final_state()["orders"][0]["lines"][0]["unopened"]


def ask_first(make_runtime, model):
    return make_runtime(model, ask_item_condition=True)


def refund_script(*more):
    return (FakeModel().add(CUSTOMER, handoff("refund", "O100", "L1"), *more)
            .add(OPS, CHECK_REFUND, recommend("refund"), CHECK_REFUND, recommend("refund"),
                 CHECK_REFUND, recommend("refund")))


@pg
def test_the_desk_asks_about_the_condition_before_any_proposal(pg_store, load, make_runtime):
    load("public_01_refund_boundary")
    result = turn(ask_first(make_runtime, refund_script()), "Refund line L1 of my order O100.", 1)
    assert (result.decision, result.response) == ("clarify", ITEM_CONDITION_QUESTION)
    assert result.proposal is None and count(pg_store, "proposals") == 0


@pg
@pytest.mark.parametrize("answer", ["unopened", "It is still sealed", "I haven't opened it", "never been opened"])
def test_unopened_gets_the_proposal(pg_store, load, make_runtime, answer):
    load("public_01_refund_boundary")
    runtime = ask_first(make_runtime, refund_script(say("I can refund it. Please confirm the proposal.")))
    turn(runtime, "Refund line L1 of my order O100.", 1)
    result = turn(runtime, answer, 2)
    assert result.decision == "propose" and result.proposal["line_id"] == "L1"
    assert UNOPENED_QUESTION not in result.response and line_unopened(pg_store) is True


@pg
@pytest.mark.parametrize("answer", ["opened", "yes it is open", "I opened it"])
def test_opened_gets_an_apology_and_updates_the_store(pg_store, load, make_runtime, answer):
    load("public_01_refund_boundary")
    runtime = ask_first(make_runtime, refund_script())
    turn(runtime, "Refund line L1 of my order O100.", 1)
    result = turn(runtime, answer, 2)
    assert (result.decision, result.response) == ("deny", ITEM_OPENED_APOLOGY)
    assert result.proposal is None and count(pg_store, "proposals") == 0 and count(pg_store, "refunds") == 0
    assert line_unopened(pg_store) is False
    audit = [e for e in pg_store.audit_log() if e["tool"] == "item_condition"]
    assert [e["outcome"] for e in audit] == ["marked_opened"]
    assert audit[0]["effects"] == [{"type": "line_marked_opened", "order_id": "O100", "line_id": "L1"}]


@pg
@pytest.mark.parametrize("answer", ["yes", "no", "I don't know", "opened it but not used"])
def test_an_unclear_answer_is_asked_again(pg_store, load, make_runtime, answer):
    load("public_01_refund_boundary")
    model = refund_script(handoff("refund", "O100", "L1") | {"item_opened": True})  # the model's guess isn't used
    runtime = ask_first(make_runtime, model)
    turn(runtime, "Refund line L1 of my order O100.", 1)
    result = turn(runtime, answer, 2)
    assert (result.decision, result.response) == ("clarify", ITEM_CONDITION_QUESTION)
    assert line_unopened(pg_store) is True


@pg
def test_saying_it_up_front_needs_no_question(pg_store, load, make_runtime):
    load("public_01_refund_boundary")
    runtime = ask_first(make_runtime, refund_script(say("I can refund it. Please confirm the proposal.")))
    assert turn(runtime, "Refund line L1 of my order O100. It is unopened.", 1).decision == "propose"


@pg
def test_an_ineligible_item_is_denied_without_asking(pg_store, load, make_runtime):
    load("public_03_expired")
    result = turn(ask_first(make_runtime, refund_script(say("Sorry, it is too late."))), "Refund L1 of O100.", 1)
    assert result.decision == "deny" and ITEM_CONDITION_QUESTION not in result.response


@pg
def test_saying_unopened_never_overrides_a_store_that_says_opened(pg_store, load, make_runtime):
    load(make_state(customer_id="C100", order_id="O100", lines=[make_line("L1", unopened=False)]))
    runtime = ask_first(make_runtime, refund_script(say("Sorry, it has been opened.")))
    assert turn(runtime, "Refund L1 of O100, it is unopened", 1).decision == "deny"
    assert line_unopened(pg_store) is False


@pg
def test_an_exchange_asks_for_the_size_first_and_keeps_it(pg_store, load, make_runtime):
    load("public_02_exchange")
    model = script_missing_size(FakeModel(), ASK)
    model.add(CUSTOMER, handoff("unknown", None, None, requested_size="L"), say("Please confirm the proposal."))
    model.add(OPS, *[tool("check_return_eligibility", order_id="O100", line_id="L1"),
                     recommend("exchange", replacement_sku="TEE-L")] * 2)
    runtime = ask_first(make_runtime, model)
    assert turn(runtime, "I want to exchange my tee", 1).decision == "clarify"
    assert turn(runtime, "size L", 2).response == ITEM_CONDITION_QUESTION
    result = turn(runtime, "unopened", 3)
    assert result.decision == "propose" and result.proposal["replacement_sku"] == "TEE-L"


@pg
def test_saying_opened_while_a_proposal_waits_retires_it(pg_store, load, make_runtime):
    load("public_01_refund_boundary")
    runtime = ask_first(make_runtime, refund_script(say("Please confirm the proposal."), handoff("refund", "O100", "L1")))
    proposal = turn(runtime, "Refund L1 of O100. It is unopened.", 1).proposal
    assert turn(runtime, "actually I opened it", 2).response == ITEM_OPENED_APOLOGY
    confirm = turn(runtime, "Confirm", 3, proposal["proposal_id"])
    assert confirm.decision != "completed" and count(pg_store, "refunds") == 0 and line_unopened(pg_store) is False


@pg
def test_the_adapter_path_never_asks_or_writes(pg_store, load, make_runtime):
    load("public_01_refund_boundary")
    model = refund_script(say("Please confirm the proposal."))
    result = turn(make_runtime(model), "Refund line L1 of my order O100.", 1)  # the contract's turn 1 proposes
    assert result.decision == "propose" and UNOPENED_QUESTION in result.response


@pytest.mark.parametrize("message,answering,expected", [
    ("unopened", False, False), ("It is unopened.", False, False), ("not opened", False, False),
    ("I didn't open it", False, False), ("it's sealed", False, False), ("opened", False, True),
    ("I opened the box", False, True), ("I want to open a return", False, None), ("it is open", False, None),
    ("it is open", True, True), ("used", True, True), ("yes", True, None), ("no", True, None),
    ("I opened it, not used", True, None), ("", True, None),
])
def test_item_condition_is_read_by_fixed_rules(message, answering, expected):
    assert item_condition_in(message, answering) is expected


def _handoff(intent="unknown", order=None, line=None, size=None):
    return CustomerHandoff(intent=intent, order_id=order, line_id=line, requested_size=size, needs_clarification=False)


def test_merge_keeps_what_the_answer_leaves_out():
    merged = merge_pending_request(_handoff(size="L"), _handoff("exchange", "O1", "L1").model_dump())
    assert (merged.intent, merged.order_id, merged.line_id, merged.requested_size) == ("exchange", "O1", "L1", "L")


def test_merge_lets_the_customer_change_their_mind():
    pending = _handoff("exchange", "O1", "L1", "L").model_dump()
    merged = merge_pending_request(_handoff("refund"), pending)
    assert (merged.intent, merged.line_id, merged.requested_size) == ("refund", "L1", None)
    assert merge_pending_request(_handoff(order="O2"), pending).line_id is None  # a line belongs to its order
    assert merge_pending_request(_handoff(size="S"), pending).requested_size == "S"


def test_merge_without_a_pending_request_changes_nothing():
    new = _handoff("refund", "O1")
    assert merge_pending_request(new, None) == new


def test_handoff_rows_read_clearly_in_the_timeline():
    dto = audit_event_dto({"event_id": 1, "actor": "trusted_controller", "tool": "agent_handoff",
                           "outcome": "clarification_needed", "result": None,
                           "arguments": {"from": OPS, "to": CUSTOMER, "remedy": "exchange", "order_id": "O1",
                                         "line_id": "L1", "missing": "requested_size", "options": ["S", "L"]}})
    assert dto.kind == "handoff"
    assert dto.summary == ("Retail Operations Agent → Customer Resolution Agent: exchange, O1, L1; "
                           "needs requested_size (S, L)")
