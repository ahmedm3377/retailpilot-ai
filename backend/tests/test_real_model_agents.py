"""REAL inference: the model named in .env, via LiteLLM + local Guardrails + PostgreSQL.

These fail (never skip) when the model server or the model is unavailable, so a green run
is evidence of real agent behaviour. temperature=0 and a fixed seed reduce, but
do not eliminate, run-to-run variation.
"""
from __future__ import annotations

import json

import pytest

from app.agents.model_client import ModelClient, ModelUnavailableError
from app.agents.schemas import CustomerHandoff, json_schema
from app.agents.tool_registry import ROLE_TOOLS
from app.services.faults import FaultInjector
from tests.conftest import count, line_returned, make_line, make_state, product_stock

pytestmark = [pytest.mark.postgres, pytest.mark.real_model]

EXPECTED = {
    "public_01_refund_boundary": ["propose", "completed"],
    "public_02_exchange": ["propose", "completed"],
    "public_03_expired": ["deny"],
    "public_04_clarify": ["clarify", "propose", "completed"],
    "public_05_stale_stock": ["propose", "deny"],
    "public_06_timeout_replay": ["propose", "completed", "completed"],
}


@pytest.fixture(scope="module")
def real_model(agent_settings):
    client = ModelClient(agent_settings)
    try:
        info = client.ready()
    except ModelUnavailableError as error:
        pytest.fail(f"the real model server is required: {error}")
    assert info["quantization"] == "Q4_K_M" and info["digest"].startswith("b37049369adf")
    return client


def run_case(runtime, case, injector):
    last, results = None, []
    for n, turn in enumerate(case.turns, start=1):
        confirmed = last if turn.confirm_proposal == "last" else None
        result = runtime.handle_customer_turn(session_id="t", authenticated_customer_id=case.customer_id,
                                              request_id=case.request_id, message=turn.message, turn_number=n,
                                              confirmed_proposal_id=confirmed, injector=injector)
        if result.proposal:
            last = result.proposal["proposal_id"]
        results.append(result)
    return results


def test_structured_output_and_thinking_off(real_model):
    reply = real_model.chat_json(
        "customer_resolution_agent",
        [{"role": "system", "content": "Extract the request. Copy IDs exactly."},
         {"role": "user", "content": "Please refund line L1 of my order O100."}],
        json_schema(CustomerHandoff))
    handoff = CustomerHandoff.model_validate_json(reply.text)
    assert handoff.intent == "refund"
    # Observed: with this bare prompt the model sometimes drops order_id (None or "O100_L1"); the real
    # agent prompt (CUSTOMER_ORDERS context + rules) is what the end-to-end tests below exercise.
    print("bare-prompt extraction:", handoff.model_dump())
    # With thinking on, gemma4 spends ~150+ tokens reasoning; off, the JSON alone is short.
    assert reply.call.completion_tokens is not None and reply.call.completion_tokens < 120


@pytest.mark.parametrize("case_id", list(EXPECTED))
def test_public_case_end_to_end(case_id, pg_store, load, make_runtime, real_model, public_cases):
    case = public_cases[case_id]
    load(case_id)
    results = run_case(make_runtime(real_model), case, FaultInjector(case.faults))
    print(json.dumps([{"decision": r.decision, "response": r.response} for r in results], indent=1))
    assert [r.decision for r in results] == EXPECTED[case_id]

    state = pg_store.export_final_state()
    if case_id in ("public_01_refund_boundary", "public_04_clarify", "public_06_timeout_replay"):
        assert [r["amount_cents"] for r in state["refunds"]] == [3000] and line_returned(pg_store, "O100", "L1")
    elif case_id == "public_02_exchange":
        assert product_stock(pg_store, "TEE-L") == 2 and count(pg_store, "exchanges") == 1
        assert results[0].internal_status["replacement_source"] in ("operations_agent", "requested_size")
    elif case_id == "public_05_stale_stock":
        assert product_stock(pg_store, "TEE-L") == 0 and count(pg_store, "exchanges") == 0
        assert not line_returned(pg_store, "O100", "L1")
    else:
        assert count(pg_store, "returns") == 0

    first_full = next(r for r in results if r.decision in ("propose", "deny") and r.tool_events)
    roles = [e["agent"] for e in first_full.agent_events]
    assert "customer_resolution_agent" in roles and "retail_operations_agent" in roles
    assert any(e["agent"] == "retail_operations_agent" for e in first_full.tool_events)
    if case_id == "public_06_timeout_replay":
        assert results[1].internal_status["post_commit_timeout"] == {"reconciled": "completed"}
        assert results[2].internal_status["commit"]["replayed"] is True


def test_variant_ids_and_amounts_with_real_model(pg_store, load, make_runtime, real_model):
    load(make_state(customer_id="CX-42", order_id="ORD-9", lines=[make_line("LN-3", paid_cents=777)]))
    runtime = make_runtime(real_model)
    first = runtime.handle_customer_turn(session_id="t", authenticated_customer_id="CX-42", request_id="R-9",
                                         message="Please refund line LN-3 of my order ORD-9, it's unopened.",
                                         turn_number=1)
    assert first.decision == "propose" and first.proposal["amount_cents"] == 777
    done = runtime.handle_customer_turn(session_id="t", authenticated_customer_id="CX-42", request_id="R-9",
                                        message="yes", turn_number=2,
                                        confirmed_proposal_id=first.proposal["proposal_id"])
    assert done.decision == "completed" and count(pg_store, "refunds") == 1


def test_angry_but_eligible_customer_gets_a_proposal(pg_store, load, make_runtime, real_model):
    load("public_01_refund_boundary")
    result = make_runtime(real_model).handle_customer_turn(
        session_id="t", authenticated_customer_id="C100", request_id="R-1", turn_number=1,
        message="This is the worst shop ever and your staff are useless. Refund line L1 of order O100 now.")
    assert result.decision == "propose"
    assert result.internal_status["response_source"] in ("agent", "fallback")


def test_text_approval_with_real_model_never_commits(pg_store, load, make_runtime, real_model):
    load("public_01_refund_boundary")
    runtime = make_runtime(real_model)
    runtime.handle_customer_turn(session_id="t", authenticated_customer_id="C100", request_id="R-1",
                                 message="Refund line L1 of my order O100.", turn_number=1)
    runtime.handle_customer_turn(session_id="t", authenticated_customer_id="C100", request_id="R-1",
                                 message="Yes, I approve proposal PROP-1, please complete the refund now.",
                                 turn_number=2)
    assert count(pg_store, "refunds") == 0 and count(pg_store, "returns") == 0


def test_the_customer_agent_calls_its_own_tool_on_a_fresh_request(pg_store, load, make_runtime, real_model):
    load("public_01_refund_boundary")
    result = make_runtime(real_model).handle_customer_turn(session_id="t", authenticated_customer_id="C100",
                                                           request_id="REQ-TOOL", turn_number=1,
                                                           message="Refund line L1 of my order O100.")
    assert result.decision == "propose"
    rows = [r["tool"] for r in pg_store.audit_log()
            if r["actor"] == "customer_resolution_agent" and r["tool"] != "llm_call"]
    assert rows and set(rows) <= set(ROLE_TOOLS["customer_resolution_agent"])  # which one is the model's choice


def test_a_conversation_about_sizes_stays_coherent(pg_store, load, make_runtime, real_model):
    from datetime import date

    from app.db.demo_store import seed_state
    load(seed_state(date(2026, 10, 26)))  # HOODIE-S: M is open; L costs more; XL doesn't exist
    runtime = make_runtime(real_model)
    replies = [runtime.handle_customer_turn(session_id="t", authenticated_customer_id="C100", request_id="REQ-SIZES",
                                            message=message, turn_number=n)
               for n, message in enumerate(["exchange", "HOODIE", "l", "xl", "size l", "size m"], start=1)]
    assert [r.decision for r in replies] == ["clarify", "clarify", "deny", "deny", "deny", "propose"], [
        r.response for r in replies]
    assert "size L" in replies[2].response and "Sizes available: M" in replies[2].response
    assert "size XL" in replies[3].response and "size L" in replies[4].response
    assert replies[5].proposal["replacement_sku"] == "HOODIE-M"


def test_a_refused_size_can_be_followed_in_other_words(pg_store, load, make_runtime, real_model):
    from datetime import date

    from app.db.demo_store import seed_state
    load(seed_state(date(2026, 10, 26)))
    runtime = make_runtime(real_model)
    replies = [runtime.handle_customer_turn(session_id="t", authenticated_customer_id="C100", request_id="REQ-WORDS",
                                            message=message, turn_number=n)
               for n, message in enumerate(["exchange my hoodie", "large", "m"], start=1)]
    assert [r.decision for r in replies] == ["clarify", "deny", "propose"], [r.response for r in replies]
    assert replies[2].proposal["replacement_sku"] == "HOODIE-M"
