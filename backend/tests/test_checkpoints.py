"""LangGraph PostgreSQL checkpointing: restart, isolation, separation from business tables."""
from __future__ import annotations

import pytest

from tests.conftest import count
from tests.fakes import CUSTOMER, FakeModel, handoff, refund_turn, say

pytestmark = pytest.mark.postgres


def run(runtime, message, n, customer="C100", session="s1", confirmed=None):
    return runtime.handle_customer_turn(session_id=session, authenticated_customer_id=customer, request_id="REQ-1",
                                        message=message, turn_number=n, confirmed_proposal_id=confirmed)


def clarify_first(model):
    return model.add(CUSTOMER, handoff("refund", "O100", None, needs_clarification=True), say("Which line, L1 or L2?"))


def test_conversation_resumes_after_runtime_restart(pg_store, load, make_runtime):
    load("public_04_clarify")
    schema = "retail_t_ckpt_restart"
    first = make_runtime(clarify_first(FakeModel()), checkpoint_schema=schema)
    assert run(first, "I want to return an item from O100.", 1).decision == "clarify"
    first.close()

    model = refund_turn(FakeModel())
    second = make_runtime(model, checkpoint_schema=schema)  # new process-equivalent: new connection + graph
    result = run(second, "Line L1 please, refund.", 2)
    assert result.decision == "propose"
    handoff_prompt = [m for role, m in model.seen if role == CUSTOMER][0][1]["content"]
    assert "question_asked" in handoff_prompt  # state came back from PostgreSQL
    assert second.resolve_last_proposal("C100", "s1", "REQ-1") == result.proposal["proposal_id"]


def test_threads_are_isolated_per_customer_and_session(pg_store, load, make_runtime):
    load("public_04_clarify")
    model = clarify_first(FakeModel())
    model.add(CUSTOMER, handoff("refund", "O100", "L1"), say("We could not find that order."))
    runtime = make_runtime(model)
    run(runtime, "I want to return an item from O100.", 1, customer="C100")
    other = run(runtime, "Line L1 please.", 1, customer="C999")
    # C999's turn did not inherit C100's clarification context.
    second_handoff_prompt = [m for role, m in model.seen if role == CUSTOMER][2][1]["content"]
    assert "question_asked" not in second_handoff_prompt
    assert other.decision in {"clarify", "deny"} and count(pg_store, "proposals") == 0
    assert runtime.resolve_last_proposal("C999", "s1", "REQ-1") is None


