"""Faults, timeout recovery and concurrent approvals through the agent runtime (SCRIPTED fake model)."""
from __future__ import annotations

import threading

import pytest

from app.services.faults import FaultInjector
from tests.conftest import count, line_returned, product_stock
from tests.fakes import CUSTOMER, FakeModel, exchange_turn, refund_turn, say

pytestmark = pytest.mark.postgres


def run(runtime, message, n, confirmed=None, injector=None, session="s1"):
    return runtime.handle_customer_turn(session_id=session, authenticated_customer_id="C100", request_id="REQ-1",
                                        message=message, turn_number=n, confirmed_proposal_id=confirmed,
                                        injector=injector)


def test_stock_change_before_approval_denies_exchange(pg_store, load, make_runtime, public_cases):
    case = public_cases["public_05_stale_stock"]
    load(case.case_id)
    injector = FaultInjector(case.faults)
    model = exchange_turn(FakeModel(), "L", "TEE-L").add(CUSTOMER, say("Sorry, size L is out of stock."))
    runtime = make_runtime(model)
    proposal = run(runtime, case.turns[0].message, 1, injector=injector).proposal
    result = run(runtime, case.turns[1].message, 2, confirmed=proposal["proposal_id"], injector=injector)
    assert result.decision == "deny"
    assert result.internal_status["faults_applied"][0]["new_stock"] == 0
    assert product_stock(pg_store, "TEE-L") == 0 and count(pg_store, "exchanges") == 0
    assert not line_returned(pg_store, "O100", "L1")


def test_timeout_after_commit_reconciles_then_replay_is_idempotent(pg_store, load, make_runtime, public_cases):
    case = public_cases["public_06_timeout_replay"]
    load(case.case_id)
    injector = FaultInjector(case.faults)
    model = refund_turn(FakeModel()).add(CUSTOMER, say("Your refund of $30.00 is complete."),
                                         say("Your refund of $30.00 is complete."))
    runtime = make_runtime(model)
    pid = run(runtime, case.turns[0].message, 1, injector=injector).proposal["proposal_id"]
    second = run(runtime, case.turns[1].message, 2, confirmed=pid, injector=injector)
    assert second.decision == "completed"
    assert second.internal_status["post_commit_timeout"] == {"reconciled": "completed"}
    third = run(runtime, case.turns[2].message, 3, confirmed=pid, injector=injector)
    assert third.decision == "completed" and third.internal_status["commit"]["replayed"] is True
    assert count(pg_store, "refunds") == 1 and count(pg_store, "quarantine") == 1
    # The recovery is visible in the audit trail: a read between the timeout and the replay, with no effects.
    trail = [(e["actor"], e["tool"], e["outcome"]) for e in pg_store.audit_log()
             if e["tool"] in ("commit_confirmed_proposal", "timeout_after_commit", "reconcile_request")]
    assert trail == [("trusted_controller", "commit_confirmed_proposal", "committed"),
                     ("external", "timeout_after_commit", "raised"),
                     ("trusted_controller", "reconcile_request", "completed"),
                     ("trusted_controller", "commit_confirmed_proposal", "replayed")]
    reconciled = next(e for e in pg_store.audit_log() if e["tool"] == "reconcile_request")
    assert reconciled["result"]["proposal_id"] == pid and not reconciled["effects"]


def test_bad_or_missing_confirmation_is_an_error(pg_store, load, make_runtime):
    load("public_01_refund_boundary")
    model = FakeModel().add(CUSTOMER, say("Sorry, nothing was changed."), say("Sorry, nothing was changed."))
    runtime = make_runtime(model)
    assert run(runtime, "confirm", 1, confirmed="PROP-404").decision == "error"
    assert run(runtime, "confirm", 2, confirmed="yes").decision == "error"
    assert count(pg_store, "refunds") == 0


def test_concurrent_approvals_through_two_runtimes_commit_once(pg_store, load, make_runtime):
    load("public_02_exchange")
    first = make_runtime(exchange_turn(FakeModel(), "L", "TEE-L"))
    pid = run(first, "Exchange line L1 of O100 for size L.", 1).proposal["proposal_id"]
    runtimes = [make_runtime(FakeModel().add(CUSTOMER, say("Done."))) for _ in range(2)]
    barrier = threading.Barrier(2)
    results = [None, None]

    def go(i):
        barrier.wait()
        results[i] = run(runtimes[i], "yes", 2, confirmed=pid, session=f"s{i}")

    threads = [threading.Thread(target=go, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert [r.decision for r in results] == ["completed", "completed"]
    assert sorted(r.internal_status["commit"]["replayed"] for r in results) == [False, True]
    assert count(pg_store, "exchanges") == 1 and product_stock(pg_store, "TEE-L") == 2
