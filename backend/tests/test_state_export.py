from __future__ import annotations


import pytest

from retail_domain.models import State
from app.services.transactions import commit_confirmed_proposal
from tests.conftest import make_state

pytestmark = pytest.mark.postgres


def _walk(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)
    else:
        yield value


def test_export_after_commit_keeps_contract_shape(pg_store, ops, load, public_cases):
    initial = load("public_02_exchange")
    proposal = ops.create_proposal("C100", "REQ-1", "O100", "L1", "exchange", "TEE-L").proposal
    commit_confirmed_proposal(pg_store, "C100", "REQ-1", proposal.proposal_id)
    exported = pg_store.export_final_state()

    assert set(exported) == set(initial.model_dump(mode="json"))
    assert exported["as_of"] == "2026-10-26" and exported["currency"] == "USD"
    assert not any(isinstance(v, float) for v in _walk(exported))
    assert State.model_validate(exported)  # still parses as contract state
    record = exported["idempotency"]["REQ-1"]
    assert set(record) == {"status", "customer_id", "proposal_id", "result"}
    assert record["result"]["record_ids"]["exchange_id"] == exported["exchanges"][0]["exchange_id"]
    for key in ("returns", "exchanges", "quarantine"):
        assert {"request_id", "proposal_id", "customer_id", "order_id", "line_id"} <= set(exported[key][0])


def test_unknown_fields_round_trip_through_postgres(pg_store, load):
    raw = make_state().model_dump(mode="json")
    raw["store_region"] = "north"
    raw["products"][0]["color"] = "teal"
    raw["orders"][0]["channel"] = {"web": True}
    raw["orders"][0]["lines"][0]["gift"] = False
    load(State.model_validate(raw))
    assert pg_store.export_final_state() == raw


def test_reset_rejects_nonempty_transaction_records(pg_store):
    raw = make_state().model_dump(mode="json")
    raw["refunds"] = [{"refund_id": "x"}]
    with pytest.raises(ValueError, match="refunds"):
        pg_store.reset_from_initial_state(State.model_validate(raw))


