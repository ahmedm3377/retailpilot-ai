"""Evaluation runner with a SCRIPTED fake model (structure, isolation, audit; not real inference)."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import psycopg
import pytest

from app.db import repositories
from retail_domain.models import FixtureRoot
from app.evaluation import runner
from app.evaluation.runner import EvalContext, run_case, run_fixture
from app.evaluation.serialization import audit_entries, dumps, write_atomic
from tests.conftest import make_line, make_state
from tests.fakes import CUSTOMER, FakeModel, exchange_turn, refund_turn, say
from evals.oracle import check_case

pytestmark = pytest.mark.postgres
ROOT = Path(__file__).resolve().parents[2]


def hp_validator():
    spec = importlib.util.spec_from_file_location("hp_validator", ROOT / "candidate_assets" / "validate_submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def make_ctx(pg_url, guards, agent_settings):
    def _make(model):
        return EvalContext(model, guards, agent_settings, pg_url,
                           {"model": "fake/scripted", "digest": None, "quantization": None,
                            "engine": None, "engine_version": None, "decoding": None}, cold_start_ms=None,
                           notes=["FAKE MODEL: deterministic test run, not real inference"])
    return _make


def case_dict(case_id="CASE-A", customer="CX-1", request="RQ-1", turns=None, faults=None, **state_kwargs):
    state = make_state(customer_id=customer, order_id="ORD-1", **state_kwargs).model_dump(mode="json")
    return {"case_id": case_id, "customer_id": customer, "request_id": request, "initial_state": state,
            "turns": turns or [{"message": "refund ORD-1 LN-1"}, {"message": "yes", "confirm_proposal": "last"}],
            "faults": faults or []}


def schemas_left(pg_url):
    with psycopg.connect(pg_url) as conn:
        return conn.execute("SELECT count(*) FROM pg_namespace WHERE nspname LIKE 'retail_eval_%'").fetchone()[0]


def test_document_structure_passes_hp_validator(tmp_path, make_ctx, pg_url):
    cases = [case_dict("CASE-A"), case_dict("CASE-B", request="RQ-2")]
    raw = json.dumps({"contract_version": 1, "cases": cases}).encode()
    input_path = tmp_path / "cases.json"
    input_path.write_bytes(raw)
    model = FakeModel()
    for _ in cases:
        refund_turn(model, "ORD-1", "LN-1")  # confirmations use the template (E3): no extra LLM reply
    root = FixtureRoot.model_validate_json(raw)
    import hashlib
    document, raws = run_fixture(make_ctx(model), root, hashlib.sha256(raw).hexdigest())
    output = tmp_path / "out" / "results.json"
    write_atomic(output, document)
    assert hp_validator().check(input_path, output) == 2
    assert [r["case_id"] for r in document["results"]] == ["CASE-A", "CASE-B"]
    for case, result in zip(cases, document["results"]):
        assert check_case(case, result) == []
        assert [t["decision"] for t in result["turns"]] == ["propose", "completed"]
        m = result["metrics"]
        assert math.isfinite(m["elapsed_ms"]) and m["elapsed_ms"] > 0 and m["llm_calls"] == 4  # 4 on the proposal turn; confirm uses the template (E3)
        assert m["peak_ram_mb"] is None or m["peak_ram_mb"] > 0
    assert document["hardware"]["cpu"] and document["configuration"]["database"]["server_version"]
    assert "retail:retail@" not in dumps(document) and "password" not in dumps(document).lower()
    assert schemas_left(pg_url) == 0


def test_last_means_latest_earlier_non_null_proposal(make_ctx):
    case = FixtureRoot.model_validate({"contract_version": 1, "cases": [case_dict(turns=[
        {"message": "refund ORD-1 LN-1"}, {"message": "yes", "confirm_proposal": "last"},
        {"message": "did it work? yes again", "confirm_proposal": "last"}])]}).cases[0]
    model = refund_turn(FakeModel(), "ORD-1", "LN-1").add(CUSTOMER, say("Done."), say("Done."))
    result, raw = run_case(make_ctx(model), case)
    assert [t["decision"] for t in result["turns"]] == ["propose", "completed", "completed"]
    assert result["turns"][1]["proposal"] is None and result["turns"][2]["proposal"] is None
    assert [t["confirmed_proposal_id"] for t in raw["turns"]] == [None, "PROP-1", "PROP-1"]
    assert raw["turns"][2]["internal_status"]["commit"]["replayed"] is True
    assert len(result["final_state"]["refunds"]) == 1


def test_confirmation_before_any_proposal_is_an_error_without_llm_handoff(make_ctx):
    case = FixtureRoot.model_validate({"contract_version": 1, "cases": [case_dict(turns=[
        {"message": "Yes, refund it.", "confirm_proposal": "last"}])]}).cases[0]
    model = FakeModel().add(CUSTOMER, say("There is nothing to confirm yet. Nothing was changed."))
    result, _ = run_case(make_ctx(model), case)
    assert result["turns"][0]["decision"] == "error" and result["turns"][0]["proposal"] is None
    assert model.calls_for("retail_operations_agent") == 0 and model.calls_for(CUSTOMER) == 0  # template (E3)
    assert result["final_state"]["returns"] == [] and check_case(case.model_dump(mode="json"), result) == []


def test_cases_with_repeated_ids_are_isolated(make_ctx, pg_url):
    a = case_dict(lines=[make_line(paid_cents=500)])
    b = case_dict(lines=[make_line(paid_cents=900)])  # same case, customer, request, order and line IDs
    root = FixtureRoot.model_validate({"contract_version": 1, "cases": [a, b]})
    model = FakeModel()
    for _ in range(2):
        refund_turn(model, "ORD-1", "LN-1")
    document, _ = run_fixture(make_ctx(model), root, "x")
    amounts = [[r["amount_cents"] for r in res["final_state"]["refunds"]] for res in document["results"]]
    assert amounts == [[500], [900]]
    assert all(res["final_state"]["idempotency"]["RQ-1"]["status"] == "completed" for res in document["results"])
    assert schemas_left(pg_url) == 0


def test_audit_is_sourced_tagged_and_not_duplicated(make_ctx):
    case = FixtureRoot.model_validate({"contract_version": 1, "cases": [case_dict(
        turns=[{"message": "exchange ORD-1 LN-1 for size B"}, {"message": "yes", "confirm_proposal": "last"}],
        faults=[{"type": "stock_change", "before_turn": 2, "sku": "WIDGET-B", "new_stock": 4}])]}).cases[0]
    model = exchange_turn(FakeModel(), "B", "WIDGET-B", "ORD-1", "LN-1").add(CUSTOMER, say("Done."))
    result, _ = run_case(make_ctx(model), case)
    audit = result["action_audit"]
    assert all(e["turn"] in (1, 2) for e in audit)
    assert len({e["event_id"] for e in audit}) == len(audit)
    actors = {e["actor"] for e in audit}
    assert {"customer_resolution_agent", "retail_operations_agent", "trusted_controller", "guardrails",
            "external"} <= actors
    fault = [e for e in audit if e["actor"] == "external"]
    assert [(e["tool"], e["turn"]) for e in fault] == [("stock_change", 2)]
    llm_calls = [e for e in audit if e["tool"] == "llm_call"]
    assert len(llm_calls) == result["metrics"]["llm_calls"]  # TurnResult events not appended twice
    assert next(p["stock"] for p in result["final_state"]["products"] if p["sku"] == "WIDGET-B") == 3
    assert check_case(case.model_dump(mode="json"), result) == []


def test_failure_during_commit_propagates_and_cleans_up(make_ctx, pg_url, monkeypatch):
    case = FixtureRoot.model_validate({"contract_version": 1, "cases": [case_dict()]}).cases[0]
    model = refund_turn(FakeModel(), "ORD-1", "LN-1").add(CUSTOMER, say("Done."))

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(repositories, "insert_idempotency", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        run_case(make_ctx(model), case)
    assert schemas_left(pg_url) == 0


def test_oracle_flags_tampered_results(make_ctx):
    raw_case = case_dict()
    case = FixtureRoot.model_validate({"contract_version": 1, "cases": [raw_case]}).cases[0]
    model = refund_turn(FakeModel(), "ORD-1", "LN-1").add(CUSTOMER, say("Done."))
    result, _ = run_case(make_ctx(model), case)
    assert check_case(raw_case, result) == []

    def tampered(edit):
        copy = json.loads(json.dumps(result))
        edit(copy)
        return check_case(raw_case, copy)

    assert tampered(lambda r: r["final_state"]["refunds"][0].update(amount_cents=1235))
    assert tampered(lambda r: r["final_state"]["products"][0].update(stock=0))
    assert tampered(lambda r: r["final_state"].pop("quarantine"))
    assert tampered(lambda r: r["final_state"]["quarantine"].clear())
    assert tampered(lambda r: r["turns"].pop())
    assert tampered(lambda r: r["turns"][1].update(decision="deny"))
    unconfirmed = json.loads(json.dumps(raw_case))
    unconfirmed["turns"][1].pop("confirm_proposal")
    assert check_case(unconfirmed, result)  # a commit on a turn that was not a confirmation


def _root(*cases):
    return FixtureRoot.model_validate({"contract_version": 1, "cases": list(cases)})


def test_a_failed_case_is_included_with_every_turn(make_ctx, monkeypatch):
    """Contract: "Include failed runs". The finished turn stays; the rest are marked as errors."""
    original = runner.AgentRuntime

    class Broken(original):
        def handle_customer_turn(self, *args, **kwargs):
            if kwargs["turn_number"] == 2:
                raise RuntimeError("runtime exploded")
            return super().handle_customer_turn(*args, **kwargs)

    monkeypatch.setattr(runner, "AgentRuntime", Broken)
    case = _root(case_dict()).cases[0]
    result, raw = run_case(make_ctx(refund_turn(FakeModel(), "ORD-1", "LN-1")), case, include_failed=True)
    assert result["run_failed"] is True and "runtime exploded" in result["error"]
    assert [t["decision"] for t in result["turns"]] == ["propose", "error"]
    assert result["action_audit"] and result["metrics"]["elapsed_ms"] > 0
    assert set(case.initial_state.model_dump()) <= set(result["final_state"])
    assert raw["decisions"] == ["propose"]
    with pytest.raises(RuntimeError):  # evidence runs still stop on a failure
        run_case(make_ctx(refund_turn(FakeModel(), "ORD-1", "LN-1")), case)


def test_failed_document_passes_the_structure_validator(tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"contract_version": 1, "cases": [case_dict()]}), encoding="utf-8")
    root = _root(case_dict())
    document = runner.failed_document(root, __import__("hashlib").sha256(cases.read_bytes()).hexdigest(), "model down")
    output = tmp_path / "results.json"
    write_atomic(output, document)
    assert hp_validator().check(cases, output) == 1
    assert document["results"][0]["run_failed"] and document["results"][0]["turns"][0]["decision"] == "error"


def test_audit_entries_name_the_agent():
    entry = audit_entries([{"event_id": 1, "actor": "retail_operations_agent", "tool": "t", "arguments": {},
                            "outcome": "ok", "result": None, "error": None, "effects": []}], [(1, 0, 1)])[0]
    assert entry["agent"] == "retail_operations_agent" and entry["turn"] == 1
