"""REAL end-to-end adapter runs: the model named in .env, local Guardrails, PostgreSQL.

Each test runs adapter.py in a fresh process, then checks the HP structural validator, the
independent oracle and, for public cases, the expected outcomes.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals.oracle import check_case
from tests.test_adapter_runner import hp_validator

pytestmark = [pytest.mark.postgres, pytest.mark.real_model]
ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "candidate_assets" / "public_cases.json"

PUBLIC_EXPECTED = {
    "public_01_refund_boundary": ["propose", "completed"],
    "public_02_exchange": ["propose", "completed"],
    "public_03_expired": ["deny"],
    "public_04_clarify": ["clarify", "propose", "completed"],
    "public_05_stale_stock": ["propose", "deny"],
    "public_06_timeout_replay": ["propose", "completed", "completed"],
}


def run_adapter(input_path: Path, output: Path) -> dict:
    proc = subprocess.run([sys.executable, str(ROOT / "adapter.py"), "--input", str(input_path), "--output",
                           str(output)], cwd=ROOT, capture_output=True, text=True, timeout=1500)
    print(proc.stdout[-2000:])
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
    assert hp_validator().check(input_path, output) == len(json.loads(input_path.read_bytes())["cases"])
    document = json.loads(output.read_text(encoding="utf-8"))
    return document


def stock(result, sku):
    return next(p["stock"] for p in result["final_state"]["products"] if p["sku"] == sku)


def test_public_cases_through_adapter(tmp_path):
    document = run_adapter(PUBLIC, tmp_path / "results.json")
    cases = {c["case_id"]: c for c in json.loads(PUBLIC.read_bytes())["cases"]}
    by_id = {r["case_id"]: r for r in document["results"]}
    assert list(by_id) == list(cases)
    for case_id, expected in PUBLIC_EXPECTED.items():
        result = by_id[case_id]
        assert [t["decision"] for t in result["turns"]] == expected, case_id
        assert check_case(cases[case_id], result) == [], case_id
    refunds = {k: [r["amount_cents"] for r in v["final_state"]["refunds"]] for k, v in by_id.items()}
    assert refunds["public_01_refund_boundary"] == refunds["public_04_clarify"] == [3000]
    assert refunds["public_06_timeout_replay"] == [3000]
    assert stock(by_id["public_02_exchange"], "TEE-L") == 2 and stock(by_id["public_05_stale_stock"], "TEE-L") == 0
    assert by_id["public_03_expired"]["final_state"]["returns"] == []
    six = by_id["public_06_timeout_replay"]["action_audit"]
    assert [(e["turn"], e["tool"], e["outcome"]) for e in six
            if e["tool"] in ("commit_confirmed_proposal", "timeout_after_commit")] == [
        (2, "commit_confirmed_proposal", "committed"), (2, "timeout_after_commit", "raised"),
        (3, "commit_confirmed_proposal", "replayed")]
    for result in document["results"]:
        if result["case_id"] != "public_03_expired":
            actors = {e["actor"] for e in result["action_audit"] if e["tool"] == "llm_call"}
            assert actors == {"customer_resolution_agent", "retail_operations_agent"}
        assert result["metrics"]["llm_calls"] >= 3


def variant_cases() -> dict:
    """Schema-compatible variants: new IDs, products, prices, dates, wording and faults."""
    def state(as_of, products, orders):
        return {"as_of": as_of, "currency": "USD", "products": products, "orders": orders, "returns": [],
                "refunds": [], "exchanges": [], "quarantine": [], "idempotency": {}}

    hoodies = [{"sku": f"HOODIE-{s}", "product_id": "HOODIE", "size": s, "price_cents": 5500, "stock": n}
               for s, n in (("S", 1), ("M", 2), ("XL", 3))]
    return {"contract_version": 1, "cases": [
        {"case_id": "variant_refund", "customer_id": "K-77", "request_id": "R-alpha",
         "initial_state": state("2027-03-31", hoodies, [
             {"order_id": "ORD-5501", "customer_id": "K-77", "status": "delivered", "delivered_on": "2027-03-01",
              "lines": [{"line_id": "A", "sku": "HOODIE-M", "paid_cents": 1999, "unopened": True,
                         "final_sale": False, "returned": False}]}]),
         "turns": [{"message": "Hi, I'd like my money back for item A in order ORD-5501. It's still sealed."},
                   {"message": "Go ahead.", "confirm_proposal": "last"}], "faults": []},
        {"case_id": "variant_exchange_fault", "customer_id": "K-88", "request_id": "R-beta",
         "initial_state": state("2027-03-31", hoodies, [
             {"order_id": "ORD-77", "customer_id": "K-88", "status": "delivered", "delivered_on": "2027-03-20",
              "lines": [{"line_id": "A", "sku": "HOODIE-M", "paid_cents": 4500, "unopened": True,
                         "final_sale": False, "returned": False}]}]),
         "turns": [{"message": "Could I swap the hoodie on line A of order ORD-77 for size XL instead?"},
                   {"message": "Yes please.", "confirm_proposal": "last"}],
         "faults": [{"type": "stock_change", "before_turn": 2, "sku": "HOODIE-XL", "new_stock": 5}]},
        {"case_id": "variant_future_delivery", "customer_id": "K-99", "request_id": "R-gamma",
         "initial_state": state("2027-03-31", hoodies, [
             {"order_id": "ORD-9", "customer_id": "K-99", "status": "delivered", "delivered_on": "2027-04-02",
              "lines": [{"line_id": "A", "sku": "HOODIE-S", "paid_cents": 5500, "unopened": True,
                         "final_sale": False, "returned": False}]}]),
         "turns": [{"message": "Please refund line A of order ORD-9."}], "faults": []},
        {"case_id": "variant_timeout_replay", "customer_id": "K-11", "request_id": "R-delta",
         "initial_state": state("2027-03-31", hoodies, [
             {"order_id": "ORD-3", "customer_id": "K-11", "status": "delivered", "delivered_on": "2027-03-30",
              "lines": [{"line_id": "Z", "sku": "HOODIE-S", "paid_cents": 4250, "unopened": True,
                         "final_sale": False, "returned": False}]}]),
         "turns": [{"message": "I want a refund for line Z of order ORD-3."},
                   {"message": "Confirmed.", "confirm_proposal": "last"},
                   {"message": "No answer came back, confirming again.", "confirm_proposal": "last"}],
         "faults": [{"type": "timeout_after_commit"}]},
    ]}


VARIANT_EXPECTED = {"variant_refund": ["propose", "completed"],
                    "variant_exchange_fault": ["propose", "completed"],
                    "variant_future_delivery": ["deny"],
                    "variant_timeout_replay": ["propose", "completed", "completed"]}


def test_changed_values_and_wording(tmp_path):
    data = variant_cases()
    input_path = tmp_path / "variants.json"
    input_path.write_text(json.dumps(data), encoding="utf-8")
    document = run_adapter(input_path, tmp_path / "variant_results.json")
    for case, result in zip(data["cases"], document["results"]):
        assert check_case(case, result) == [], case["case_id"]
        assert [t["decision"] for t in result["turns"]] == VARIANT_EXPECTED[case["case_id"]], case["case_id"]
    by_id = {r["case_id"]: r for r in document["results"]}
    assert [r["amount_cents"] for r in by_id["variant_refund"]["final_state"]["refunds"]] == [1999]
    assert stock(by_id["variant_exchange_fault"], "HOODIE-XL") == 4  # fault set 5, exchange took 1
    assert [r["amount_cents"] for r in by_id["variant_timeout_replay"]["final_state"]["refunds"]] == [4250]
