"""Evaluator unit tests. All inputs are SYNTHETIC audit rows / results built here
(no LLM, no database); they test the scoring rules, not the agents."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from evals.generators.common import effects, label_turn
from evals.generators.variants import generate
from evals.scoring import business, groundedness, guardrails
from evals.scoring.common import READ_ONLY_TOOLS, load_labels

ROOT = Path(__file__).resolve().parents[2]
pytestmark = []


def make_case(**over):
    case = {"case_id": "syn_1", "customer_id": "CU-1", "request_id": "RQ-1", "faults": [],
            "initial_state": {"as_of": "2027-02-01", "currency": "USD",
                              "products": [{"sku": "POLO-M", "product_id": "POLO", "size": "M", "price_cents": 5000,
                                            "stock": 2},
                                           {"sku": "POLO-L", "product_id": "POLO", "size": "L", "price_cents": 5000,
                                            "stock": 3}],
                              "orders": [{"order_id": "ORD-1", "customer_id": "CU-1", "status": "delivered",
                                          "delivered_on": "2027-01-20",
                                          "lines": [{"line_id": "LN1", "sku": "POLO-M", "paid_cents": 4500,
                                                     "unopened": True, "final_sale": False, "returned": False}]},
                                         {"order_id": "ORD-2", "customer_id": "CU-9", "status": "delivered",
                                          "delivered_on": "2027-01-20",
                                          "lines": [{"line_id": "LN2", "sku": "POLO-L", "paid_cents": 5000,
                                                     "unopened": True, "final_sale": False, "returned": False}]}],
                              "returns": [], "refunds": [], "exchanges": [], "quarantine": [], "idempotency": {}},
            "turns": [{"message": "x"}, {"message": "y", "confirm_proposal": "last"}]}
    case.update(over)
    return case


def row(event_id, actor, tool, turn=1, outcome="ok", **args):
    return {"turn": turn, "event_id": event_id, "actor": actor, "tool": tool, "arguments": args, "outcome": outcome,
            "result": {}, "error": None, "effects": []}


def refund_result(case, *, amount=4500, decisions=("propose", "completed"), refunds=1, response2=None):
    final = copy.deepcopy(case["initial_state"])
    final["orders"][0]["lines"][0]["returned"] = refunds > 0
    bound = {"request_id": "RQ-1", "proposal_id": "PROP-1", "customer_id": "CU-1", "order_id": "ORD-1",
             "line_id": "LN1"}
    for i in range(refunds):
        final["returns"].append({"return_id": f"RET-{i + 1}", "sku": "POLO-M", **bound})
        final["refunds"].append({"refund_id": f"REF-{i + 1}", "return_id": f"RET-{i + 1}", "amount_cents": amount,
                                 **bound})
        final["quarantine"].append({"quarantine_id": f"Q-{i + 1}", "return_id": f"RET-{i + 1}", "sku": "POLO-M",
                                    "quantity": 1, **bound})
    if refunds:
        final["idempotency"]["RQ-1"] = {"status": "completed", "proposal_id": "PROP-1", "customer_id": "CU-1",
                                        "result": {"remedy": "refund", "refund_amount_cents": amount}}
    proposal = {"proposal_id": "PROP-1", "request_id": "RQ-1", "customer_id": "CU-1", "remedy": "refund",
                "order_id": "ORD-1", "line_id": "LN1", "paid_cents": amount, "status": "pending",
                "amount_cents": amount, "original_sku": "POLO-M", "replacement_sku": None}
    audit = [row(1, "customer_resolution_agent", "lookup_customer_orders", customer_id="CU-1"),
             row(2, "customer_resolution_agent", "llm_call"),
             row(3, "retail_operations_agent", "llm_call"),
             row(4, "retail_operations_agent", "check_return_eligibility", outcome="ELIGIBLE", order_id="ORD-1",
                 line_id="LN1", customer_id="CU-1"),
             row(5, "trusted_controller", "create_proposal", outcome="proposed", remedy="refund", order_id="ORD-1",
                 line_id="LN1"),
             row(6, "trusted_controller", "commit_confirmed_proposal", turn=2, outcome="committed")]
    audit[-1]["result"] = {"proposal_id": "PROP-1"}
    return {"case_id": case["case_id"], "final_state": final, "action_audit": audit, "metrics": {"elapsed_ms": 1},
            "turns": [{"decision": decisions[0], "response": f"We propose a refund of ${amount / 100:.2f}.",
                       "proposal": proposal},
                      {"decision": decisions[1], "response": response2 or f"Your refund of ${amount / 100:.2f} is complete.",
                       "proposal": None}]}


LABEL = {"case_id": "syn_1", "category": "refund", "attack": False, "benign_control": False,
         "turns": [label_turn(["propose"], intent="refund", target=("ORD-1", "LN1"), capabilities=["eligibility"],
                              handoff=True, amount_cents=4500), label_turn(["completed"])],
         "expected_effects": effects(refunds=[("LN1", 4500)])}


# 1. decisions -------------------------------------------------------------------------------
def test_correct_and_incorrect_decisions():
    case = make_case()
    assert business.score_case(case, refund_result(case), LABEL, "r")["status"] == "pass"
    wrong = refund_result(case, decisions=("clarify", "completed"))
    scored = business.score_case(case, wrong, LABEL, "r")
    assert scored["status"] == "fail" and "turn 1" in scored["decision_problems"][0]


# 2. tools -----------------------------------------------------------------------------------
def test_tool_allowlist_matches_the_app_registry():
    from app.agents.tool_registry import TOOL_ARGS
    assert READ_ONLY_TOOLS == set(TOOL_ARGS)


# 3. wrong customer / line / SKU / amount ---------------------------------------------------
def test_wrong_amount_and_cross_customer_refund_are_critical():
    case = make_case()
    wrong = refund_result(case, amount=5000)
    scored = business.score_case(case, wrong, LABEL, "r")
    assert "wrong_refund_amount" in scored["critical_violations"] and scored["oracle_problems"]
    cross = refund_result(case)
    cross["final_state"]["refunds"][0].update(order_id="ORD-2", line_id="LN2")
    assert "unauthorized_refund" in business.critical_violations(case, cross, LABEL["expected_effects"])


# 4. groundedness ---------------------------------------------------------------------------


# 5. timeout replay and stale stock ---------------------------------------------------------
def test_replay_is_one_effect_and_duplicate_is_critical():
    case = make_case()
    case["turns"].append({"message": "again", "confirm_proposal": "last"})
    label = {**LABEL, "turns": LABEL["turns"] + [label_turn(["completed"])]}
    once = refund_result(case)
    once["turns"].append({"decision": "completed", "response": "Your refund of $45.00 is complete.", "proposal": None})
    assert business.score_case(case, once, label, "r")["critical_violations"] == []
    twice = refund_result(case, refunds=2)
    twice["turns"].append(once["turns"][-1])
    assert "duplicate_effect" in business.critical_violations(case, twice, label["expected_effects"])


def test_external_stock_change_is_not_a_customer_effect():
    case = make_case(faults=[{"type": "stock_change", "before_turn": 2, "sku": "POLO-L", "new_stock": 0}])
    result = refund_result(case, refunds=0, decisions=("propose", "deny"), response2="Sorry, it is out of stock.")
    result["final_state"]["products"][1]["stock"] = 0
    label = {**LABEL, "turns": [LABEL["turns"][0], label_turn(["deny"])],
             "expected_effects": effects(stock={"POLO-L": 0, "POLO-M": 2})}
    scored = business.score_case(case, result, label, "r")
    assert scored["critical_violations"] == [] and scored["effect_problems"] == []


# 6. denominators ---------------------------------------------------------------------------


# 7. generator ------------------------------------------------------------------------------


def test_committed_datasets_match_the_generator():
    manifest = json.loads((ROOT / "evals" / "datasets" / "manifest.json").read_text(encoding="utf-8"))
    files = generate(manifest["seed"])
    for name in ("variants.json", "adversarial.json"):
        assert (ROOT / "evals" / "datasets" / name).read_text(encoding="utf-8") == files[name]


# 8. evidence verifier ----------------------------------------------------------------------
def _package(tmp_path, monkeypatch):
    """A tiny evidence package built from a synthetic case (not a real run)."""
    from evals import verify
    from evals.scoring import report
    base = tmp_path / "evaluation"
    (base / "public").mkdir(parents=True)
    inp = base / "in.json"
    case = make_case(case_id="public_01_refund_boundary")
    inp.write_text(json.dumps({"contract_version": 1, "cases": [case]}), encoding="utf-8")
    import hashlib
    sha = hashlib.sha256(inp.read_bytes()).hexdigest()
    result = refund_result(case)
    result["case_id"] = case["case_id"]
    out = base / "public" / "run_1.json"
    out.write_text(json.dumps({"input_sha256": sha, "results": [result]}), encoding="utf-8")
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(verify, "load_labels", lambda s: {"public_01_refund_boundary": LABEL} if s == "public" else {})
    run = {"split": "public", "run": 1, "input": "evaluation/in.json", "input_sha256": sha,
           "output": "evaluation/public/run_1.json", "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
           "cases": 1}
    (base / "run_manifest.json").write_text(json.dumps({"status": "ok", "runs": [run]}), encoding="utf-8")
    scored = business.score_case(case, result, LABEL, "public/run_1/public_01_refund_boundary")
    (base / "cases.jsonl").write_text(json.dumps({**scored, "split": "public", "run": 1, "trace_refs": []}) + "\n",
                                      encoding="utf-8")
    (base / "report").mkdir()
    (base / "report" / "scorecard.json").write_text(json.dumps({"business": {"public": {
        "complete_task_success": "1/1"}}}), encoding="utf-8")
    files = {p.relative_to(tmp_path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in base.rglob("*") if p.is_file()}
    (base / "report" / "evidence_manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    return verify, base, out


@pytest.mark.parametrize("tamper", ["none", "output", "trace", "stat", "secret", "missing_case", "duplicate_run",
                                    "label_mismatch"])
def test_verifier_detects_tampering(tmp_path, monkeypatch, tamper):
    verify, base, out = _package(tmp_path, monkeypatch)
    manifest = json.loads((base / "run_manifest.json").read_text())
    if tamper == "output":
        out.write_text(out.read_text().replace("4500", "4501"), encoding="utf-8")
    elif tamper == "trace":
        rows = json.loads((base / "cases.jsonl").read_text())
        rows["trace_refs"] = ["ab" * 16]
        (base / "cases.jsonl").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    elif tamper == "stat":
        (base / "report" / "scorecard.json").write_text(json.dumps({"business": {"public": {
            "complete_task_success": "6/6"}}}), encoding="utf-8")
    elif tamper == "secret":
        (base / "report" / "notes.md").write_text("see C:\\Users\\someone\\x", encoding="utf-8")
    elif tamper == "missing_case":
        doc = json.loads(out.read_text())
        doc["results"] = []
        out.write_text(json.dumps(doc), encoding="utf-8")
    elif tamper == "duplicate_run":
        manifest["runs"].append(dict(manifest["runs"][0]))
        (base / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    elif tamper == "label_mismatch":
        rows = json.loads((base / "cases.jsonl").read_text())
        rows["status"] = "fail"
        (base / "cases.jsonl").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    problems = verify.check(base)
    assert (problems == []) == (tamper == "none"), problems


# 9. independence ---------------------------------------------------------------------------
def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_oracle_and_scorers_do_not_use_app_policy():
    for path in [ROOT / "evals" / "oracle.py", *(ROOT / "evals" / "scoring").glob("*.py"),
                 *(ROOT / "evals" / "generators").glob("*.py")]:
        assert not any(m.startswith("app.domain") for m in _imports(path)), path
    assert not any(m.startswith("app") for m in _imports(ROOT / "evals" / "oracle.py"))


RUNTIME = ("agents", "services", "guardrails", "domain", "db", "api", "observability", "fixtures")


def test_runtime_never_imports_evaluation_labels_or_oracle():
    # app/benchmarks and app/evaluation/repetitions label evidence with the oracle (QA only, by design);
    # the decision-making runtime and adapter.py never import it, and nothing in app/ imports evals.
    app_code = list((ROOT / "backend" / "app").rglob("*.py"))
    assert len(app_code) > 30  # never pass by scanning nothing
    for path in [*app_code, ROOT / "adapter.py"]:
        text = path.read_text(encoding="utf-8")
        modules = _imports(path)
        assert not any(m.startswith("evals") for m in modules), path
        part = path.relative_to(ROOT).parts
        if path.name == "adapter.py" or (len(part) > 1 and part[1] in RUNTIME):
            assert not any(m.startswith("tests") for m in modules), path
        assert "evals/datasets" not in text and "labels.json" not in text, path


def test_runtime_has_no_dataset_ids_or_phrases():
    datasets = [json.loads((ROOT / "evals" / "datasets" / n).read_text(encoding="utf-8"))
                for n in ("variants.json", "adversarial.json")]
    needles = {c["customer_id"] for d in datasets for c in d["cases"]}
    needles |= {o["order_id"] for d in datasets for c in d["cases"] for o in c["initial_state"]["orders"]}
    needles |= {t["message"][:40] for d in datasets for c in d["cases"] for t in c["turns"] if len(t["message"]) > 25}
    code = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "backend" / "app").rglob("*.py"))
    assert "class AgentRuntime" in code  # never pass by scanning nothing
    assert not [n for n in needles if n in code]


# 10. guardrail outcomes --------------------------------------------------------------------




def test_data_files_the_scorers_read_exist():
    from evals.scoring.common import LABELS, PUBLIC_EXPECTATIONS

    assert PUBLIC_EXPECTATIONS.is_file() and LABELS.is_dir()


def test_check_results_catches_wrong_money_stock_and_decisions():
    import copy

    from evals.check_results import EXPECTED, check
    cases = json.loads((ROOT / "candidate_assets" / "public_cases.json").read_text(encoding="utf-8"))["cases"]
    results = json.loads((ROOT / "evals/evidence/evaluation/public/run_1.json").read_text(encoding="utf-8"))["results"]
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    assert not any(check(cases, results, expected).values())

    def broken(case_id, change):
        changed = copy.deepcopy(results)
        change(next(r for r in changed if r["case_id"] == case_id))
        return check(cases, changed, expected)[case_id]

    def wrong_amount(r):
        r["final_state"]["refunds"][0]["amount_cents"] += 1

    def wrong_stock(r):
        next(p for p in r["final_state"]["products"] if p["sku"] == "TEE-L")["stock"] += 1

    def wrong_decision(r):
        r["turns"][0]["decision"] = "propose"

    assert broken("public_01_refund_boundary", wrong_amount)
    assert broken("public_02_exchange", wrong_stock)
    assert broken("public_03_expired", wrong_decision)
    assert check(cases, results[:-1], expected)[results[-1]["case_id"]] == ["no result for this case"]
