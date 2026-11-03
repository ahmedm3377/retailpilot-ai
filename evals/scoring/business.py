"""Business correctness per case: independent oracle + labeled decisions + exact expected effects.

The oracle (evals/oracle.py) checks contract invariants; this module adds the
labeled expectations (which decision, which exact effects) and classifies the
critical invariant violations. It never imports retail_domain.policy.
"""
from __future__ import annotations

from typing import Any

from evals.oracle import check_case


def _effects_problems(case: dict[str, Any], result: dict[str, Any], expected: dict[str, Any] | None) -> list[str]:
    """Compare final_state with the label's exact expected effects; extra effects are prohibited."""
    if expected is None:
        return []
    final = result.get("final_state") or {}
    problems = []
    got_refunds = sorted((r.get("line_id"), r.get("amount_cents")) for r in final.get("refunds", []))
    want_refunds = sorted((r["line_id"], r["amount_cents"]) for r in expected["refunds"])
    if got_refunds != want_refunds:
        problems.append(f"refunds {got_refunds} != expected {want_refunds}")
    got_ex = sorted((r.get("line_id"), r.get("replacement_sku")) for r in final.get("exchanges", []))
    want_ex = sorted((r["line_id"], r["replacement_sku"]) for r in expected["exchanges"])
    if got_ex != want_ex:
        problems.append(f"exchanges {got_ex} != expected {want_ex}")
    stock = {p["sku"]: p.get("stock") for p in final.get("products", [])}
    for sku, value in expected["stock"].items():
        if stock.get(sku) != value:
            problems.append(f"stock {sku}={stock.get(sku)} != expected {value}")
    return problems


def critical_violations(case: dict[str, Any], result: dict[str, Any], expected: dict[str, Any] | None) -> list[str]:
    """Violations that no score may offset (see skill §3A)."""
    init, final = case["initial_state"], result.get("final_state") or {}
    out = []
    start = {p["sku"]: p["stock"] for p in init["products"]}
    for product in final.get("products", []):
        if isinstance(product.get("stock"), int) and product["stock"] < 0:
            out.append(f"negative_stock:{product['sku']}")
    lines = {(o["order_id"], l["line_id"]): l for o in init["orders"] for l in o["lines"]}
    owner = {o["order_id"]: o["customer_id"] for o in init["orders"]}
    for refund in final.get("refunds", []):
        line = lines.get((refund.get("order_id"), refund.get("line_id")))
        if line is None or owner.get(refund.get("order_id")) != case["customer_id"]:
            out.append("unauthorized_refund")
        elif refund.get("amount_cents") != line["paid_cents"]:
            out.append("wrong_refund_amount")
    effects = [(r.get("order_id"), r.get("line_id")) for r in final.get("returns", [])]
    if len(effects) != len(set(effects)) or len(final.get("refunds", [])) + len(final.get("exchanges", [])) > len(
            set(effects)):
        out.append("duplicate_effect")
    for item in final.get("quarantine", []):
        sku = item.get("sku")
        if sku and isinstance(start.get(sku), int):
            sold = sum(1 for e in final.get("exchanges", []) if e.get("replacement_sku") == sku)
            external = [f for f in case.get("faults", []) if f.get("type") == "stock_change" and f.get("sku") == sku]
            current = next((p["stock"] for p in final.get("products", []) if p["sku"] == sku), None)
            if not external and current is not None and current > start[sku] - sold:
                out.append(f"original_back_in_stock:{sku}")
    completed = any(t.get("decision") == "completed" for t in result.get("turns", []))
    if completed and case["request_id"] not in (final.get("idempotency") or {}):
        out.append("completed_without_idempotency")
    if expected is not None:
        want = len(expected["refunds"]) + len(expected["exchanges"])
        got = len(final.get("refunds", [])) + len(final.get("exchanges", []))
        if got > want:
            out.append("unexpected_economic_effect")
    return sorted(set(out))


def score_case(case: dict[str, Any], result: dict[str, Any] | None, label: dict[str, Any], ref: str) -> dict[str, Any]:
    if result is None:
        return {"ref": ref, "case_id": case["case_id"], "status": "unverifiable", "reason": "no result"}
    oracle = check_case(case, result)
    decisions = [t.get("decision") for t in result.get("turns", [])]
    decision_problems = [f"turn {i}: {d} not in {lt['accepted_decisions']}"
                         for i, (d, lt) in enumerate(zip(decisions, label["turns"]), start=1)
                         if d not in lt["accepted_decisions"]]
    if len(decisions) != len(label["turns"]):
        decision_problems.append(f"{len(decisions)} turns for {len(label['turns'])} labeled")
    effects = _effects_problems(case, result, label.get("expected_effects"))
    critical = critical_violations(case, result, label.get("expected_effects"))
    success = not oracle and not decision_problems and not effects and not critical
    return {"ref": ref, "case_id": case["case_id"], "status": "pass" if success else "fail",
            "decisions": decisions, "oracle_problems": oracle, "decision_problems": decision_problems,
            "effect_problems": effects, "critical_violations": critical}
