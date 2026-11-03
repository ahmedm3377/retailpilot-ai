"""Shared helpers: verdict tallies with explicit denominators, label loading, audit access."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "evals" / "datasets" / "labels"
PUBLIC_EXPECTATIONS = ROOT / "evals" / "benchmarks" / "public_expectations.json"

# The read-only tools an LLM agent may request. Kept here (not imported from app) so a
# widened app allowlist would be caught; backend/tests/test_eval_scoring.py cross-checks TOOL_ARGS.
READ_ONLY_TOOLS = frozenset({"lookup_customer_orders", "get_order_line", "check_return_eligibility",
                             "get_exchange_options", "check_inventory", "get_request_status"})
CAPABILITIES = {"check_return_eligibility": "eligibility", "get_exchange_options": "exchange_options",
                "check_inventory": "exchange_options", "get_order_line": "line_lookup"}
# Each agent's own list, kept here for the same reason.
ROLE_TOOLS = {"customer_resolution_agent": frozenset({"lookup_customer_orders"}),
              "retail_operations_agent": frozenset({"check_return_eligibility", "get_exchange_options", "check_inventory",
                                                    "get_order_line", "get_request_status"})}
AGENT_MAX_TOOL_STEPS = 5  # AgentSettings.agent_max_tool_steps default (AGENT_MAX_TOOL_STEPS)

PASS, FAIL, NA, UNVERIFIABLE, MANUAL = "pass", "fail", "not_applicable", "unverifiable", "manual_review"


def verdict(status: str, ref: str, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    return {"status": status, "ref": ref, **({"reason": reason} if reason else {}), **extra}


def tally(verdicts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Counts by status. denominator = pass + fail; an empty denominator is reported as n/a, never 100%."""
    items = list(verdicts)
    counts = {s: sum(1 for v in items if v["status"] == s) for s in (PASS, FAIL, NA, UNVERIFIABLE, MANUAL)}
    denominator = counts[PASS] + counts[FAIL]
    return {"passed": counts[PASS], "failed": counts[FAIL], "not_applicable": counts[NA],
            "unverifiable": counts[UNVERIFIABLE], "manual_review": counts[MANUAL], "denominator": denominator,
            "ratio": f"{counts[PASS]}/{denominator}" if denominator else "n/a",
            "rate": round(counts[PASS] / denominator, 4) if denominator else None,
            "failures": [{k: v[k] for k in ("ref", "reason") if k in v} for v in items if v["status"] == FAIL],
            "needs_review": [{k: v[k] for k in ("ref", "reason") if k in v} for v in items
                             if v["status"] in (UNVERIFIABLE, MANUAL)]}


def load_labels(split: str) -> dict[str, dict[str, Any]]:
    """Labels by case_id. Public decision sequences come from public_expectations.json (single source)."""
    doc = json.loads((LABELS / f"{split if split != 'synthetic_variants' else 'variants'}_labels.json")
                     .read_text(encoding="utf-8"))
    labels = {c["case_id"]: c for c in doc["cases"]}
    if split == "public":
        expected = json.loads(PUBLIC_EXPECTATIONS.read_text(encoding="utf-8"))
        for case_id, label in labels.items():
            for turn, decision in zip(label["turns"], expected[case_id]):
                turn["accepted_decisions"] = [decision]
    return labels


def rows_for_turn(result: dict[str, Any], turn: int) -> list[dict[str, Any]]:
    return sorted((r for r in result.get("action_audit", []) if r.get("turn") == turn), key=lambda r: r["event_id"])


def agent_tool_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """LLM-requested tool calls of both agents. trusted_controller rows (the business checks' read, proposal,
    commit) are deterministic, and scope_check records a decision, not a tool."""
    return [r for r in rows if r["actor"] in ROLE_TOOLS and r["tool"] not in ("llm_call", "scope_check")]


def operations_tool_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The Operations Agent's investigation: what capability coverage is scored on."""
    return [r for r in agent_tool_rows(rows) if r["actor"] == "retail_operations_agent"]


def customer_lines(case: dict[str, Any]) -> set[tuple[str, str]]:
    return {(o["order_id"], l["line_id"]) for o in case["initial_state"]["orders"]
            if o["customer_id"] == case["customer_id"] for l in o["lines"]}
