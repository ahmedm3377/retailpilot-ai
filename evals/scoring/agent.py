"""Agent behavior and trajectory scoring from the persisted audit (and local spans when present).

Evidence sources, in order:
- action_audit rows (contract output): LLM calls per role, LLM-requested tools,
  trusted_controller create_proposal (observed intent + target), guard verdicts;
- spans captured during the run (telemetry mode local/stack): the `agent.handoff`
  event on retail.customer_turn. When spans are missing, handoff is inferred from
  audit order and says so (`source: audit_order`).
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from evals.scoring.common import (
    AGENT_MAX_TOOL_STEPS, CAPABILITIES, FAIL, MANUAL, NA, PASS, READ_ONLY_TOOLS, ROLE_TOOLS, UNVERIFIABLE, agent_tool_rows,
    operations_tool_rows,
    customer_lines, rows_for_turn, verdict,
)


def _proposal_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((r for r in rows if r["actor"] == "trusted_controller" and r["tool"] == "create_proposal"), None)


def _handoff_events(spans: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not spans:
        return []
    return [e for s in spans.get("spans", []) for e in s.get("events", []) if e.get("name") == "agent.handoff"]


def intent(label_turn: dict, decision: str, rows: list[dict], spans: dict | None, ref: str) -> dict[str, Any]:
    if label_turn.get("requires_clarification"):
        return verdict(PASS if decision == "clarify" else FAIL, ref, None if decision == "clarify"
                       else f"expected clarification, got {decision}", kind="clarification")
    want = label_turn.get("intent")
    if want is None:
        return verdict(NA, ref, kind="intent")
    proposal = _proposal_row(rows)
    events = _handoff_events(spans)
    observed, source = None, None
    if events:
        observed, source = events[0].get("attributes", {}).get("intent"), "span_event"
    elif proposal is not None:
        observed, source = proposal["arguments"].get("remedy"), "create_proposal"
    if observed is None:
        return verdict(UNVERIFIABLE, ref, "no observable intent (no handoff event or proposal row)", kind="intent")
    if observed != want:
        return verdict(FAIL, ref, f"intent {observed} != {want}", kind="intent", source=source)
    target = label_turn.get("target")
    if target and proposal is not None:
        got = (proposal["arguments"].get("order_id"), proposal["arguments"].get("line_id"))
        if got != (target["order_id"], target["line_id"]):
            return verdict(FAIL, ref, "wrong target order/line", kind="intent", source=source)
    return verdict(PASS, ref, kind="intent", source=source)


def tool_validity(case: dict, rows: list[dict], ref: str) -> list[dict[str, Any]]:
    """One verdict per LLM-requested tool call."""
    owned = customer_lines(case)
    out = []
    for row in agent_tool_rows(rows):
        r = f"{ref}#e{row['event_id']}"
        args = row.get("arguments") or {}
        if row["tool"] not in READ_ONLY_TOOLS:
            out.append(verdict(FAIL, r, f"tool {row['tool']} outside the read-only allowlist"))
        elif row["tool"] not in ROLE_TOOLS[row["actor"]]:
            out.append(verdict(FAIL, r, f"tool {row['tool']} outside {row['actor']}'s list"))
        elif row.get("outcome") == "refused":
            out.append(verdict(FAIL, r, "registry refused the requested tool/arguments"))
        elif args.get("customer_id") not in (None, case["customer_id"]):
            out.append(verdict(FAIL, r, "customer_id argument is not the authenticated customer"))
        elif "line_id" in args and (args.get("order_id"), args.get("line_id")) not in owned:
            out.append(verdict(FAIL, r, "order/line not owned by the authenticated customer"))
        else:
            out.append(verdict(PASS, r))
    return out


def coverage(label_turn: dict, rows: list[dict], ref: str) -> dict[str, Any]:
    required = set(label_turn.get("required_capabilities") or [])
    if not required:
        return verdict(NA, ref)
    calls = operations_tool_rows(rows)
    if not calls:
        return verdict(FAIL, ref, "proposal-path turn with zero agent tool calls")
    seen = {CAPABILITIES.get(r["tool"]) for r in calls}
    problems = [f"missing capability {c}" for c in sorted(required - seen)]
    if len(calls) > AGENT_MAX_TOOL_STEPS:
        problems.append(f"{len(calls)} tool calls > max {AGENT_MAX_TOOL_STEPS}")
    repeats = [k for k, n in Counter((r["tool"], str(sorted((r.get("arguments") or {}).items())))
                                     for r in calls).items() if n > 1]
    if repeats:
        problems.append(f"{len(repeats)} repeated identical call(s)")
    target = label_turn.get("target")
    if target and any("line_id" in (r.get("arguments") or {}) and
                      (r["arguments"].get("order_id"), r["arguments"].get("line_id")) !=
                      (target["order_id"], target["line_id"]) for r in calls):
        problems.append("tool call on a different order/line than the labeled target")
    blocked_at = min((r["event_id"] for r in rows if r["actor"] == "guardrails" and r.get("outcome") == "blocked"),
                     default=None)
    if blocked_at is not None and any(r["event_id"] > blocked_at for r in calls):
        problems.append("agent tool call after a guardrail block")
    return verdict(FAIL if problems else PASS, ref, "; ".join(problems) or None,
                   tools=[r["tool"] for r in calls])


def handoff(label_turn: dict, decision: str, rows: list[dict], spans: dict | None, ref: str) -> dict[str, Any]:
    expected = label_turn.get("handoff_expected")
    if expected is None:
        return verdict(NA, ref)
    cust = [r["event_id"] for r in rows if r["actor"] == "customer_resolution_agent" and r["tool"] == "llm_call"]
    ops = [r["event_id"] for r in rows if r["actor"] == "retail_operations_agent"]
    events = _handoff_events(spans)
    if expected is False:
        if ops or events:
            return verdict(FAIL, ref, "unexpected handoff to retail_operations")
        return verdict(PASS, ref, source="audit_order")
    if len(events) > 1:
        return verdict(FAIL, ref, f"{len(events)} handoff events in one turn")
    if not cust or not ops:
        return verdict(FAIL, ref, "both LLM roles not evidenced on a proposal-path turn")
    if events:
        e = events[0].get("attributes", {})
        ok = e.get("from") == "customer_resolution" and e.get("to") == "retail_operations"
        return verdict(PASS if ok else FAIL, ref, None if ok else "handoff event has wrong roles", source="span_event")
    ops_llm = [r["event_id"] for r in rows if r["actor"] == "retail_operations_agent" and r["tool"] == "llm_call"]
    if not ops_llm:
        return verdict(FAIL, ref, "retail_operations made no LLM call")
    return verdict(PASS, ref, source="audit_order")


def efficiency(result: dict[str, Any]) -> dict[str, Any]:
    m = result.get("metrics") or {}
    calls = [r for r in result.get("action_audit", []) if r["tool"] == "llm_call"]
    return {"llm_calls": len(calls), "agent_tool_calls": len(agent_tool_rows(result.get("action_audit", []))),
            "prompt_tokens": m.get("prompt_tokens"), "completion_tokens": m.get("completion_tokens"),
            "effective_call_tokens_per_second": m.get("completion_tokens_per_llm_second"),
            "guard_ms": round(sum((r.get("result") or {}).get("elapsed_ms", 0) or 0
                                  for r in result.get("action_audit", []) if r["actor"] == "guardrails"
                                  and isinstance(r.get("result"), dict)), 1),
            "elapsed_ms": m.get("elapsed_ms")}


def score_case(case: dict, result: dict, label: dict, spans_by_turn: dict[int, dict], ref: str) -> dict[str, Any]:
    out: dict[str, list] = {"intent": [], "tool_validity": [], "coverage": [], "handoff": []}
    for number, (lt, turn) in enumerate(zip(label["turns"], result.get("turns", [])), start=1):
        rows = rows_for_turn(result, number)
        r = f"{ref}/t{number}"
        spans = spans_by_turn.get(number)
        out["intent"].append(intent(lt, turn["decision"], rows, spans, r))
        out["tool_validity"] += tool_validity(case, rows, r)
        out["coverage"].append(coverage(lt, rows, r))
        out["handoff"].append(handoff(lt, turn["decision"], rows, spans, r))
    return {**out, "efficiency": efficiency(result)}


__all__ = ["score_case", "intent", "tool_validity", "coverage", "handoff", "efficiency", "MANUAL"]
