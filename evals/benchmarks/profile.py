"""Per-turn component breakdown and per-run benchmark rows, derived only from existing traces.

For each turn of a result document:
  llm_ms   = sum of `llm_call` audit events' latency_ms (actors: the two agents)
  guard_ms = sum of guardrail audit events' elapsed_ms
  other_ms = turn_ms - llm_ms - guard_ms   (graph, PostgreSQL, tools, serialization)
turn_ms comes from metrics.turn_ms (wall time of handle_customer_turn). No new
telemetry is generated; other_ms is a remainder, not a direct measurement.
"""
from __future__ import annotations

from typing import Any

FAILURE_CATEGORIES = ("guardrail_block", "model_unavailable", "schema_invalid", "wrong_decision", "oracle_fail",
                      "infrastructure")


def breakdown(result: dict[str, Any]) -> list[dict[str, Any]]:
    turn_ms = result["metrics"].get("turn_ms") or []
    rows = []
    for index, wall in enumerate(turn_ms, start=1):
        events = [e for e in result["action_audit"] if e.get("turn") == index]
        llm = [e["result"].get("latency_ms", 0.0) for e in events
               if e.get("tool") == "llm_call" and isinstance(e.get("result"), dict)]
        guard = [e["result"].get("elapsed_ms", 0.0) for e in events
                 if e.get("actor") == "guardrails" and isinstance(e.get("result"), dict)]
        llm_ms, guard_ms = round(sum(llm), 1), round(sum(guard), 1)
        rows.append({"turn": index, "turn_ms": wall, "llm_calls": len(llm), "llm_ms": llm_ms, "guard_ms": guard_ms,
                     "other_ms": round(wall - llm_ms - guard_ms, 1)})
    return rows


def _failure(decisions: list[str], expected: list[str] | None, oracle_problems: list[str],
             audit: list[dict[str, Any]]) -> str | None:
    decisions_ok = expected is None or decisions == expected
    if not oracle_problems and decisions_ok:
        return None
    if any(e.get("actor") == "guardrails" and e.get("outcome") == "blocked" for e in audit):
        return "guardrail_block"
    if any(e.get("tool") == "llm_call" and e.get("outcome") != "ok" for e in audit):
        return "model_unavailable"
    if any(e.get("actor") == "guardrails" and e.get("outcome") == "validation_error" for e in audit) \
            and "error" in decisions:
        return "schema_invalid"
    return "wrong_decision" if not decisions_ok else "oracle_fail"


def run_row(*, variant: str, repetition: int, source_id: str, config_id: str, input_sha256: str,
            result: dict[str, Any], raw: dict[str, Any], oracle_problems: list[str] | None,
            expected: list[str] | None, run_file: str) -> dict[str, Any]:
    m = result["metrics"]
    turns = breakdown(result)
    decisions = [t["decision"] for t in result["turns"]]
    audit = result["action_audit"]
    oracle_pass = oracle_problems is not None and not oracle_problems
    llm_ms = sum(t["llm_ms"] for t in turns)
    completion = m.get("completion_tokens")
    return {
        "variant": variant, "repetition": repetition, "case_id": result["case_id"], "source_id": source_id,
        "config_id": config_id, "input_sha256": input_sha256, "run_file": run_file, "cold_or_warm": "warm",
        "decisions": decisions, "expected_decisions": expected,
        "oracle_pass": oracle_pass, "oracle_problems": oracle_problems,
        "completed_task_success": oracle_pass and (expected is None or decisions == expected),
        "failure_category": _failure(decisions, expected, oracle_problems or [], audit),
        "elapsed_ms": m["elapsed_ms"], "turn_ms": m.get("turn_ms"),
        "llm_calls": m.get("llm_calls"), "llm_calls_failed": m.get("llm_calls_failed", 0),
        "llm_ms": round(llm_ms, 1),
        "guard_ms": round(sum(t["guard_ms"] for t in turns), 1),
        "other_ms": round(sum(t["other_ms"] for t in turns), 1), "turns": turns,
        "prompt_tokens": m.get("prompt_tokens"), "completion_tokens": completion,
        "effective_output_tokens_per_call_second": (round(completion / (llm_ms / 1000), 1)
                                                    if completion is not None and llm_ms > 0 else None),
        "peak_ram_mb": m.get("peak_ram_mb"), "peak_python_rss_mb": m.get("peak_python_rss_mb"),
        "peak_model_server_rss_mb": m.get("peak_model_server_rss_mb"), "peak_vram_mb": m.get("peak_vram_mb"),
        "peak_gpu_device_used_mb": m.get("peak_gpu_device_used_mb"),
        "peak_mcp_server_rss_mb": m.get("peak_mcp_server_rss_mb"),
        "error_turns": decisions.count("error"),
        "injected_timeouts": sum(1 for e in audit if e.get("tool") == "timeout_after_commit"),
        "recovered_timeouts": sum(1 for t in raw.get("turns", [])
                                  if (t.get("internal_status") or {}).get("post_commit_timeout", {}).get("reconciled")
                                  == "completed"),
    }
