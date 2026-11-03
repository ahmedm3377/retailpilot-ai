"""Summaries of benchmark rows (runs.jsonl), shared by run.py and compare.py."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from evals.benchmarks.stats import PERCENTILE_METHOD, median, summarize


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r["completed_task_success"]]
    per_case: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        per_case[r["case_id"]].append(r["elapsed_ms"])
    tokens_c = [r["completion_tokens"] for r in rows]
    tokens_p = [r["prompt_tokens"] for r in rows]
    return {
        "task_runs": len(rows),
        "complete_task_success": len(ok),
        "success_rate": round(len(ok) / len(rows), 4) if rows else None,
        "failures_by_category": dict(Counter(r["failure_category"] for r in rows if r["failure_category"])),
        "oracle_pass": sum(1 for r in rows if r["oracle_pass"]),
        "failed_tasks": len(rows) - len(ok),
        "error_turns": sum(r["error_turns"] for r in rows),
        "llm_call_failures": sum(r.get("llm_calls_failed") or 0 for r in rows),
        "injected_timeouts": sum(r["injected_timeouts"] for r in rows),
        "recovered_timeouts": sum(r["recovered_timeouts"] for r in rows),
        "successful_task_latency_ms": {**summarize(r["elapsed_ms"] for r in ok), "method": PERCENTILE_METHOD,
                                       "scope": "first turn start to last turn end; successful runs only"},
        "per_case_median_ms": {case: median(values) for case, values in sorted(per_case.items())},
        "component_median_ms_per_task": {
            "llm_ms": median(r["llm_ms"] for r in ok), "guard_ms": median(r["guard_ms"] for r in ok),
            "other_ms": median(r["other_ms"] for r in ok),
            "derivation": "llm = sum of llm_call latency; guard = sum of guardrail elapsed; other = remainder"},
        "llm_calls_per_task_median": median(r["llm_calls"] for r in ok),
        "tokens": {
            "prompt_total": None if None in tokens_p else sum(tokens_p),
            "completion_total": None if None in tokens_c else sum(tokens_c),
            "effective_output_tokens_per_call_second_median":
                median(r["effective_output_tokens_per_call_second"] for r in ok),
            "note": "engine-reported completion tokens / summed LLM call wall time (not pure decode time)"},
        "memory": {
            "peak_ram_mb_max": max((r["peak_ram_mb"] for r in rows if r["peak_ram_mb"] is not None), default=None),
            "peak_python_rss_mb_max": max((r["peak_python_rss_mb"] for r in rows
                                           if r["peak_python_rss_mb"] is not None), default=None),
            "peak_model_server_rss_mb_max": max((r["peak_model_server_rss_mb"] for r in rows
                                                 if r["peak_model_server_rss_mb"] is not None), default=None),
            "peak_vram_mb_max": max((r["peak_vram_mb"] for r in rows if r["peak_vram_mb"] is not None), default=None),
            "peak_mcp_server_rss_mb_max": max((r.get("peak_mcp_server_rss_mb") for r in rows
                                                 if r.get("peak_mcp_server_rss_mb") is not None), default=None),
            "peak_gpu_device_used_mb_max": max((r["peak_gpu_device_used_mb"] for r in rows
                                                if r["peak_gpu_device_used_mb"] is not None), default=None),
            "scope": {"peak_ram_mb": "adapter Python RSS + its child processes (MCP servers) + the processes named in "
                                     "LLM_SERVER_PROCESSES, summed per sample (shared pages double-counted)",
                      "peak_vram_mb": "nvidia-smi memory.used, whole GPU incl. other applications",
                      "peak_gpu_device_used_mb": "whole GPU incl. other applications"}},
    }
