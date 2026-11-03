"""Correctness-gated comparison of two measured variants, plus the evidence manifest.

    uv run python -m evals.benchmarks.compare --baseline evals/evidence/final/baseline \
        --optimized evals/evidence/final/final --output-dir evals/evidence/final/comparison

Reads only run directories written by evals.benchmarks.run (summary.json,
environment.json, runs.jsonl). Writes comparison.json, comparison.md and
evidence_manifest.json (repository-relative paths + SHA-256 of every evidence file).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from evals.benchmarks.stats import PERCENTILE_METHOD, improvement_pct, increase_pct

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).resolve().parents[2] / "candidate_assets" / "public_cases.json"
MANIFEST_SCHEMA = 1
GUARD_KEYS = ("guardrails_enabled", "jailbreak_threshold", "toxicity_threshold", "fail_closed", "max_input_chars")


def load_variant(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in (path / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return {"dir": path, "summary": json.loads((path / "summary.json").read_text(encoding="utf-8")),
            "env": json.loads((path / "environment.json").read_text(encoding="utf-8")), "rows": rows}


def comparability(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    ea, eb = a["env"], b["env"]
    sa, sb = ea["config"]["settings"], eb["config"]["settings"]
    hw = ("cpu", "gpu", "ram", "os", "power")
    checks = {
        "same_input": ea["input_sha256"] == eb["input_sha256"],
        "same_model_digest": ea["model"]["digest"] == eb["model"]["digest"],
        "same_model_route_and_decoding": (sa["model"], sa["temperature"], sa["top_p"], sa["seed"], sa["think"])
        == (sb["model"], sb["temperature"], sb["top_p"], sb["seed"], sb["think"]),
        "same_guardrail_policy": all(sa[k] == sb[k] for k in GUARD_KEYS),
        "same_hardware_and_power": all(ea["hardware"].get(k) == eb["hardware"].get(k) for k in hw),
        "same_percentile_method": a["summary"]["successful_task_latency_ms"]["method"]
        == b["summary"]["successful_task_latency_ms"]["method"] == PERCENTILE_METHOD,
        "neither_contended": not ea["contended"] and not eb["contended"],
        "same_timeouts": all(sa.get(k) == sb.get(k) for k in ("timeout_s", "max_tokens", "max_tool_steps")),
        "at_least_3_repetitions": a["summary"]["repetitions"] >= 3 and b["summary"]["repetitions"] >= 3,
        "same_case_set": sorted({r["case_id"] for r in a["rows"]}) == sorted({r["case_id"] for r in b["rows"]}),
    }
    return {"comparable": all(checks.values()), "checks": checks,
            "reasons": [k for k, ok in checks.items() if not ok]}


def _cold_start_notes(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    na = (a["env"].get("cold_start") or {}).get("notes")
    nb = (b["env"].get("cold_start") or {}).get("notes")
    return {"baseline": na or [], "optimized": nb or [], "comparable": na is not None and na == nb}


def _pair(a: Any, b: Any, lower_is_better: bool = True) -> dict[str, Any]:
    change = improvement_pct(a, b) if lower_is_better else increase_pct(a, b)
    return {"baseline": a, "optimized": b,
            ("improvement_pct" if lower_is_better else "increase_pct"): change}


def build(baseline: dict[str, Any], optimized: dict[str, Any], experiments: list[dict[str, Any]]) -> dict[str, Any]:
    sa, sb = baseline["summary"], optimized["summary"]
    comp = comparability(baseline, optimized)
    quality_preserved = sb["complete_task_success"] >= sa["complete_task_success"] and \
        sb["success_rate"] >= sa["success_rate"]
    la, lb = sa["successful_task_latency_ms"], sb["successful_task_latency_ms"]
    cases = sorted(set(sa["per_case_median_ms"]) | set(sb["per_case_median_ms"]))
    return {
        "schema_version": 1,
        "methodology": {
            "percentile": PERCENTILE_METHOD,
            "latency_scope": la.get("scope"),
            "formulas": {"lower_is_better": "improvement_pct = 100 * (baseline - optimized) / baseline",
                         "higher_is_better": "increase_pct = 100 * (optimized - baseline) / baseline",
                         "unavailable": "null when either value is missing or the baseline is <= 0"},
            "sample": f"{sa['repetitions']} and {sb['repetitions']} repetitions x {sa['cases']} cases; "
                      f"n(successful) = {la['n']} and {lb['n']}; p95 of n<=20 is a near-maximum, no "
                      "significance claimed"},
        "variants": {name: {"dir": v["dir"].resolve().relative_to(ROOT).as_posix(),
                            "source_id": v["summary"]["source_id"], "config_id": v["summary"]["config_id"],
                            "started_at": v["env"]["started_at"], "contended": v["env"]["contended"],
                            "load_before": v["env"]["load_before"], "load_after": v["env"]["load_after"]}
                     for name, v in (("baseline", baseline), ("optimized", optimized))},
        "input_sha256": sa["input_sha256"], "model_digest": baseline["env"]["model"]["digest"],
        "comparability": comp,
        "quality": {"baseline_success": f"{sa['complete_task_success']}/{sa['task_runs']}",
                    "optimized_success": f"{sb['complete_task_success']}/{sb['task_runs']}",
                    "quality_preserved": quality_preserved,
                    "failures": {"baseline": sa["failures_by_category"], "optimized": sb["failures_by_category"]},
                    "error_turns": _pair(sa["error_turns"], sb["error_turns"]),
                    "llm_call_failures": {"baseline": sa.get("llm_call_failures"),
                                          "optimized": sb.get("llm_call_failures")},
                    "injected_timeouts": {"baseline": sa["injected_timeouts"], "optimized": sb["injected_timeouts"]},
                    "recovered_timeouts": {"baseline": sa["recovered_timeouts"],
                                           "optimized": sb["recovered_timeouts"]},
                    "validator": {"baseline": sa["validator"], "optimized": sb["validator"]}},
        "latency_ms": {"p50": _pair(la["p50"], lb["p50"]), "p95": _pair(la["p95"], lb["p95"]),
                       "mean": _pair(la["mean"], lb["mean"]), "n": {"baseline": la["n"], "optimized": lb["n"]},
                       "per_case_median": {c: _pair(sa["per_case_median_ms"].get(c), sb["per_case_median_ms"].get(c))
                                           for c in cases}},
        "components_median_ms_per_task": {
            k: _pair(sa["component_median_ms_per_task"][k], sb["component_median_ms_per_task"][k])
            for k in ("llm_ms", "guard_ms", "other_ms")},
        "llm_calls_per_task_median": _pair(sa["llm_calls_per_task_median"], sb["llm_calls_per_task_median"]),
        "tokens": {"completion_total": _pair(sa["tokens"]["completion_total"], sb["tokens"]["completion_total"]),
                   "prompt_total": _pair(sa["tokens"]["prompt_total"], sb["tokens"]["prompt_total"]),
                   "effective_output_tokens_per_call_second_median": _pair(
                       sa["tokens"]["effective_output_tokens_per_call_second_median"],
                       sb["tokens"]["effective_output_tokens_per_call_second_median"], lower_is_better=False),
                   "note": sa["tokens"]["note"]},
        "memory": {k: _pair(sa["memory"].get(k), sb["memory"].get(k)) for k in
                   ("peak_ram_mb_max", "peak_python_rss_mb_max", "peak_model_server_rss_mb_max", "peak_vram_mb_max",
                    "peak_gpu_device_used_mb_max", "peak_mcp_server_rss_mb_max")} | {"scope": sa["memory"]["scope"]},
        "cold_start_ms": _pair(sa["cold_start_ms"], sb["cold_start_ms"]),
        "cold_start_notes": _cold_start_notes(baseline, optimized),
        "declarations": {"baseline": baseline["env"].get("declarations"),
                         "optimized": optimized["env"].get("declarations")},
        "experiments": experiments,
        "claims_allowed": comp["comparable"] and quality_preserved,
    }


def _fmt(v: Any, unit: str = "") -> str:
    if v is None:
        return "unavailable"
    return f"{v:,.1f}{unit}" if isinstance(v, float) else f"{v}{unit}"


def render_md(c: dict[str, Any]) -> str:
    lat, q = c["latency_ms"], c["quality"]
    lines = [
        "# Benchmark comparison: baseline versus final", "",
        f"Input SHA-256 `{c['input_sha256']}` · model digest `{(c['model_digest'] or '')[:12]}` · "
        f"baseline `{c['variants']['baseline']['source_id']}` (config `{c['variants']['baseline']['config_id']}`) "
        f"→ optimized `{c['variants']['optimized']['source_id']}` (config `{c['variants']['optimized']['config_id']}`)",
        "",
        f"**Comparable:** {c['comparability']['comparable']}"
        + (f" (failed checks: {', '.join(c['comparability']['reasons'])})" if c["comparability"]["reasons"] else ""),
        f"**Quality preserved:** {q['quality_preserved']} · **claims allowed:** {c['claims_allowed']}", "",
        "## Correctness", "",
        "| | Baseline | Optimized |", "|---|---|---|",
        f"| Complete-task success | {q['baseline_success']} | {q['optimized_success']} |",
        f"| Failures by category | {q['failures']['baseline'] or 'none'} | {q['failures']['optimized'] or 'none'} |",
        f"| Error turns | {q['error_turns']['baseline']} | {q['error_turns']['optimized']} |",
        f"| Failed model calls | {q['llm_call_failures']['baseline']} | {q['llm_call_failures']['optimized']} |",
        f"| Injected / recovered post-commit timeouts | {q['injected_timeouts']['baseline']} / "
        f"{q['recovered_timeouts']['baseline']} | {q['injected_timeouts']['optimized']} / "
        f"{q['recovered_timeouts']['optimized']} |", "",
        "## Latency of successful tasks (ms)", "",
        "| Metric | Baseline | Optimized | Improvement |", "|---|---|---|---|",
    ]
    for key in ("p50", "p95", "mean"):
        p = lat[key]
        lines.append(f"| {key} | {_fmt(p['baseline'])} | {_fmt(p['optimized'])} | {_fmt(p['improvement_pct'], '%')} |")
    lines += [f"| n | {lat['n']['baseline']} | {lat['n']['optimized']} | |", "",
              "| Case (median ms) | Baseline | Optimized | Improvement |", "|---|---|---|---|"]
    for case, p in lat["per_case_median"].items():
        lines.append(f"| {case} | {_fmt(p['baseline'])} | {_fmt(p['optimized'])} | {_fmt(p['improvement_pct'], '%')} |")
    lines += ["", "## Where the time goes (median per task, ms)", "",
              "| Component | Baseline | Optimized | Improvement |", "|---|---|---|---|"]
    for key, p in c["components_median_ms_per_task"].items():
        lines.append(f"| {key} | {_fmt(p['baseline'])} | {_fmt(p['optimized'])} | {_fmt(p['improvement_pct'], '%')} |")
    p = c["llm_calls_per_task_median"]
    lines.append(f"| LLM calls per task | {_fmt(p['baseline'])} | {_fmt(p['optimized'])} | "
                 f"{_fmt(p['improvement_pct'], '%')} |")
    t, m = c["tokens"], c["memory"]
    lines += ["", "## Tokens, memory, cold start", "",
              "| Metric | Baseline | Optimized | Change |", "|---|---|---|---|",
              f"| Completion tokens (total) | {_fmt(t['completion_total']['baseline'])} | "
              f"{_fmt(t['completion_total']['optimized'])} | {_fmt(t['completion_total']['improvement_pct'], '%')} |",
              f"| Effective output tokens per call-second (median) | "
              f"{_fmt(t['effective_output_tokens_per_call_second_median']['baseline'])} | "
              f"{_fmt(t['effective_output_tokens_per_call_second_median']['optimized'])} | "
              f"{_fmt(t['effective_output_tokens_per_call_second_median']['increase_pct'], '%')} (higher is better) |"]
    for key in ("peak_ram_mb_max", "peak_vram_mb_max", "peak_gpu_device_used_mb_max"):
        lines.append(f"| {key} | {_fmt(m[key]['baseline'])} | {_fmt(m[key]['optimized'])} | "
                     f"{_fmt(m[key]['improvement_pct'], '%')} |")
    cs = c["cold_start_ms"]
    lines += [f"| Cold start ms | {_fmt(cs['baseline'])} | {_fmt(cs['optimized'])} | {_fmt(cs['improvement_pct'], '%')} |",
              ""]
    cn = c["cold_start_notes"]
    if not cn["comparable"]:
        lines += ["**Cold start is not comparable between the variants:** "
                  f"baseline: {'; '.join(cn['baseline'])} / optimized: {'; '.join(cn['optimized'])}. "
                  "Its difference reflects whether the engine already held the model, not the code changes.", ""]
    declared = c.get("declarations") or {}
    if declared.get("baseline") and declared.get("optimized"):
        lines += ["## Declared seeds, concurrency, caches and timeouts", "",
                  "| Group | Setting | Baseline | Optimized |", "|---|---|---|---|"]
        for group, values in declared["baseline"].items():
            for key, value in values.items():
                lines.append(f"| {group} | {key} | {value} | {declared['optimized'].get(group, {}).get(key)} |")
        lines.append("")
    if c["experiments"]:
        lines += ["## Experiments", "", "| ID | Decision | Effect | Reason |", "|---|---|---|---|"]
        for e in c["experiments"]:
            lines.append(f"| {e['id']} | {e['decision']} | {e.get('effect', '')} | {e.get('reason', '')} |")
    lines += ["", "## Method and limits", "",
              f"- Percentiles: {c['methodology']['percentile']}.",
              f"- Latency scope: {c['methodology']['latency_scope']}.",
              f"- Sample: {c['methodology']['sample']}.",
              f"- Formulas: {c['methodology']['formulas']['lower_is_better']}; "
              f"{c['methodology']['formulas']['higher_is_better']}.",
              f"- Memory scope: {c['memory']['scope']}.",
              f"- Throughput: {t['note']}.",
              "- Experiment effects were measured one change at a time against the then-accepted state; they are "
              "not additive. The headline numbers are a fresh baseline-versus-final measurement.",
              "- Reproduce: see the README's Performance section (benchmark run, compare and verify commands).", ""]
    return "\n".join(lines)


def manifest(evidence: Path) -> dict[str, Any]:
    files = []
    for path in sorted(p for p in evidence.rglob("*") if p.is_file() and p.name != "evidence_manifest.json"):
        files.append({"path": path.resolve().relative_to(ROOT).as_posix(),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    fixture = FIXTURE
    return {"schema_version": MANIFEST_SCHEMA, "fixture": {"path": "candidate_assets/public_cases.json",
                                                          "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest()},
            "files": files}


def load_experiments(evidence: Path) -> list[dict[str, Any]]:
    out = []
    for decision in sorted((evidence / "experiments").glob("*/decision.json")):
        out.append(json.loads(decision.read_text(encoding="utf-8")))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--optimized", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    baseline, optimized = load_variant(Path(args.baseline).resolve()), load_variant(Path(args.optimized).resolve())
    evidence = out.parent
    comparison = build(baseline, optimized, load_experiments(evidence.parent))
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    (out / "comparison.md").write_text(render_md(comparison), encoding="utf-8")
    (out / "evidence_manifest.json").write_text(json.dumps(manifest(evidence), indent=2) + "\n", encoding="utf-8")
    print(f"comparable={comparison['comparability']['comparable']} "
          f"quality_preserved={comparison['quality']['quality_preserved']} "
          f"p50 {comparison['latency_ms']['p50']} p95 {comparison['latency_ms']['p95']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
