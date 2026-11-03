"""Benchmark one variant: N clean repetitions of every case through the real runtime.

    uv run python -m evals.benchmarks.run --variant baseline --input candidate_assets/public_cases.json \
        --repetitions 3 --output-dir evals/evidence/final/baseline [--allow-contended] [--cold-start]

Uses app.evaluation.runner (prepare = cold start, run_fixture = fresh schemas per
case per run). Writes run_<i>.json (HP contract documents), runs.jsonl,
summary.json and environment.json (source/config identity, model, hardware,
load snapshots before/after, declared seeds, concurrency, caches and timeouts). --cold-start evicts
the model first (LLM_UNLOAD_COMMAND), so that variants' cold starts are comparable. Exit codes: 0 ok; 2 usage; 3 infrastructure;
4 machine not quiet (unless --allow-contended, which marks the variant contended).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPECTATIONS = Path(__file__).with_name("public_expectations.json")


def _validator():
    spec = importlib.util.spec_from_file_location("hp_validator", ROOT / "candidate_assets" / "validate_submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _oracle():
    sys.path.insert(0, str(ROOT))
    from evals.oracle import check_case  # QA labelling only; never influences a run
    return check_case


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def declarations(settings: Any, cold_start: bool) -> dict[str, Any]:
    """What the brief asks every run to declare: seeds, concurrency, caches and timeouts."""
    from app.db.connection import pooling_enabled
    return {
        "seeds": {"llm_seed": settings.seed, "temperature": settings.temperature, "top_p": settings.top_p,
                  "note": "the only random input to decisions; schema names and session IDs are random but unused "
                          "by any decision"},
        "concurrency": {"cases": 1, "turns": 1, "llm_requests": 1,
                        "note": "cases run one after another in one process"},
        "caches": {"model_load": "evicted before the run (LLM_UNLOAD_COMMAND), so cold start includes the model load"
                                 if cold_start else "not controlled: the model may already be loaded",
                   "warm_runs": "the model stays loaded between cases (one warm-up call before the first)",
                   "engine_prompt_cache": "engine default, not cleared between cases",
                   "guardrail_models": "loaded once per process from local caches",
                   "database_connections": "per-schema pool" if pooling_enabled() else "new connection per use",
                   "business_state": "none carried over: fresh schemas for every case in every repetition",
                   "os_file_cache": "not controlled, so a later variant may read model files from memory"},
        "timeouts": {"llm_call_s": settings.timeout_s, "llm_max_tokens": settings.max_tokens,
                     "agent_tool_steps": settings.max_tool_steps, "tool_transport": "mcp_http",
                     "task": "no overall task timeout; injected post-commit timeouts are counted separately"},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", required=True, help="label, e.g. baseline | optimized | E1-pool")
    parser.add_argument("--input", required=True)
    parser.add_argument("--repetitions", type=int, default=3, choices=range(1, 11), metavar="N (1-10)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-contended", action="store_true", help="run anyway; variant is marked contended")
    parser.add_argument("--cold-start", action="store_true", help="evict the model with LLM_UNLOAD_COMMAND before starting")
    args = parser.parse_args(argv)
    out = Path(args.output_dir).resolve()
    if (ROOT / "candidate_assets") in [out, *out.parents]:
        print("refusing to write under candidate_assets/", file=sys.stderr)
        return 2
    from app.agents.settings import AgentSettings
    from evals.benchmarks import machine
    from evals.benchmarks.identity import config_identity, source_identity
    from evals.benchmarks.probe import decode_probe
    from evals.benchmarks.profile import run_row
    from evals.benchmarks.summary import summarize_rows
    from app.evaluation import environment
    from app.evaluation.runner import InfrastructureError, prepare, run_fixture
    from app.evaluation.serialization import write_atomic
    from app.fixtures.loader import load_fixtures

    settings = AgentSettings.from_env()
    if args.cold_start and not machine.unload_model(settings.unload_command):
        print("could not evict the model: check LLM_UNLOAD_COMMAND in .env", file=sys.stderr)
        return 3
    before = machine.load_snapshot(settings.server_process_names())
    problems = machine.quiet_problems(before)
    if problems and not args.allow_contended:
        print("machine not quiet:\n  " + "\n  ".join(problems), file=sys.stderr)
        print("close the listed apps (or pass --allow-contended to record a contended run)", file=sys.stderr)
        return 4
    root, input_sha = load_fixtures(args.input)
    expectations = {k: v for k, v in json.loads(EXPECTATIONS.read_text()).items() if not k.startswith("_")}
    raw_cases = {c["case_id"]: c for c in json.loads(Path(args.input).read_bytes())["cases"]}
    source = source_identity()
    config = config_identity(settings)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        ctx = prepare(settings)
    except InfrastructureError as error:
        print(f"infrastructure failure: {error}", file=sys.stderr)
        return 3
    validator, check_case = _validator(), _oracle()
    probe_before = decode_probe(ctx.model)  # warm model, before the cases
    rows, validator_results = [], {}
    for rep in range(1, args.repetitions + 1):
        try:
            document, raws = run_fixture(ctx, root, input_sha)
        except InfrastructureError as error:
            print(f"repetition {rep}: infrastructure failure: {error}", file=sys.stderr)
            return 3
        run_path = out / f"run_{rep}.json"
        write_atomic(run_path, document)
        try:
            validator_results[run_path.name] = {"ok": validator.check(Path(args.input), run_path) ==
                                                len(root.cases), "error": None}
        except (ValueError, KeyError, TypeError) as error:
            validator_results[run_path.name] = {"ok": False, "error": str(error)}
        for result, raw in zip(document["results"], raws):
            rows.append(run_row(variant=args.variant, repetition=rep, source_id=source["source_id"],
                                config_id=config["config_id"], input_sha256=input_sha, result=result, raw=raw,
                                oracle_problems=check_case(raw_cases[result["case_id"]], result),
                                expected=expectations.get(result["case_id"]), run_file=_rel(run_path)))
        done = [r for r in rows if r["repetition"] == rep]
        print(f"rep {rep}: " + ", ".join(f"{r['case_id']}={'ok' if r['completed_task_success'] else 'FAIL'}"
                                        f"({r['elapsed_ms'] / 1000:.1f}s)" for r in done))
    probe = decode_probe(ctx.model)  # after the cases, so it cannot affect them
    after = machine.load_snapshot(settings.server_process_names())
    after_problems = machine.quiet_problems(after) + machine.probe_problems(probe_before, probe)

    from app.observability import telemetry as tel
    tel.shutdown()  # flush (no-op when telemetry is off)
    (out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary = {"variant": args.variant, "source_id": source["source_id"], "config_id": config["config_id"],
               "input_sha256": input_sha, "repetitions": args.repetitions, "cases": len(root.cases),
               "cold_start_ms": ctx.cold_start_ms, "validator": validator_results,
               "decode_probe": probe, "decode_probe_before": probe_before, **summarize_rows(rows)}
    write_atomic(out / "summary.json", summary)
    write_atomic(out / "environment.json", {
        "variant": args.variant, "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source, "config": config, "input_sha256": input_sha, "input_file": _rel(Path(args.input)),
        "model": {k: ctx.model_info.get(k) for k in ("model", "digest", "quantization", "engine", "engine_version",
                                                     "decoding")},
        "cold_start": {"cold_start_ms": ctx.cold_start_ms, "notes": ctx.notes, "samples": 1},
        "declarations": declarations(settings, args.cold_start),
        "hardware": environment.hardware(), "packages": environment.package_versions(),
        "guardrails": ctx.guards.describe(),
        "load_before": before, "load_after": after,
        "contended": bool(problems or after_problems), "contention": problems + after_problems,
        "concurrency": 1, "prewarm": "runner.prepare(): validators + model server reachable + one warm-up call"})
    print(json.dumps({k: summary[k] for k in ("complete_task_success", "task_runs", "successful_task_latency_ms",
                                               "component_median_ms_per_task")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
