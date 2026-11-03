"""Evidence checker for a benchmark folder (exit 1 on any failure).

    uv run python -m evals.benchmarks.verify evals/evidence/final

Re-checks every measured variant directory (any folder with runs.jsonl):
fixture hash, ordered case IDs and turn counts per run file, >= 3 repetitions
per case with no gaps or duplicates, the HP validator, an independent oracle
re-run per result, finite latencies, contention flags, baseline provenance
(tooling-only changes, measured before every experiment), and that no
credentials, tokens, raw blocked text or absolute user paths appear anywhere.
Optionally checks retail_test for leftover retail_eval_* schemas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parents[2] / "candidate_assets"  # never patched in tests
FIXTURE_SHA = "f3f9dc73ac2d846de3af5c8647afcccd58d0335b80cd06f416f4dd08a1ca6165"
# Measurement-only code may differ from HEAD in the baseline; app behaviour may not.
TOOLING_PREFIXES = ("evals/", "backend/tests/", "README", "CLAUDE")
FORBIDDEN = [re.compile(p, re.IGNORECASE) for p in (
    r"retail:retail@", r"X-Demo-Session\"?\s*:\s*\"[A-Za-z0-9_-]{20,}", r"[A-Za-z]:(\\\\|\\|/)+Users(\\\\|\\|/)+\w+",
    r"/home/\w+/", r"ignore all rules", r"password\s*[:=]")]


def _validator():
    import importlib.util
    spec = importlib.util.spec_from_file_location("hp_validator", ASSETS / "validate_submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_variant(path: Path, input_path: Path, problems: list[str]) -> dict[str, Any]:
    sys.path.insert(0, str(ASSETS.parent))
    from evals.oracle import check_case
    rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
    rows = [json.loads(line) for line in (path / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line]
    env = json.loads((path / "environment.json").read_text(encoding="utf-8"))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    cases = json.loads(input_path.read_bytes())["cases"]
    case_ids = [c["case_id"] for c in cases]
    reps = summary["repetitions"]
    if env["input_sha256"] != hashlib.sha256(input_path.read_bytes()).hexdigest():
        problems.append(f"{rel}: input hash differs from {input_path.name}")
    if reps < 3:
        problems.append(f"{rel}: only {reps} repetitions")
    seen = sorted((r["repetition"], r["case_id"]) for r in rows)
    if seen != sorted((rep, cid) for rep in range(1, reps + 1) for cid in case_ids):
        problems.append(f"{rel}: runs.jsonl does not hold exactly {reps} x {len(case_ids)} unique case runs")
    validator = _validator()
    for rep in range(1, reps + 1):
        run_file = path / f"run_{rep}.json"
        if not run_file.is_file():
            problems.append(f"{rel}: missing {run_file.name}")
            continue
        doc = json.loads(run_file.read_text(encoding="utf-8"))
        try:
            validator.check(input_path, run_file)
        except (ValueError, KeyError, TypeError) as error:
            problems.append(f"{rel}/{run_file.name}: HP validator: {error}")
        if [r["case_id"] for r in doc["results"]] != case_ids:
            problems.append(f"{rel}/{run_file.name}: case order differs")
        for case, result in zip(cases, doc["results"]):
            if len(result["turns"]) != len(case["turns"]):
                problems.append(f"{rel}/{run_file.name}: {case['case_id']} turn count differs")
            row = next((r for r in rows if r["repetition"] == rep and r["case_id"] == case["case_id"]), None)
            if row is None:
                continue
            oracle_pass = not check_case(case, result)
            if oracle_pass != row["oracle_pass"]:
                problems.append(f"{rel}: {case['case_id']} rep {rep}: recorded oracle_pass={row['oracle_pass']} "
                                f"but re-check gives {oracle_pass}")
            if row["elapsed_ms"] != result["metrics"]["elapsed_ms"]:
                problems.append(f"{rel}: {case['case_id']} rep {rep}: elapsed_ms differs from the run file")
    for r in rows:
        if not (isinstance(r["elapsed_ms"], (int, float)) and math.isfinite(r["elapsed_ms"]) and r["elapsed_ms"] > 0):
            problems.append(f"{rel}: {r['case_id']} rep {r['repetition']}: bad elapsed_ms")
        for key in ("peak_ram_mb", "peak_vram_mb", "prompt_tokens", "completion_tokens"):
            v = r.get(key)
            if v is not None and not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0):
                problems.append(f"{rel}: {r['case_id']} rep {r['repetition']}: bad {key}")
    return {"dir": rel, "variant": env["variant"], "started_at": env["started_at"], "source": env["source"],
            "contended": env["contended"], "success": f"{summary['complete_task_success']}/{summary['task_runs']}"}


def scan_secrets(evidence: Path, problems: list[str]) -> None:
    for path in evidence.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".md", ".txt", ".diff"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in FORBIDDEN:
                if pattern.search(text):
                    problems.append(f"{path.relative_to(ROOT).as_posix()}: forbidden content /{pattern.pattern}/")


def leftover_schemas() -> int | None:
    try:
        import psycopg
        from app.db.connection import test_database_url
        with psycopg.connect(test_database_url(), connect_timeout=3) as conn:
            return conn.execute("SELECT count(*) FROM pg_namespace WHERE nspname LIKE 'retail_eval_%'").fetchone()[0]
    except Exception:  # noqa: BLE001 - reported as unknown
        return None


def verify(evidence: Path, input_path: Path, *, check_db: bool = True) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    measured = sorted(evidence.rglob("runs.jsonl"))
    # Folders starting with "_" hold runs kept for transparency but invalidated (with a reason file).
    excluded = [p.parent for p in measured if any(part.startswith("_") for part in p.relative_to(evidence).parts)]
    for folder in excluded:
        if not (folder / "INVALID.md").is_file():
            problems.append(f"{folder.name}: excluded run folder has no INVALID.md explaining why")
    variants = [check_variant(p.parent, input_path, problems) for p in measured if p.parent not in excluded]
    by_name = {v["dir"].split("/")[-1]: v for v in variants}
    baseline = by_name.get("baseline")
    if baseline is None:
        problems.append("no baseline variant")
    else:
        extra = [f for f in baseline["source"]["changed_files"] if not f.startswith(TOOLING_PREFIXES)]
        # A variant on the same source differs only by configuration, so uncommitted app code can't favour either.
        final = by_name.get("final")
        same_source = final is not None and final["source"]["source_id"] == baseline["source"]["source_id"]
        if extra and not same_source:
            problems.append(f"baseline ran with non-tooling changes: {extra}")
        if baseline["contended"]:
            problems.append("baseline is contended")
        later = [v for v in variants if v is not baseline and v["started_at"] < baseline["started_at"]]
        if later:
            problems.append(f"variants measured before the baseline: {[v['dir'] for v in later]}")
    for name in ("optimized", "final"):
        if name in by_name and by_name[name]["contended"]:
            problems.append(f"{name} variant is contended")
    scan_secrets(evidence, problems)
    leftovers = leftover_schemas() if check_db else 0
    if leftovers:
        problems.append(f"{leftovers} retail_eval_* schemas left in retail_test")
    return problems, {"variants": variants, "excluded": [f.relative_to(evidence).as_posix() for f in excluded],
                      "leftover_eval_schemas": leftovers}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("evidence_dir", nargs="?", default="evals/evidence/final")
    parser.add_argument("--input", default="candidate_assets/public_cases.json")
    args = parser.parse_args(argv)
    input_path = (ROOT / args.input).resolve()
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != FIXTURE_SHA:
        print("public fixture hash changed!", file=sys.stderr)
        return 1
    problems, report = verify((ROOT / args.evidence_dir).resolve(), input_path)
    for v in report["variants"]:
        print(f"{v['dir']}: {v['success']} success, source {v['source']['source_id']}, contended={v['contended']}")
    for folder in report["excluded"]:
        print(f"excluded (invalid, kept for transparency): {folder}")
    print(f"leftover retail_eval_* schemas: {report['leftover_eval_schemas']}")
    if problems:
        print("VERIFY FAILED:\n  " + "\n  ".join(problems))
        return 1
    print("VERIFY PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
