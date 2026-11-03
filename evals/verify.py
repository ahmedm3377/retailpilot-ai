"""Independent consistency check of the evaluation evidence package.

    uv run python -m evals.verify evals/evidence/evaluation

Fails (exit 1) on: a missing case or run output, an altered input or output hash,
a duplicated run, a recorded case outcome that rescoring does not reproduce, a
claimed trace reference without captured spans, a scorecard statistic not backed
by cases.jsonl, a home path / credential in any artifact, or an evidence-manifest
hash mismatch. Exit 2 when the package is absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from evals.scoring import business
from evals.scoring.common import ROOT, load_labels

PUBLIC_SHA256 = "f3f9dc73ac2d846de3af5c8647afcccd58d0335b80cd06f416f4dd08a1ca6165"
SECRET = re.compile(r"(?i)([A-Z]:\\+Users\\+[^\\\"]+|/home/[a-z]|/Users/[A-Za-z]|password\s*[=:]|api[_-]?key\s*[=:]"
                    r"|sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|X-Demo-Session\"?:\s*\"[A-Za-z0-9])")
SPLIT_DIRS = {"public": "public", "synthetic_variants": "variants", "adversarial": "adversarial"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(base: Path) -> list[str]:
    problems: list[str] = []
    bad = problems.append
    manifest_path = base / "run_manifest.json"
    if not manifest_path.exists():
        return ["run_manifest.json missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ok":
        bad(f"run status {manifest.get('status')}")
    if (ROOT / "candidate_assets" / "public_cases.json").exists() and \
            sha(ROOT / "candidate_assets" / "public_cases.json") != PUBLIC_SHA256:
        bad("candidate_assets/public_cases.json changed")
    seen_outputs: set[str] = set()
    seen_runs: set[tuple] = set()
    for run in manifest.get("runs", []):
        key = (run["split"], run.get("run"))
        if key in seen_runs:
            bad(f"duplicated run {key}")
        seen_runs.add(key)
        inp, out = ROOT / run["input"], ROOT / run["output"]
        if not inp.exists() or sha(inp) != run["input_sha256"]:
            bad(f"{run['input']}: input hash differs from the run")
        if not out.exists():
            bad(f"{run['output']}: missing")
            continue
        if sha(out) != run["output_sha256"]:
            bad(f"{run['output']}: output altered after the run")
        if run["output_sha256"] in seen_outputs:
            bad(f"{run['output']}: identical to another run (duplicated)")
        seen_outputs.add(run["output_sha256"])
        document = json.loads(out.read_text(encoding="utf-8"))
        if document.get("input_sha256") != run["input_sha256"]:
            bad(f"{run['output']}: input_sha256 field does not match the input")
        cases = json.loads(inp.read_text(encoding="utf-8"))["cases"]
        if [r.get("case_id") for r in document.get("results", [])] != [c["case_id"] for c in cases]:
            bad(f"{run['output']}: case list/order differs from the input")
    rows = [json.loads(x) for x in (base / "cases.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] \
        if (base / "cases.jsonl").exists() else []
    if not rows:
        bad("cases.jsonl missing or empty (run evals.scoring.report)")
    expected_rows = sum(r["cases"] for r in manifest.get("runs", []))
    if rows and len(rows) != expected_rows:
        bad(f"cases.jsonl has {len(rows)} rows for {expected_rows} case runs")
    spans: dict[str, set[str]] = {}
    for path in (base / "diagnostics").glob("spans_*.jsonl"):
        spans[path.name] = {json.loads(x)["trace_id"] for x in path.read_text(encoding="utf-8").splitlines() if x}
    all_traces = set().union(*spans.values()) if spans else set()
    labels = {s: load_labels(s) for s in SPLIT_DIRS}
    for row in rows:
        for ref in row.get("trace_refs", []):
            if ref != "trace_unavailable" and ref not in all_traces:
                bad(f"{row['ref']}: trace {ref[:8]}… claimed but no captured spans")
        run = next((r for r in manifest["runs"] if r["split"] == row["split"] and r.get("run") == row.get("run")), None)
        if run is None or row["case_id"] not in labels[row["split"]]:
            bad(f"{row['ref']}: no matching run or label")
            continue
        case = next(c for c in json.loads((ROOT / run["input"]).read_text(encoding="utf-8"))["cases"]
                    if c["case_id"] == row["case_id"])
        result = next((r for r in json.loads((ROOT / run["output"]).read_text(encoding="utf-8"))["results"]
                       if r["case_id"] == row["case_id"]), None)
        again = business.score_case(case, result, labels[row["split"]][row["case_id"]], row["ref"])
        if again["status"] != row["status"]:
            bad(f"{row['ref']}: recorded {row['status']} but rescoring gives {again['status']}")
    card_path = base / "report" / "scorecard.json"
    if card_path.exists():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        for split, stats in card.get("business", {}).items():
            mine = [r for r in rows if r["split"] == split]
            if stats["complete_task_success"] != f"{sum(r['status'] == 'pass' for r in mine)}/{len(mine)}":
                bad(f"scorecard business.{split} not backed by cases.jsonl")
    else:
        bad("report/scorecard.json missing")
    evidence = base / "report" / "evidence_manifest.json"
    if evidence.exists():
        for name, digest in json.loads(evidence.read_text(encoding="utf-8"))["files"].items():
            path = ROOT / name
            if not path.exists() or sha(path) != digest:
                bad(f"evidence manifest mismatch: {name}")
    else:
        bad("report/evidence_manifest.json missing")
    for path in base.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".md", ".txt", ".xml"):
            if SECRET.search(path.read_text(encoding="utf-8", errors="replace")):
                bad(f"{path.relative_to(base).as_posix()}: home path or credential-like text")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", nargs="?", default="evals/evidence/evaluation")
    args = parser.parse_args(argv)
    base = Path(args.directory).resolve()
    if not base.exists():
        print(f"{args.directory} does not exist", file=sys.stderr)
        return 2
    problems = check(base)
    for p in problems:
        print(f"FAIL {p}")
    print("VERIFY PASSED" if not problems else f"VERIFY FAILED: {len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
