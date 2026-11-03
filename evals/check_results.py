"""Check the business correctness of one adapter results file (what the HP validator doesn't score).

    uv run python -m evals.check_results --input candidate_assets/public_cases.json --output results.json

Every case goes through the independent oracle (evals/oracle.py). Public cases also have their turn
decisions compared with evals/benchmarks/public_expectations.json. Exit 0 if all pass, 1 if not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evals.oracle import check_case

EXPECTED = Path(__file__).with_name("benchmarks") / "public_expectations.json"


def check(cases: list[dict], results: list[dict], expected: dict) -> dict[str, list[str]]:
    """case_id -> problems (empty when the case is correct)."""
    by_id = {r.get("case_id"): r for r in results}
    report = {}
    for case in cases:
        result = by_id.get(case["case_id"])
        if result is None:
            report[case["case_id"]] = ["no result for this case"]
            continue
        problems = list(check_case(case, result))
        decisions = [t.get("decision") for t in result.get("turns", [])]
        wanted = expected.get(case["case_id"])
        if wanted is not None and decisions != wanted:
            problems.append(f"decisions {decisions}, expected {wanted}")
        report[case["case_id"]] = problems
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="the cases JSON the adapter ran")
    parser.add_argument("--output", required=True, help="the adapter's results JSON")
    args = parser.parse_args(argv)
    cases = json.loads(Path(args.input).read_text(encoding="utf-8"))["cases"]
    results = json.loads(Path(args.output).read_text(encoding="utf-8"))["results"]
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    by_id = {r.get("case_id"): r for r in results}
    report = check(cases, results, expected)
    for case_id, problems in report.items():
        decisions = " -> ".join(t.get("decision", "?") for t in by_id.get(case_id, {}).get("turns", []))
        print(f"{'PASS' if not problems else 'FAIL'}  {case_id}: {decisions}")
        for problem in problems:
            print(f"      {problem}")
    failed = sum(1 for problems in report.values() if problems)
    known = sum(1 for case_id in report if case_id in expected)
    print(f"{len(report) - failed}/{len(report)} cases correct. Decisions compared with known answers for {known} of them.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
