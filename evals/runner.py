"""Run the evaluation splits through the REAL runtime (app.evaluation.runner) and keep raw evidence.

    uv run python -m evals.runner --suite all --public-runs 3 --telemetry local \
        --output-dir evals/evidence/evaluation

No evaluation logic runs here: each split goes through prepare() + run_fixture()
(the same engine as adapter.py and app.evaluation.repetitions), then the HP
validator. Scoring is a separate pass (evals.scoring.report) over these files.

Telemetry modes: `local` (default) captures spans in memory, with no stack needed,
and writes allowlisted span names, attributes and events to diagnostics/; `stack`
exports to this project's collector (127.0.0.1:14318) and fetches traces from
Tempo (127.0.0.1:13200); `off` records no spans (handoff then comes from audit order).

Exit codes: 0 all splits ran, 2 usage, 3 infrastructure failure (diagnostics written).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "candidate_assets" / "public_cases.json"
SPLITS = {"public": PUBLIC, "synthetic_variants": "variants.json", "adversarial": "adversarial.json"}
SPLIT_DIRS = {"public": "public", "synthetic_variants": "variants", "adversarial": "adversarial"}
SPAN_ATTR_PREFIXES = ("retail.", "gen_ai.", "error.type")
SPAN_ATTR_DROP = ("retail.customer.", "retail.proposal_ref")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hp_validator():
    spec = importlib.util.spec_from_file_location("hp_validator", ROOT / "candidate_assets" / "validate_submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(input_path: Path, output_path: Path) -> dict[str, Any]:
    """The HP structural check, called exactly as its CLI does (platform default text encoding)."""
    try:
        return {"ok": True, "cases": hp_validator().check(input_path, output_path), "error": None}
    except (ValueError, KeyError, TypeError, OSError, UnicodeDecodeError) as error:
        return {"ok": False, "cases": None, "error": f"{type(error).__name__}: {str(error)[:200]}"}


def safe_span(span: Any) -> dict[str, Any]:
    keep = lambda attrs: {k: v for k, v in dict(attrs or {}).items()  # noqa: E731
                          if k.startswith(SPAN_ATTR_PREFIXES) and not k.startswith(SPAN_ATTR_DROP)}
    return {"name": span.name, "attributes": keep(span.attributes),
            "events": [{"name": e.name, "attributes": keep(e.attributes) if e.name != "agent.handoff"
                        else dict(e.attributes)} for e in span.events],
            "error": span.status.status_code.name == "ERROR"}


class SpanCapture:
    def __init__(self, mode: str) -> None:
        self.mode, self.exporter = mode, None

    def start(self) -> bool:
        from app.observability import telemetry as tel
        from app.observability.config import TelemetrySettings
        if self.mode == "off":
            return False
        if self.mode == "local":
            from opentelemetry.sdk.metrics.export import InMemoryMetricReader
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
            self.exporter = InMemorySpanExporter()
            return tel.init(TelemetrySettings(enabled=True), span_exporter=self.exporter,
                            metric_reader=InMemoryMetricReader())
        os.environ["RETAIL_OTEL_ENABLED"] = "true"
        return tel.init()

    def collect(self, trace_ids: list[str]) -> dict[str, dict[str, Any]]:
        """{trace_id: {"spans": [...]}} for the given turns; missing ones are simply absent."""
        if self.mode == "local" and self.exporter is not None:
            grouped: dict[str, list] = {}
            for span in self.exporter.get_finished_spans():
                grouped.setdefault(format(span.context.trace_id, "032x"), []).append(safe_span(span))
            self.exporter.clear()
            return {t: {"spans": grouped[t]} for t in trace_ids if t in grouped}
        if self.mode == "stack":
            from app.observability import telemetry as tel
            from app.tools.verify_observability import wait_for_traces
            for provider in ("tracer_provider",):
                state = getattr(tel, "_STATE", None)
                if state is not None:
                    getattr(state, provider).force_flush(timeout_millis=5000)
            found = wait_for_traces([t for t in trace_ids if t], deadline_s=60)
            return {t: {"spans": [{"name": s["name"], "attributes": {k: v for k, v in s["attributes"].items()
                                                                      if k.startswith(SPAN_ATTR_PREFIXES)
                                                                      and not k.startswith(SPAN_ATTR_DROP)},
                                   "events": [{"name": "agent.handoff", "attributes": e} for e in s["events"]
                                              if {"from", "to"} <= set(e)], "error": s["error"]}
                                  for s in spans]} for t, spans in found.items()}
        return {}


def run_split(ctx: Any, split: str, input_path: Path, out_dir: Path, run: int | None,
              capture: SpanCapture) -> dict[str, Any]:
    from app.evaluation.runner import run_fixture
    from app.evaluation.serialization import write_atomic
    from app.fixtures.loader import load_fixtures

    root, input_sha = load_fixtures(input_path)
    started = time.perf_counter()
    document, raws = run_fixture(ctx, root, input_sha)
    name = f"run_{run}.json" if run else "results.json"
    path = out_dir / name
    write_atomic(path, document)
    trace_ids = [t["internal_status"].get("trace_id") for r in raws for t in r["turns"]]
    spans = capture.collect([t for t in trace_ids if t])
    stem = name.removesuffix(".json")
    raw_rows = [{"case_id": r["case_id"], "decisions": r["decisions"],
                 "turns": [{"turn": t["turn"], "decision": t["decision"],
                            "trace_id": t["internal_status"].get("trace_id"),
                            "spans_captured": t["internal_status"].get("trace_id") in spans}
                           for t in r["turns"]],
                 "cleanup_errors": r.get("cleanup_errors", [])} for r in raws]
    (out_dir / f"{stem}.raw.jsonl").write_text("".join(json.dumps(x) + "\n" for x in raw_rows), encoding="utf-8")
    diag = out_dir.parent / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    (diag / f"spans_{SPLIT_DIRS[split]}_{stem}.jsonl").write_text(
        "".join(json.dumps({"trace_id": t, **v}) + "\n" for t, v in spans.items()), encoding="utf-8")
    return {"split": split, "run": run, "input": rel(input_path), "input_sha256": input_sha,
            "output": rel(path), "output_sha256": sha256_file(path), "cases": len(document["results"]),
            "turns": len(trace_ids), "traces_captured": len(spans), "validator": validate(input_path, path),
            "wall_s": round(time.perf_counter() - started, 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=["all", "public", "synthetic_variants", "adversarial"], default="all")
    parser.add_argument("--public-runs", type=int, default=3, choices=range(1, 11), metavar="N (1-10)")
    parser.add_argument("--datasets-dir", default=str(ROOT / "evals" / "datasets"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--telemetry", choices=["local", "stack", "off"], default="local")
    parser.add_argument("--only", help="comma-separated case IDs (nondeterminism reruns; writes a filtered input)")
    args = parser.parse_args(argv)
    out = Path(args.output_dir).resolve()
    if (ROOT / "candidate_assets") in [out, *out.parents] or any(
            (ROOT / "evals" / "evidence" / p) in [out, *out.parents] for p in ("final", "experiments")):
        print("refusing to write into candidate_assets/ or earlier evidence directories", file=sys.stderr)
        return 2
    datasets = Path(args.datasets_dir)
    splits = list(SPLITS) if args.suite == "all" else [args.suite]
    inputs = {s: (SPLITS[s] if s == "public" else datasets / SPLITS[s]) for s in splits}
    missing = [str(p) for p in inputs.values() if not Path(p).exists()]
    if missing:
        print(f"missing inputs: {missing} (run evals.generators.variants first)", file=sys.stderr)
        return 2
    if args.only:
        wanted = set(args.only.split(","))
        for split, path in list(inputs.items()):
            document = json.loads(Path(path).read_text(encoding="utf-8"))
            document["cases"] = [c for c in document["cases"] if c["case_id"] in wanted]
            filtered = out / "inputs" / f"{SPLIT_DIRS[split]}.json"
            filtered.parent.mkdir(parents=True, exist_ok=True)
            filtered.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            inputs[split] = filtered

    from evals.benchmarks.identity import config_identity, source_identity
    from app.evaluation.runner import InfrastructureError, prepare

    capture = SpanCapture(args.telemetry)
    telemetry_active = capture.start()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    runs: list[dict[str, Any]] = []
    status, error = "ok", None
    try:
        ctx = prepare()
        for split in splits:
            target = out / SPLIT_DIRS[split]
            target.mkdir(parents=True, exist_ok=True)
            for run in (range(1, args.public_runs + 1) if split == "public" else [None]):
                runs.append(run_split(ctx, split, Path(inputs[split]), target, run, capture))
                print(f"{split}{f' run {run}' if run else ''}: {runs[-1]['cases']} cases, "
                      f"validator {'ok' if runs[-1]['validator']['ok'] else runs[-1]['validator']['error']}")
    except InfrastructureError as exc:
        status, error = "infrastructure_failure", str(exc)[:300]
    from app.observability import telemetry as tel
    tel.shutdown()
    manifest = {
        "schema_version": 1, "status": status, "error": error, "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source_identity(), "config": config_identity(ctx.settings) if status == "ok" else None,
        "model": ctx.model_info if status == "ok" else None, "cold_start_ms": ctx.cold_start_ms if status == "ok" else None,
        "telemetry": {"mode": args.telemetry, "active": telemetry_active, "content_capture": False},
        "datasets": {s: {"path": rel(Path(p)), "sha256": sha256_file(Path(p))} for s, p in inputs.items()},
        "runs": runs,
    }
    from app.evaluation.serialization import write_atomic
    write_atomic(out / "run_manifest.json", manifest)
    if status != "ok":
        (out / "diagnostics").mkdir(parents=True, exist_ok=True)
        write_atomic(out / "diagnostics" / "infrastructure_error.json", {"error": error})
        print(f"infrastructure failure: {error}", file=sys.stderr)
        return 3
    print(f"wrote {rel(out / 'run_manifest.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
