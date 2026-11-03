"""End-to-end check of the local observability stack against real cases (real model server, real PostgreSQL).

Runs public cases with telemetry ON, then checks that:
- every turn produced a trace in Tempo with the expected spans, and no business IDs or raw text in any attribute;
- Prometheus counts for THIS process (instance = service.instance.id) equal what the run itself observed;
- the provisioned Grafana dashboard exists and every Prometheus panel query returns data.

Writes a sanitized summary to evals/evidence/observability/verification.json (no customer/order/request/proposal IDs,
no message text, no absolute paths). Only talks to 127.0.0.1 ports of this project's stack.

Usage: uv run python -m app.tools.verify_observability [--cases public_01_refund_boundary,...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("RETAIL_OTEL_ENABLED", "true")

from app.evaluation.runner import prepare, run_case  # noqa: E402
from app.evaluation.serialization import write_atomic  # noqa: E402
from app.fixtures.loader import load_fixtures  # noqa: E402
from app.observability import telemetry as tel  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
PROMETHEUS, TEMPO, GRAFANA = "http://127.0.0.1:19090", "http://127.0.0.1:13200", "http://127.0.0.1:13000"
DASHBOARD = ROOT / "observability" / "grafana" / "dashboards" / "retail-ai-overview.json"
OUT = ROOT / "evals" / "evidence" / "observability" / "verification.json"
DEFAULT_CASES = ["public_01_refund_boundary", "public_02_exchange", "public_05_stale_stock",
                 "public_06_timeout_replay"]
CHAT_SPANS = {"retail.customer_turn", "retail.context.load", "retail.guardrail.input",
              "invoke_agent customer_resolution", "retail.response", "retail.guardrail.output"}


def get_json(url: str, timeout: float = 10) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed loopback URLs
        return json.loads(response.read())


def health() -> dict[str, bool]:
    checks = {}
    for name, url in (("prometheus", f"{PROMETHEUS}/-/ready"), ("tempo", f"{TEMPO}/ready"),
                      ("grafana", f"{GRAFANA}/api/health")):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
                checks[name] = response.status == 200
        except OSError:
            checks[name] = False
    try:
        targets = get_json(f"{PROMETHEUS}/api/v1/targets")["data"]["activeTargets"]
        checks["collector_scrape"] = any(t["health"] == "up" for t in targets)
    except (OSError, KeyError):
        checks["collector_scrape"] = False
    return checks


def tempo_spans(trace_id: str) -> list[dict[str, Any]] | None:
    try:
        body = get_json(f"{TEMPO}/api/traces/{trace_id}")
    except OSError:
        return None
    spans = []
    for batch in body.get("batches") or body.get("resourceSpans") or []:
        for scope in batch.get("scopeSpans") or batch.get("instrumentationLibrarySpans") or []:
            for span in scope.get("spans", []):
                attrs = {a["key"]: next(iter(a["value"].values()), None) for a in span.get("attributes", [])}
                events = [{a["key"]: next(iter(a["value"].values()), None) for a in e.get("attributes", [])}
                          for e in span.get("events", [])]
                spans.append({"name": span["name"], "attributes": attrs, "events": events,
                              "error": (span.get("status") or {}).get("code") in ("STATUS_CODE_ERROR", 2)})
    return spans


def wait_for_traces(trace_ids: list[str], deadline_s: float = 60) -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = {}
    end = time.monotonic() + deadline_s
    while time.monotonic() < end and len(found) < len(trace_ids):
        for trace_id in trace_ids:
            if trace_id not in found:
                spans = tempo_spans(trace_id)
                if spans and any(s["name"] == "retail.customer_turn" for s in spans):
                    found[trace_id] = spans
        time.sleep(2)
    return found


def prom(expr: str) -> float | None:
    url = f"{PROMETHEUS}/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    result = get_json(url)["data"]["result"]
    return sum(float(r["value"][1]) for r in result) if result else None


def prom_series(expr: str) -> list[dict[str, Any]]:
    url = f"{PROMETHEUS}/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    return get_json(url)["data"]["result"]


def wait_for_metrics(expected: dict[str, float], instance: str, deadline_s: float = 60) -> dict[str, float | None]:
    observed: dict[str, float | None] = {}
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        observed = {name: prom(f'sum({name}{{instance="{instance}"}})') for name in expected}
        if all(observed[n] == v for n, v in expected.items()):
            break
        time.sleep(3)
    return observed


def dashboard_check(instance: str) -> dict[str, Any]:
    local = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    try:
        served = get_json(f"{GRAFANA}/api/dashboards/uid/{local['uid']}")["dashboard"]
    except OSError:
        return {"provisioned": False}
    panels = [p for top in served["panels"] for p in [top, *top.get("panels", [])]]
    queries, empty = 0, []
    for p in panels:
        for t in p.get("targets", []):
            if "expr" not in t:
                continue
            queries += 1
            expr = (t["expr"].replace('job=~"$job"', f'instance="{instance}"').replace("$__range", "1h")
                    .replace("$__rate_interval", "1h"))
            if not prom_series(expr):
                empty.append(p["title"])
    return {"provisioned": True, "title": served["title"], "panels": len(panels), "prometheus_queries": queries,
            "panels_without_data": sorted(set(empty))}


def forbidden_values(case) -> set[str]:
    ids = {case.customer_id, case.request_id}
    for order in case.initial_state.orders:
        ids.add(order.order_id)
    return {i for i in ids if i}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--input", default=str(ROOT / "candidate_assets" / "public_cases.json"))
    args = parser.parse_args(argv)

    stack = health()
    if not all(stack.values()):
        print(f"observability stack not healthy: {stack}\n  start it: docker compose -f observability/compose.yaml up -d")
        return 2
    root, _ = load_fixtures(args.input)
    wanted = args.cases.split(",")
    cases = [c for c in root.cases if c.case_id in wanted]
    ctx = prepare()
    if not tel.active():
        print("telemetry did not start (check RETAIL_OTEL_ENDPOINT)")
        return 2
    instance = tel.instance_id()

    runs, forbidden, expected = [], set(), {"retail_turn_duration_seconds_count": 0.0,
                                            "retail_llm_call_duration_seconds_count": 0.0}
    for case in cases:
        result, raw = run_case(ctx, case)
        proposal_ids = {t["proposal"]["proposal_id"] for t in result["turns"] if t.get("proposal")}
        forbidden |= forbidden_values(case) | proposal_ids
        expected["retail_turn_duration_seconds_count"] += len(raw["turns"])
        expected["retail_llm_call_duration_seconds_count"] += raw["metrics"]["llm_calls"]
        runs.append({"case_id": case.case_id, "decisions": raw["decisions"],
                     "trace_ids": [t["internal_status"].get("trace_id") for t in raw["turns"]]})
    # prepare()'s warm-up call has role "warmup", which is not an agent role, so it is not in the metrics.
    tel.shutdown()  # flush spans and the final metric export

    trace_ids = [t for r in runs for t in r["trace_ids"] if t]
    traces = wait_for_traces(trace_ids)
    observed = wait_for_metrics(expected, instance)

    problems: list[str] = []
    if len(trace_ids) != int(expected["retail_turn_duration_seconds_count"]):
        problems.append("a turn returned no trace_id")
    for run in runs:
        run["turns"] = []
        for number, trace_id in enumerate(run.pop("trace_ids"), start=1):
            spans = traces.get(trace_id) or []
            names = {s["name"] for s in spans}
            root_span = next((s for s in spans if s["name"] == "retail.customer_turn"), None)
            kind = root_span["attributes"].get("retail.turn.kind") if root_span else None
            entry = {"turn": number, "in_tempo": bool(spans), "spans": len(spans), "kind": kind,
                     "has_proposal_ref": bool(root_span and root_span["attributes"].get("retail.proposal_ref")),
                     "business_outcome": next((s["attributes"].get("retail.business.outcome") for s in spans
                                               if s["name"] == "retail.transaction.commit"), None),
                     "error_spans": sorted({s["name"] for s in spans if s["error"]})}
            if not spans:
                problems.append(f"{run['case_id']} turn {number}: trace not found in Tempo")
            elif kind == "chat" and not CHAT_SPANS <= names:
                problems.append(f"{run['case_id']} turn {number}: missing spans {sorted(CHAT_SPANS - names)}")
            elif kind == "confirm" and "retail.transaction.commit" not in names:
                problems.append(f"{run['case_id']} turn {number}: confirmation without commit span")
            values = [str(v) for s in spans for v in [*s["attributes"].values(),
                                                      *(x for e in s["events"] for x in e.values())]]
            leaks = sorted({f for f in forbidden for v in values if f in v})
            if leaks:
                problems.append(f"{run['case_id']} turn {number}: {len(leaks)} business ID(s) found in span attributes")
            run["turns"].append(entry)
    for name, value in expected.items():
        if observed.get(name) != value:
            problems.append(f"Prometheus {name} = {observed.get(name)} but the run observed {value}")
    outcomes = {f'{s["metric"].get("remedy")}/{s["metric"].get("outcome")}': float(s["value"][1])
                for s in prom_series(f'sum by (remedy, outcome) (retail_business_outcomes_total{{instance="{instance}"}})')}
    board = dashboard_check(instance)
    if not board.get("provisioned"):
        problems.append("Grafana dashboard not provisioned")
    elif board["panels_without_data"]:
        problems.append(f"dashboard panels without data: {board['panels_without_data']}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "PASS" if not problems else "FAIL",
        "stack": stack,
        "endpoint": "http://127.0.0.1:14318 (OTLP/HTTP, this project's collector)",
        "model": ctx.model_info.get("model"),
        "cases": runs,
        "prometheus": {"expected_from_run": expected, "observed": observed, "business_outcomes": outcomes,
                       "note": "series filtered to this process via instance=service.instance.id"},
        "dashboard": board,
        "privacy": {"business_ids_checked": len(forbidden), "content_capture": os.environ.get(
            "RETAIL_OTEL_CAPTURE_CONTENT", "false")},
        "problems": problems,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(OUT, report)
    print(json.dumps({k: report[k] for k in ("status", "problems")}, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
