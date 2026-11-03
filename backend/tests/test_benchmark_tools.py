"""Benchmark tooling on SYNTHETIC data (statistics, profiling, identity, gate, verify, compare).

No model or database is used; the artifact trees here are fabricated test fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.benchmarks import compare, identity, machine, profile, stats, verify
from evals.benchmarks.summary import summarize_rows
from tests.conftest import make_state

ROOT = Path(__file__).resolve().parents[2]


# --- statistics ----------------------------------------------------------------------------

@pytest.mark.parametrize("values, q, expected", [
    (list(range(1, 19)), 0.95, 18),     # ceil(17.1) = 18 -> the maximum for n = 18
    (list(range(1, 19)), 0.50, 9),
    (list(range(1, 21)), 0.95, 19),
    (list(range(1, 21)), 0.50, 10),
    ([42.0], 0.95, 42.0),
    ([5, 1, 5, 3, 5], 0.5, 5),          # ties and unsorted input
    ([3, 1, 2], 1.0, 3),
])
def test_nearest_rank(values, q, expected):
    assert stats.nearest_rank(values, q) == expected


def test_summarize_and_formulas():
    s = stats.summarize([100.0, 200.0, None, float("nan"), 300.0])
    assert (s["n"], s["p50"], s["p95"], s["min"], s["max"], s["mean"]) == (3, 200.0, 300.0, 100.0, 300.0, 200.0)
    assert stats.summarize([])["p50"] is None
    assert stats.improvement_pct(10000, 7500) == 25.0
    assert stats.improvement_pct(10000, 12000) == -20.0
    assert stats.increase_pct(30, 36) == 20.0
    for fn in (stats.improvement_pct, stats.increase_pct):
        assert fn(None, 1) is None and fn(1, None) is None and fn(0, 1) is None


# --- profiling --------------------------------------------------------------------------------

def synthetic_result(case_id="CASE-A", decisions=("deny",), turn_ms=(1000.0,), audit=None, state=None):
    state = state or make_state().model_dump(mode="json")
    return {"case_id": case_id,
            "turns": [{"decision": d, "response": "x", "proposal": None} for d in decisions],
            "action_audit": audit if audit is not None else [
                {"event_id": 1, "turn": 1, "actor": "customer_resolution_agent", "tool": "llm_call",
                 "outcome": "ok", "result": {"latency_ms": 400.0}},
                {"event_id": 2, "turn": 1, "actor": "retail_operations_agent", "tool": "llm_call",
                 "outcome": "ok", "result": {"latency_ms": 300.0}},
                {"event_id": 3, "turn": 1, "actor": "guardrails", "tool": "guard:input:DetectJailbreak",
                 "outcome": "passed", "result": {"elapsed_ms": 150.0}}],
            "final_state": state,
            "metrics": {"elapsed_ms": sum(turn_ms), "turn_ms": list(turn_ms), "llm_calls": 2,
                        "prompt_tokens": 800, "completion_tokens": 70, "peak_ram_mb": 2000.0,
                        "peak_python_rss_mb": 1500.0, "peak_model_server_rss_mb": 500.0, "peak_vram_mb": 315.5,
                        "peak_gpu_device_used_mb": 3000.0, "peak_vram_note": None}}


def test_run_row_and_failure_categories():
    row = profile.run_row(variant="v", repetition=1, source_id="s", config_id="c", input_sha256="h",
                          result=synthetic_result(), raw={"turns": []}, oracle_problems=[], expected=["deny"],
                          run_file="run_1.json")
    assert row["completed_task_success"] and row["failure_category"] is None
    assert row["effective_output_tokens_per_call_second"] == 100.0  # 70 tokens / 0.7 s
    wrong = profile.run_row(variant="v", repetition=1, source_id="s", config_id="c", input_sha256="h",
                            result=synthetic_result(), raw={"turns": []}, oracle_problems=[],
                            expected=["propose", "completed"], run_file="run_1.json")
    assert wrong["failure_category"] == "wrong_decision" and not wrong["completed_task_success"]
    blocked_audit = [{"event_id": 1, "turn": 1, "actor": "guardrails", "tool": "guard:input:TaskInjection",
                      "outcome": "blocked", "result": {"elapsed_ms": 2.0}}]
    blocked = profile.run_row(variant="v", repetition=1, source_id="s", config_id="c", input_sha256="h",
                              result=synthetic_result(decisions=("error",), audit=blocked_audit),
                              raw={"turns": []}, oracle_problems=["x"], expected=["deny"], run_file="r")
    assert blocked["failure_category"] == "guardrail_block"


# --- identity and machine gate -------------------------------------------------------------------


def test_quiet_gate():
    snap = {"model_server_rss_mb": 0, "ram_free_mb": 8000, "gpu_util_pct": 0}
    assert machine.quiet_problems(snap) == []
    resident = {"model_server_rss_mb": 2800, "ram_free_mb": 3500, "gpu_util_pct": 0}
    assert machine.quiet_problems(resident) == []                  # model memory is workload, not contention
    assert machine.quiet_problems({**resident, "model_server_rss_mb": 0})  # same free RAM, model not loaded yet
    during = {**resident, "ram_free_mb": 1700, "own_rss_mb": 2300}  # the benchmark's own memory is not contention
    assert machine.quiet_problems(during) == []
    busy = {"model_server_rss_mb": 0, "ram_free_mb": 2000, "gpu_util_pct": 55, "on_ac_power": False}
    assert len(machine.quiet_problems(busy)) == 3
    probe = lambda r: {"median_decode_tokens_per_s": r}  # noqa: E731
    assert machine.probe_problems(probe(64.5), probe(60.0)) == []
    assert machine.probe_problems(probe(64.5), probe(19.2))   # the battery slowdown seen in E1
    assert machine.probe_problems(None, probe(60.0)) == ["decode probe unavailable"]


# --- verify and compare on fabricated trees -------------------------------------------------------------

def fabricate(tmp_path: Path, name: str, *, reps=3, started="2026-10-30T10:00:00+00:00", changed=None,
              elapsed=1000.0, contended=False):
    """One-case input + a variant directory whose results satisfy the oracle (a denial with no effects)."""
    state = make_state(age_days=99).model_dump(mode="json")
    case = {"case_id": "CASE-A", "customer_id": "CUST-9", "request_id": "R-1", "initial_state": state,
            "turns": [{"message": "refund please"}], "faults": []}
    input_path = tmp_path / "cases.json"
    input_path.write_text(json.dumps({"contract_version": 1, "cases": [case]}))
    import hashlib
    sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
    out = tmp_path / "evidence" / name
    out.mkdir(parents=True)
    rows = []
    for rep in range(1, reps + 1):
        result = synthetic_result(turn_ms=(elapsed,), state=state)
        doc = {"contract_version": 1, "input_sha256": sha, "hardware": {"cpu": "x"}, "configuration": {"m": "y"},
               "results": [result]}
        (out / f"run_{rep}.json").write_text(json.dumps(doc))
        rows.append(profile.run_row(variant=name, repetition=rep, source_id="abc", config_id="cfg",
                                    input_sha256=sha, result=result, raw={"turns": []}, oracle_problems=[],
                                    expected=None, run_file=f"run_{rep}.json"))
    (out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {"variant": name, "source_id": "abc", "config_id": "cfg", "input_sha256": sha, "repetitions": reps,
               "cases": 1, "cold_start_ms": 30000.0, "validator": {}, **summarize_rows(rows)}
    (out / "summary.json").write_text(json.dumps(summary))
    settings = {"model": "ollama_chat/gemma4:e2b", "temperature": 0.0, "top_p": 1.0, "seed": 7, "think": False,
                "guardrails_enabled": True, "jailbreak_threshold": 0.8, "toxicity_threshold": 0.5,
                "fail_closed": True, "max_input_chars": 2000}
    env = {"variant": name, "started_at": started, "input_sha256": sha, "contended": contended,
           "source": {"source_id": "abc", "changed_files": changed or []},
           "config": {"settings": settings}, "model": {"digest": "b37049369adf"},
           "hardware": {"cpu": "cpu", "gpu": "gpu", "ram": "16 GiB", "os": "os", "power": {"p": 1}},
           "load_before": {}, "load_after": {},
           "cold_start": {"cold_start_ms": 30000.0, "notes": ["model was already loaded in Ollama before the run"]}}
    (out / "environment.json").write_text(json.dumps(env))
    return input_path, out


def test_verify_accepts_a_good_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    input_path, _ = fabricate(tmp_path, "baseline", changed=["evals/benchmarks/run.py"])
    problems, report = verify.verify(tmp_path / "evidence", input_path, check_db=False)
    assert problems == [] and report["variants"][0]["success"] == "3/3"


def test_verify_catches_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    input_path, out = fabricate(tmp_path, "baseline", changed=["app/agents/graph.py"], reps=2)
    rows = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]
    rows[0]["oracle_pass"] = False
    (out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "notes.md").write_text(r"trace from C:\Users\someone\project")
    problems, _ = verify.verify(tmp_path / "evidence", input_path, check_db=False)
    text = "\n".join(problems)
    assert "only 2 repetitions" in text
    assert "recorded oracle_pass=False" in text
    assert "non-tooling changes" in text
    assert "forbidden content" in text


def test_compare_flags_incomparable_and_quality(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "ROOT", tmp_path)
    _, base = fabricate(tmp_path, "baseline", elapsed=1000.0)
    _, opt = fabricate(tmp_path, "optimized", elapsed=800.0)
    result = compare.build(compare.load_variant(base), compare.load_variant(opt), [])
    assert result["comparability"]["comparable"] and result["claims_allowed"]
    assert result["latency_ms"]["p50"]["improvement_pct"] == 20.0
    env = json.loads((opt / "environment.json").read_text())
    env["model"]["digest"] = "different"
    env["contended"] = True
    (opt / "environment.json").write_text(json.dumps(env))
    result = compare.build(compare.load_variant(base), compare.load_variant(opt), [])
    assert not result["claims_allowed"]
    assert set(result["comparability"]["reasons"]) == {"same_model_digest", "neither_contended"}
    assert result["cold_start_notes"]["comparable"]
    md = compare.render_md(result)
    assert "Comparable:** False" in md and "| p50 | 1,000.0 | 800.0 | 20.0% |" in md
    m = compare.manifest(tmp_path / "evidence")
    assert all(not f["path"].startswith(("C:", "/")) for f in m["files"])


def test_every_run_declares_seeds_concurrency_caches_and_timeouts(agent_settings):
    from evals.benchmarks.run import declarations
    declared = declarations(agent_settings, True)
    assert set(declared) == {"seeds", "concurrency", "caches", "timeouts"}
    assert declared["seeds"]["llm_seed"] == agent_settings.seed
    assert declared["timeouts"]["llm_call_s"] == agent_settings.timeout_s
    assert declared["concurrency"]["cases"] == 1 and "evicted" in declared["caches"]["model_load"]
    assert "not controlled" in declarations(agent_settings, False)["caches"]["model_load"]


def test_summary_counts_failed_tasks_and_model_call_failures():
    from evals.benchmarks.summary import summarize_rows
    row = {"completed_task_success": True, "case_id": "A", "elapsed_ms": 10.0, "completion_tokens": 1,
           "prompt_tokens": 1, "failure_category": None, "oracle_pass": True, "error_turns": 0, "injected_timeouts": 0,
           "recovered_timeouts": 0, "llm_ms": 1.0, "guard_ms": 1.0, "other_ms": 1.0, "llm_calls": 1,
           "llm_calls_failed": 0, "effective_output_tokens_per_call_second": 1.0, "peak_ram_mb": None,
           "peak_python_rss_mb": None, "peak_model_server_rss_mb": None, "peak_vram_mb": None,
           "peak_gpu_device_used_mb": None}
    failed = {**row, "completed_task_success": False, "failure_category": "error_turn", "llm_calls_failed": 2}
    summary = summarize_rows([row, failed])
    assert summary["failed_tasks"] == 1 and summary["llm_call_failures"] == 2
    assert summary["failures_by_category"] == {"error_turn": 1}


def test_verify_allows_uncommitted_code_when_baseline_and_final_share_the_source(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    fabricate(tmp_path, "baseline", changed=["app/agents/graph.py"])
    problems, _ = verify.verify(tmp_path / "evidence", tmp_path / "cases.json", check_db=False)
    assert any("non-tooling" in p for p in problems)  # a baseline alone must still be clean
    input_path, _ = fabricate(tmp_path, "final", changed=["app/agents/graph.py"], started="2026-10-30T11:00:00+00:00")
    problems, _ = verify.verify(tmp_path / "evidence", input_path, check_db=False)
    assert not any("non-tooling" in p for p in problems)
