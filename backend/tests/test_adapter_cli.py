"""adapter.py CLI safety (no model or database needed) and no public-case shortcuts in runtime code."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CODE = [ROOT / "adapter.py", *[p for d in ("agents", "guardrails", "services", "domain", "db",
                                                     "evaluation", "fixtures")
                                       for p in (ROOT / "backend" / "app" / d).rglob("*.py")],
                *(ROOT / "backend" / "mcp_server").rglob("*.py"), *(ROOT / "backend" / "retail_domain").rglob("*.py")]


def adapter(*args, cwd=ROOT):
    return subprocess.run([sys.executable, str(ROOT / "adapter.py"), *args], cwd=cwd, capture_output=True,
                          text=True, timeout=120)


def test_help():
    proc = adapter("--help")
    assert proc.returncode == 0 and "--input" in proc.stdout


@pytest.mark.parametrize("output", ["candidate_assets/results.json", "candidate_assets/public_cases.json",
                                    "candidate_assets"])
def test_refuses_to_write_supplied_assets(output):
    proc = adapter("--input", "candidate_assets/public_cases.json", "--output", output)
    assert proc.returncode == 2 and "refusing" in proc.stderr or "must not be the input" in proc.stderr


def test_missing_and_invalid_input(tmp_path):
    assert adapter("--input", str(tmp_path / "nope.json"), "--output", str(tmp_path / "o.json")).returncode == 2
    bad = tmp_path / "bad.json"
    bad.write_text('{"contract_version": 2, "cases": []}')
    proc = adapter("--input", str(bad), "--output", str(tmp_path / "o.json"))
    assert proc.returncode == 2 and "invalid input" in proc.stderr
    assert not (tmp_path / "o.json").exists()


def test_missing_required_arguments():
    assert adapter().returncode == 2


def test_no_public_case_shortcuts_in_runtime_code():
    forbidden = re.compile(r"public_0\d|\bC100\b|\bO100\b|TEE-[SML]\b|REQ-public|Refund line L1|"
                           r"JACKET-|2026-09-2[56]")
    assert len(RUNTIME_CODE) > 30  # never pass by scanning nothing
    hits = [f"{p.relative_to(ROOT)}:{n}" for p in RUNTIME_CODE
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if forbidden.search(line)]
    assert hits == []


def test_relaunch_uses_the_project_environment(monkeypatch):
    """`python adapter.py` from any Python: the project's .venv, else uv, else a clear failure."""
    import adapter
    command = adapter.relaunch_command("adapter.py", ["--input", "a", "--output", "b"])
    assert command[-5:] == ["adapter.py", "--input", "a", "--output", "b"]
    monkeypatch.setattr(adapter, "ROOT", adapter.ROOT / "no-such-dir")
    monkeypatch.setattr(adapter.shutil, "which", lambda name: None)
    assert adapter.relaunch_command("adapter.py", []) is None


def test_outside_the_environment_it_relaunches_once(monkeypatch):
    import adapter
    calls = []
    monkeypatch.setattr(adapter, "in_project_environment", lambda: False)
    monkeypatch.setattr(adapter.subprocess, "run", lambda cmd, env: calls.append(env) or type("R", (), {"returncode": 7})())
    monkeypatch.delenv("RETAIL_ADAPTER_RELAUNCHED", raising=False)
    assert adapter.main(["--input", "a", "--output", "b"]) == 7
    assert calls[0]["RETAIL_ADAPTER_RELAUNCHED"] == "1"
    monkeypatch.setenv("RETAIL_ADAPTER_RELAUNCHED", "1")  # no loop if the environment is really broken
    assert adapter.main(["--input", "a", "--output", "b"]) == adapter.EXIT_INFRA
