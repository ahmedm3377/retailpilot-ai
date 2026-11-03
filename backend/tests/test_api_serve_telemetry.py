"""The API turns telemetry on at startup, starts only its own stack, and never fails because of it."""
from __future__ import annotations


import pytest

from app.api import serve
from app.observability import stack


@pytest.fixture
def calls(monkeypatch):
    seen: list[tuple[str, bool]] = []
    monkeypatch.delenv("RETAIL_OTEL_ENABLED", raising=False)

    def fake(endpoint, *, start=True, wait_s=60.0):
        seen.append((endpoint, start))
        return True, "ok"
    monkeypatch.setattr(stack, "ensure_stack", fake)
    return seen


def test_on_by_default_and_ensures_the_stack(calls, monkeypatch):
    serve.configure_telemetry(True, True)
    assert serve.os.environ["RETAIL_OTEL_ENABLED"] == "true"
    assert calls == [("http://127.0.0.1:14318", True)]


def test_explicit_env_off_wins(calls, monkeypatch):
    monkeypatch.setenv("RETAIL_OTEL_ENABLED", "false")
    serve.configure_telemetry(True, True)
    assert serve.os.environ["RETAIL_OTEL_ENABLED"] == "false" and calls == []


def test_unreachable_stack_keeps_serving_without_telemetry(monkeypatch, capsys):
    monkeypatch.delenv("RETAIL_OTEL_ENABLED", raising=False)
    monkeypatch.setattr(stack, "ensure_stack", lambda endpoint, *, start=True: (False, "not reachable"))
    serve.configure_telemetry(True, False)
    assert serve.os.environ["RETAIL_OTEL_ENABLED"] == "false"
    assert "Telemetry: off - not reachable" in capsys.readouterr().out


def test_starts_only_this_projects_compose_file(monkeypatch):
    state = {"up": False}
    run_args: list[list[str]] = []

    def run(args, **kwargs):
        run_args.append(args)
        state["up"] = True
        return type("Done", (), {"returncode": 0})()
    monkeypatch.setattr(stack, "collector_reachable", lambda endpoint, timeout_s=1.0: state["up"])
    monkeypatch.setattr(stack.shutil, "which", lambda name: "docker")
    monkeypatch.setattr(stack.subprocess, "run", run)
    ok, _ = stack.ensure_stack("http://127.0.0.1:14318")
    assert ok and run_args == [["docker", "compose", "-f", str(stack.COMPOSE_FILE), "up", "-d"]]
    assert stack.COMPOSE_FILE.name == "compose.yaml" and stack.COMPOSE_FILE.parent.name == "observability"


