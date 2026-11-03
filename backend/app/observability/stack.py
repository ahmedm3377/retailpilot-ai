"""Bring up this project's own local telemetry stack when the API starts with telemetry on.

Only observability/compose.yaml (Compose project hp-retail-observability) is ever started,
with `up -d`: never `down`, never another project's containers. The images are pinned and
local, so this works offline. Telemetry must never stop the app: every failure here is
reported and the server continues.
"""
from __future__ import annotations

import shutil
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = ROOT / "observability" / "compose.yaml"
GRAFANA_URL = "http://127.0.0.1:13000"


def collector_reachable(endpoint: str, timeout_s: float = 1.0) -> bool:
    parsed = urlparse(endpoint)
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=timeout_s):
            return True
    except OSError:
        return False


def ensure_stack(endpoint: str, *, start: bool = True, wait_s: float = 60.0) -> tuple[bool, str]:
    """Returns (collector reachable, one-line status for the console)."""
    if collector_reachable(endpoint):
        return True, f"collector reachable at {endpoint}; Grafana {GRAFANA_URL}"
    manual = f"docker compose -f {COMPOSE_FILE.relative_to(ROOT).as_posix()} up -d"
    if not start:
        return False, f"collector not reachable at {endpoint}; start it with: {manual}"
    if shutil.which("docker") is None:
        return False, f"collector not reachable and docker is not on PATH; start it with: {manual}"
    try:
        done = subprocess.run(["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d"], cwd=ROOT,
                              capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"could not start the stack ({type(exc).__name__}); start it with: {manual}"
    if done.returncode != 0:
        return False, f"`{manual}` failed (exit {done.returncode}); is Docker running?"
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if collector_reachable(endpoint):
            return True, f"started the local stack; collector at {endpoint}; Grafana {GRAFANA_URL}"
        time.sleep(1.0)
    return False, f"started the stack but the collector at {endpoint} did not answer within {wait_s:.0f}s"
