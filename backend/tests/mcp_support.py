"""Run a real standalone MCP server process for tests, on a free port, and stop it afterwards."""
from __future__ import annotations

import contextlib
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

BACKEND = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _kill_tree(process: subprocess.Popen) -> None:
    import psutil
    try:
        parent = psutil.Process(process.pid)
        family = [*parent.children(recursive=True), parent]
    except psutil.Error:
        return
    for member in family:
        with contextlib.suppress(psutil.Error):
            member.kill()
    psutil.wait_procs(family, timeout=5)


@contextlib.contextmanager
def running_server(test_db_url: str, *, module: str = "mcp_server", extra_env: dict[str, str] | None = None,
                   wait: bool = True) -> Iterator[tuple[str, str, subprocess.Popen]]:
    """Yields (url, token, process). `module` lets the failure tests run a misbehaving server instead."""
    port, token = free_port(), secrets.token_urlsafe(16)
    env = {**os.environ, "RETAIL_SKIP_DOTENV": "1", "MCP_AUTH_TOKEN": token, "MCP_SERVER_PORT": str(port),
           "MCP_DATABASE_URL": test_db_url, "MCP_TEST_DATABASE_URL": test_db_url, **(extra_env or {})}
    process = subprocess.Popen([sys.executable, "-m", module], cwd=BACKEND, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 60
        while wait:
            if process.poll() is not None:
                raise RuntimeError(f"{module} exited with {process.returncode}")
            with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f"{module} did not start")
            time.sleep(0.05)
        yield f"http://127.0.0.1:{port}/mcp", token, process
    finally:
        _kill_tree(process)
