"""Start the standalone MCP server for one run, when none is running.

Only the adapter uses this: the contract allows `python adapter.py` to launch what it needs, and a checker may
run nothing else. The API, benchmarks and evaluation runner still expect the server to be running already.
A server that is already up is used and left running.
"""
from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import psutil

from app.mcp.client import McpToolError, check_server, server_url

BACKEND = Path(__file__).resolve().parents[2]
START_TIMEOUT_S = 60.0


def _port_open(port: int) -> bool:
    with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), timeout=0.5):
        return True
    return False


def _stop(process: subprocess.Popen) -> None:
    """Stop the process and its children: on Windows a virtual environment's python.exe is a launcher."""
    try:
        parent = psutil.Process(process.pid)
        family = [*parent.children(recursive=True), parent]
    except psutil.Error:
        return
    for member in family:
        with contextlib.suppress(psutil.Error):
            member.terminate()
    _, alive = psutil.wait_procs(family, timeout=5)
    for member in alive:
        with contextlib.suppress(psutil.Error):
            member.kill()


def _wait_until_ready(process: subprocess.Popen, port: int) -> None:
    deadline = time.monotonic() + START_TIMEOUT_S
    while not _port_open(port):
        if process.poll() is not None:
            raise McpToolError("MCP_START_FAILED")
        if time.monotonic() > deadline:
            raise McpToolError("MCP_START_TIMEOUT")
        time.sleep(0.1)
    check_server()


@contextlib.contextmanager
def server_running() -> Iterator[None]:
    """Yields with a usable server. Raises McpToolError if one can't be reached or started."""
    parsed = urlparse(server_url())
    try:
        check_server()
    except McpToolError as error:
        # Something else holds the port (or the token or URL is wrong): starting another can't help.
        if error.args[0] == "MCP_NOT_CONFIGURED" or parsed.port is None or _port_open(parsed.port):
            raise
    else:
        yield
        return
    process = subprocess.Popen([sys.executable, "-m", "mcp_server", "--port", str(parsed.port)], cwd=BACKEND,
                               env=dict(os.environ), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    try:
        _wait_until_ready(process, parsed.port)
        print(f"adapter: started the MCP server on port {parsed.port} for this run", file=sys.stderr, flush=True)
        yield
    finally:
        _stop(process)
