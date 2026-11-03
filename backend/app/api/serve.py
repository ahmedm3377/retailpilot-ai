"""Run the API on loopback only.

    uv run python -m app.api.serve [--port 8080] [--no-telemetry] [--no-start-stack]

The standalone MCP server must already be running (uv run python -m mcp_server): the agents' read-only
tools live there, so the API refuses to start without it.

People sign in with a customer ID and the desk works on the persistent PostgreSQL store, which
is seeded the first time (see app/db/demo_store.py).

Telemetry is ON by default for the API: it exports to this project's own local stack
(observability/compose.yaml, OTLP on 127.0.0.1:14318) and starts that stack with
`docker compose up -d` if the collector is not answering. --no-telemetry (or an explicit
RETAIL_OTEL_ENABLED=false) turns it off; --no-start-stack only reports a missing collector.
Traces also carry each step's input and output (RETAIL_OTEL_CAPTURE_CONTENT=false turns that off).
If the stack cannot be started, the API still runs, without telemetry. The adapter,
benchmarks and tests keep their own default (off).

"""
from __future__ import annotations

import argparse
import os
import sys



def configure_telemetry(enabled: bool, start_stack: bool) -> None:
    """Decide telemetry before any app import: apply_offline_defaults() reads it at import time."""
    if not enabled:
        os.environ["RETAIL_OTEL_ENABLED"] = "false"
    os.environ.setdefault("RETAIL_OTEL_ENABLED", "true")  # an explicit env value wins
    from app.observability.config import TelemetrySettings, telemetry_enabled
    if not telemetry_enabled():
        print("Telemetry: off", flush=True)
        return
    from app.observability.stack import ensure_stack
    endpoint = TelemetrySettings.from_env().endpoint
    ok, status = ensure_stack(endpoint, start=start_stack)
    if not ok:
        os.environ["RETAIL_OTEL_ENABLED"] = "false"  # keep serving; nothing to export to
    if ok:
        os.environ.setdefault("RETAIL_OTEL_CAPTURE_CONTENT", "true")  # an explicit env value wins
    print(f"Telemetry: {'on' if ok else 'off'} - {status}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-telemetry", action="store_true", help="do not export telemetry")
    parser.add_argument("--no-start-stack", action="store_true",
                        help="do not run `docker compose up -d` for the observability stack")
    args = parser.parse_args(argv)
    configure_telemetry(not args.no_telemetry, not args.no_start_stack)
    from app.mcp.client import McpToolError, check_server, unavailable_message
    try:
        check_server()
    except McpToolError as error:
        print(unavailable_message(error), file=sys.stderr, flush=True)
        return 3
    os.environ["RETAIL_API_NO_DEFAULT_APP"] = "1"  # build the app here
    import uvicorn

    from app.api.main import create_app

    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="info", access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
