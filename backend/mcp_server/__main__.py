"""Start the server: python -m mcp_server [--port 8765]. Settings come from the environment or .env."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from mcp_server.server import ConfigError, Config, build_server, serve_http, start_telemetry


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    if not os.environ.get("RETAIL_SKIP_DOTENV"):
        load_dotenv(root / ".env" if (root / ".env").exists() else root / ".env.example", override=False)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_SERVER_PORT", "8765")))
    args = parser.parse_args(argv)
    try:
        config = Config.from_env(os.environ, args.port)
    except ConfigError as error:
        print(f"mcp server not started: {error}", file=sys.stderr)
        return 3
    shutdown = start_telemetry(os.environ)
    print(f"mcp server listening on http://127.0.0.1:{config.port}/mcp", file=sys.stderr, flush=True)
    try:
        serve_http(build_server(config), config.port, config.token)
    finally:
        shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
