"""HP candidate interface v1 adapter.

    python adapter.py --input candidate_assets/public_cases.json --output results.json

Any Python can launch it: outside the project's environment it relaunches itself there (.venv, else uv).
If the standalone MCP server isn't running, it starts one for the run and stops it afterwards.

Runs every case sequentially through the same AgentRuntime the UI uses (two
local LLM agents on the model named in .env, local Guardrails AI, PostgreSQL business
engine), each case from a fresh isolated state in the test database, and writes
the contract results document atomically.

Exit codes: 0 all cases ran (business outcomes are in the file, right or wrong);
2 bad usage or invalid input; 3 infrastructure failure (database, model, MCP server
or validators unavailable). On exit 3 the results file
still lists every case, with the failed ones marked ("Include failed runs"), and
a <output>.diagnostics.json explains why.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTECTED_DIR = ROOT / "candidate_assets"
EXIT_OK, EXIT_USAGE, EXIT_INFRA = 0, 2, 3


def in_project_environment() -> bool:
    return sys.version_info >= (3, 13) and all(importlib.util.find_spec(m) for m in ("psycopg", "langgraph", "litellm"))


def relaunch_command(script: str, argv: list[str]) -> list[str] | None:
    """How to run this script in the project's environment: its .venv, else uv; None if neither exists."""
    for python in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"):
        if python.exists():
            return [str(python), script, *argv]
    uv = shutil.which("uv")
    return [uv, "run", "--project", str(ROOT), "python", script, *argv] if uv else None


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="adapter.py", description="Run HP contract cases through the local agentic returns desk.",
        epilog="Requires: docker compose up -d postgres; the model server named in .env; cached Guardrails models.")
    parser.add_argument("--input", required=True, help="contract v1 cases JSON (read, never modified)")
    parser.add_argument("--output", required=True, help="results JSON to write (atomically)")
    parser.add_argument("--keep-schemas", action="store_true",
                        help="debug: keep each case's retail_eval_* schemas in the test database")
    return parser.parse_args(argv)


def check_paths(input_path: Path, output_path: Path) -> str | None:
    if not input_path.is_file():
        return f"input not found: {input_path}"
    if output_path == input_path:
        return "output must not be the input file"
    if output_path == PROTECTED_DIR or PROTECTED_DIR in output_path.parents:
        return f"refusing to write under {PROTECTED_DIR} (supplied assets are read-only)"
    if output_path.is_dir():
        return f"output is a directory: {output_path}"
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not in_project_environment():
        command = None if os.environ.get("RETAIL_ADAPTER_RELAUNCHED") else relaunch_command(__file__, sys.argv[1:])
        if command is None:
            print("adapter: the project's environment is missing; run `uv sync --extra dev` first", file=sys.stderr)
            return EXIT_INFRA
        return subprocess.run(command, env={**os.environ, "RETAIL_ADAPTER_RELAUNCHED": "1"}).returncode
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    diagnostics_path = output_path.with_name(output_path.name + ".diagnostics.json")
    problem = check_paths(input_path, output_path)
    if problem:
        print(f"adapter: {problem}", file=sys.stderr)
        return EXIT_USAGE

    from pydantic import ValidationError

    from app.evaluation.runner import InfrastructureError, failed_document, prepare, run_fixture
    from app.evaluation.serialization import write_atomic
    from app.fixtures.loader import load_fixtures
    from app.mcp.client import McpToolError, unavailable_message
    from app.mcp.launch import server_running

    try:
        root, input_sha256 = load_fixtures(input_path)
    except (ValidationError, ValueError, OSError) as error:
        print(f"adapter: invalid input: {str(error).splitlines()[0]}", file=sys.stderr)
        return EXIT_USAGE
    document, raws, failure = None, [], None
    try:
        with server_running():
            ctx = prepare()
            document, raws = run_fixture(ctx, root, input_sha256, keep_schemas=args.keep_schemas, include_failed=True)
        failed = [r["case_id"] for r in document["results"] if r.get("run_failed")]
        if failed:
            failure = f"cases failed: {', '.join(failed)}"
    except InfrastructureError as error:
        failure = f"infrastructure: {error}"
    except McpToolError as error:
        failure = f"infrastructure: {unavailable_message(error)}"
    except Exception as error:  # noqa: BLE001 - any unexpected failure is reported, never hidden
        failure = f"unexpected {type(error).__name__}: {error}"
        traceback.print_exc()
    if document is None:  # nothing could run: still report every case as failed
        document = failed_document(root, input_sha256, failure or "not run")

    from app.observability import telemetry as tel
    tel.shutdown()  # flush (no-op when telemetry is off)
    if failure:
        write_atomic(output_path, document)
        write_atomic(diagnostics_path, {"status": "failed", "error": failure, "input": str(input_path)})
        print(f"adapter: FAILED ({failure}); see {diagnostics_path}. Results list the failed runs.", file=sys.stderr)
        return EXIT_INFRA

    write_atomic(output_path, document)
    diagnostics_path.unlink(missing_ok=True)  # an older failure note no longer applies
    for raw in raws:
        print(f"{raw['case_id']}: {' -> '.join(raw['decisions'])} ({raw['metrics']['elapsed_ms']:.0f} ms)")
    print(f"wrote {output_path} ({len(document['results'])} cases, input sha256 {input_sha256[:12]}...)")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
