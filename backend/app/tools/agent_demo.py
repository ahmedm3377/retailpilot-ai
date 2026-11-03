"""Run public cases through the real agents: the model named in .env + local Guardrails + PostgreSQL.

Each case runs in throwaway schemas (business state and checkpoints) that are
dropped afterwards. "last" confirmations are resolved here, as the trusted
caller, to the most recent non-null proposal ID returned by earlier turns.

Usage: python -m app.tools.agent_demo [--case public_02_exchange] [--trace]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid

from app.agents.model_client import ModelClient
from app.agents.settings import AgentSettings
from app.db.connection import database_url
from app.db.fixture_state import RetailStore
from app.fixtures.loader import load_fixtures
from app.guardrails.service import GuardrailService
from app.services.agent_runtime import AgentRuntime, describe_configuration
from app.services.faults import FaultInjector


def run_case(case, model, guards, settings, trace: bool) -> dict:
    suffix = uuid.uuid4().hex[:8]
    store = RetailStore(database_url(), f"retail_ui_demo_{suffix}")
    store.create_schema()
    runtime = AgentRuntime(store, model, guards, settings, database_url(), f"retail_demo_ckpt_{suffix}")
    try:
        store.reset_from_initial_state(case.initial_state)
        injector = FaultInjector(case.faults)
        last_proposal = None
        turns = []
        for number, turn in enumerate(case.turns, start=1):
            confirmed = last_proposal if turn.confirm_proposal == "last" else None
            start = time.perf_counter()
            result = runtime.handle_customer_turn(
                session_id="demo", authenticated_customer_id=case.customer_id, request_id=case.request_id,
                message=turn.message, turn_number=number, confirmed_proposal_id=confirmed, injector=injector)
            elapsed = (time.perf_counter() - start) * 1000
            if result.proposal:
                last_proposal = result.proposal["proposal_id"]
            turns.append(result.decision)
            print(f"  turn {number}{' [confirm ' + str(confirmed) + ']' if turn.confirm_proposal else ''}: "
                  f"{result.decision} ({elapsed:.0f} ms) -> {result.response}")
            if trace:
                print("    agents:", [(e["agent"], e["latency_ms"], e["completion_tokens"]) for e in result.agent_events])
                print("    tools:", [(e["agent"], e["tool"], e["arguments"], e["outcome"]) for e in result.tool_events])
                blocked = [(e["validator"], e["outcome"], e["reason_code"]) for e in result.guardrail_events
                           if e["outcome"] != "passed"]
                print("    guardrails (non-pass):", blocked)
                print("    internal:", json.dumps(result.internal_status, default=str))
        state = store.export_final_state()
        summary = {"decisions": turns, "refunds": len(state["refunds"]), "exchanges": len(state["exchanges"]),
                   "quarantine": len(state["quarantine"]),
                   "stock": {p["sku"]: p["stock"] for p in state["products"]}}
        print("  final:", json.dumps(summary))
        return summary
    finally:
        runtime.close()
        with store.connect() as conn:
            conn.execute(f"DROP SCHEMA IF EXISTS retail_demo_ckpt_{suffix} CASCADE")
        store.drop_schema()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="candidate_assets/public_cases.json")
    parser.add_argument("--case", action="append", help="case_id to run (repeatable; default all)")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args(argv)
    settings = AgentSettings.from_env()
    model = ModelClient(settings)
    start = time.perf_counter()
    guards = GuardrailService(settings)
    print(f"guardrails loaded in {time.perf_counter() - start:.1f} s")
    print(json.dumps(describe_configuration(model, guards), indent=1))
    root, sha256 = load_fixtures(args.input)
    print(f"input sha256: {sha256}")
    for case in root.cases:
        if args.case and case.case_id not in args.case:
            continue
        print(case.case_id)
        run_case(case, model, guards, settings, args.trace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
