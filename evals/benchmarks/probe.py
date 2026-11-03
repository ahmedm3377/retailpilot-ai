"""Throughput probe: output tokens per second for a fixed prompt (measurement only).

Works with any engine, through the same client as the agents, so it measures the whole call
(request overhead included), not pure decoding. It is a model-level measurement under the
variant's conditions, not part of any agent or task.
"""
from __future__ import annotations

from typing import Any

from app.agents.model_client import ChatModel
from app.agents.schemas import CustomerMessage, json_schema
from evals.benchmarks.stats import median

PROMPT = "Reply with the message: the days of the week, in order."


def decode_probe(model: ChatModel, runs: int = 3) -> dict[str, Any]:
    results = []
    for _ in range(runs):
        call = model.chat_json("probe", [{"role": "user", "content": PROMPT}], json_schema(CustomerMessage)).call
        tokens = call.completion_tokens
        results.append({"completion_tokens": tokens, "call_ms": round(call.latency_ms, 1),
                        "decode_tokens_per_s": round(tokens / (call.latency_ms / 1000), 1)
                        if tokens and call.latency_ms > 0 else None})
    return {"runs": results, "median_decode_tokens_per_s": median(r["decode_tokens_per_s"] for r in results),
            "method": "fixed prompt through the agents' model client; tokens the engine reported / whole call time"}
