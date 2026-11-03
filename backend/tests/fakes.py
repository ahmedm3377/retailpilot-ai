"""Scripted stand-in for the LLM, for deterministic unit tests ONLY.

Tests using FakeModel exercise graph routing, guardrails, permissions and trusted
integration; they are not evidence of real agent inference (see the `real_model`
tests for that).
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Any, Callable

from app.agents.model_client import ModelCall, ModelReply, ModelUnavailableError

Reply = str | dict[str, Any] | Callable[[list[dict[str, str]]], str] | Exception


class FakeModel:
    model = "fake/scripted"

    def __init__(self, script: dict[str, list[Reply]] | None = None) -> None:
        self.queues: dict[str, deque[Reply]] = defaultdict(deque)
        self.seen: list[tuple[str, list[dict[str, str]]]] = []
        for role, replies in (script or {}).items():
            self.queues[role].extend(replies)

    def add(self, role: str, *replies: Reply) -> "FakeModel":
        self.queues[role].extend(replies)
        return self

    def chat_json(self, role: str, messages: list[dict[str, str]], schema: dict[str, Any]) -> ModelReply:
        self.seen.append((role, messages))
        if not self.queues[role]:
            raise ModelUnavailableError(f"fake model has no scripted reply for {role}")
        reply = self.queues[role].popleft()
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            reply = reply(messages)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return ModelReply(text, ModelCall(role, self.model, 1.0, True, 10, 5))

    def calls_for(self, role: str) -> int:
        return sum(1 for r, _ in self.seen if r == role)


CUSTOMER = "customer_resolution_agent"
OPS = "retail_operations_agent"


def handoff(intent: str = "refund", order_id: str | None = "O100", line_id: str | None = "L1",
            requested_size: str | None = None, needs_clarification: bool = False,
            question: str | None = None) -> dict[str, Any]:
    return {"intent": intent, "order_id": order_id, "line_id": line_id, "requested_size": requested_size,
            "needs_clarification": needs_clarification, "clarification_question": question}


def tool(name: str, **args: str) -> dict[str, Any]:
    return {"action": "call_tool", "tool": name, **args}


def recommend(remedy: str, order_id: str = "O100", line_id: str = "L1", replacement_sku: str | None = None,
              reason: str = "ok") -> dict[str, Any]:
    return {"action": "recommend", "remedy": remedy, "order_id": order_id, "line_id": line_id,
            "replacement_sku": replacement_sku, "reason": reason}


def say(message: str) -> dict[str, Any]:
    return {"message": message}


def refund_turn(model: FakeModel, order_id: str = "O100", line_id: str = "L1",
                explanation: str = "I can refund that item. Please confirm the proposal.") -> FakeModel:
    """Script one unconfirmed refund turn: handoff, eligibility tool, recommend, explanation."""
    return (model.add(CUSTOMER, handoff("refund", order_id, line_id), say(explanation))
            .add(OPS, tool("check_return_eligibility", order_id=order_id, line_id=line_id),
                 recommend("refund", order_id, line_id)))


def exchange_turn(model: FakeModel, size: str, sku: str, order_id: str = "O100", line_id: str = "L1") -> FakeModel:
    return (model.add(CUSTOMER, handoff("exchange", order_id, line_id, requested_size=size),
                      say("I can exchange that item. Please confirm the proposal."))
            .add(OPS, tool("check_return_eligibility", order_id=order_id, line_id=line_id),
                 tool("get_exchange_options", order_id=order_id, line_id=line_id),
                 recommend("exchange", order_id, line_id, replacement_sku=sku)))
