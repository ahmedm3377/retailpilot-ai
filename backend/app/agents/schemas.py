"""Strict typed outputs of the two LLM agents, and the per-turn result.

Every model output is parsed by the guardrail schema gate into one of these
(`extra="forbid"`). Passing the schema says nothing about eligibility or
approval; the trusted services decide those.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.domain.requests import ID, CustomerHandoff, Strict  # noqa: F401 - CustomerHandoff is one of the agents' outputs

READ_ONLY_TOOLS = ("lookup_customer_orders", "get_order_line", "check_return_eligibility",
                   "get_exchange_options", "check_inventory", "get_request_status")
ToolName = Literal["lookup_customer_orders", "get_order_line", "check_return_eligibility",
                   "get_exchange_options", "check_inventory", "get_request_status"]


class CustomerStep(CustomerHandoff):
    """One step of the Customer Resolution Agent: call a read-only tool, or hand on what it understood."""

    action: Literal["call_tool", "handoff"] = "handoff"
    tool: ToolName | None = None
    intent: Literal["refund", "exchange", "unknown", "out_of_scope"] = "unknown"
    needs_clarification: bool = False

    def handoff(self) -> CustomerHandoff:
        return CustomerHandoff.model_validate(self.model_dump(exclude={"action", "tool"}))


class OperationsAction(Strict):
    """One step of the Retail Operations Agent: call a read-only tool, recommend, or hand the
    request back to the Customer Resolution Agent because the customer must answer something."""

    action: Literal["call_tool", "recommend", "ask_customer"]
    tool: ToolName | None = None
    order_id: str | None = ID
    line_id: str | None = ID
    sku: str | None = ID
    remedy: Literal["refund", "exchange", "none"] | None = None
    replacement_sku: str | None = ID
    reason: str | None = Field(default=None, max_length=300)


class CustomerMessage(Strict):
    """Customer Resolution Agent's wording of trusted facts."""

    message: str = Field(min_length=1, max_length=600)


Decision = Literal["propose", "clarify", "deny", "completed", "error"]


class TurnResult(BaseModel):
    """Contract-relevant turn outcome plus safe traces (no prompts or reasoning)."""

    decision: Decision
    response: str
    proposal: dict[str, Any] | None = None
    agent_events: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []
    guardrail_events: list[dict[str, Any]] = []
    internal_status: dict[str, Any] = {}


def json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Schema for structured model output."""
    return model.model_json_schema()
