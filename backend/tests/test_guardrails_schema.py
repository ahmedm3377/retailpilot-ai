from __future__ import annotations

import json

import pytest

from app.agents.schemas import READ_ONLY_TOOLS, CustomerHandoff, OperationsAction
from app.agents.tool_registry import TOOL_ARGS, ReadOnlyToolRegistry, ToolNotAllowed

GOOD_HANDOFF = {"intent": "refund", "order_id": "O100", "line_id": "L1", "requested_size": None,
                "needs_clarification": False, "clarification_question": None}


@pytest.mark.parametrize("raw", [
    "not json at all",
    "{'intent': 'refund'}",                                            # python repr, not JSON
    json.dumps({**GOOD_HANDOFF, "extra": 1}),                          # extra key
    json.dumps({**GOOD_HANDOFF, "intent": "steal"}),                    # wrong enum
    json.dumps({**GOOD_HANDOFF, "needs_clarification": "maybe"}),       # wrong type
    json.dumps({k: v for k, v in GOOD_HANDOFF.items() if k != "intent"}),  # missing field
    json.dumps({**GOOD_HANDOFF, "order_id": "O100; DROP TABLE"}),       # bad ID characters
    json.dumps({**GOOD_HANDOFF, "clarification_question": "x" * 301}),  # oversized text
    json.dumps([GOOD_HANDOFF]),
    "x" * 9000,
])
def test_invalid_handoffs_rejected(guards, raw):
    result = guards.validate_agent_action(raw, CustomerHandoff)
    assert not result.ok and result.value is None


def test_valid_handoff_parsed(guards):
    result = guards.validate_agent_action(json.dumps(GOOD_HANDOFF), CustomerHandoff)
    assert result.ok and result.value == CustomerHandoff(**GOOD_HANDOFF)


@pytest.mark.parametrize("action", [
    {"action": "call_tool", "tool": "commit_confirmed_proposal"},
    {"action": "call_tool", "tool": "create_proposal", "order_id": "O100", "line_id": "L1"},
    {"action": "call_tool", "tool": "set_stock", "sku": "TEE-L"},
    {"action": "approve", "proposal_id": "PROP-1"},
    {"action": "call_tool", "tool": "check_inventory", "sku": "TEE-L", "customer_id": "C999"},
    {"action": "recommend", "remedy": "store_credit"},
])
def test_unknown_tools_and_forged_fields_rejected(guards, action):
    assert not guards.validate_agent_action(json.dumps(action), OperationsAction).ok


def test_registry_exposes_only_read_only_tools():
    assert set(READ_ONLY_TOOLS) == set(TOOL_ARGS) == {
        "lookup_customer_orders", "get_order_line", "check_return_eligibility", "get_exchange_options",
        "check_inventory", "get_request_status"}
    schema_text = json.dumps(OperationsAction.model_json_schema())
    for name in ("commit", "create_proposal", "reset", "set_stock", "apply_before_turn", "sql", "approve"):
        assert name not in schema_text


def test_registry_requires_exact_arguments():
    registry = ReadOnlyToolRegistry(ops=None, authenticated_customer_id="C1", request_id="R1", transport=lambda: None)  # type: ignore[arg-type]
    with pytest.raises(ToolNotAllowed):
        registry.arguments_for(OperationsAction(action="call_tool", tool="get_order_line", order_id="O1"))
    with pytest.raises(ToolNotAllowed):
        registry.arguments_for(OperationsAction(action="call_tool", tool=None))
    assert registry.arguments_for(OperationsAction(action="call_tool", tool="check_inventory", sku="X-1",
                                                   order_id="ignored")) == {"sku": "X-1"}
