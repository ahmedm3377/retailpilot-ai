"""Task-specific prompt-injection validator (a Guardrails AI custom validator).

DetectJailbreak covers generic jailbreaks. This adds the attacks that matter to
a returns desk: overriding instructions, impersonating system/tool messages,
forged approvals, switching customer, revealing prompts, calling tools the
agents do not have, and smuggling structured action payloads.

It is a heuristic layer, not a guarantee. Patterns target instruction-like
phrasing ("ignore the policy") rather than the bare words "ignore" or "refund",
but quoted attack text is still flagged (fail closed); see the false-positive
tests. Authorization never depends on this validator passing.
"""
from __future__ import annotations

import re
from typing import Any

from guardrails.validator_base import FailResult, PassResult, ValidationResult, Validator, register_validator

_I = re.IGNORECASE | re.MULTILINE

PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "instruction_override": [
        re.compile(r"\b(ignore|disregard|forget|override|bypass|skip|disable)\b.{0,40}?\b(instructions?|rules?|"
                   r"polic(?:y|ies)|system prompt|guidelines|guardrails?|restrictions?|checks?|validation)\b", _I),
        re.compile(r"\b(you are now|from now on you|act as|pretend to be|new instructions?)\b", _I),
    ],
    "role_impersonation": [
        re.compile(r"^\s*(system|assistant|tool|developer)\s*:", _I),
        re.compile(r"</?\s*(system|tool|assistant|developer|instructions?)\s*>", _I),
        re.compile(r"\[(system|tool|inst|/inst)\]", _I),
    ],
    "forged_approval": [
        re.compile(r"\b(system|admin|administrator|manager|supervisor|policy|staff)\b.{0,30}?"
                   r"\b(has |have )?(approved|authori[sz]ed|confirmed|overrid(?:den|e))\b", _I),
        re.compile(r"\bpre-?(approved|authori[sz]ed)\b", _I),
        re.compile(r"\bproposal\b.{0,25}?\b(is|was|has been|already)\s+(approved|confirmed|authori[sz]ed)\b", _I),
    ],
    "identity_switch": [
        re.compile(r"\b(another|other|different)\s+customer'?s?\b", _I),
        re.compile(r"\b(as|for|on behalf of|switch to|i am|act as|log ?in as)\s+customer\s+[A-Z0-9_-]+", _I),
        re.compile(r"\b(customer_?id|authenticated_customer_id)\s*[:=]", _I),
    ],
    "prompt_extraction": [
        re.compile(r"\b(reveal|show|print|repeat|leak|output)\b.{0,30}?\b(system prompt|your (instructions|prompt)|"
                   r"hidden (instructions|prompt))\b", _I),
    ],
    "unlisted_tool": [
        re.compile(r"\b(commit_\w+|commit_confirmed_proposal|create_proposal|reset_from_initial_state|"
                   r"apply_before_turn|set_stock|stock_update|refund_now|approve_\w+)\b", _I),
        re.compile(r"\b(call|invoke|run|execute|trigger)\b.{0,20}?\b(tool|function|api)\b.{0,30}?"
                   r"\b(commit|refund|approve|exchange|reset)\w*", _I),
        re.compile(r"\b(drop\s+table|delete\s+from|update\s+\w+\s+set|insert\s+into|truncate\s+\w+)\b", _I),
    ],
    "structured_payload": [
        re.compile(r'"\s*(action|tool|proposal_id|confirmed_proposal_id|customer_id|'
                   r'authenticated_customer_id|decision|status)"\s*:', _I),
    ],
}


def find_injection(text: str) -> list[str]:
    """Names of the pattern categories that match `text`."""
    return [category for category, patterns in PATTERNS.items() if any(p.search(text) for p in patterns)]


@register_validator(name="retail/task_injection", data_type="string")
class TaskInjection(Validator):
    """Fails when text contains returns-desk specific injection patterns."""

    def __init__(self, on_fail: Any = None, **kwargs: Any) -> None:
        super().__init__(on_fail=on_fail, use_local=True, **kwargs)

    def validate(self, value: Any, metadata: dict[str, Any]) -> ValidationResult:
        categories = find_injection(str(value))
        if categories:
            return FailResult(error_message="task injection: " + ", ".join(categories),
                              metadata={"categories": categories})
        return PassResult()
