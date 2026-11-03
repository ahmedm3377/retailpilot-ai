"""Trusted, immutable scope of one MCP connection, sent to the standalone server in request headers.

The host (AgentRuntime) builds it from values it already verified (the evaluation case or the
server-owned UI registry). Nothing here is ever taken from tool arguments, prompts or the browser.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
SCHEMA_PATTERN = re.compile(r"retail_[a-z0-9_]+")
MODES = {"ui": ("retail_ui_", "retail_store"), "evaluation": ("retail_eval_",), "test": ("retail_t_",)}


class ScopeError(RuntimeError):
    """The trusted scope is malformed."""


@dataclass(frozen=True)
class ToolScope:
    schema: str
    customer_id: str
    request_id: str
    mode: str

    def validate(self) -> "ToolScope":
        if self.mode not in MODES:
            raise ScopeError("unknown scope mode")
        if not SCHEMA_PATTERN.fullmatch(self.schema) or not self.schema.startswith(MODES[self.mode]):
            raise ScopeError(f"schema not allowed for the {self.mode} mode")
        if not ID_PATTERN.fullmatch(self.customer_id) or not ID_PATTERN.fullmatch(self.request_id):
            raise ScopeError("customer_id/request_id missing or malformed")
        return self

    def headers(self) -> dict[str, str]:
        return {"X-Retail-Scope-Schema": self.schema, "X-Retail-Scope-Customer": self.customer_id,
                "X-Retail-Scope-Request": self.request_id, "X-Retail-Scope-Mode": self.mode}

    def key(self) -> tuple[str, str, str, str]:
        """Identity of the scope: a connection is reused only for exactly the same key."""
        return (self.schema, self.customer_id, self.request_id, self.mode)


def mode_for_schema(schema: str) -> str:
    for mode, prefixes in MODES.items():
        if schema.startswith(prefixes):
            return mode
    raise ScopeError("schema has no MCP scope mode")
