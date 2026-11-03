"""A customer's request once understood: what the business layer works on, whoever produced it."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ID = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CustomerHandoff(Strict):
    """Customer Resolution Agent -> Retail Operations Agent."""

    intent: Literal["refund", "exchange", "unknown", "out_of_scope"]
    order_id: str | None = ID
    line_id: str | None = ID
    requested_size: str | None = Field(default=None, max_length=20)
    all_items: bool = False  # every item of the order, or several of them ("both", "all", "the whole order")
    item_opened: bool | None = None  # what the customer says; it can only refuse, never make an item eligible
    needs_clarification: bool
    clarification_question: str | None = Field(default=None, max_length=300)
