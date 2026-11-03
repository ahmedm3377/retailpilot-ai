"""Pydantic v2 models for the candidate interface v1 fixture schema (CONTRACT.md).

Models accept extra fields so schema-compatible private fixtures and fields
reserved for later steps survive a load/dump round trip. Money and stock are
strict integers; floats are rejected.
"""
from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class Product(ContractModel):
    sku: str
    product_id: str
    size: str
    price_cents: StrictInt
    stock: StrictInt


class Line(ContractModel):
    """One order line; each line is exactly one unit."""

    line_id: str
    sku: str
    paid_cents: StrictInt
    unopened: StrictBool
    final_sale: StrictBool
    returned: StrictBool


class Order(ContractModel):
    order_id: str
    customer_id: str
    status: str
    delivered_on: date | None = None
    lines: list[Line]


class State(ContractModel):
    as_of: date
    currency: str
    products: list[Product]
    orders: list[Order]
    returns: list[dict[str, Any]] = Field(default_factory=list)
    refunds: list[dict[str, Any]] = Field(default_factory=list)
    exchanges: list[dict[str, Any]] = Field(default_factory=list)
    quarantine: list[dict[str, Any]] = Field(default_factory=list)
    idempotency: dict[str, Any] = Field(default_factory=dict)


class Turn(ContractModel):
    message: str
    confirm_proposal: str | None = None


class StockChangeFault(ContractModel):
    type: Literal["stock_change"]
    before_turn: StrictInt
    sku: str
    new_stock: StrictInt


class TimeoutAfterCommitFault(ContractModel):
    type: Literal["timeout_after_commit"]


Fault = Annotated[Union[StockChangeFault, TimeoutAfterCommitFault], Field(discriminator="type")]


class Case(ContractModel):
    case_id: str
    customer_id: str
    request_id: str
    initial_state: State
    turns: list[Turn]
    faults: list[Fault] = Field(default_factory=list)


class FixtureRoot(ContractModel):
    contract_version: Literal[1]
    cases: list[Case]
