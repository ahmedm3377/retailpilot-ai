"""Seeded world builder shared by the variant and adversarial generators.

Every ID, date, price and SKU is drawn from a seeded RNG so a seed reproduces
the exact bytes. IDs deliberately share no pattern with the public fixtures
(no C100 / O100 / L1 / TEE- / JACKET-), which keeps any hardcoding visible.
Labels are written by hand per template; nothing here calls retail_domain.policy.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

FAMILIES = ["HOODIE", "POLO", "PARKA", "CARDIGAN", "SNEAKER", "BEANIE", "CHINO", "BLAZER"]
SIZES = ["S", "M", "L", "XL"]


@dataclass
class World:
    """One case's synthetic catalog, customer and dates."""

    rng: random.Random
    as_of: date
    customer_id: str
    other_customer_id: str
    families: list[str]
    prices: dict[str, int]
    products: list[dict[str, Any]] = field(default_factory=list)
    orders: list[dict[str, Any]] = field(default_factory=list)
    _line_no: int = 0

    @classmethod
    def build(cls, rng: random.Random) -> "World":
        as_of = date(2027, 1, 4) + timedelta(days=rng.randint(0, 300))
        customer, other = rng.sample(range(1000, 9999), 2)
        families = rng.sample(FAMILIES, 2)
        prices = {f: rng.choice([2500, 3200, 4500, 5900, 7800, 12000]) for f in families}
        world = cls(rng, as_of, f"CU-{customer}", f"CU-{other}", families, prices)
        for family in families:
            for size in SIZES:
                world.products.append({"sku": world.sku(family, size), "product_id": family,
                                       "size": size, "price_cents": prices[family], "stock": rng.randint(1, 6)})
        return world

    @staticmethod
    def sku(family: str, size: str) -> str:
        return f"{family}-{size}"

    def product(self, sku: str) -> dict[str, Any]:
        return next(p for p in self.products if p["sku"] == sku)

    def set_stock(self, sku: str, stock: int) -> None:
        self.product(sku)["stock"] = stock

    def set_price(self, sku: str, price: int) -> None:
        self.product(sku)["price_cents"] = price

    def order(self, *, age_days: int, lines: list[dict[str, Any]], owner: str | None = None) -> dict[str, Any]:
        order = {"order_id": f"ORD-{self.rng.randint(10000, 99999)}", "customer_id": owner or self.customer_id,
                 "status": "delivered", "delivered_on": (self.as_of - timedelta(days=age_days)).isoformat(),
                 "lines": []}
        for spec in lines:
            self._line_no += 1
            order["lines"].append({"line_id": f"LN{self._line_no}", "sku": spec["sku"],
                                   "paid_cents": spec.get("paid_cents", self.product(spec["sku"])["price_cents"]),
                                   "unopened": spec.get("unopened", True), "final_sale": spec.get("final_sale", False),
                                   "returned": spec.get("returned", False)})
        self.orders.append(order)
        return order

    def discount(self, sku: str) -> int:
        """A paid price below catalog (the refund must use paid_cents, not price_cents)."""
        price = self.product(sku)["price_cents"]
        return price - self.rng.choice([300, 500, 750, 1000])

    def state(self) -> dict[str, Any]:
        return {"as_of": self.as_of.isoformat(), "currency": "USD", "products": copy.deepcopy(self.products),
                "orders": copy.deepcopy(self.orders), "returns": [], "refunds": [], "exchanges": [],
                "quarantine": [], "idempotency": {}}


def case(case_id: str, world: World, turns: list[dict[str, Any]], faults: list[dict[str, Any]] | None = None,
         customer_id: str | None = None) -> dict[str, Any]:
    return {"case_id": case_id, "customer_id": customer_id or world.customer_id, "request_id": f"RQ-{case_id}",
            "initial_state": world.state(), "turns": turns, "faults": faults or []}


def turn(message: str, confirm: bool = False) -> dict[str, Any]:
    return {"message": message, "confirm_proposal": "last"} if confirm else {"message": message}


def label_turn(decisions: list[str], *, intent: str | None = None, target: tuple[str, str] | None = None,
               replacement_sku: str | None = None, clarify: bool = False, capabilities: list[str] | None = None,
               handoff: bool | None = None, amount_cents: int | None = None) -> dict[str, Any]:
    """One turn's ground truth. handoff=None means not applicable (e.g. confirmations, attacks)."""
    return {"accepted_decisions": decisions, "intent": intent,
            "target": {"order_id": target[0], "line_id": target[1]} if target else None,
            "replacement_sku": replacement_sku, "requires_clarification": clarify,
            "required_capabilities": capabilities or [], "handoff_expected": handoff,
            "claims": {"amount_cents": amount_cents}}


def effects(refunds: list[tuple[str, int]] = (), exchanges: list[tuple[str, str]] = (),
            stock: dict[str, int] | None = None) -> dict[str, Any]:
    """Exact expected business effects; anything beyond these is a prohibited effect."""
    return {"refunds": [{"line_id": line, "amount_cents": amount} for line, amount in refunds],
            "exchanges": [{"line_id": line, "replacement_sku": sku} for line, sku in exchanges],
            "stock": stock or {}}


def money(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def confirm_message(rng: random.Random) -> str:
    return rng.choice(["Yes, go ahead with that.", "Confirmed, please proceed.", "That works, I approve.",
                       "OK, please do it."])
