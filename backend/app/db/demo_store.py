"""The persistent store behind the web UI: one PostgreSQL schema that outlives sessions.

Customers, orders and stock are read from this schema, and commits stay in it. The seed file is
used only to fill it the first time or on an explicit reset.

    uv run python -m app.db.demo_store           # create and seed if empty
    uv run python -m app.db.demo_store --reset   # back to the seed data; deletes all transactions
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

from app.db.connection import database_url
from app.db.fixture_state import RetailStore
from retail_domain.models import State

SEED_FILE = Path(__file__).resolve().parent / "demo_store_seed.json"
DEFAULT_SCHEMA = "retail_store"


def store_schema() -> str:
    return os.environ.get("RETAIL_STORE_SCHEMA", DEFAULT_SCHEMA)


def seed_state(as_of: date | None = None, seed_file: Path = SEED_FILE) -> State:
    """The seed as a contract-shaped State, with delivery dates counted back from `as_of`."""
    seed = json.loads(seed_file.read_text(encoding="utf-8"))
    as_of = as_of or date.today()
    orders = [{**{k: v for k, v in o.items() if k != "delivered_days_ago"},
               "delivered_on": (as_of - timedelta(days=o["delivered_days_ago"])).isoformat()}
              for o in seed["orders"]]
    return State.model_validate({"as_of": as_of.isoformat(), "currency": seed["currency"],
                                 "products": seed["products"], "orders": orders})


def ensure_store(store: RetailStore, *, reset: bool = False) -> bool:
    """Create the schema and seed it if it is empty, or if `reset`. Returns whether it seeded."""
    store.create_schema()
    with store.connect() as conn:
        seeded = conn.execute("SELECT count(*) AS n FROM fixture_meta").fetchone()["n"] > 0
    if seeded and not reset:
        return False
    store.reset_from_initial_state(seed_state())
    return True


def find_customer(store: RetailStore, typed_id: str) -> str | None:
    """The customer ID as stored; small or capital letters both match. None if unknown or ambiguous."""
    with store.connect() as conn:
        rows = conn.execute("SELECT DISTINCT customer_id FROM orders WHERE lower(customer_id) = lower(%s)",
                            (typed_id,)).fetchall()
    ids = [r["customer_id"] for r in rows]
    if typed_id in ids:
        return typed_id
    return ids[0] if len(ids) == 1 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reset", action="store_true", help="replace everything with the seed data")
    args = parser.parse_args(argv)
    store = RetailStore(database_url(), store_schema())
    seeded = ensure_store(store, reset=args.reset)
    print(f"store {store.schema}: " + ("seeded" if seeded else "already has data (use --reset to start over)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
