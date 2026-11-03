"""One case's business state in an isolated PostgreSQL schema: reset, load, export."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from app.db import repositories as repo
from app.db.connection import UnsafeDatabaseError, check_schema, check_url, close_pool, connect, store_connection
from app.db.schema import RESETTABLE_TABLES, SCHEMA_MARKER, apply_migrations
from retail_domain.models import State

TRANSACTION_KEYS = ("returns", "refunds", "exchanges", "quarantine", "idempotency")

# (table, contract key, ID column) for the transaction-record lists.
_RECORD_TABLES = (
    ("returns", "returns", "return_id"),
    ("refunds", "refunds", "refund_id"),
    ("exchanges", "exchanges", "exchange_id"),
    ("quarantine", "quarantine", "quarantine_id"),
)


class StateNotLoadedError(RuntimeError):
    """The run schema has no fixture state; call reset_from_initial_state first."""


def _product(row: dict[str, Any]) -> dict[str, Any]:
    return {**row["extra"], "sku": row["sku"], "product_id": row["product_id"], "size": row["size"],
            "price_cents": row["price_cents"], "stock": row["stock"]}


def _line(row: dict[str, Any]) -> dict[str, Any]:
    return {**row["extra"], "line_id": row["line_id"], "sku": row["sku"], "paid_cents": row["paid_cents"],
            "unopened": row["unopened"], "final_sale": row["final_sale"], "returned": row["returned"]}


def _order(row: dict[str, Any], lines: list[dict[str, Any]]) -> dict[str, Any]:
    delivered = row["delivered_on"]
    return {**row["extra"], "order_id": row["order_id"], "customer_id": row["customer_id"],
            "status": row["status"], "delivered_on": delivered.isoformat() if delivered else None,
            "lines": [_line(line) for line in lines]}


def business_state_dict(meta: dict[str, Any], products: list[dict[str, Any]],
                        orders: list[dict[str, Any]], lines: list[dict[str, Any]]) -> dict[str, Any]:
    """Contract-shaped products/orders from rows, without transaction records."""
    lines_by_order: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in lines:
        lines_by_order[line["order_id"]].append(line)
    return {**meta["extra"], "as_of": meta["as_of"].isoformat(), "currency": meta["currency"],
            "products": [_product(p) for p in products],
            "orders": [_order(o, lines_by_order[o["order_id"]]) for o in orders]}


def state_from_rows(meta: dict[str, Any], products: list[dict[str, Any]],
                    orders: list[dict[str, Any]], lines: list[dict[str, Any]]) -> State:
    """A State for policy checks, built from (possibly a subset of) rows."""
    return State.model_validate(business_state_dict(meta, products, orders, lines))


class RetailStore:
    """Handle to one run schema in a local PostgreSQL database."""

    def __init__(self, url: str, schema: str, *, require_test_db: bool = False) -> None:
        check_url(url, require_test_db=require_test_db)
        check_schema(schema)
        self.url = url
        self.schema = schema

    def connect(self):
        """Context manager yielding a connection pinned to this schema (pooled unless RETAIL_DB_POOL=0)."""
        return store_connection(self.url, self.schema)

    def create_schema(self) -> list[int]:
        """Create the schema and apply pending migrations; returns versions applied."""
        with connect(self.url, self.schema) as conn:
            return apply_migrations(conn, self.schema)

    def drop_schema(self) -> None:
        """Drop the schema, but only if this code created it (checked via its comment)."""
        close_pool(self.url, self.schema)
        with connect(self.url, self.schema) as conn:
            row = conn.execute(
                "SELECT obj_description(oid, 'pg_namespace') AS marker FROM pg_namespace WHERE nspname = %s",
                (self.schema,)).fetchone()
            if row is None:
                return
            if row["marker"] != SCHEMA_MARKER:
                raise UnsafeDatabaseError(f"schema {self.schema!r} was not created by this application")
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.schema)))

    def reset_from_initial_state(self, initial_state: State) -> None:
        """Atomically replace all state in this schema with a copy of `initial_state`."""
        state = initial_state.model_copy(deep=True)
        nonempty = [key for key in TRANSACTION_KEYS if getattr(state, key)]
        if nonempty:
            raise ValueError(f"initial_state must have empty {', '.join(nonempty)} (contract v1)")
        with self.connect() as conn, conn.transaction():
            conn.execute(sql.SQL("TRUNCATE {} RESTART IDENTITY").format(
                sql.SQL(", ").join(sql.Identifier(t) for t in RESETTABLE_TABLES)))
            conn.execute("INSERT INTO fixture_meta (as_of, currency, extra) VALUES (%s, %s, %s)",
                         (state.as_of, state.currency, Jsonb(state.model_extra or {})))
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO products (sku, product_id, size, price_cents, stock, position, extra)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    [(p.sku, p.product_id, p.size, p.price_cents, p.stock, i, Jsonb(p.model_extra or {}))
                     for i, p in enumerate(state.products)])
                cur.executemany(
                    """INSERT INTO orders (order_id, customer_id, status, delivered_on, position, extra)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    [(o.order_id, o.customer_id, o.status, o.delivered_on, i, Jsonb(o.model_extra or {}))
                     for i, o in enumerate(state.orders)])
                cur.executemany(
                    """INSERT INTO order_lines (order_id, line_id, sku, paid_cents, unopened, final_sale,
                                                returned, position, extra)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    [(o.order_id, ln.line_id, ln.sku, ln.paid_cents, ln.unopened, ln.final_sale,
                      ln.returned, i, Jsonb(ln.model_extra or {}))
                     for o in state.orders for i, ln in enumerate(o.lines)])

    def _export(self, conn: psycopg.Connection) -> dict[str, Any]:
        meta = repo.get_meta(conn)
        if meta is None:
            raise StateNotLoadedError(f"schema {self.schema!r} has no fixture state")
        out = business_state_dict(meta, repo.list_products(conn), repo.list_orders(conn), repo.list_lines(conn))
        for table, key, id_col in _RECORD_TABLES:
            out[key] = [dict(r) for r in repo.list_table(conn, table, f"length({id_col}), {id_col}")]
        out["idempotency"] = {
            r["request_id"]: {"status": r["status"], "customer_id": r["customer_id"],
                              "proposal_id": r["proposal_id"], "result": r["result"]}
            for r in repo.list_table(conn, "idempotency", "request_id")}
        return out

    def export_final_state(self) -> dict[str, Any]:
        """Contract-shaped state: input keys plus current transaction records."""
        with self.connect() as conn, conn.transaction():
            # One snapshot for every table read.
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            return self._export(conn)

    def load_state(self) -> State:
        """The current state as a domain model, for read-only policy checks."""
        return State.model_validate(self.export_final_state())

    def audit_log(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(r) for r in repo.list_audit(conn)]
