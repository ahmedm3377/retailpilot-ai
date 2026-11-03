"""The server's own PostgreSQL access: read-only SQL, on read-only transactions, one schema per call."""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import psycopg
from psycopg.rows import dict_row
from retail_domain.models import State

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
SCHEMA = re.compile(r"retail_[a-z0-9_]+")
SCHEMA_MARKER = "hp-retail-desk run schema"


class StoreError(RuntimeError):
    """The database or schema is not one this server may read."""


def check_url(url: str, *, test_database: bool) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("postgresql", "postgres") or parsed.hostname not in LOCAL_HOSTS:
        raise StoreError("only a local PostgreSQL database is allowed")
    if parsed.path.lstrip("/").endswith("_test") != test_database:
        raise StoreError("wrong database for this mode")
    return url


def _connect(url: str, schema: str) -> psycopg.Connection:
    if not SCHEMA.fullmatch(schema):
        raise StoreError("schema name not allowed")
    return psycopg.connect(url, connect_timeout=5, row_factory=dict_row,
                           options=f"-c search_path={schema} -c default_transaction_read_only=on")


def _has_marker(conn: psycopg.Connection, schema: str) -> bool:
    row = conn.execute("SELECT obj_description(oid, 'pg_namespace') AS marker FROM pg_namespace WHERE nspname = %s",
                       (schema,)).fetchone()
    return row is not None and row["marker"] == SCHEMA_MARKER


def load_state(url: str, schema: str) -> State:
    """Products and orders as the shared State model, read in one snapshot."""
    with _connect(url, schema) as conn:
        if not _has_marker(conn, schema):
            raise StoreError("schema missing or not created by this application")
        meta = conn.execute("SELECT as_of, currency FROM fixture_meta").fetchone()
        if meta is None:
            raise StoreError("schema has no state")
        products = conn.execute("SELECT sku, product_id, size, price_cents, stock FROM products "
                                "ORDER BY position").fetchall()
        orders = conn.execute("SELECT order_id, customer_id, status, delivered_on FROM orders "
                              "ORDER BY position").fetchall()
        lines = conn.execute("SELECT order_id, line_id, sku, paid_cents, unopened, final_sale, returned "
                             "FROM order_lines ORDER BY order_id, position").fetchall()
    by_order: dict[str, list[dict[str, Any]]] = {}
    for line in lines:
        by_order.setdefault(line["order_id"], []).append({k: v for k, v in line.items() if k != "order_id"})
    return State.model_validate({
        "as_of": meta["as_of"].isoformat(), "currency": meta["currency"], "products": products,
        "orders": [{**o, "delivered_on": o["delivered_on"].isoformat() if o["delivered_on"] else None,
                    "lines": by_order.get(o["order_id"], [])} for o in orders]})


def get_product(url: str, schema: str, sku: str) -> dict[str, Any] | None:
    with _connect(url, schema) as conn:
        if not _has_marker(conn, schema):
            raise StoreError("schema missing or not created by this application")
        return conn.execute("SELECT sku, product_id, size, price_cents, stock FROM products WHERE sku = %s",
                            (sku,)).fetchone()


def get_proposal(url: str, schema: str, proposal_id: str) -> dict[str, Any] | None:
    with _connect(url, schema) as conn:
        if not _has_marker(conn, schema):
            raise StoreError("schema missing or not created by this application")
        return conn.execute(
            """SELECT proposal_id, request_id, customer_id, order_id, line_id, remedy, amount_cents, paid_cents,
                      original_sku, replacement_sku, status
               FROM proposals WHERE proposal_id = %s""", (proposal_id,)).fetchone()


def get_request(url: str, schema: str, request_id: str) -> dict[str, Any] | None:
    with _connect(url, schema) as conn:
        if not _has_marker(conn, schema):
            raise StoreError("schema missing or not created by this application")
        return conn.execute("SELECT customer_id, result FROM idempotency WHERE request_id = %s",
                            (request_id,)).fetchone()
