"""Parameterized SQL for the run schema. No business decisions live here.

Functions take an open connection; callers own transaction boundaries. The
`*_for_update` variants must be called inside a transaction.
"""
from __future__ import annotations

from typing import Any

import psycopg
from psycopg.types.json import Jsonb

Row = dict[str, Any]


# --- catalog and orders -------------------------------------------------------

def get_meta(conn: psycopg.Connection) -> Row | None:
    return conn.execute("SELECT as_of, currency, extra FROM fixture_meta").fetchone()


def list_products(conn: psycopg.Connection) -> list[Row]:
    return conn.execute("SELECT * FROM products ORDER BY position").fetchall()


def get_product(conn: psycopg.Connection, sku: str, *, for_update: bool = False) -> Row | None:
    lock = " FOR UPDATE" if for_update else ""
    return conn.execute("SELECT * FROM products WHERE sku = %s" + lock, (sku,)).fetchone()


def list_orders(conn: psycopg.Connection, customer_id: str | None = None) -> list[Row]:
    if customer_id is None:
        return conn.execute("SELECT * FROM orders ORDER BY position").fetchall()
    return conn.execute("SELECT * FROM orders WHERE customer_id = %s ORDER BY position",
                        (customer_id,)).fetchall()


def get_order(conn: psycopg.Connection, order_id: str) -> Row | None:
    return conn.execute("SELECT * FROM orders WHERE order_id = %s", (order_id,)).fetchone()


def list_lines(conn: psycopg.Connection, order_ids: list[str] | None = None) -> list[Row]:
    if order_ids is None:
        return conn.execute("SELECT * FROM order_lines ORDER BY order_id, position").fetchall()
    return conn.execute("SELECT * FROM order_lines WHERE order_id = ANY(%s) ORDER BY order_id, position",
                        (order_ids,)).fetchall()


def get_line_for_update(conn: psycopg.Connection, order_id: str, line_id: str) -> Row | None:
    return conn.execute("SELECT * FROM order_lines WHERE order_id = %s AND line_id = %s FOR UPDATE",
                        (order_id, line_id)).fetchone()


def set_stock(conn: psycopg.Connection, sku: str, new_stock: int) -> Row | None:
    """Overwrite stock inside a transaction; returns old/new stock, or None if the SKU is unknown."""
    old = get_product(conn, sku, for_update=True)
    if old is None:
        return None
    conn.execute("UPDATE products SET stock = %s WHERE sku = %s", (new_stock, sku))
    return {"old_stock": old["stock"], "new_stock": new_stock}


def mark_line_opened(conn: psycopg.Connection, customer_id: str, order_id: str, line_id: str) -> bool:
    """Record that this customer's own line was opened; returns whether anything changed."""
    return conn.execute(
        """UPDATE order_lines SET unopened = FALSE
           WHERE order_id = %s AND line_id = %s AND unopened AND NOT returned
             AND order_id IN (SELECT order_id FROM orders WHERE customer_id = %s)
           RETURNING line_id""", (order_id, line_id, customer_id)).fetchone() is not None


def decrement_stock(conn: psycopg.Connection, sku: str) -> int | None:
    """Take one unit if any is left; returns the new stock, or None if none was available."""
    row = conn.execute("UPDATE products SET stock = stock - 1 WHERE sku = %s AND stock > 0 RETURNING stock",
                       (sku,)).fetchone()
    return None if row is None else row["stock"]


def mark_line_returned(conn: psycopg.Connection, order_id: str, line_id: str) -> bool:
    cur = conn.execute("UPDATE order_lines SET returned = TRUE WHERE order_id = %s AND line_id = %s AND NOT returned",
                       (order_id, line_id))
    return cur.rowcount == 1


# --- proposals ----------------------------------------------------------------

def insert_proposal(conn: psycopg.Connection, **fields: Any) -> Row:
    return conn.execute(
        """INSERT INTO proposals (request_id, customer_id, order_id, line_id, remedy,
                                  amount_cents, paid_cents, original_sku, replacement_sku)
           VALUES (%(request_id)s, %(customer_id)s, %(order_id)s, %(line_id)s, %(remedy)s,
                   %(amount_cents)s, %(paid_cents)s, %(original_sku)s, %(replacement_sku)s)
           RETURNING *""", fields).fetchone()


def get_proposal(conn: psycopg.Connection, proposal_id: str, *, for_update: bool = False) -> Row | None:
    lock = " FOR UPDATE" if for_update else ""
    return conn.execute("SELECT * FROM proposals WHERE proposal_id = %s" + lock, (proposal_id,)).fetchone()


def set_proposal_status(conn: psycopg.Connection, proposal_id: str, status: str) -> None:
    conn.execute("UPDATE proposals SET status = %s WHERE proposal_id = %s", (status, proposal_id))


# --- transaction records ------------------------------------------------------

def insert_return(conn: psycopg.Connection, **fields: Any) -> str:
    return conn.execute(
        """INSERT INTO returns (request_id, proposal_id, customer_id, order_id, line_id, sku, remedy)
           VALUES (%(request_id)s, %(proposal_id)s, %(customer_id)s, %(order_id)s, %(line_id)s,
                   %(sku)s, %(remedy)s)
           RETURNING return_id""", fields).fetchone()["return_id"]


def insert_refund(conn: psycopg.Connection, **fields: Any) -> str:
    return conn.execute(
        """INSERT INTO refunds (return_id, request_id, proposal_id, customer_id, order_id, line_id,
                                amount_cents, currency)
           VALUES (%(return_id)s, %(request_id)s, %(proposal_id)s, %(customer_id)s, %(order_id)s,
                   %(line_id)s, %(amount_cents)s, %(currency)s)
           RETURNING refund_id""", fields).fetchone()["refund_id"]


def insert_exchange(conn: psycopg.Connection, **fields: Any) -> str:
    return conn.execute(
        """INSERT INTO exchanges (return_id, request_id, proposal_id, customer_id, order_id, line_id,
                                  original_sku, replacement_sku)
           VALUES (%(return_id)s, %(request_id)s, %(proposal_id)s, %(customer_id)s, %(order_id)s,
                   %(line_id)s, %(original_sku)s, %(replacement_sku)s)
           RETURNING exchange_id""", fields).fetchone()["exchange_id"]


def insert_quarantine(conn: psycopg.Connection, **fields: Any) -> str:
    return conn.execute(
        """INSERT INTO quarantine (return_id, request_id, proposal_id, customer_id, order_id, line_id,
                                   sku, quantity)
           VALUES (%(return_id)s, %(request_id)s, %(proposal_id)s, %(customer_id)s, %(order_id)s,
                   %(line_id)s, %(sku)s, 1)
           RETURNING quarantine_id""", fields).fetchone()["quarantine_id"]


def get_idempotency(conn: psycopg.Connection, request_id: str) -> Row | None:
    return conn.execute("SELECT * FROM idempotency WHERE request_id = %s", (request_id,)).fetchone()


def insert_idempotency(conn: psycopg.Connection, request_id: str, customer_id: str,
                       proposal_id: str, result: dict[str, Any]) -> None:
    conn.execute(
        """INSERT INTO idempotency (request_id, customer_id, proposal_id, status, result)
           VALUES (%s, %s, %s, 'completed', %s)""",
        (request_id, customer_id, proposal_id, Jsonb(result)))


def list_table(conn: psycopg.Connection, table: str, order_by: str) -> list[Row]:
    """Rows of one transaction-record table; names come from internal constants only."""
    return conn.execute(f"SELECT * FROM {table} ORDER BY {order_by}").fetchall()


# --- audit --------------------------------------------------------------------

def append_audit(conn: psycopg.Connection, *, actor: str, tool: str, arguments: dict[str, Any],
                 outcome: str, result: Any = None, error: str | None = None,
                 effects: list[dict[str, Any]] | None = None) -> None:
    conn.execute(
        """INSERT INTO audit_events (actor, tool, arguments, outcome, result, error, effects)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (actor, tool, Jsonb(arguments), outcome, None if result is None else Jsonb(result), error,
         Jsonb(effects or [])))


def list_audit(conn: psycopg.Connection) -> list[Row]:
    return conn.execute("SELECT * FROM audit_events ORDER BY event_id").fetchall()
