"""Durable demo-session registry in its own PostgreSQL schema (default `retail_ui_registry`).

Stores only what the server needs to stay authoritative across requests and
restarts: token hashes (never tokens), trusted customer/request bindings, turn
counters, per-turn audit event ranges and stored responses for idempotent
client actions. Business state lives in each session's own `retail_ui_<id>` schema.
"""
from __future__ import annotations

import json
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.db.connection import check_schema, check_url
from app.db.schema import SCHEMA_MARKER

DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id         TEXT PRIMARY KEY,
    token_sha256       TEXT NOT NULL UNIQUE,
    scenario_id        TEXT NOT NULL,
    customer_id        TEXT NOT NULL,
    active_request_id  TEXT NOT NULL,
    turn_number        INTEGER NOT NULL DEFAULT 0 CHECK (turn_number >= 0),
    business_schema    TEXT NOT NULL,
    checkpoint_schema  TEXT NOT NULL,
    status             TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'ended')),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS history (
    session_id      TEXT NOT NULL REFERENCES sessions (session_id),
    seq             INTEGER NOT NULL,
    kind            TEXT NOT NULL CHECK (kind IN ('message', 'confirm', 'cancel', 'new_request')),
    turn_number     INTEGER,
    request_id      TEXT NOT NULL,
    message         TEXT,
    decision        TEXT,
    response        TEXT,
    first_event_id  BIGINT,
    last_event_id   BIGINT,
    elapsed_ms      DOUBLE PRECISION,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, seq)
);
CREATE TABLE IF NOT EXISTS actions (
    session_id        TEXT NOT NULL REFERENCES sessions (session_id),
    client_action_id  TEXT NOT NULL,
    status_code       INTEGER NOT NULL,
    response_json     JSONB NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, client_action_id)
);
-- Sessions on the persistent store share one business schema.
ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_business_schema_key;
-- So do their checkpoints.
ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_checkpoint_schema_key;
-- An uploaded case exists nowhere else, so its faults are kept here to survive a restart.
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS case_faults JSONB;
"""


class SessionBusy(Exception):
    """Another request currently holds this session's lock."""


class Registry:
    def __init__(self, url: str, schema: str, *, require_test_db: bool = False) -> None:
        check_url(url, require_test_db=require_test_db)
        check_schema(schema)
        self.url, self.schema = url, schema

    def connect(self) -> psycopg.Connection:
        return psycopg.connect(self.url, autocommit=True, row_factory=dict_row,
                               options=f"-c search_path={self.schema}")

    def migrate(self) -> None:
        ident = sql.Identifier(self.schema)
        with psycopg.connect(self.url, autocommit=True) as conn, conn.transaction():
            conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(ident))
            conn.execute(sql.SQL("COMMENT ON SCHEMA {} IS {}").format(ident, sql.Literal(SCHEMA_MARKER)))
        with self.connect() as conn, conn.transaction():
            conn.execute(DDL)

    def drop(self) -> None:
        """Drop the registry schema (tests only); refuses schemas without the app marker."""
        with self.connect() as conn:
            row = conn.execute("SELECT obj_description(oid, 'pg_namespace') AS m FROM pg_namespace WHERE nspname=%s",
                               (self.schema,)).fetchone()
            if row is None:
                return
            if row["m"] != SCHEMA_MARKER:
                raise RuntimeError(f"schema {self.schema} is not owned by this app")
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.schema)))

    # --- sessions -----------------------------------------------------------------------

    def create(self, **fields: Any) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO sessions (session_id, token_sha256, scenario_id, customer_id, active_request_id,
                                         business_schema, checkpoint_schema, case_faults)
                   VALUES (%(session_id)s, %(token_sha256)s, %(scenario_id)s, %(customer_id)s,
                           %(active_request_id)s, %(business_schema)s, %(checkpoint_schema)s, %(case_faults)s)""",
                {"case_faults": None, **fields})

    def by_token(self, token_sha256: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM sessions WHERE token_sha256 = %s AND status = 'active'",
                                (token_sha256,)).fetchone()

    def by_id(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM sessions WHERE session_id = %s", (session_id,)).fetchone()

    def lock(self, conn: psycopg.Connection, session_id: str) -> dict[str, Any]:
        """Row-lock the session for the rest of `conn`'s transaction; fail fast if already locked."""
        try:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = %s AND status = 'active' FOR UPDATE NOWAIT",
                               (session_id,)).fetchone()
        except psycopg.errors.LockNotAvailable as error:
            raise SessionBusy(session_id) from error
        if row is None:
            raise LookupError(session_id)
        return row

    @staticmethod
    def update(conn: psycopg.Connection, session_id: str, **fields: Any) -> None:
        sets = sql.SQL(", ").join(sql.SQL("{} = {}").format(sql.Identifier(k), sql.Placeholder(k)) for k in fields)
        conn.execute(sql.SQL("UPDATE sessions SET {}, last_active_at = now() WHERE session_id = %(sid)s")
                     .format(sets), {**fields, "sid": session_id})

    def expired(self, older_than_hours: float) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM sessions WHERE status = 'active' AND last_active_at < "
                                "now() - make_interval(secs => %s)", (older_than_hours * 3600,)).fetchall()

    def mark_ended(self, session_id: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE sessions SET status = 'ended', last_active_at = now() WHERE session_id = %s",
                         (session_id,))

    # --- history and idempotent actions ------------------------------------------------------

    @staticmethod
    def add_history(conn: psycopg.Connection, session_id: str, **fields: Any) -> int:
        seq = conn.execute("SELECT coalesce(max(seq), 0) + 1 AS n FROM history WHERE session_id = %s",
                           (session_id,)).fetchone()["n"]
        conn.execute(
            """INSERT INTO history (session_id, seq, kind, turn_number, request_id, message, decision, response,
                                    first_event_id, last_event_id, elapsed_ms)
               VALUES (%(sid)s, %(seq)s, %(kind)s, %(turn_number)s, %(request_id)s, %(message)s, %(decision)s,
                       %(response)s, %(first_event_id)s, %(last_event_id)s, %(elapsed_ms)s)""",
            {"turn_number": None, "message": None, "decision": None, "response": None, "first_event_id": None,
             "last_event_id": None, "elapsed_ms": None, **fields, "sid": session_id, "seq": seq})
        return seq

    def history(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM history WHERE session_id = %s ORDER BY seq", (session_id,)).fetchall()

    @staticmethod
    def stored_action(conn: psycopg.Connection, session_id: str, client_action_id: str) -> tuple[int, Any] | None:
        row = conn.execute("SELECT status_code, response_json FROM actions WHERE session_id = %s "
                           "AND client_action_id = %s", (session_id, client_action_id)).fetchone()
        return None if row is None else (row["status_code"], row["response_json"])

    @staticmethod
    def store_action(conn: psycopg.Connection, session_id: str, client_action_id: str, status_code: int,
                     response: Any) -> None:
        conn.execute("INSERT INTO actions (session_id, client_action_id, status_code, response_json) "
                     "VALUES (%s, %s, %s, %s)", (session_id, client_action_id, status_code,
                                                 Jsonb(json.loads(json.dumps(response)))))
