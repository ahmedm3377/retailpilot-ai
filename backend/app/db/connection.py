"""psycopg 3 connections pinned to one run schema, with fail-closed safety checks.

Connections are autocommit; every write happens inside an explicit
`with conn.transaction():` block so transaction boundaries are visible.
"""
from __future__ import annotations

import os
import re
from urllib.parse import urlparse

import psycopg
from psycopg.rows import dict_row

DEFAULT_DATABASE_URL = "postgresql://retail:retail@127.0.0.1:55432/retail"
DEFAULT_TEST_DATABASE_URL = "postgresql://retail:retail@127.0.0.1:55432/retail_test"

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
SCHEMA_PATTERN = re.compile(r"^retail_[a-z0-9_]{1,50}$")


class UnsafeDatabaseError(RuntimeError):
    """Refused to touch a database or schema outside the allowed local scope."""


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def test_database_url() -> str:
    return os.environ.get("RETAIL_TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL)


def check_url(url: str, *, require_test_db: bool = False) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("postgresql", "postgres"):
        raise UnsafeDatabaseError(f"not a postgresql:// URL: scheme {parsed.scheme!r}")
    if parsed.hostname not in LOCAL_HOSTS:
        raise UnsafeDatabaseError(f"refusing non-local database host {parsed.hostname!r}")
    if require_test_db and not parsed.path.lstrip("/").endswith("_test"):
        raise UnsafeDatabaseError("test database name must end in '_test'")


def check_schema(schema: str) -> None:
    if not SCHEMA_PATTERN.fullmatch(schema):
        raise UnsafeDatabaseError(f"schema {schema!r} must match {SCHEMA_PATTERN.pattern}")


def connect(url: str, schema: str) -> psycopg.Connection:
    """Open an autocommit connection whose search_path is only `schema`."""
    check_url(url)
    check_schema(schema)
    # The schema name is regex-validated above, so it is safe in the options string.
    return psycopg.connect(url, autocommit=True, row_factory=dict_row,
                           options=f"-c search_path={schema}")


# --- Per-schema connection pools -------------------------------------
# A new PostgreSQL connection costs ~22 ms on the reference machine versus ~1 ms for a query on
# an open one, and a task opens 25-35 of them. Pooled connections keep the exact same settings
# (autocommit, dict rows, search_path pinned to one schema); callers keep their explicit
# `with conn.transaction():` blocks. RETAIL_DB_POOL=0 restores one-connection-per-operation.

import atexit  # noqa: E402
import threading  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from typing import Iterator  # noqa: E402

from psycopg_pool import ConnectionPool  # noqa: E402

_POOLS: dict[tuple[str, str], ConnectionPool] = {}
_POOLS_LOCK = threading.Lock()
POOL_MAX_SIZE = 8


def pooling_enabled() -> bool:
    return os.environ.get("RETAIL_DB_POOL", "1").strip().lower() not in ("0", "false", "no", "off")


def _pool(url: str, schema: str) -> ConnectionPool:
    key = (url, schema)
    with _POOLS_LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            pool = ConnectionPool(url, min_size=1, max_size=POOL_MAX_SIZE, timeout=30, open=True,
                                  name=f"pool-{schema}",
                                  kwargs={"autocommit": True, "row_factory": dict_row,
                                          "options": f"-c search_path={schema}"})
            _POOLS[key] = pool
        return pool


@contextmanager
def store_connection(url: str, schema: str) -> Iterator[psycopg.Connection]:
    """A connection pinned to `schema`: pooled (default) or freshly opened (RETAIL_DB_POOL=0)."""
    check_url(url)
    check_schema(schema)
    if not pooling_enabled():
        with connect(url, schema) as conn:
            yield conn
        return
    with _pool(url, schema).connection() as conn:
        yield conn


def close_pool(url: str, schema: str) -> None:
    with _POOLS_LOCK:
        pool = _POOLS.pop((url, schema), None)
    if pool is not None:
        pool.close()


@atexit.register
def _close_all_pools() -> None:
    with _POOLS_LOCK:
        pools = list(_POOLS.values())
        _POOLS.clear()
    for pool in pools:
        pool.close()
