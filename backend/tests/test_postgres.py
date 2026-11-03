from __future__ import annotations

import psycopg
import pytest
from psycopg import errors

from app.db import repositories
from app.db.connection import UnsafeDatabaseError
from app.db.fixture_state import RetailStore
from app.services.transactions import commit_confirmed_proposal
from tests.conftest import count

pytestmark = pytest.mark.postgres


def test_migrations_run_on_empty_schema_and_are_repeatable(pg_url):
    store = RetailStore(pg_url, "retail_t_migrate_check", require_test_db=True)
    store.drop_schema()
    try:
        assert store.create_schema() == [1]
        assert store.create_schema() == []
        with store.connect() as conn:
            tables = {r["table_name"] for r in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s",
                (store.schema,))}
        assert {"products", "orders", "order_lines", "proposals", "returns", "refunds", "exchanges",
                "quarantine", "idempotency", "audit_events", "schema_migrations"} <= tables
    finally:
        store.drop_schema()


@pytest.mark.parametrize("url, schema", [
    ("postgresql://retail:retail@db.example.com:5432/retail_test", "retail_t_x"),
    ("mysql://retail:retail@127.0.0.1:55432/retail_test", "retail_t_x"),
    ("postgresql://retail:retail@127.0.0.1:55432/retail", "retail_t_x"),
    ("postgresql://retail:retail@127.0.0.1:55432/retail_test", "public"),
    ("postgresql://retail:retail@127.0.0.1:55432/retail_test", "retail_x; DROP TABLE y"),
])
def test_unsafe_targets_are_refused(url, schema):
    with pytest.raises(UnsafeDatabaseError):
        RetailStore(url, schema, require_test_db=True)


def test_drop_refuses_schema_it_did_not_create(pg_url):
    with psycopg.connect(pg_url, autocommit=True) as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS retail_t_foreign")
    try:
        with pytest.raises(UnsafeDatabaseError):
            RetailStore(pg_url, "retail_t_foreign", require_test_db=True).drop_schema()
    finally:
        with psycopg.connect(pg_url, autocommit=True) as conn:
            conn.execute("DROP SCHEMA retail_t_foreign")


def test_reset_isolates_sequential_cases(pg_store, ops, load, public_cases):
    load("public_02_exchange")
    proposal = ops.create_proposal("C100", "REQ-a", "O100", "L1", "exchange", "TEE-L").proposal
    assert commit_confirmed_proposal(pg_store, "C100", "REQ-a", proposal.proposal_id).status == "completed"

    load("public_01_refund_boundary")
    assert pg_store.export_final_state() == public_cases["public_01_refund_boundary"].initial_state.model_dump(mode="json")
    assert count(pg_store, "proposals") == 0 and count(pg_store, "audit_events") == 0
    # ID sequences restart, so each case run is reproducible.
    again = ops.create_proposal("C100", "REQ-b", "O100", "L1", "refund").proposal
    assert again.proposal_id == "PROP-1"


def test_failure_before_commit_rolls_back_every_table(pg_store, ops, load, monkeypatch):
    load("public_02_exchange")
    proposal = ops.create_proposal("C100", "REQ-x", "O100", "L1", "exchange", "TEE-L").proposal
    before = pg_store.export_final_state()

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(repositories, "insert_idempotency", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        commit_confirmed_proposal(pg_store, "C100", "REQ-x", proposal.proposal_id)

    assert pg_store.export_final_state() == before  # stock, returned flag, records, idempotency
    with pg_store.connect() as conn:
        assert repositories.get_proposal(conn, proposal.proposal_id)["status"] == "pending"


def test_proposals_are_immutable(pg_store, ops, load):
    load("public_01_refund_boundary")
    proposal = ops.create_proposal("C100", "REQ-i", "O100", "L1", "refund").proposal
    with pg_store.connect() as conn:
        with pytest.raises(errors.RaiseException, match="immutable"):
            conn.execute("UPDATE proposals SET amount_cents = 1 WHERE proposal_id = %s", (proposal.proposal_id,))
        with pytest.raises(errors.RaiseException, match="cannot be deleted"):
            conn.execute("DELETE FROM proposals WHERE proposal_id = %s", (proposal.proposal_id,))


def test_audit_log_is_append_only(pg_store, ops, load):
    load("public_01_refund_boundary")
    ops.record_refusal("C100", "REQ-1", "O100", "L1", "refund", "NOT_ELIGIBLE")
    with pg_store.connect() as conn:
        with pytest.raises(errors.RaiseException, match="append-only"):
            conn.execute("UPDATE audit_events SET outcome = 'x'")
        with pytest.raises(errors.RaiseException, match="append-only"):
            conn.execute("DELETE FROM audit_events")


def test_database_rejects_negative_stock(pg_store, load):
    load("public_01_refund_boundary")
    with pg_store.connect() as conn:
        with pytest.raises(errors.CheckViolation):
            conn.execute("UPDATE products SET stock = -1 WHERE sku = 'TEE-M'")
