from __future__ import annotations

import os
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest

from app.db.connection import check_url, test_database_url
from app.db.fixture_state import RetailStore
from retail_domain.models import State
from app.fixtures.loader import load_fixtures
from app.services.operations import RetailOperations

PUBLIC_CASES = Path(__file__).resolve().parents[2] / "candidate_assets" / "public_cases.json"


@pytest.fixture(scope="session")
def public_fixtures():
    return load_fixtures(PUBLIC_CASES)


@pytest.fixture
def public_cases(public_fixtures):
    """Public cases keyed by case_id; use only to look up data in assertions."""
    root, _ = public_fixtures
    return {case.case_id: case for case in root.cases}


@pytest.fixture(scope="session")
def pg_url() -> str:
    """The local test database; fails (never skips or falls back) when unavailable."""
    url = test_database_url()
    check_url(url, require_test_db=True)
    try:
        psycopg.connect(url, connect_timeout=3).close()
    except psycopg.OperationalError as error:
        pytest.fail(f"PostgreSQL test database unreachable at {url}. "
                    f"Start it with `docker compose up -d postgres`.\n{error}")
    return url


@pytest.fixture
def pg_store(pg_url):
    """A migrated, empty schema unique to this test; dropped afterwards."""
    store = RetailStore(pg_url, f"retail_t_{uuid.uuid4().hex[:12]}", require_test_db=True)
    store.create_schema()
    yield store
    store.drop_schema()


@pytest.fixture
def ops(pg_store):
    return RetailOperations(pg_store, actor="test_agent")


@pytest.fixture
def load(pg_store, public_cases):
    """Reset the test schema from a public case ID or a State; returns the State."""
    def _load(case_or_state: str | State) -> State:
        state = public_cases[case_or_state].initial_state if isinstance(case_or_state, str) else case_or_state
        pg_store.reset_from_initial_state(state)
        return state
    return _load


@pytest.fixture(scope="session")
def agent_settings():
    from app.agents.settings import AgentSettings
    return AgentSettings.from_env()  # the model named in .env; FakeModel tests never call it


@pytest.fixture(scope="session")
def guards(agent_settings):
    """The real local Guardrails service (loaded once; ~20-35 s)."""
    from app.guardrails.service import GuardrailService
    service = GuardrailService(agent_settings)
    assert service.unavailable == {}, service.unavailable
    return service


@pytest.fixture
def make_runtime(pg_store, pg_url, guards, agent_settings):
    """Factory for AgentRuntime on the test store; checkpoint schemas are dropped afterwards."""
    from app.services.agent_runtime import AgentRuntime
    created = []

    def _make(model, checkpoint_schema: str | None = None, **options):
        schema = checkpoint_schema or f"retail_t_ckpt_{uuid.uuid4().hex[:10]}"
        runtime = AgentRuntime(pg_store, model, guards, agent_settings, pg_url, schema, **options)
        created.append((runtime, schema))
        return runtime

    yield _make
    for runtime, schema in created:
        runtime.close()
    with psycopg.connect(pg_url, autocommit=True) as conn:
        for _, schema in created:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


def count(store: RetailStore, table: str) -> int:
    with store.connect() as conn:
        return conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()["n"]


def product_stock(store: RetailStore, sku: str) -> int:
    return next(p["stock"] for p in store.export_final_state()["products"] if p["sku"] == sku)


def line_returned(store: RetailStore, order_id: str, line_id: str) -> bool:
    order = next(o for o in store.export_final_state()["orders"] if o["order_id"] == order_id)
    return next(ln["returned"] for ln in order["lines"] if ln["line_id"] == line_id)


def make_line(line_id: str = "LN-1", sku: str = "WIDGET-A", paid_cents: int = 1234,
              **overrides: Any) -> dict[str, Any]:
    return {"line_id": line_id, "sku": sku, "paid_cents": paid_cents, "unopened": True,
            "final_sale": False, "returned": False, **overrides}


def make_state(*, as_of: date = date(2031, 3, 15), age_days: int = 5, customer_id: str = "CUST-9",
               order_id: str = "ORD-77", status: str = "delivered",
               lines: list[dict[str, Any]] | None = None,
               products: list[dict[str, Any]] | None = None) -> State:
    """Synthetic state with IDs, dates and amounts unrelated to the public fixtures."""
    if products is None:
        products = [
            {"sku": "WIDGET-A", "product_id": "WIDGET", "size": "A", "price_cents": 2500, "stock": 1},
            {"sku": "WIDGET-B", "product_id": "WIDGET", "size": "B", "price_cents": 2500, "stock": 7},
            {"sku": "WIDGET-C", "product_id": "WIDGET", "size": "C", "price_cents": 2600, "stock": 3},
            {"sku": "WIDGET-D", "product_id": "WIDGET", "size": "D", "price_cents": 2500, "stock": 0},
            {"sku": "GADGET-B", "product_id": "GADGET", "size": "B", "price_cents": 2500, "stock": 9},
        ]
    return State.model_validate({
        "as_of": as_of.isoformat(),
        "currency": "USD",
        "products": products,
        "orders": [{
            "order_id": order_id,
            "customer_id": customer_id,
            "status": status,
            "delivered_on": (as_of - timedelta(days=age_days)).isoformat(),
            "lines": lines if lines is not None else [make_line()],
        }],
        "returns": [], "refunds": [], "exchanges": [], "quarantine": [], "idempotency": {},
    })


@pytest.fixture(scope="session", autouse=True)
def mcp_server():
    """One real standalone server for the session, as in production: the app has no other read tools.
    Yields (url, token), which are also set for code that reads them from the environment."""
    from tests.mcp_support import running_server
    with running_server(test_database_url()) as (server_url, token, _process):
        os.environ["MCP_SERVER_URL"] = server_url
        os.environ["MCP_AUTH_TOKEN"] = token
        yield server_url, token
