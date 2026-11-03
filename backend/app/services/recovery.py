"""Read persisted request outcomes; used after a timeout instead of re-executing."""
from __future__ import annotations

from typing import Any

from app.db import repositories as repo
from app.db.fixture_state import RetailStore
from app.domain.outcomes import CommitResult, RequestStatus


def stored_result(row: dict[str, Any]) -> CommitResult:
    return CommitResult.model_validate(row["result"])


def reconcile_request(store: RetailStore, authenticated_customer_id: str, request_id: str) -> RequestStatus:
    """Status of `request_id` from a fresh connection. Never commits anything.

    Another customer's request reports `conflict` without its details.
    """
    with store.connect() as conn:
        row = repo.get_idempotency(conn, request_id)
    if row is None:
        return RequestStatus(state="not_found", request_id=request_id)
    if row["customer_id"] != authenticated_customer_id:
        return RequestStatus(state="conflict", request_id=request_id)
    return RequestStatus(state="completed", request_id=request_id, result=stored_result(row))
