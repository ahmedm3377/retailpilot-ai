"""Contract faults, applied at the business-tool boundary (never via prompts).

- stock_change: overwrite one SKU's stock immediately before a 1-based turn.
  It is an external event and persists even if a later exchange is refused.
- timeout_after_commit: the first *successful* commit stays durable, then
  PostCommitTimeout is raised instead of returning the result.
"""
from __future__ import annotations

from typing import Any, Sequence

from app.db import repositories as repo
from app.db.fixture_state import RetailStore
from retail_domain.models import Fault, StockChangeFault, TimeoutAfterCommitFault
from app.domain.outcomes import CommitResult, PostCommitTimeout

EXTERNAL_ACTOR = "external"


class FaultError(ValueError):
    """A fault could not be applied as specified."""


class FaultInjector:
    """Per-run fault state; create one per case run."""

    def __init__(self, faults: Sequence[Fault] = ()) -> None:
        self._stock_changes = [f for f in faults if isinstance(f, StockChangeFault)]
        self._applied: set[int] = set()
        self._timeout_pending = any(isinstance(f, TimeoutAfterCommitFault) for f in faults)

    @classmethod
    def restore(cls, faults: Sequence[Fault], audit_rows: Sequence[dict[str, Any]]) -> "FaultInjector":
        """Rebuild per-run fault state from the persisted audit log (e.g. after an API restart).

        A stock_change already logged as an external event is marked applied, and a
        logged timeout_after_commit is marked consumed, so neither ever fires twice.
        """
        injector = cls(faults)
        external = [r for r in audit_rows if r.get("actor") == EXTERNAL_ACTOR]
        for index, fault in enumerate(injector._stock_changes):
            if any(r.get("tool") == "stock_change" and (r.get("arguments") or {}).get("sku") == fault.sku
                   and (r.get("arguments") or {}).get("before_turn") == fault.before_turn for r in external):
                injector._applied.add(index)
        if any(r.get("tool") == "timeout_after_commit" for r in external):
            injector._timeout_pending = False
        return injector

    @property
    def timeout_pending(self) -> bool:
        return self._timeout_pending

    def apply_before_turn(self, store: RetailStore, turn_number: int) -> list[dict[str, Any]]:
        """Apply each stock_change scheduled before `turn_number` (1-based), at most once."""
        events = []
        for index, fault in enumerate(self._stock_changes):
            if fault.before_turn != turn_number or index in self._applied:
                continue
            if fault.new_stock < 0:
                raise FaultError(f"stock_change for {fault.sku}: new_stock must be >= 0")
            arguments = {"sku": fault.sku, "new_stock": fault.new_stock, "before_turn": turn_number}
            with store.connect() as conn, conn.transaction():
                change = repo.set_stock(conn, fault.sku, fault.new_stock)
                if change is None:
                    raise FaultError(f"stock_change for unknown SKU {fault.sku!r}")
                effect = {"type": "stock_set", "sku": fault.sku, **change}
                repo.append_audit(conn, actor=EXTERNAL_ACTOR, tool="stock_change", arguments=arguments,
                                  outcome="applied", effects=[effect])
            self._applied.add(index)
            events.append(effect)
        return events

    def after_commit(self, store: RetailStore, result: CommitResult) -> None:
        """Called only after a successful COMMIT; raises the one-shot timeout if configured."""
        if not self._timeout_pending:
            return
        self._timeout_pending = False
        with store.connect() as conn:
            repo.append_audit(conn, actor=EXTERNAL_ACTOR, tool="timeout_after_commit",
                              arguments={"request_id": result.request_id, "proposal_id": result.proposal_id},
                              outcome="raised", error="simulated timeout after durable commit")
        raise PostCommitTimeout(result.request_id, result.proposal_id)
