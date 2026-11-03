"""Contract output mapping and atomic, strict JSON writing."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from app.agents.schemas import TurnResult
from app.services.audit import jsonable

AUDIT_FIELDS = ("event_id", "actor", "tool", "arguments", "outcome", "result", "error", "effects")


def turn_output(result: TurnResult) -> dict[str, Any]:
    """The contract's per-turn object: decision, response, persisted proposal or null."""
    return {"decision": result.decision, "response": result.response,
            "proposal": jsonable(result.proposal) if result.proposal is not None else None}


def audit_entries(rows: list[dict[str, Any]], turn_bounds: list[tuple[int, int, int]]) -> list[dict[str, Any]]:
    """Persisted audit_events rows, tagged with the turn whose event_id range contains them.

    turn_bounds: (turn_number, first_event_id_exclusive, last_event_id_inclusive).
    Rows outside every turn (e.g. none today) keep turn=None.
    """
    out = []
    for row in rows:
        turn = next((n for n, lo, hi in turn_bounds if lo < row["event_id"] <= hi), None)
        entry = {"turn": turn, "agent": row["actor"], **{k: jsonable(row[k]) for k in AUDIT_FIELDS}}
        out.append(entry)
    return out


def dumps(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_atomic(path: Path, document: dict[str, Any]) -> None:
    """Write JSON to a temp file in the same directory, fsync, then replace."""
    text = dumps(document)  # serialize first: a failure never leaves a partial file
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
