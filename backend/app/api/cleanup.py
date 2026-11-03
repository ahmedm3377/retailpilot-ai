"""Drop expired demo sessions' own schemas (never automatic).

    uv run python -m app.api.cleanup --older-than-hours 24 [--dry-run]

Only sessions recorded in the registry are touched; each schema must carry the
app's ownership marker (RetailStore.drop_schema refuses anything else). The persistent
store and the shared checkpoint schema stay; a session's rows in the latter are deleted.
"""
from __future__ import annotations

import argparse
import sys

from app.api.registry import Registry
from app.api.service import SHARED_CHECKPOINTS, release_session_data
from app.db.connection import database_url
from app.db.demo_store import store_schema


def clean(registry: Registry, url: str, store: str, shared_checkpoints: str, older_than_hours: float,
          *, dry_run: bool = False) -> int:
    """End expired sessions and remove what is only theirs (see release_session_data)."""
    expired = registry.expired(older_than_hours)
    for session in expired:
        print(f"{session['session_id']}: {session['business_schema']}, {session['checkpoint_schema']}")
        if not dry_run:
            release_session_data(url, session, store, shared_checkpoints)
            registry.mark_ended(session["session_id"])
    return len(expired)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--older-than-hours", type=float, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--registry-schema", default="retail_ui_registry")
    args = parser.parse_args(argv)
    count = clean(Registry(database_url(), args.registry_schema), database_url(), store_schema(), SHARED_CHECKPOINTS,
                  args.older_than_hours, dry_run=args.dry_run)
    print(f"{count} session(s) {'would be ' if args.dry_run else ''}cleaned up")
    return 0


if __name__ == "__main__":
    sys.exit(main())
