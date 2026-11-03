"""Read contract fixtures from exact bytes and hand out isolated state copies."""
from __future__ import annotations

import hashlib
from pathlib import Path

from retail_domain.models import FixtureRoot

def load_fixtures(path: Path | str) -> tuple[FixtureRoot, str]:
    """Parse a fixture file; return it with the SHA-256 of the exact input bytes."""
    raw = Path(path).read_bytes()
    return FixtureRoot.model_validate_json(raw), hashlib.sha256(raw).hexdigest()
