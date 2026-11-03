"""Local retail returns and exchanges desk.

Settings come from the environment. A `.env` file at the repository root fills in whatever the
environment does not set; without one, `.env.example` supplies working defaults, so that
`python adapter.py` runs from a fresh checkout. MCP tool servers set RETAIL_SKIP_DOTENV: they
receive only their scope, and must not read settings they don't need, such as an API key.
"""
from __future__ import annotations

import os
from pathlib import Path

if not os.environ.get("RETAIL_SKIP_DOTENV"):
    from dotenv import load_dotenv

    _root = Path(__file__).resolve().parents[2]
    load_dotenv(_root / ".env" if (_root / ".env").exists() else _root / ".env.example", override=False)
