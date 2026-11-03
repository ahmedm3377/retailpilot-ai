"""Source and configuration identity for a benchmark variant.

source_id ties results to code: git HEAD, whether the tree is dirty, and hashes
of the working-tree diff and of untracked source files. config_id ties results
to runtime settings: decoding, guardrail thresholds and the agent prompt texts.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.agents import customer_agent, operations_agent
from app.agents.settings import AgentSettings

ROOT = Path(__file__).resolve().parents[2]



def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True, timeout=30).stdout


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_identity() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD").decode().strip()
    status = _git("status", "--porcelain", "--untracked-files=all").decode()
    relevant = [line for line in status.splitlines()
                if not line[3:].startswith(("evals/evidence/", ".claude/", "frontend/test-results/"))]
    untracked = sorted(line[3:] for line in relevant if line.startswith("??"))
    untracked_hash = hashlib.sha256()
    for rel in untracked:
        path = ROOT / rel
        if rel.startswith(("backend/", "evals/")) and path.is_file():
            untracked_hash.update(rel.encode() + b"\0" + path.read_bytes())
    diff = _git("diff", "HEAD", "--binary", "--", ".", ":(exclude)evals/evidence", ":(exclude).claude")
    dirty = bool(relevant)
    short = head[:10] + ("+" + _sha(diff + untracked_hash.digest())[:8] if dirty else "")
    return {"source_id": short, "head": head, "dirty": dirty, "diff_sha256": _sha(diff),
            "untracked_source_files": [u for u in untracked if u.startswith(("backend/", "evals/"))],
            "untracked_sha256": untracked_hash.hexdigest(), "changed_files": [line[3:] for line in relevant]}


def config_identity(settings: AgentSettings) -> dict[str, Any]:
    prompts = {"customer.handoff": customer_agent.HANDOFF_PROMPT, "customer.explain": customer_agent.EXPLAIN_PROMPT,
               "operations": operations_agent.PROMPT}
    from app.db.connection import pooling_enabled
    payload = {"settings": asdict(settings), "db_pool": pooling_enabled(),
               "prompt_sha256": {k: _sha(v.encode()) for k, v in prompts.items()}}
    return {"config_id": _sha(json.dumps(payload, sort_keys=True).encode())[:12], **payload}
