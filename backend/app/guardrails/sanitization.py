"""Deterministic input normalization and safe embedding of untrusted text in prompts.

Sanitization keeps the customer's wording: Unicode NFKC, removal of control and
invisible formatting characters (zero-width, bidi overrides), CRLF -> LF.
It never HTML-escapes (that would corrupt SKU-style IDs); escaping belongs at
HTML sinks. Untrusted text enters prompts only as a JSON string literal.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from typing import Any

# Zero-width, bidi embedding/override/isolate, word joiner, BOM.
_INVISIBLE = {0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2028, 0x2029, *range(0x202A, 0x202F),
              *range(0x2060, 0x2065), *range(0x2066, 0x206A), 0xFEFF}


@dataclass(frozen=True)
class SanitizedText:
    text: str
    removed_chars: int
    reason_code: str | None = None  # set when the input is rejected


def sanitize_text(raw: Any, max_chars: int) -> SanitizedText:
    if not isinstance(raw, str):
        return SanitizedText("", 0, "INPUT_NOT_TEXT")
    if len(raw) > max_chars * 4:  # refuse before doing work on huge inputs
        return SanitizedText("", 0, "INPUT_TOO_LONG")
    text = unicodedata.normalize("NFKC", raw.replace("\r\n", "\n").replace("\r", "\n"))
    kept = [ch for ch in text
            if ch in "\n\t" or (unicodedata.category(ch)[0] != "C" and ord(ch) not in _INVISIBLE)]
    cleaned = "".join(kept).strip()
    removed = len(text) - len("".join(kept))
    if not cleaned:
        return SanitizedText("", removed, "INPUT_EMPTY")
    if len(cleaned) > max_chars:
        return SanitizedText("", removed, "INPUT_TOO_LONG")
    return SanitizedText(cleaned, removed)


def quote_untrusted(text: str) -> str:
    """Untrusted text as a JSON string literal: no tag or instruction boundary can be forged."""
    return json.dumps(text, ensure_ascii=False)


def safe_preview(text: str, limit: int = 60) -> str:
    """Short, single-line preview for audit records (never the full untrusted string)."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
