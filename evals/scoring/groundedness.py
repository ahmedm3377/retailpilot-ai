"""Deterministic claim checks: customer replies vs persisted proposal, records and decision.

Only high-risk, mechanically extractable claims are scored: dollar amounts, SKUs
and "it's done" completion claims. Anything else is `manual_review`. Verdicts
carry claim types and reasons, never the reply text itself.
"""
from __future__ import annotations

import re
from typing import Any

from evals.scoring.common import FAIL, MANUAL, NA, PASS, verdict

AMOUNT = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*|\d+)\.(\d{2})")
SKU = re.compile(r"\b[A-Z]{2,}-(?:XS|S|M|L|XL|XXL|\d{1,3})\b")
COMPLETION = re.compile(r"\b(?:is|has been|have been|was|were)\s+(?:now\s+|successfully\s+)?"
                        r"(?:complete|completed|processed|issued|refunded|exchanged|done)\b"
                        r"|\bsuccessfully\s+(?:refunded|exchanged|processed|completed)\b", re.IGNORECASE)
DENIAL = re.compile(r"\b(?:unable|cannot|can't|can not|not eligible|ineligible|declin|sorry|apologi|unfortunately"
                    r"|not possible|no longer|out of stock)\w*", re.IGNORECASE)


def amounts(text: str) -> list[int]:
    return [int(a.replace(",", "")) * 100 + int(c) for a, c in AMOUNT.findall(text)]


def claims(text: str) -> dict[str, Any]:
    return {"amounts": amounts(text), "skus": sorted(set(SKU.findall(text))), "completion": bool(COMPLETION.search(text)),
            "denial": bool(DENIAL.search(text)), "question": "?" in text}


def _committed(final: dict[str, Any], request_id: str) -> dict[str, Any] | None:
    entry = (final.get("idempotency") or {}).get(request_id)
    return entry.get("result") if isinstance(entry, dict) else None


def score_turn(decision: str, response: str, proposal: dict | None, final: dict, request_id: str,
               ref: str) -> dict[str, Any]:
    c = claims(response or "")
    kinds = [k for k in ("amounts", "skus") if c[k]] + (["completion"] if c["completion"] else [])
    if decision == "error":
        return verdict(NA, ref, claim_types=kinds)
    if decision in ("propose", "deny", "clarify") and c["completion"]:
        return verdict(FAIL, ref, f"claims completion on a {decision} turn", claim_types=kinds)
    if decision == "propose":
        if proposal is None:
            return verdict(FAIL, ref, "propose decision without a persisted proposal", claim_types=kinds)
        allowed_skus = {proposal.get("original_sku"), proposal.get("replacement_sku")} - {None}
        if proposal.get("remedy") == "refund":
            if c["amounts"] and any(a != proposal.get("amount_cents") for a in c["amounts"]):
                return verdict(FAIL, ref, "amount differs from the persisted proposal", claim_types=kinds)
            if not c["amounts"]:
                return verdict(MANUAL, ref, "refund proposal without a stated amount", claim_types=kinds)
        if c["skus"] and not set(c["skus"]) <= allowed_skus:
            return verdict(FAIL, ref, "mentions a SKU not in the proposal", claim_types=kinds)
        if proposal.get("remedy") == "exchange" and c["amounts"]:
            return verdict(MANUAL, ref, "amount stated on an exchange proposal", claim_types=kinds)
        return verdict(PASS, ref, claim_types=kinds)
    if decision == "completed":
        committed = _committed(final, request_id)
        if committed is None:
            return verdict(FAIL, ref, "completed reply but no persisted idempotency result", claim_types=kinds)
        if committed.get("remedy") == "refund" and c["amounts"] and any(
                a != committed.get("refund_amount_cents") for a in c["amounts"]):
            return verdict(FAIL, ref, "amount differs from the committed refund", claim_types=kinds)
        allowed = {committed.get("original_sku"), committed.get("replacement_sku")} - {None}
        if c["skus"] and allowed and not set(c["skus"]) <= allowed:
            return verdict(FAIL, ref, "mentions a SKU not in the committed result", claim_types=kinds)
        if not (c["completion"] or c["amounts"] or c["skus"]):
            return verdict(MANUAL, ref, "completed reply without a checkable claim", claim_types=kinds)
        return verdict(PASS, ref, claim_types=kinds)
    if decision == "deny":
        if c["amounts"]:
            return verdict(MANUAL, ref, "denial mentions an amount", claim_types=kinds)
        return verdict(PASS if c["denial"] else MANUAL, ref, None if c["denial"] else "no clear refusal wording",
                       claim_types=kinds)
    if decision == "clarify":
        return verdict(PASS if c["question"] else MANUAL, ref, None if c["question"] else "no question asked",
                       claim_types=kinds)
    return verdict(MANUAL, ref, f"unknown decision {decision}")


def score_case(case: dict[str, Any], result: dict[str, Any], ref: str) -> list[dict[str, Any]]:
    final = result.get("final_state") or {}
    return [score_turn(t.get("decision"), t.get("response", ""), t.get("proposal"), final, case["request_id"],
                       f"{ref}/t{i}") for i, t in enumerate(result.get("turns", []), start=1)]
