"""Builders for API DTOs from trusted persisted data (proposal rows, store state, audit rows)."""
from __future__ import annotations

from typing import Any

from app.agents.messages import money
from app.api.schemas import AuditEventDTO, OrderDTO, OrderLineDTO, ProposalDTO
from app.db.fixture_state import RetailStore
from retail_domain.models import State

ACTOR_LABELS = {
    "customer_resolution_agent": "Customer Resolution Agent",
    "retail_operations_agent": "Retail Operations Agent",
    "trusted_controller": "Trusted controller",
    "guardrails": "Guardrails",
    "external": "External event",
    "ui_controller": "UI controller",
}

READ_TOOLS = {"lookup_customer_orders", "get_order_line", "check_return_eligibility", "get_exchange_options",
              "check_inventory", "get_request_status"}


def proposal_dto(row: dict[str, Any] | None) -> ProposalDTO | None:
    if row is None:
        return None
    return ProposalDTO(proposal_id=row["proposal_id"], remedy=row["remedy"], order_id=row["order_id"],
                       line_id=row["line_id"], original_sku=row["original_sku"],
                       replacement_sku=row["replacement_sku"], amount_cents=row["amount_cents"],
                       status=row["status"])


def latest_proposal_row(store: RetailStore, request_id: str) -> dict[str, Any] | None:
    with store.connect() as conn:
        return conn.execute("SELECT * FROM proposals WHERE request_id = %s "
                            "ORDER BY length(proposal_id) DESC, proposal_id DESC LIMIT 1", (request_id,)).fetchone()


def _line_outcome(refund: dict[str, Any] | None, exchange: dict[str, Any] | None) -> dict[str, Any]:
    """What was committed for a line, from the transaction records."""
    if refund:
        return {"outcome": "refund", "refund_cents": refund["amount_cents"]}
    if exchange:
        return {"outcome": "exchange", "replacement_sku": exchange["replacement_sku"]}
    return {}


def orders_dto(state: State, customer_id: str) -> list[OrderDTO]:
    products = {p.sku: p for p in state.products}
    refunds = {(r["order_id"], r["line_id"]): r for r in state.refunds}
    exchanges = {(x["order_id"], x["line_id"]): x for x in state.exchanges}
    out = []
    for order in state.orders:
        if order.customer_id != customer_id:
            continue
        out.append(OrderDTO(
            order_id=order.order_id, status=order.status,
            delivered_on=order.delivered_on.isoformat() if order.delivered_on else None,
            lines=[OrderLineDTO(line_id=ln.line_id, sku=ln.sku,
                                product_id=products[ln.sku].product_id if ln.sku in products else None,
                                size=products[ln.sku].size if ln.sku in products else None,
                                paid_cents=ln.paid_cents, returned=ln.returned,
                                **_line_outcome(refunds.get((order.order_id, ln.line_id)),
                                                exchanges.get((order.order_id, ln.line_id)))) for ln in order.lines]))
    return out


def _args(args: dict[str, Any], *keys: str) -> str:
    return ", ".join(str(args[k]) for k in keys if args.get(k) is not None)


def audit_event_dto(entry: dict[str, Any]) -> AuditEventDTO:
    """One sanitized line per persisted audit event (no prompts, payloads or raw blocked text)."""
    actor, tool, outcome = entry["actor"], entry["tool"], entry["outcome"]
    args = entry.get("arguments") or {}
    result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
    kind, severity = "other", "info"
    if tool == "llm_call":
        kind = "llm"
        tokens = result.get("completion_tokens")
        summary = (f"Model call ({result.get('model', 'local model')}) · {result.get('latency_ms', 0) / 1000:.1f} s"
                   + (f" · {tokens} output tokens" if tokens is not None else ""))
        if outcome != "ok":
            severity, summary = "error", f"Model call failed ({entry.get('error') or 'unavailable'})"
    elif actor == "guardrails":
        kind = "guardrail"
        gate, validator = (tool.split(":") + ["", ""])[1:3]
        summary = f"{gate} check · {validator}: {outcome}" + (f" ({result.get('reason_code')})" if result else "")
        severity = "info" if outcome in ("passed", "sanitized") else "warning"
        if outcome == "validator_unavailable":
            severity = "error"
    elif actor == "external":
        kind, severity = "fault", "warning"
        if tool == "stock_change":
            effect = (entry.get("effects") or [{}])[0]
            summary = (f"Stock of {args.get('sku')} changed {effect.get('old_stock')} → {effect.get('new_stock')} "
                       f"(before turn {args.get('before_turn')})")
        elif tool == "timeout_after_commit":
            summary = "Simulated timeout after the commit was saved; the result is read back, not re-run"
        else:
            summary = tool
    elif tool in READ_TOOLS:
        kind = "tool"
        summary = f"{tool}({_args(args, 'order_id', 'line_id', 'sku')}) → {outcome}"
    elif tool == "agent_handoff":
        kind = "handoff"
        source, target = (ACTOR_LABELS.get(args.get(k), args.get(k)) for k in ("from", "to"))
        summary = f"{source} → {target}: {_args(args, 'remedy', 'order_id', 'line_id', 'requested_size')}"
        if args.get("missing"):
            summary += f"; needs {args['missing']} ({', '.join(args.get('options') or [])})"
    elif tool == "get_business_context":
        kind = "tool"
        summary = ("Business checks read the store through the MCP server" if outcome == "ok"
                   else f"Business checks could not read the store ({outcome})")
    elif tool == "scope_check":
        summary = "Message read as out of scope: answered with the fixed reply, and nothing changed"
    elif tool == "item_condition":
        item = f"line {args.get('line_id')} of order {args.get('order_id')}"
        summary = (f"Customer said {item} was opened; the store now records it as opened"
                   if outcome == "marked_opened" else f"Customer said {item} was opened; the store is unchanged")
    elif tool == "create_proposal":
        kind = "proposal"
        proposal = (result.get("proposal") or {}) if result else {}
        if proposal:
            what = (f"refund {money(proposal.get('amount_cents'))}" if proposal.get("remedy") == "refund"
                    else f"exchange {proposal.get('original_sku')} → {proposal.get('replacement_sku')}")
            summary = f"Proposal {proposal.get('proposal_id')} created: {what} (no business effect yet)"
        else:
            summary = f"No proposal: {outcome} ({result.get('reason_code') if result else ''})"
    elif tool == "reconcile_request":
        kind = "commit"
        if outcome == "completed":
            summary = (f"Recovered after the timeout: read the saved result of {result.get('proposal_id')} "
                       "from the store, nothing re-run")
        else:
            severity, summary = "error", f"Recovery after the timeout found no saved result ({outcome})"
    elif tool in ("commit_confirmed_proposal", "cancel_proposal"):
        kind = "commit"
        if outcome == "committed":
            what = (f"refund {money(result.get('refund_amount_cents'))}" if result.get("remedy") == "refund"
                    else f"exchange {result.get('original_sku')} → {result.get('replacement_sku')}")
            stock = next((e for e in entry.get("effects") or [] if e.get("type") == "stock_decrement"), None)
            if stock:
                what += f" ({stock.get('sku')} stock now {stock.get('new_stock')})"
            summary = f"Committed {args.get('proposal_id')}: {what}"
        elif outcome == "replayed":
            summary = f"Repeat approval of {args.get('proposal_id')}: original result returned, nothing re-run"
        elif tool == "cancel_proposal":
            who = "by the customer" if actor == "ui_controller" else "and replaced by the customer's new request"
            summary = f"Proposal {args.get('proposal_id')} cancelled {who} ({outcome})"
        else:
            severity = "warning"
            summary = f"Commit refused for {args.get('proposal_id')}: {entry.get('error') or outcome}"
    else:
        summary = f"{tool}: {outcome}"
    return AuditEventDTO(event_id=entry["event_id"], turn=entry.get("turn"), actor=actor,
                         actor_label=ACTOR_LABELS.get(actor, actor), kind=kind, summary=summary, outcome=outcome,
                         severity=severity)
