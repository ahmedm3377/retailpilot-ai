"""Audit rows for the read tools (which run on the standalone MCP server) and proposal creation.

Nothing here moves money or stock. `authenticated_customer_id` must come from a
trusted caller (UI session or fixture adapter), never from model output.
Eligibility and amounts come from the policy in retail_domain.policy.
"""
from __future__ import annotations

from typing import Any

from app.db import repositories as repo
from app.db.fixture_state import RetailStore
from retail_domain import policy
from app.domain.outcomes import RequestStatus, WorkflowCode
from app.domain.proposals import Proposal, ProposalOutcome
from app.services.audit import jsonable
from app.services.recovery import reconcile_request

REMEDIES = ("refund", "exchange")


class RetailOperations:
    """Everything here is audit-logged as `actor`."""

    def __init__(self, store: RetailStore, actor: str = "retail_operations") -> None:
        self.store = store
        self.actor = actor

    def _audit(self, tool: str, arguments: dict[str, Any], outcome: str, result: Any) -> None:
        with self.store.connect() as conn:
            repo.append_audit(conn, actor=self.actor, tool=tool, arguments=arguments,
                              outcome=outcome, result=jsonable(result))



    def record_remote_tool(self, tool: str, authenticated_customer_id: str, request_id: str,
                           arguments: dict[str, str], result: Any) -> None:
        """Audit a read tool that ran on the standalone MCP server."""
        identity = {"customer_id": authenticated_customer_id}
        if tool == "get_business_context":  # the outcome only: the result is the whole schema's state
            self._audit(tool, {**identity, "request_id": request_id}, "ok", None)
        elif tool == "lookup_customer_orders":
            self._audit(tool, identity, "ok", {"order_ids": [o["order_id"] for o in result]})
        elif tool == "get_order_line" or tool == "check_return_eligibility":
            self._audit(tool, {**identity, **arguments}, result["reason_code"], result)
        elif tool == "get_exchange_options":
            self._audit(tool, {**identity, **arguments}, "ok", result)
        elif tool == "check_inventory":
            self._audit(tool, arguments, "ok" if result["found"] else "SKU_NOT_FOUND", result)
        else:
            self._audit(tool, {**identity, "request_id": request_id}, result["state"], result)

    def record_refused_tool(self, tool: str, arguments: dict[str, str | None], error: str) -> None:
        """Audit a tool request the agent's registry refused; nothing was called."""
        with self.store.connect() as conn:
            repo.append_audit(conn, actor=self.actor, tool=tool, arguments=arguments, outcome="refused", error=error)

    def record_failed_tool(self, tool: str, arguments: dict[str, str], error: str) -> None:
        """Audit a tool the standalone MCP server couldn't run, so that the failure shows in the trail."""
        with self.store.connect() as conn:
            repo.append_audit(conn, actor=self.actor, tool=tool, arguments=arguments, outcome="failed", error=error)

    # --- proposal creation (no business effects) ------------------------------

    def create_proposal(self, authenticated_customer_id: str, request_id: str, order_id: str | None,
                        line_id: str | None, remedy: str,
                        replacement_sku: str | None = None) -> ProposalOutcome:
        """Persist an immutable proposal if policy allows; amounts are computed here, never passed in."""
        arguments = {"customer_id": authenticated_customer_id, "request_id": request_id,
                     "order_id": order_id, "line_id": line_id, "remedy": remedy,
                     "replacement_sku": replacement_sku}
        outcome = self._create_proposal(authenticated_customer_id, request_id, order_id, line_id,
                                        remedy, replacement_sku)
        self._audit("create_proposal", arguments, outcome.kind, outcome)
        return outcome

    def record_refusal(self, customer_id: str, request_id: str, order_id: str | None, line_id: str | None,
                       remedy: str, reason_code: str, replacement_sku: str | None = None) -> None:
        """Audit a refusal the controller made before asking for a proposal; it has no business effect."""
        self._audit("create_proposal", {"customer_id": customer_id, "request_id": request_id, "order_id": order_id,
                                        "line_id": line_id, "remedy": remedy, "replacement_sku": replacement_sku},
                    "deny", {"reason_code": reason_code, "request_id": request_id})

    def record_out_of_scope(self, customer_id: str, request_id: str) -> None:
        """The agent read the message as unrelated to returns, so the desk answered with its fixed scope reply."""
        self._audit("scope_check", {"customer_id": customer_id, "request_id": request_id}, "out_of_scope", None)

    def record_reconciliation(self, customer_id: str, request_id: str, status: RequestStatus) -> None:
        """After a post-commit timeout, the desk read the stored outcome instead of committing again (a read: no effects)."""
        self._audit("reconcile_request", {"customer_id": customer_id, "request_id": request_id}, status.state,
                    status.result)

    def record_handoff(self, source: str, target: str, outcome: str, details: dict[str, Any]) -> None:
        """A visible row per handoff between the agents."""
        with self.store.connect() as conn:
            repo.append_audit(conn, actor=self.actor, tool="agent_handoff", outcome=outcome,
                              arguments={"from": source, "to": target, **details})

    def record_item_opened(self, customer_id: str, request_id: str, order_id: str, line_id: str) -> None:
        """The customer said the item was opened, so the store now says so too. This only ever makes an
        item ineligible: nothing a customer says can mark an item unopened."""
        arguments = {"customer_id": customer_id, "request_id": request_id, "order_id": order_id, "line_id": line_id}
        with self.store.connect() as conn, conn.transaction():
            changed = repo.mark_line_opened(conn, customer_id, order_id, line_id)
            repo.append_audit(conn, actor=self.actor, tool="item_condition", arguments=arguments,
                              outcome="marked_opened" if changed else "unchanged",
                              effects=[{"type": "line_marked_opened", "order_id": order_id, "line_id": line_id}]
                              if changed else [])

    def _create_proposal(self, customer_id: str, request_id: str, order_id: str | None,
                         line_id: str | None, remedy: str, replacement_sku: str | None) -> ProposalOutcome:
        status = reconcile_request(self.store, customer_id, request_id)
        if status.state == "completed":
            return ProposalOutcome(kind="already_completed", reason_code=WorkflowCode.REQUEST_COMPLETED,
                                   request_id=request_id, existing_result=status.result)
        if status.state == "conflict":
            return ProposalOutcome(kind="conflict", reason_code=WorkflowCode.REQUEST_CONFLICT,
                                   request_id=request_id)
        if remedy not in REMEDIES:
            return ProposalOutcome(kind="deny", reason_code=WorkflowCode.UNSUPPORTED_REMEDY,
                                   request_id=request_id)

        state = self.store.load_state()
        if remedy == "refund":
            assessment = policy.assess_refund(state, customer_id, order_id, line_id)
            sku, amount, replacement = assessment.sku, assessment.refund_amount_cents, None
        else:
            assessment = policy.assess_exchange(state, customer_id, order_id, line_id, replacement_sku)
            sku, amount, replacement = assessment.original_sku, 0, assessment.replacement_sku
        if not assessment.eligible:
            kind = "clarify" if assessment.reason_code is policy.ReasonCode.CLARIFICATION_REQUIRED else "deny"
            return ProposalOutcome(kind=kind, reason_code=assessment.reason_code, request_id=request_id,
                                   candidate_line_ids=assessment.candidate_line_ids)

        paid = policy.check_return_eligibility(state, customer_id, order_id, line_id).paid_cents
        with self.store.connect() as conn, conn.transaction():
            row = repo.insert_proposal(conn, request_id=request_id, customer_id=customer_id,
                                       order_id=assessment.order_id, line_id=assessment.line_id,
                                       remedy=remedy, amount_cents=amount, paid_cents=paid,
                                       original_sku=sku, replacement_sku=replacement)
        return ProposalOutcome(kind="proposed", reason_code=assessment.reason_code, request_id=request_id,
                               proposal=Proposal.from_row(row))
