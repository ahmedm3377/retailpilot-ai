"""BusinessDesk: every deterministic business step of a turn, usable with no model and no graph.

The graph decides only which step comes next. What is allowed, what a request means once checked,
and what gets written all live here and in request_checks; the agents never see this module.
Reads for the checks come in as a BusinessContext. Proposal creation, commit and recovery go to the
database directly: they are the write path and must read what they write.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.business import request_checks as checks
from app.business.facts import facts_from_commit, facts_from_proposal_outcome
from app.db.fixture_state import RetailStore
from app.domain.outcomes import CommitResult, PostCommitTimeout, RequestStatus
from app.domain.proposals import Proposal, ProposalOutcome
from app.domain.requests import CustomerHandoff
from app.observability import telemetry as tel
from app.services.faults import FaultInjector
from app.services.operations import RetailOperations
from app.services.recovery import reconcile_request
from app.services.transactions import TRUSTED_ACTOR, cancel_proposal, commit_confirmed_proposal
from retail_domain import policy
from retail_domain.models import State
from retail_domain.policy import ReasonCode

POLICY_REASONS = {code.value for code in ReasonCode}
PROPOSAL_DECISIONS = {"proposed": "propose", "clarify": "clarify", "deny": "deny",
                      "already_completed": "completed", "conflict": "error"}
REPLACEMENT_REFUSALS = {ReasonCode.INVALID_REPLACEMENT, ReasonCode.REPLACEMENT_OUT_OF_STOCK}


@dataclass(frozen=True)
class BusinessContext:
    """What the checks read for one turn: the store's state, this request's status, the waiting proposal."""

    state: State
    status: RequestStatus
    waiting: dict[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BusinessContext":
        """Rebuild the context from the MCP server's get_business_context result."""
        status = payload["request_status"]
        result = CommitResult.model_validate(status["result"]) if status["result"] else None
        proposal = payload["proposal"]
        return cls(State.model_validate(payload["state"]),
                   RequestStatus(state=status["state"], request_id=status["request_id"], result=result),
                   proposal if proposal and proposal["status"] == "pending" else None)


@dataclass
class DeskResult:
    """One step's outcome. `ready` means the request can go on to the next step; otherwise `decision` is final."""

    ready: bool = False
    decision: str | None = None
    facts: dict[str, Any] | None = None
    proposal: dict[str, Any] | None = None
    handoff: dict[str, Any] | None = None
    remaining_items: list[dict[str, Any]] | None = None
    last_proposal_id: str | None = None
    asked_customer: bool = False  # the desk needs the customer's answer about something an agent handed back
    missing_sizes: list[str] = field(default_factory=list)
    internal: dict[str, Any] = field(default_factory=dict)

    def state_update(self) -> dict[str, Any]:
        values = {"decision": self.decision, "facts": self.facts, "proposal": self.proposal, "handoff": self.handoff,
                  "remaining_items": self.remaining_items, "last_proposal_id": self.last_proposal_id}
        return {k: v for k, v in values.items() if v is not None}


class BusinessDesk:
    def __init__(self, store: RetailStore, *, ask_item_condition: bool = False) -> None:
        self.store = store
        # Ask "opened or unopened?" before proposing (the UI). The adapter can't: contract cases confirm on turn 2.
        self.ask_item_condition = ask_item_condition

    def _trusted(self) -> RetailOperations:
        return RetailOperations(self.store, actor=TRUSTED_ACTOR)

    # --- reading the customer's answer -----------------------------------------------------------

    def condition_answer(self, message: str, pending: dict[str, Any] | None, last_asked: str | None) -> dict[str, Any] | None:
        """The kept request with a plain answer to the desk's own question, or None if this isn't one."""
        if not (self.ask_item_condition and pending and last_asked == "condition"):
            return None
        answer = checks.item_condition_in(message, answering=True)
        return None if answer is None else {**pending, "item_opened": answer}

    def read_answer(self, context: BusinessContext, handoff: CustomerHandoff, message: str,
                    pending: dict[str, Any] | None, last_asked: str | None, answering_condition: bool) -> CustomerHandoff:
        """What fixed rules add to the model's reading of the message."""
        if self.ask_item_condition:  # the model's reading of this can't be used, since "opened" changes the store
            handoff = handoff.model_copy(update={"item_opened": checks.item_condition_in(
                message, answering=answering_condition or bool(context.waiting))})
        elif not context.waiting:
            # A refusal must rest on the customer's own words. The model reads an `unopened` field in a tool
            # result as their claim. While a proposal waits its reading stands, because "no" answers the question.
            handoff = handoff.model_copy(update={"item_opened": checks.item_condition_in(message)})
        if pending and last_asked == "size" and not handoff.requested_size and handoff.intent != "refund":
            size = checks.size_in_answer(message, context.state, pending)
            if size:
                handoff = handoff.model_copy(update={"intent": "exchange", "requested_size": size})
        return handoff

    # --- checking the request --------------------------------------------------------------------

    def check_request(self, context: BusinessContext, customer_id: str, request_id: str, handoff: CustomerHandoff,
                      message: str, pending: dict[str, Any] | None, last_asked: str | None,
                      remaining_items: list[dict[str, Any]] | None, history: list[dict[str, Any]] | None) -> DeskResult:
        """Resolve the request against the store and what the customer named. `ready`: it can go to the agent."""
        state = context.state
        orders = checks.customer_orders(state, customer_id)
        words = checks.words_in(message)
        foreign = [o.order_id for o in state.orders if o.customer_id != customer_id and o.order_id.upper() in words]
        picking_line = bool(pending and last_asked == "line")
        internal: dict[str, Any] = {}
        handoff = checks.ground_target(handoff, message, orders, pending, picking_line)
        first, others, declined = checks.several_items(handoff, message, orders, state, picking_line)
        if first:
            handoff = handoff.model_copy(update={"line_id": first, "all_items": False})
            internal["already_returned_lines"] = declined
        remaining = others or [i for i in (remaining_items or [])
                               if i["order_id"] == handoff.order_id and i["line_id"] != handoff.line_id]
        grounded = handoff.model_dump(mode="json")

        def ask(facts: dict[str, Any], kept: dict[str, Any]) -> DeskResult:
            # When the desk moved on to one order's lines, that order is settled for the customer's answer.
            if facts.get("asked") == "line":
                kept = {**kept, "order_id": facts["order_id"]}
            return DeskResult(decision="clarify", handoff=kept, remaining_items=remaining, internal=internal,
                              facts={"reason_code": "CLARIFICATION_REQUIRED", **facts})

        if foreign:  # same answer as an unknown order, so nothing leaks
            internal["named_reference_mismatch"] = True
            return ask(checks.ask_for_order(state, customer_id, wrong=True), grounded)
        answering = bool(pending and last_asked in ("order", "line"))
        resolved, clarify = checks.resolve_handoff(handoff, orders, state, customer_id, message, answering)
        if resolved is None:
            return ask(clarify, grounded)
        resolved = checks.catalog_size(resolved, state)
        if (self.ask_item_condition and pending and pending.get("line_id") and resolved.item_opened is False
                and (resolved.order_id, resolved.line_id) != (pending.get("order_id"), pending["line_id"])
                and checks.item_condition_in(message) is None):
            resolved = resolved.model_copy(update={"item_opened": None})  # that answer was about another item
        mismatch, named_sku = checks.check_named_references(message, resolved, customer_id, state)
        if mismatch is not None:
            internal["named_reference_mismatch"] = True
            return ask(mismatch, grounded)
        internal["named_replacement_sku"] = named_sku
        kept = resolved.model_dump(mode="json")
        if context.waiting and not resolved.item_opened and checks.same_as_waiting_proposal(
                resolved, context.waiting, named_sku, state):
            # Nothing changed: show the same proposal again rather than making another one.
            outcome = ProposalOutcome(kind="proposed", reason_code=ReasonCode.ELIGIBLE, request_id=request_id,
                                      proposal=Proposal.from_row(context.waiting))
            return DeskResult(decision="propose", handoff=kept, remaining_items=remaining, internal=internal,
                              facts=facts_from_proposal_outcome(outcome),
                              proposal=outcome.proposal.model_dump(mode="json"))
        internal["case_state"] = checks.build_case_state(state, customer_id, request_id, resolved, history)
        return DeskResult(ready=True, handoff=kept, remaining_items=remaining, internal=internal)

    def check_size(self, context: BusinessContext, customer_id: str, request_id: str, handoff: CustomerHandoff,
                   named_sku: str | None) -> DeskResult:
        """Whether an answer about the size is really missing, whatever the agent said. `ready`: nothing is."""
        verdict, sizes = checks.check_exchange_size(context.state, customer_id, handoff, named_sku)
        if verdict == "ok":
            return DeskResult(ready=True)
        item = {"remedy": "exchange", "order_id": handoff.order_id, "line_id": handoff.line_id,
                "available_sizes": sizes, "asked": "size", "question": checks.ask_for_size(sizes)}
        # The size is cleared from the kept request, so that the customer can still answer with another one.
        kept = handoff.model_copy(update={"requested_size": None}).model_dump(mode="json")
        if verdict == "refuse":
            self._trusted().record_refusal(customer_id, request_id, handoff.order_id, handoff.line_id, "exchange",
                                           "SIZE_NOT_AVAILABLE")
            return DeskResult(decision="deny", handoff=kept, facts={"reason_code": "SIZE_NOT_AVAILABLE",
                                                                    "requested_size": handoff.requested_size, **item})
        return DeskResult(decision="clarify", handoff=kept, asked_customer=True, missing_sizes=sizes,
                          facts={"reason_code": "CLARIFICATION_REQUIRED", **item})

    # --- proposing --------------------------------------------------------------------------------

    def propose(self, context: BusinessContext, customer_id: str, request_id: str, handoff: CustomerHandoff,
                recommendation: dict[str, Any] | None, named_sku: str | None) -> DeskResult:
        """The handoff fixes target and remedy; the agent's SKU choice is checked; policy decides."""
        trusted = self._trusted()
        waiting = context.waiting

        def retire_waiting_proposal() -> None:
            if waiting:
                cancel_proposal(self.store, customer_id, request_id, waiting["proposal_id"])

        if handoff.item_opened:  # the customer's word can refuse, never approve; the store is checked anyway
            trusted.record_refusal(customer_id, request_id, handoff.order_id, handoff.line_id, handoff.intent,
                                   ReasonCode.ITEM_OPENED.value)
            retire_waiting_proposal()
            if self.ask_item_condition:
                trusted.record_item_opened(customer_id, request_id, handoff.order_id, handoff.line_id)
            return DeskResult(decision="deny", facts={"reason_code": ReasonCode.ITEM_OPENED.value, "remedy": handoff.intent,
                                                      "order_id": handoff.order_id, "line_id": handoff.line_id,
                                                      "fixed_wording": self.ask_item_condition})
        replacement, internal = None, {}
        if handoff.intent == "exchange":
            options = policy.list_exchange_options(context.state, customer_id, handoff.order_id, handoff.line_id)
            replacement, internal["replacement_source"] = checks.choose_replacement(options, context.state, handoff,
                                                                                    recommendation, named_sku)
        if self.ask_item_condition and handoff.item_opened is None:
            # Only when an offer is really possible: anything else goes on to be denied with its own reason.
            assessment = (policy.assess_exchange(context.state, customer_id, handoff.order_id, handoff.line_id, replacement)
                          if handoff.intent == "exchange"
                          else policy.assess_refund(context.state, customer_id, handoff.order_id, handoff.line_id))
            if assessment.eligible:
                size = next((p.size for p in context.state.products if p.sku == replacement), None)
                kept = handoff.model_copy(update={"requested_size": handoff.requested_size or size})
                return DeskResult(decision="clarify", handoff=kept.model_dump(mode="json"), internal=internal,
                                  facts={"reason_code": "CLARIFICATION_REQUIRED", "asked": "condition",
                                         "remedy": handoff.intent, "order_id": handoff.order_id,
                                         "line_id": handoff.line_id, "question": checks.ITEM_CONDITION_QUESTION,
                                         "fixed_wording": True})
        outcome = trusted.create_proposal(customer_id, request_id, handoff.order_id, handoff.line_id, handoff.intent,
                                          replacement)
        facts = facts_from_proposal_outcome(outcome)
        proposal = outcome.proposal.model_dump(mode="json") if outcome.proposal else None
        result = DeskResult(decision=PROPOSAL_DECISIONS[outcome.kind], facts=facts, proposal=proposal, internal=internal)
        if outcome.reason_code in REPLACEMENT_REFUSALS:
            _, sizes = checks.check_exchange_size(context.state, customer_id, handoff, None)
            if sizes:  # only this size was refused: the customer can still choose another one
                facts.update(remedy="exchange", order_id=handoff.order_id, line_id=handoff.line_id, available_sizes=sizes,
                             requested=f"size {handoff.requested_size}" if handoff.requested_size else replacement,
                             asked="size", question=checks.ask_for_size(sizes), fixed_wording=True)
                result.handoff = handoff.model_copy(update={"requested_size": None}).model_dump(mode="json")
        tel.set_attributes(tel.current_span(), {
            "retail.proposal.outcome": outcome.kind, "retail.remedy": handoff.intent,
            "retail.decision": result.decision, "retail.reason_code": outcome.reason_code.value,
            "retail.proposal_ref": tel.proposal_ref(proposal and proposal["proposal_id"], request_id)})
        if proposal:
            retire_waiting_proposal()  # only the newest proposal can be confirmed
            result.last_proposal_id = proposal["proposal_id"]
        return result

    # --- committing -------------------------------------------------------------------------------

    def commit(self, customer_id: str, request_id: str, proposal_id: str | None,
               injector: FaultInjector | None) -> DeskResult:
        """The only step that moves money or stock. After a timeout it reads the result, never commits again."""
        try:
            result = commit_confirmed_proposal(self.store, customer_id, request_id, proposal_id, injector=injector)
        except PostCommitTimeout:
            # The commit itself timed out after saving: keep that visible as an error on the commit span;
            # only the separate reconciliation (a read) can show the stored outcome.
            commit_span = tel.current_span()
            tel.set_attributes(commit_span, {"retail.timeout.injected": True})
            tel.mark_error(commit_span, "PostCommitTimeout")
            with tel.span("retail.transaction.reconcile", {"retail.reconcile.reason": "post_commit_timeout"}) as rs:
                status = reconcile_request(self.store, customer_id, request_id)
                tel.set_attributes(rs, {"retail.reconcile.state": status.state})
            self._trusted().record_reconciliation(customer_id, request_id, status)
            internal = {"post_commit_timeout": {"reconciled": status.state}}
            if status.state == "completed":
                tel.set_attributes(commit_span, {"retail.business.outcome": "reconciled",
                                                 "retail.timeout.recovered": True})
                tel.record_business(status.result.remedy, "reconciled")
                return DeskResult(decision="completed", internal=internal,
                                  facts={**facts_from_commit(status.result), "recovered_after_timeout": True})
            tel.record_business(None, "error")
            return DeskResult(decision="error", internal=internal, facts={"reason_code": "TIMEOUT_UNRESOLVED"})
        internal = {"commit": {"status": result.status, "reason_code": result.reason_code.value,
                               "replayed": result.replayed}}
        if result.status == "completed":
            decision = "completed"
            business = "replayed" if result.replayed else "committed"
        elif result.status == "rejected" and result.reason_code.value in POLICY_REASONS:
            decision, business = "deny", "denied"
        else:
            decision, business = "error", "conflict" if result.status == "conflict" else "error"
        tel.set_attributes(tel.current_span(), {
            "retail.commit.status": result.status, "retail.commit.replayed": result.replayed,
            "retail.reason_code": result.reason_code.value, "retail.business.outcome": business,
            "retail.remedy": result.remedy or "none",
            "retail.proposal_ref": tel.proposal_ref(proposal_id, request_id)})
        tel.record_business(result.remedy, business)
        return DeskResult(decision=decision, facts=facts_from_commit(result), internal=internal)
