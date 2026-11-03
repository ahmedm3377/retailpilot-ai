"""The LangGraph StateGraph for one customer turn.

    START -> guard_input --blocked--> present -> END
                         --confirmed proposal--> commit -> present -> END
                         --otherwise--> load_context --tools unavailable--> present -> END
                                                     --request already completed--> present -> END (no agents)
                                                     --otherwise--> customer_agent -> check_handoff
        check_handoff --needs info--> present -> END
                      --resolved--> operations_agent --customer must answer--> present -> END
                                                     --otherwise--> create_proposal -> present -> END

The clarification loop: when either agent needs an answer, the Customer Resolution Agent asks and
the partial request stays in the checkpoint. On the next turn merge_pending_request updates it with
the answer, and the updated request is handed to the Operations Agent again.

This module only routes. Each LLM node runs an agent with its role's read-only tools; each deterministic
node calls app.business.BusinessDesk, which holds every business rule and write.

LLM nodes: customer_agent, operations_agent, present (wording only).
Deterministic nodes: everything else. Only `commit` can change money or stock,
and only for a proposal ID the trusted caller passed as confirmed_proposal_id. A turn the
caller marks confirmation_requested never reaches the LLM agents, even without an ID.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from app.agents import customer_agent, operations_agent
from app.agents.messages import (
    UNOPENED_QUESTION, already_returned_note, facts_for_wording, fallback_message, remaining_items_note,
)
from app.agents.model_client import ChatModel, ModelUnavailableError
from app.agents.schemas import CustomerHandoff
from app.agents.settings import AgentSettings
from app.agents.state import TurnContext, TurnState
from app.agents.tool_registry import ReadOnlyToolRegistry, ToolFailed, call_read_tool
from app.business.desk import BusinessContext, BusinessDesk
from app.business.facts import facts_already_completed
from app.business.request_checks import customer_orders, merge_pending_request
from app.db import repositories as repo
from app.db.fixture_state import RetailStore
from app.guardrails.service import GuardrailService
from app.observability import telemetry as tel
from app.services.operations import RetailOperations
from app.services.transactions import TRUSTED_ACTOR

# Span per graph node (None: no own span; the node's work is traced by its callees).
NODE_SPANS: dict[str, tuple[str, dict[str, Any]] | None] = {
    "load_context": ("retail.context.load", {}),
    "guard_input": ("retail.input.check", {}),
    "customer_agent": ("invoke_agent customer_resolution", {"gen_ai.operation.name": "invoke_agent",
                                                            "gen_ai.agent.name": "customer_resolution",
                                                            "retail.agent.role": "customer_resolution",
                                                            "retail.agent.task": "understand"}),
    "check_handoff": ("retail.handoff.check", {}),
    "operations_agent": ("invoke_agent retail_operations", {"gen_ai.operation.name": "invoke_agent",
                                                            "gen_ai.agent.name": "retail_operations",
                                                            "retail.agent.role": "retail_operations",
                                                            "retail.agent.task": "investigate"}),
    "create_proposal": ("retail.proposal.create", {}),
    "commit": ("retail.transaction.commit", {}),
    "present": ("retail.response", {}),
}


@dataclass
class GraphDeps:
    store: RetailStore
    model: ChatModel
    guards: GuardrailService
    settings: AgentSettings
    ctx: TurnContext | None = None  # set by the runtime for the turn being run
    # (customer_id, request_id) -> scoped client of the standalone MCP server; set by the runtime.
    read_tools: Callable[[str, str], Any] | None = None
    # Ask "opened or unopened?" before proposing (the UI). The adapter can't: contract cases confirm on turn 2.
    ask_item_condition: bool = False
    desk: BusinessDesk = field(init=False)

    def __post_init__(self) -> None:
        self.desk = BusinessDesk(self.store, ask_item_condition=self.ask_item_condition)

    @property
    def turn(self) -> TurnContext:
        assert self.ctx is not None, "graph invoked outside AgentRuntime.handle_customer_turn"
        return self.ctx


def model_call_event(call: Any) -> dict[str, Any]:
    return {"agent": call.role, "model": call.model, "latency_ms": round(call.latency_ms, 1), "ok": call.ok,
            "prompt_tokens": call.prompt_tokens, "completion_tokens": call.completion_tokens, "error": call.error}


def audit_new_events(store: RetailStore, ctx: TurnContext) -> None:
    """Add the turn's model calls and guardrail verdicts that aren't in the audit log yet (no prompts, no
    reasoning). Called after every node, so the log shows the steps in the order they ran."""
    calls, guards = ctx.model_calls[ctx.audited_calls:], ctx.guardrail_events[ctx.audited_guards:]
    if not calls and not guards:
        return
    with store.connect() as conn, conn.transaction():
        for event in map(model_call_event, calls):
            repo.append_audit(conn, actor=event["agent"], tool="llm_call", arguments={"model": event["model"]},
                              outcome="ok" if event["ok"] else "error", result=event, error=event["error"])
        for event in guards:
            repo.append_audit(conn, actor="guardrails", tool=f"guard:{event['gate']}:{event['validator']}",
                              arguments={}, outcome=event["outcome"], result=event)
    ctx.audited_calls, ctx.audited_guards = len(ctx.model_calls), len(ctx.guardrail_events)


def build_graph(deps: GraphDeps) -> StateGraph:
    def ops(actor: str) -> RetailOperations:
        return RetailOperations(deps.store, actor=actor)

    def record_calls(calls: list[Any]) -> None:
        deps.turn.model_calls.extend(calls)

    def record_handoff(state: TurnState, source: str, target: str, handoff: CustomerHandoff,
                       outcome: str, **extra: Any) -> None:
        """A visible audit row and span event per handoff. The controller routes it, so it is its row."""
        request = {**deps.turn.internal["case_state"], "remedy": handoff.intent, "order_id": handoff.order_id,
                   "line_id": handoff.line_id, "requested_size": handoff.requested_size}
        ops(TRUSTED_ACTOR).record_handoff(source, target, outcome, {**request, **extra})
        tel.add_event(tel.current_span(), "agent.handoff", {"from": source.removesuffix("_agent"),
                                                            "to": target.removesuffix("_agent"),
                                                            "intent": handoff.intent})

    def transport(state: TurnState) -> Callable[[], Any]:
        return lambda: deps.read_tools(state["customer_id"], state["request_id"])

    def load_context(state: TurnState) -> dict[str, Any]:
        """What the business checks read for this turn, in one call to the MCP server."""
        try:
            payload = call_read_tool(ops(TRUSTED_ACTOR), transport(state), state["customer_id"], state["request_id"],
                                     "get_business_context", {"proposal_id": state.get("last_proposal_id") or ""})
        except ToolFailed:
            tel.record_tool("get_business_context", "failed")
            tel.mark_error(tel.current_span(), "McpToolError")
            return {"route": "done", "decision": "error", "facts": {"reason_code": "TOOLS_UNAVAILABLE"}}
        context = deps.turn.internal["business_context"] = BusinessContext.from_payload(payload)
        tel.capture(tel.current_span(), output=customer_orders(context.state, state["customer_id"]))
        # One request = one transaction. A chat turn on a request that already committed reports the
        # stored result deterministically (no agents) and points to a new request; nothing is redone.
        if context.status.state == "completed":
            deps.turn.internal["request_already_completed"] = True
            return {"route": "done", "decision": "completed", "facts": facts_already_completed(context.status.result)}
        return {"route": "understand"}

    def guard_input(state: TurnState) -> dict[str, Any]:
        """Always the first step: nothing else runs for a message until it has passed every input check."""
        fresh = {"handoff": None, "facts": {}, "proposal": None}
        result = deps.guards.validate_and_sanitize_customer_input(deps.turn.message)
        deps.turn.guardrail_events += [e.as_dict() for e in result.events]
        if not result.allowed:
            return {**fresh, "route": "blocked", "decision": "error", "facts": {"reason_code": result.reason_code}}
        deps.turn.safe_message = result.text
        tel.capture(tel.current_span(), output=result.text)
        deps.turn.internal["input_allowed"] = True
        # A trusted confirmation turn goes to commit even without an ID (-> CONFIRMATION_REQUIRED),
        # so its text is never treated as a new request.
        confirming = state.get("confirmed_proposal_id") or state.get("confirmation_requested")
        if confirming:
            return {**fresh, "route": "commit"}
        return {**fresh, "route": "understand"}

    def business() -> BusinessContext:
        return deps.turn.internal["business_context"]

    def agent_step(call: Any, verdicts: list[dict[str, Any]]) -> None:
        """Into the audit log as it happens, so model calls, tool calls and verdicts keep their order."""
        if call:
            record_calls([call])
        deps.turn.guardrail_events += verdicts
        audit_new_events(deps.store, deps.turn)

    def customer_agent_node(state: TurnState) -> dict[str, Any]:
        message = deps.turn.safe_message
        context = business()
        tel.capture(tel.current_span(), input=message)
        asked = state.get("last_asked")
        answering_condition = bool(state.get("last_question") and asked == "condition" and state.get("last_handoff"))
        if answering_condition:
            answered = deps.desk.condition_answer(message, state["last_handoff"], asked)
            if answered is not None:  # the desk's own question, plainly answered: nothing for the model to read
                return {"route": "check", "handoff": answered}
        previous = None
        if state.get("last_question") or context.waiting:
            previous = {"earlier_request": state.get("last_handoff"),
                        "question_asked": state.get("last_question") or UNOPENED_QUESTION}
        registry = ReadOnlyToolRegistry(ops(customer_agent.ROLE), state["customer_id"], state["request_id"],
                                        transport(state), role=customer_agent.ROLE)
        try:
            run = customer_agent.extract_handoff(deps.model, deps.guards, registry, message, previous, agent_step)
        except ModelUnavailableError:
            return {"route": "done", "decision": "error", "facts": {"reason_code": "MODEL_UNAVAILABLE"}}
        deps.turn.tool_events += run.tool_events
        if any(e["outcome"] == "failed" for e in run.tool_events):
            return {"route": "done", "decision": "error", "facts": {"reason_code": "TOOLS_UNAVAILABLE"}}
        if run.handoff is None:
            return {"route": "done", "decision": "error", "facts": {"reason_code": run.error}}
        handoff = run.handoff
        pending = state.get("last_handoff") if state.get("last_question") or context.waiting else None
        handoff = deps.desk.read_answer(context, handoff, message, pending, asked, answering_condition)
        if handoff.intent == "out_of_scope":  # fixed reply; a request being clarified stays open
            audit_new_events(deps.store, deps.turn)  # this agent's model call decided it, so it comes first
            ops(customer_agent.ROLE).record_out_of_scope(state["customer_id"], state["request_id"])
            return {"route": "done", "decision": "clarify", "handoff": pending,
                    "facts": {"reason_code": "OUT_OF_SCOPE", "asked": asked, "question": state.get("last_question")}}
        handoff = merge_pending_request(handoff, pending, asked in ("order", "line"))
        return {"route": "check", "handoff": handoff.model_dump(mode="json")}

    def check_handoff(state: TurnState) -> dict[str, Any]:
        context = business()
        pending = state.get("last_handoff") if state.get("last_question") or context.waiting else None
        result = deps.desk.check_request(context, state["customer_id"], state["request_id"],
                                         CustomerHandoff.model_validate(state["handoff"]), deps.turn.safe_message,
                                         pending, state.get("last_asked"), state.get("remaining_items"),
                                         state.get("history"))
        deps.turn.internal.update(result.internal)
        if not result.ready:
            return {"route": "done", **result.state_update()}
        record_handoff(state, customer_agent.ROLE, operations_agent.ROLE, CustomerHandoff.model_validate(result.handoff),
                       "request_ready")
        return {"route": "operate", **result.state_update()}

    def operations_agent_node(state: TurnState) -> dict[str, Any]:
        handoff = CustomerHandoff.model_validate(state["handoff"])
        registry = ReadOnlyToolRegistry(ops(operations_agent.ROLE), state["customer_id"], state["request_id"],
                                        transport(state), role=operations_agent.ROLE)
        try:
            run = operations_agent.run_operations_agent(deps.model, deps.guards, registry, handoff,
                                                        deps.settings.max_tool_steps, deps.turn.internal["case_state"],
                                                        on_step=agent_step)
        except ModelUnavailableError:
            return {"route": "done", "decision": "error", "facts": {"reason_code": "MODEL_UNAVAILABLE"}}
        deps.turn.tool_events += run.tool_events
        recommendation = run.recommendation.model_dump(mode="json") if run.recommendation else None
        deps.turn.internal["operations_recommendation"] = recommendation
        if run.error:
            deps.turn.internal["operations_error"] = run.error
        if any(e["outcome"] == "failed" for e in run.tool_events):
            # The tools couldn't be reached, so nobody checked this request: say so instead of proposing.
            return {"route": "done", "decision": "error", "facts": {"reason_code": "TOOLS_UNAVAILABLE"}}
        # The agent may hand back, but deterministic code decides whether an answer is really missing:
        # it asks even if the agent guessed a size, and an ineligible line goes on to be denied.
        result = deps.desk.check_size(business(), state["customer_id"], state["request_id"], handoff,
                                      deps.turn.internal.get("named_replacement_sku"))
        deps.turn.internal["operations_asked_customer"] = run.asked_customer
        if result.ready:
            return {"route": "propose"}
        if result.asked_customer:
            record_handoff(state, operations_agent.ROLE, customer_agent.ROLE, handoff, "clarification_needed",
                           missing="requested_size", options=result.missing_sizes)
        return {"route": "done", **result.state_update()}

    def create_proposal(state: TurnState) -> dict[str, Any]:
        result = deps.desk.propose(business(), state["customer_id"], state["request_id"],
                                   CustomerHandoff.model_validate(state["handoff"]),
                                   deps.turn.internal.get("operations_recommendation"),
                                   deps.turn.internal.get("named_replacement_sku"))
        deps.turn.internal.update(result.internal)
        return result.state_update()

    def commit(state: TurnState) -> dict[str, Any]:
        result = deps.desk.commit(state["customer_id"], state["request_id"], state["confirmed_proposal_id"],
                                  deps.turn.injector)
        deps.turn.internal.update(result.internal)
        return result.state_update()

    def present(state: TurnState) -> dict[str, Any]:
        decision = state["decision"]
        facts = state.get("facts") or {}
        fallback = fallback_message(decision, facts)
        text, used = fallback, "deterministic"
        confirmation_turn = bool(state.get("confirmed_proposal_id") or state.get("confirmation_requested"))
        templated = confirmation_turn and deps.settings.confirm_template
        # Blocked input and model outages get fixed wording; no LLM sees blocked text. With
        # confirm_template, a confirmation turn reports the stored result with the trusted template.
        use_llm = (state.get("route") != "blocked"
                   and facts.get("reason_code") not in ("MODEL_UNAVAILABLE", "REQUEST_COMPLETED", "OUT_OF_SCOPE",
                                                        "SIZE_NOT_AVAILABLE", "TOOLS_UNAVAILABLE")
                   and not facts.get("fixed_wording") and not templated)
        if use_llm:
            try:
                with tel.span("invoke_agent customer_resolution", {
                        "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "customer_resolution",
                        "retail.agent.role": "customer_resolution", "retail.agent.task": "explain"}):
                    draft, calls, events = customer_agent.explain(deps.model, deps.guards, decision,
                                                                 facts_for_wording(decision, facts))
                record_calls(calls)
                deps.turn.guardrail_events += events
            except ModelUnavailableError:
                draft = None
            if draft:
                checked = deps.guards.validate_customer_output(draft, {"decision": decision, **facts}, fallback)
                deps.turn.guardrail_events += [e.as_dict() for e in checked.events]
                text, used = checked.text, "fallback" if checked.used_fallback else "agent"
        if decision == "propose" and (state.get("handoff") or {}).get("item_opened") is not False:
            text = f"{text} {UNOPENED_QUESTION}"  # not again once the customer has said it is unopened
        declined = deps.turn.internal.get("already_returned_lines")
        if declined and decision in ("propose", "deny"):
            text = f"{text} {already_returned_note(declined)}"
        remaining = state.get("remaining_items") or []
        if remaining and decision in ("propose", "deny", "completed"):
            line = (state.get("proposal") or {}).get("line_id") or (state.get("handoff") or {}).get("line_id")
            text = f"{text} {remaining_items_note(decision, line, remaining)}"
        deps.turn.internal["response_source"] = used
        tel.capture(tel.current_span(), input=facts, output=text)
        tel.set_attributes(tel.current_span(), {"retail.response.source": used, "retail.decision": decision})
        history = (state.get("history") or [])[-9:] + [{"turn": state.get("turn_number"), "decision": decision,
                                                        "proposal_id": facts.get("proposal_id")}]
        carry: dict[str, Any] = {"response": text, "history": history, "last_asked": None}
        if decision == "clarify" or facts.get("asked") == "size":
            # The open question, not a refusal or an off-topic reply: the small model reads answers against it.
            carry["last_question"] = facts.get("question") or text
            carry["last_asked"] = facts.get("asked")
            carry["last_handoff"] = state.get("handoff")
        elif state.get("route") != "blocked":
            carry["last_question"] = None
            carry["last_handoff"] = state.get("handoff") or state.get("last_handoff")
        return carry

    def after_guard(state: TurnState) -> str:
        return {"blocked": "present", "commit": "commit", "done": "present"}.get(state["route"], "load_context")

    def after_context(state: TurnState) -> str:
        return "customer_agent" if state["route"] == "understand" else "present"

    def after_customer(state: TurnState) -> str:
        return "check_handoff" if state["route"] == "check" else "present"

    def after_check(state: TurnState) -> str:
        return "operations_agent" if state["route"] == "operate" else "present"

    def after_operations(state: TurnState) -> str:
        return "create_proposal" if state["route"] == "propose" else "present"

    def traced(name: str, fn: Any) -> Any:
        spec = NODE_SPANS.get(name)

        def node(state: TurnState) -> dict[str, Any]:
            with tel.use_context(deps.ctx.otel_context if deps.ctx else None):
                with tel.span(spec[0], {**spec[1], "retail.step": name}) as span:
                    tel.capture(span, state_in=state)
                    start = time.perf_counter()
                    outcome = "error"
                    try:
                        update = fn(state)
                        audit_new_events(deps.store, deps.turn)
                        outcome = "error" if update.get("decision") == "error" else "ok"
                    finally:
                        tel.record_step(name, time.perf_counter() - start, outcome)
                    tel.capture(span, state_out=update)
                    return update
        return node

    graph = StateGraph(TurnState)
    for name, fn in [("load_context", load_context), ("guard_input", guard_input),
                     ("customer_agent", customer_agent_node), ("check_handoff", check_handoff),
                     ("operations_agent", operations_agent_node), ("create_proposal", create_proposal),
                     ("commit", commit), ("present", present)]:
        graph.add_node(name, traced(name, fn))
    graph.add_edge(START, "guard_input")
    graph.add_conditional_edges("guard_input", after_guard, ["present", "commit", "load_context"])
    graph.add_conditional_edges("load_context", after_context, ["customer_agent", "present"])
    graph.add_conditional_edges("customer_agent", after_customer, ["check_handoff", "present"])
    graph.add_conditional_edges("check_handoff", after_check, ["operations_agent", "present"])
    graph.add_conditional_edges("operations_agent", after_operations, ["create_proposal", "present"])
    graph.add_edge("create_proposal", "present")
    graph.add_edge("commit", "present")
    graph.add_edge("present", END)
    return graph
