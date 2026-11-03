"""Trusted facts for the customer explanation, and deterministic fallback wording.

The fallback is used whenever the LLM wording fails a guardrail or the model is
unavailable; it is built only from trusted service results.
"""
from __future__ import annotations

from typing import Any

from app.business.facts import money  # noqa: F401 - the API's views read it here
from app.business.request_checks import ITEM_CONDITION_QUESTION  # noqa: F401

REASON_TEXT = {
    "DELIVERY_DATE_OUT_OF_RANGE": "it is outside the 30-day return window",
    "NOT_DELIVERED": "the order has not been delivered",
    "ITEM_OPENED": "the item has been opened",
    "FINAL_SALE": "the item was a final-sale purchase",
    "ALREADY_RETURNED": "this item has already been returned",
    "CUSTOMER_MISMATCH": "we could not find that order on your account",
    "ORDER_NOT_FOUND": "we could not find that order on your account",
    "LINE_NOT_FOUND": "we could not find that item in the order",
    "SKU_NOT_FOUND": "that product or size is not in our catalog",
    "INVALID_REPLACEMENT": "exchanges are only possible for a different size of the same product at the same price",
    "REPLACEMENT_OUT_OF_STOCK": "the requested size is out of stock",
    "UNSUPPORTED_REMEDY": "only refunds and size exchanges are supported",
}

SCOPE_MESSAGE = ("I'm sorry, I can only help with returns and exchanges for items you bought from us. "
                 "I can refund a delivered item, exchange it for a different size, or check whether it "
                 "qualifies. If you'd like one of those, tell me which and which order it's about.")

# Added to every proposal: the rule is checked against the store, and the customer's answer can refuse.
ITEM_OPENED_APOLOGY = "I'm sorry, I can't proceed with this request because the item has been opened."

UNOPENED_QUESTION = ("Is the item still unopened? If it is, please confirm. If it has been opened, please tell me "
                     "or cancel, because opened items can't be returned or exchanged.")


def already_returned_note(line_ids: list[str]) -> str:
    """Part of a request for several items was declined: it must not pass silently."""
    listed = " and ".join(f"line {line}" for line in line_ids)
    return f"I can't process {listed} again, because it has already been returned."


def remaining_items_note(decision: str, line_id: str | None, items: list[dict[str, Any]]) -> str:
    """Several items were asked for, but one request is one transaction: say what happens to the others."""
    listed = " and ".join(f"line {i['line_id']} ({i['sku']})" for i in items)
    where = f"{listed} of order {items[0]['order_id']}"
    if decision == "propose":
        return f"I handle one item per request, so this is line {line_id}. When it's done, ask me about {where}."
    return f"You can now ask me about {where}."


ERROR_TEXT = {
    "PROMPT_INJECTION": SCOPE_MESSAGE,
    "JAILBREAK": SCOPE_MESSAGE,
    "INPUT_TOO_LONG": "That message is too long. Please describe your request in a few sentences.",
    "INPUT_EMPTY": "I didn't receive a message. Please describe your request.",
    "INPUT_NOT_TEXT": "I didn't receive a text message. Please describe your request.",
    "VALIDATOR_UNAVAILABLE": "Our safety checks are unavailable right now, so I can't process requests. Please try later.",
    "TOOLS_UNAVAILABLE": "I can't check your order right now, because our order tools are unavailable. Nothing was "
                         "changed; please try again later.",
    "MODEL_UNAVAILABLE": "The assistant is unavailable right now. Nothing was changed; please try again later.",
    "AGENT_OUTPUT_INVALID": "I couldn't process that request reliably. Nothing was changed; please rephrase it.",
    "CONFIRMATION_REQUIRED": "There is no proposal to confirm yet. Please describe your request first.",
    "PROPOSAL_NOT_FOUND": "That proposal could not be found. Nothing was changed.",
    "PROPOSAL_MISMATCH": "That proposal does not belong to this request. Nothing was changed.",
    "PROPOSAL_NOT_PENDING": "That proposal is no longer valid. Please start a new request.",
    "REQUEST_CONFLICT": "This request reference is already in use. Nothing was changed.",
    "AMOUNT_CHANGED": "The refund amount changed since the proposal. Nothing was changed; please start again.",
    "TIMEOUT_UNRESOLVED": "We could not confirm the result of your request. Please check again shortly.",
}


def facts_for_wording(decision: str, facts: dict[str, Any]) -> dict[str, Any]:
    """What the agent words: a refusal gets the reason in plain words, not its internal code."""
    if decision != "deny":
        return facts
    plain = {k: v for k, v in facts.items() if k != "reason_code"}
    return {**plain, "reason_text": REASON_TEXT.get(facts.get("reason_code"), "it is not eligible")}


def fallback_message(decision: str, facts: dict[str, Any]) -> str:
    code = facts.get("reason_code", "")
    item = f"line {facts.get('line_id')} of order {facts.get('order_id')}"
    if decision == "propose":
        if facts.get("remedy") == "refund":
            return (f"I can refund {item} for {facts['amount_text']}. "
                    f"Please confirm proposal {facts['proposal_id']} to proceed.")
        return (f"I can exchange {item} ({facts['original_sku']}) for {facts['replacement_sku']} at no extra "
                f"charge. Please confirm proposal {facts['proposal_id']} to proceed.")
    if decision == "completed" and code == "REQUEST_COMPLETED":
        if facts.get("remedy") == "refund":
            done = f"your refund of {facts['amount_text']} for {item}"
        elif facts.get("remedy") == "exchange":
            done = f"your exchange of {facts['original_sku']} for {facts['replacement_sku']} ({item})"
        else:
            done = "it"
        return (f"This request is already complete: {done}. Nothing new was done. "
                "To ask for something else, please start a new request.")
    if decision == "completed":
        if facts.get("remedy") == "refund":
            return f"Your refund of {facts['amount_text']} for {item} is complete."
        if facts.get("remedy") == "exchange":
            return (f"Your exchange of {facts['original_sku']} for {facts['replacement_sku']} ({item}) "
                    f"is complete.")
        return "Your request is already complete."
    if decision == "clarify":
        if code == "OUT_OF_SCOPE":
            return SCOPE_MESSAGE
        if facts.get("question"):
            return facts["question"]
        options = facts.get("candidate_line_ids") or []
        if options:
            return f"Which item would you like to return? Lines in that order: {', '.join(options)}."
        if code == "CLARIFICATION_REQUIRED" and facts.get("remedy") == "exchange":
            return "Which size would you like instead?"
        return "Could you tell me the order number and which item you mean?"
    if decision == "deny" and code == "SIZE_NOT_AVAILABLE":
        sizes = facts.get("available_sizes") or []
        available = f" Sizes available: {', '.join(sizes)}." if sizes else ""
        return f"I'm sorry, size {facts.get('requested_size')} isn't available for this item, so I can't make " \
               f"this exchange.{available}"
    if decision == "deny" and code == "ITEM_OPENED" and facts.get("fixed_wording"):
        return ITEM_OPENED_APOLOGY
    if decision == "deny" and facts.get("available_sizes"):
        return f"I'm sorry, I can't exchange this item for {facts.get('requested')}, because " \
               f"{REASON_TEXT.get(code, 'it is not eligible')}. Sizes available: {', '.join(facts['available_sizes'])}."
    if decision == "deny":
        return f"Sorry, I can't process this request because {REASON_TEXT.get(code, 'it is not eligible')}."
    return ERROR_TEXT.get(code, "Something went wrong. Nothing was changed; please try again.")
