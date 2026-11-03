"""Pure business checks on a request: no model, no database, no graph. Each takes the store's state as data.

They resolve what the customer asked for against what really exists, and decide what must be asked
before anything can be proposed. Eligibility itself stays in retail_domain.policy.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from app.domain.requests import CustomerHandoff
from retail_domain import policy

NO_ORDERS_QUESTION = "I can't find any orders on your account. Which order number is this about?"
WRONG_NUMBER = "You entered a wrong number."
NO_ELIGIBLE_ORDERS = "None of your orders can be returned or exchanged right now."
ITEM_CONDITION_QUESTION = "Before I can make you an offer: is the item opened or unopened?"


def customer_orders(state: Any, customer_id: str) -> list[dict[str, Any]]:
    """This customer's orders as plain data, with each line's product and size from the catalog."""
    catalog = {p.sku: p for p in state.products}

    def line(ln: Any) -> dict[str, Any]:
        product = catalog.get(ln.sku)
        return {"line_id": ln.line_id, "sku": ln.sku, "product_id": product.product_id if product else None,
                "size": product.size if product else None}

    return [{"order_id": o.order_id, "status": o.status,
             "delivered_on": o.delivered_on.isoformat() if o.delivered_on else None,
             "lines": [line(ln) for ln in o.lines]} for o in state.orders if o.customer_id == customer_id]


def words_in(message: str) -> set[str]:
    return {t.upper() for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", message or "")}


def ground_target(handoff: CustomerHandoff, message: str, orders: list[dict[str, Any]],
                  pending: dict[str, Any] | None, picking_line: bool) -> CustomerHandoff:
    """Keep only the target the customer actually identified; the desk never picks one for them.

    An order or line counts when the message names its ID, its SKU or its product, or when an earlier
    turn of this clarification settled it. The model's own choice counts in one case only: the desk
    listed the lines and the customer is picking one ("the first one").
    """
    words = words_in(message)
    by_id = {o["order_id"]: o for o in orders}
    pending = pending or {}
    named_orders = [o for o in orders if o["order_id"].upper() in words]
    order = named_orders[0] if len(named_orders) == 1 else None
    if not named_orders:
        order = by_id.get(pending.get("order_id") or "")

    def described(line: dict[str, Any]) -> bool:
        product = (line.get("product_id") or "").upper()
        return line["sku"].upper() in words or bool(product) and (product in words or f"{product}S" in words)

    scope = [(o, ln) for o in ([order] if order else orders) for ln in o["lines"]]
    named = ([p for p in scope if p[1]["line_id"].upper() in words] or [p for p in scope if described(p[1])])
    line_id = None
    if len(named) == 1:
        order, line_id = named[0][0], named[0][1]["line_id"]
    elif named and not order and len({p[1]["line_id"] for p in named}) == 1:
        line_id = named[0][1]["line_id"]  # the same line ID in several orders: kept while the desk asks which order
    elif not named and order:
        own_lines = {ln["line_id"] for ln in order["lines"]}
        if picking_line and handoff.line_id in own_lines:
            line_id = handoff.line_id
        elif pending.get("order_id") in (None, order["order_id"]) and pending.get("line_id") in own_lines:
            line_id = pending["line_id"]
    return handoff.model_copy(update={"order_id": order["order_id"] if order else None, "line_id": line_id})


EVERY_ITEM_WORDS = {"ALL", "BOTH", "EVERYTHING"}  # only read as an answer to "which line?"


def several_items(handoff: CustomerHandoff, message: str, orders: list[dict[str, Any]], state: Any,
                  picking_line: bool = False) -> tuple[str | None, list[dict[str, Any]], list[str]]:
    """The line to handle now and the others, when the customer asked for several items of one order.

    Lines named or described in the message decide ("L1 and L2", "the tee and the jacket"); otherwise the
    model's all_items reading of "both" or "the whole order" means every line. The small model can miss a
    bare "all", so that is also read directly when the desk has just asked which line. Lines already
    returned are declined. Returns (line to handle, the others, the lines declined), or (None, [], []) for
    a single item.
    """
    order = next((o for o in orders if o["order_id"] == handoff.order_id), None)
    if order is None or len(order["lines"]) < 2:
        return None, [], []
    words = words_in(message)
    named = [ln for ln in order["lines"] if ln["line_id"].upper() in words] or [
        ln for ln in order["lines"] if ln["sku"].upper() in words or (ln.get("product_id") or "").upper() in words
        or f"{(ln.get('product_id') or '').upper()}S" in words]
    every = handoff.all_items or (picking_line and bool(words & EVERY_ITEM_WORDS))
    chosen = named if len(named) > 1 else (order["lines"] if every and not named else [])
    if not chosen:
        return None, [], []
    stored = next(o for o in state.orders if o.order_id == order["order_id"])
    returned = {ln.line_id for ln in stored.lines if ln.returned}
    open_lines = [ln for ln in chosen if ln["line_id"] not in returned] or chosen[:1]
    others = [{"order_id": order["order_id"], "line_id": ln["line_id"], "sku": ln["sku"]} for ln in open_lines[1:]]
    return open_lines[0]["line_id"], others, [ln["line_id"] for ln in chosen if ln not in open_lines]


def _names_unknown_id(words: set[str], orders: list[dict[str, Any]], state: Any, answering: bool = False) -> bool:
    """Whether the message has an ID-like word (letters and digits) that is none of these orders' IDs or a
    catalog SKU. While answering "which order or line?", a bare number counts as an ID too."""
    known = {o["order_id"].upper() for o in orders} | {p.sku.upper() for p in state.products}
    known |= {v.upper() for o in orders for ln in o["lines"] for v in (ln["line_id"], ln["sku"])}
    return any(w not in known and re.search(r"\d", w) and (answering or re.search(r"[A-Z]", w)) for w in words)


def _eligible_lines(state: Any, customer_id: str, order: Any) -> list[Any]:
    return [ln for ln in order.lines
            if policy.check_return_eligibility(state, customer_id, order.order_id, ln.line_id).eligible]


def ask_for_line(state: Any, customer_id: str, order: Any, wrong: bool = False) -> dict[str, Any]:
    """Which line? Only the eligible ones, or all of them if none is: policy then gives the real reason."""
    lines = _eligible_lines(state, customer_id, order) or order.lines
    listed = "; ".join(f"{ln.line_id} ({ln.sku})" for ln in lines)
    prefix = f"{WRONG_NUMBER} " if wrong else ""
    return {"question": f"{prefix}Which item of order {order.order_id} do you mean? {listed}.", "asked": "line",
            "order_id": order.order_id, "candidate_line_ids": [ln.line_id for ln in lines],
            "candidate_lines": [{"line_id": ln.line_id, "sku": ln.sku} for ln in lines], "fixed_wording": True}


def ask_for_order(state: Any, customer_id: str, wrong: bool = False) -> dict[str, Any]:
    """Which order? Only those with something eligible; with just one of them, go straight to its lines."""
    own = [o for o in state.orders if o.customer_id == customer_id]
    eligible = [(o, lines) for o in own if (lines := _eligible_lines(state, customer_id, o))]
    prefix = f"{WRONG_NUMBER} " if wrong else ""
    if len(eligible) == 1:
        return ask_for_line(state, customer_id, eligible[0][0], wrong)
    if not eligible:
        return {"question": prefix + (NO_ELIGIBLE_ORDERS if own else NO_ORDERS_QUESTION), "fixed_wording": True}
    listed = "; ".join(f"{o.order_id} ({', '.join(ln.sku for ln in lines)})" for o, lines in eligible)
    return {"question": f"{prefix}Which order is this about? Orders you can return or exchange: {listed}.",
            "asked": "order", "fixed_wording": True}


def resolve_handoff(handoff: CustomerHandoff, orders: list[dict[str, Any]], state: Any, customer_id: str,
                    message: str = "", answering: bool = False) -> tuple[CustomerHandoff | None, dict]:
    """Deterministically check the grounded request against the customer's real orders.

    Returns (resolved handoff, {}) to proceed, or (None, clarify facts) naming what is missing, with
    the options open to the customer. The contract asks for clarification when the target isn't unique,
    so a customer's only delivered order, or an order's only line, is filled in, unless the message
    names an ID that doesn't exist: then the desk says so instead of silently using another item.
    An order the customer names themselves goes on even if it isn't eligible, so policy gives the reason.
    """
    by_id = {o.order_id: o for o in state.orders if o.customer_id == customer_id}
    unknown_id = _names_unknown_id(words_in(message), orders, state, answering)
    delivered = [o for o in orders if o["status"] == "delivered"]
    if handoff.intent == "unknown":
        return None, {"question": "Would you like a refund, or an exchange for a different size?"}
    order = by_id.get(handoff.order_id or "")
    if order is None and len(delivered) == 1 and not unknown_id:
        order = by_id[delivered[0]["order_id"]]
    if order is None:
        return None, ask_for_order(state, customer_id, unknown_id)
    line_ids = [ln.line_id for ln in order.lines]
    line_id = handoff.line_id
    if line_id is None and len(line_ids) == 1 and not unknown_id:
        line_id = line_ids[0]
    if line_id not in line_ids:
        return None, ask_for_line(state, customer_id, order, unknown_id)
    return handoff.model_copy(update={"order_id": order.order_id, "line_id": line_id,
                                      "needs_clarification": False, "clarification_question": None}), {}


def merge_pending_request(new: CustomerHandoff, pending: dict[str, Any] | None,
                          asked_for_item: bool = False) -> CustomerHandoff:
    """Update the request being clarified with the customer's answer.

    Anything the answer states wins (so the customer can change their mind). Anything it leaves out
    keeps its earlier value, except that a line belongs to its order and a size to an exchange. An
    answer to "which item?" keeps the remedy too, unless it also gives a new size: the model tends to
    invent a remedy for a bare answer like "the jacket".
    """
    if not pending:
        return new
    earlier = pending.get("intent") or "unknown"
    keep_remedy = new.intent == "unknown" or (asked_for_item and earlier != "unknown" and not new.requested_size)
    intent = earlier if keep_remedy else new.intent
    order_id = new.order_id or pending.get("order_id")
    same_order = order_id == pending.get("order_id")
    line_id = new.line_id or (pending.get("line_id") if same_order else None)
    size = (new.requested_size or pending.get("requested_size")) if intent == "exchange" else None
    opened = new.item_opened if new.item_opened is not None else pending.get("item_opened")
    return new.model_copy(update={"intent": intent, "order_id": order_id, "line_id": line_id, "requested_size": size,
                                  "item_opened": opened, "all_items": new.all_items or bool(pending.get("all_items"))})


def ask_for_size(sizes: list[str]) -> str:
    return f"Which size would you like instead? Available: {', '.join(sizes)}."


NOT_OPENED = re.compile(r"\b(unopened|sealed|unused|closed)\b|(\bnot|\bnever|n['\u2019]t)\s+(been\s+|yet\s+)?(opened|open|used)\b")
OPENED = re.compile(r"\b(opened|worn)\b")
OPENED_IN_AN_ANSWER = re.compile(r"\b(opened|open|used|worn)\b")


def item_condition_in(message: str, answering: bool = False) -> bool | None:
    """Whether the message says the item was opened: True, False, or None if it doesn't say or says both.

    Read by fixed rules, not the model, because "opened" changes the store. Outside an answer to the
    question, only clear words count: "open a return" is not an opened item.
    """
    text = (message or "").lower()
    says_not_opened = bool(NOT_OPENED.search(text))
    says_opened = bool((OPENED_IN_AN_ANSWER if answering else OPENED).search(NOT_OPENED.sub(" ", text)))
    if says_opened == says_not_opened:
        return None
    return says_opened


def size_in_answer(message: str, state: Any, pending: dict[str, Any]) -> str | None:
    """The size a reply to "which size?" names, for when the model misses a bare answer like "l".

    Only the catalog sizes of the line's product count, and only one named on its own: "I'm" is not an M.
    """
    order = next((o for o in state.orders if o.order_id == pending.get("order_id")), None)
    sku = next((ln.sku for ln in order.lines if ln.line_id == pending.get("line_id")), None) if order else None
    product = next((p.product_id for p in state.products if p.sku == sku), None)
    sizes = {p.size.upper(): p.size for p in state.products if p.product_id == product}
    words = {w.upper() for w in re.findall(r"(?<![\w'’-])\w+(?![\w'’-])", message or "")}
    named = [sizes[w] for w in words if w in sizes]
    return named[0] if len(named) == 1 else None


SIZE_WORDS = {"SMALL": "S", "MEDIUM": "M", "LARGE": "L", "EXTRA LARGE": "XL", "EXTRA-LARGE": "XL", "XLARGE": "XL",
              "X-LARGE": "XL", "EXTRA SMALL": "XS", "EXTRA-SMALL": "XS", "XSMALL": "XS", "X-SMALL": "XS"}


def catalog_size(handoff: CustomerHandoff, state: Any) -> CustomerHandoff:
    """The requested size as this product's catalog writes it: "large" is an L if the product has one."""
    if not handoff.requested_size or not handoff.order_id or not handoff.line_id:
        return handoff
    order = next((o for o in state.orders if o.order_id == handoff.order_id), None)
    sku = next((ln.sku for ln in order.lines if ln.line_id == handoff.line_id), None) if order else None
    product = next((p.product_id for p in state.products if p.sku == sku), None)
    sizes = {p.size.upper(): p.size for p in state.products if p.product_id == product}
    said = handoff.requested_size.strip().upper()
    size = sizes.get(said) or sizes.get(SIZE_WORDS.get(said, ""))
    return handoff.model_copy(update={"requested_size": size}) if size else handoff


def check_exchange_size(state: Any, customer_id: str, handoff: CustomerHandoff,
                        named_sku: str | None) -> tuple[str, list[str]]:
    """Whether an exchange can go on: ("ok" | "ask" | "refuse", the sizes open to the customer).

    The size is always the customer's choice, even with one left: no size given means ask. A size the
    product doesn't have is refused here. A size it has but can't supply (no stock, other price, the
    current size) goes on to policy, which denies it, as does an item with nothing open. While other
    sizes are open, either refusal leaves the request open for another answer.
    """
    if handoff.intent != "exchange" or named_sku:
        return "ok", []
    options = policy.list_exchange_options(state, customer_id, handoff.order_id, handoff.line_id)
    by_sku = {p.sku: p.size for p in state.products}
    open_sizes = [by_sku[o.replacement_sku] for o in options if o.eligible]
    if not handoff.requested_size:
        return ("ask" if open_sizes else "ok"), open_sizes
    offered = {by_sku[s].lower() for o in options for s in (o.replacement_sku, o.original_sku)}
    return ("ok" if not options or handoff.requested_size.lower() in offered else "refuse"), open_sizes


def same_as_waiting_proposal(resolved: CustomerHandoff, row: dict[str, Any], named_sku: str | None,
                             state: Any) -> bool:
    """Whether the request still describes the proposal the customer is deciding on."""
    if (resolved.intent, resolved.order_id, resolved.line_id) != (row["remedy"], row["order_id"], row["line_id"]):
        return False
    if resolved.intent != "exchange" or (not resolved.requested_size and not named_sku):
        return True
    if named_sku:
        return named_sku == row["replacement_sku"]
    size = next((p.size for p in state.products if p.sku == row["replacement_sku"]), "")
    return size.lower() == resolved.requested_size.lower()


def build_case_state(state: Any, customer_id: str, request_id: str, handoff: CustomerHandoff,
                     history: list[dict[str, Any]] | None) -> dict[str, Any]:
    """The case the Customer Resolution Agent hands on, in the contract's case shape, from the database.

    Scoped to this customer. Turns hold decisions, not text: the Operations Agent can call tools, so
    customer wording never reaches it. Faults are left out: they are never disclosed to an agent.
    """
    orders = [o for o in state.orders if o.customer_id == customer_id]
    skus = {ln.sku for o in orders for ln in o.lines}
    products = {p.product_id for p in state.products if p.sku in skus}
    return {"case_id": request_id, "customer_id": customer_id, "request_id": request_id,
            "initial_state": {"as_of": state.as_of.isoformat(), "currency": state.currency,
                              "products": [p.model_dump(mode="json") for p in state.products
                                           if p.product_id in products],
                              "orders": [o.model_dump(mode="json") for o in orders]},
            "turns": [{"turn": h.get("turn"), "decision": h.get("decision")} for h in history or []]}


def check_named_references(message: str, resolved: CustomerHandoff, customer_id: str,
                           state: Any) -> tuple[dict | None, str | None]:
    """Compare what the customer explicitly named with the resolved target, using only real store IDs.

    The customer agent only sees the customer's own orders, so it can silently substitute one of
    them for an order the customer named but does not own. Returns (clarify facts or None, the
    replacement SKU the customer named for an exchange or None; policy then decides on it).
    Another customer's order gets the same answer as an unknown order, so nothing leaks.
    """
    tokens = words_in(message)
    own = {o.order_id for o in state.orders if o.customer_id == customer_id}
    named_orders = {o.order_id for o in state.orders if o.order_id.upper() in tokens}
    if named_orders - own or (named_orders and resolved.order_id not in named_orders):
        return ask_for_order(state, customer_id, wrong=True), None
    order = next(o for o in state.orders if o.order_id == resolved.order_id)
    named_lines = {ln.line_id for ln in order.lines if ln.line_id.upper() in tokens}
    if named_lines and resolved.line_id not in named_lines:
        return ask_for_line(state, customer_id, order), None
    if resolved.intent != "exchange":
        return None, None
    original = next(ln.sku for ln in order.lines if ln.line_id == resolved.line_id)
    named_skus = sorted({p.sku for p in state.products if p.sku.upper() in tokens} - {original})
    if len(named_skus) > 1:
        return {"question": "Which replacement item would you like? Please name one."}, None
    return None, (named_skus[0] if named_skus else None)


def choose_replacement(options: Sequence[Any], state: Any, handoff: CustomerHandoff,
                       recommendation: dict[str, Any] | None, named_sku: str | None) -> tuple[str | None, str]:
    """Replacement SKU for an exchange and where it came from: the SKU the customer named, the option matching
    the size they asked for, or the agent's pick if it is one of the line's options. Never guesses a size;
    policy re-checks."""
    if named_sku:  # the customer named a catalog SKU: use exactly that; policy accepts or denies it
        return named_sku, "customer_named"
    if handoff.requested_size:  # the customer's words beat the agent's pick
        skus = {s for o in options for s in (o.replacement_sku, o.original_sku)}
        sizes = [p for p in state.products if p.sku in skus and p.size.lower() == handoff.requested_size.lower()]
        if len(sizes) == 1:
            return sizes[0].sku, "requested_size"
    picked = (recommendation or {}).get("replacement_sku")
    if picked and any(o.replacement_sku == picked for o in options):
        return picked, "operations_agent"
    # check_exchange_size already asked or refused, so only an item with nothing open gets here:
    # any option lets policy give the real reason.
    if options:
        return options[0].replacement_sku, "only_option"
    return None, "none"
