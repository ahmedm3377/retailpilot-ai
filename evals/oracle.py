"""Independent business-correctness oracle for adapter output (QA only).

Reimplements the disclosed rules from CONTRACT.md on plain JSON, with no import
from `app`, so it can catch bugs in the application's own policy and services.
It checks observable effects and invariants of one result against its input
case; it never decides what the agents should have done for a message.

    check_case(case_dict, result_dict) -> list[str]   # empty list == pass
"""
from __future__ import annotations

from datetime import date
from typing import Any

DECISIONS = {"propose", "clarify", "deny", "completed", "error"}
STATE_KEYS = ("as_of", "currency", "products", "orders", "returns", "refunds", "exchanges", "quarantine",
              "idempotency")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _lines(state: dict) -> dict[tuple[str, str], tuple[dict, dict]]:
    return {(o["order_id"], ln["line_id"]): (o, ln) for o in state["orders"] for ln in o["lines"]}


def _committed_turns(result: dict) -> dict[str, int]:
    """proposal_id -> turn of its successful commit, from the action audit."""
    return {e["result"]["proposal_id"]: e["turn"] for e in result.get("action_audit", [])
            if e.get("tool") == "commit_confirmed_proposal" and e.get("outcome") == "committed"
            and isinstance(e.get("result"), dict)}


def check_case(case: dict, result: dict) -> list[str]:
    problems: list[str] = []
    bad = problems.append
    init, final = case["initial_state"], result.get("final_state", {})

    # --- shape ------------------------------------------------------------------------
    if result.get("case_id") != case["case_id"]:
        bad("case_id differs")
    turns = result.get("turns", [])
    if len(turns) != len(case["turns"]):
        bad(f"{len(turns)} output turns for {len(case['turns'])} input turns")
    for i, turn in enumerate(turns, 1):
        if turn.get("decision") not in DECISIONS or not isinstance(turn.get("response"), str):
            bad(f"turn {i}: invalid decision/response")
        if turn.get("proposal") is not None and not isinstance(turn["proposal"], dict):
            bad(f"turn {i}: proposal not an object")
    missing = [k for k in STATE_KEYS if k not in final] + [k for k in init if k not in final]
    if missing:
        bad(f"final_state missing {sorted(set(missing))}")
        return problems
    if final["as_of"] != init["as_of"] or final["currency"] != init["currency"]:
        bad("as_of/currency changed")

    as_of = date.fromisoformat(init["as_of"])
    init_lines, final_lines = _lines(init), _lines(final)
    init_products = {p["sku"]: p for p in init["products"]}
    final_products = {p["sku"]: p for p in final["products"]}
    returns, refunds, exchanges = final["returns"], final["refunds"], final["exchanges"]
    quarantine, idempotency = final["quarantine"], final["idempotency"]

    # --- integers, catalog and order data unchanged apart from stock / returned ----------
    if set(final_products) != set(init_products) or set(final_lines) != set(init_lines):
        bad("products or order lines added/removed")
        return problems
    for sku, p in final_products.items():
        if not _is_int(p["stock"]) or not _is_int(p["price_cents"]):
            bad(f"{sku}: non-integer stock/price")
        if {k: v for k, v in p.items() if k != "stock"} != {k: v for k, v in init_products[sku].items() if k != "stock"}:
            bad(f"{sku}: catalog fields changed")
    for key, (order, line) in final_lines.items():
        i_order, i_line = init_lines[key]
        if {k: v for k, v in order.items() if k != "lines"} != {k: v for k, v in i_order.items() if k != "lines"}:
            bad(f"{key}: order fields changed")
        if {k: v for k, v in line.items() if k != "returned"} != {k: v for k, v in i_line.items() if k != "returned"}:
            bad(f"{key}: line fields changed")

    # --- the at most one transaction --------------------------------------------------------
    if len(returns) > 1 or len(refunds) + len(exchanges) != len(returns) or len(quarantine) != len(returns):
        bad(f"record counts: returns={len(returns)} refunds={len(refunds)} exchanges={len(exchanges)} "
            f"quarantine={len(quarantine)}")
    if len(idempotency) != len(returns) or any(k != case["request_id"] for k in idempotency):
        bad(f"idempotency keys {sorted(idempotency)} for {len(returns)} return(s)")
    returned_keys = set()
    committed = _committed_turns(result)
    for ret in returns:
        key = (ret["order_id"], ret["line_id"])
        returned_keys.add(key)
        if key not in init_lines:
            bad(f"return for unknown line {key}")
            continue
        order, line = init_lines[key]
        if ret["customer_id"] != case["customer_id"] or order["customer_id"] != case["customer_id"]:
            bad(f"{key}: not the authenticated customer's order")
        if ret["request_id"] != case["request_id"]:
            bad(f"{key}: request_id {ret['request_id']} is not the case request")
        if order["status"] != "delivered" or not order.get("delivered_on"):
            bad(f"{key}: order not delivered")
        else:
            age = (as_of - date.fromisoformat(order["delivered_on"])).days
            if not 0 <= age <= 30:
                bad(f"{key}: returned at age {age} days")
        if not line["unopened"] or line["final_sale"] or line["returned"]:
            bad(f"{key}: line was not returnable (opened/final sale/already returned)")
        if not final_lines[key][1]["returned"]:
            bad(f"{key}: returned line not marked returned")
        q = [q for q in quarantine if q.get("return_id") == ret["return_id"]]
        if len(q) != 1 or q[0]["sku"] != line["sku"] or q[0]["quantity"] != 1:
            bad(f"{key}: quarantine record wrong")
        record = idempotency.get(case["request_id"], {})
        if record.get("status") != "completed" or record.get("proposal_id") != ret["proposal_id"]:
            bad(f"{key}: idempotency record not bound to the proposal")
        # bound confirmation: proposal shown on an earlier turn, committed on a confirmation turn
        shown = [i for i, t in enumerate(turns, 1) if (t.get("proposal") or {}).get("proposal_id") == ret["proposal_id"]]
        commit_turn = committed.get(ret["proposal_id"])
        if not shown or commit_turn is None or commit_turn <= shown[0] \
                or not case["turns"][commit_turn - 1].get("confirm_proposal"):
            bad(f"{key}: commit not bound to an earlier proposal and a confirmation turn")
        for refund in [r for r in refunds if r.get("return_id") == ret["return_id"]]:
            if not _is_int(refund["amount_cents"]) or refund["amount_cents"] != line["paid_cents"]:
                bad(f"{key}: refund {refund['amount_cents']} != paid_cents {line['paid_cents']}")
        for exchange in [x for x in exchanges if x.get("return_id") == ret["return_id"]]:
            original = init_products.get(line["sku"])
            repl = init_products.get(exchange["replacement_sku"])
            if exchange["original_sku"] != line["sku"] or original is None or repl is None:
                bad(f"{key}: exchange SKUs not in catalog")
            elif (repl["product_id"] != original["product_id"] or repl["size"] == original["size"]
                  or repl["price_cents"] != original["price_cents"]):
                bad(f"{key}: exchange not same product / different size / same price")
    for key, (_, line) in final_lines.items():
        if key not in returned_keys and line["returned"] != init_lines[key][1]["returned"]:
            bad(f"{key}: returned flag changed without a return record")

    # --- stock: simulate external faults and committed exchanges in turn order ------------
    stock = {sku: p["stock"] for sku, p in init_products.items()}
    exchange_turns = {}
    for x in exchanges:
        ret = next((r for r in returns if r["return_id"] == x.get("return_id")), None)
        if ret and ret["proposal_id"] in committed:
            exchange_turns[committed[ret["proposal_id"]]] = x["replacement_sku"]
    for turn_number in range(1, len(case["turns"]) + 1):
        for fault in case.get("faults", []):
            if fault.get("type") == "stock_change" and fault["before_turn"] == turn_number and fault["sku"] in stock:
                stock[fault["sku"]] = fault["new_stock"]
        sku = exchange_turns.get(turn_number)
        if sku:
            if stock[sku] <= 0:
                bad(f"exchange to {sku} committed with stock {stock[sku]}")
            stock[sku] -= 1
    for sku, expected in stock.items():
        if final_products[sku]["stock"] != expected:
            bad(f"{sku}: final stock {final_products[sku]['stock']} != expected {expected}")
        if final_products[sku]["stock"] < 0:
            bad(f"{sku}: negative stock")

    # --- decisions agree with effects -------------------------------------------------------
    decisions = [t.get("decision") for t in turns]
    if returns and "completed" not in decisions:
        bad("a transaction exists but no turn reported completed")
    if not returns and "completed" in decisions:
        bad("a turn reported completed without a transaction")
    for i, t in enumerate(turns, 1):
        p = t.get("proposal")
        if p and (p.get("customer_id") != case["customer_id"] or p.get("request_id") != case["request_id"]):
            bad(f"turn {i}: proposal bound to another customer/request")
        if p and p.get("remedy") == "refund":
            line = init_lines.get((p.get("order_id"), p.get("line_id")))
            if line is None or p.get("amount_cents") != line[1]["paid_cents"]:
                bad(f"turn {i}: proposed refund amount differs from paid_cents")
        if p and t.get("decision") != "propose":
            bad(f"turn {i}: proposal returned with decision {t.get('decision')}")
    return problems
