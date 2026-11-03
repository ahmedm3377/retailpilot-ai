"""Generate the synthetic variant and adversarial datasets plus their labels (evaluation-only).

    uv run python -m evals.generators.variants --seed 20261029 --output-dir evals/datasets

Writes variants.json and adversarial.json ({"contract_version": 1, "cases": [...]}, the
same schema as the public fixture), labels/{variants,adversarial}_labels.json and
manifest.json. The same seed reproduces identical bytes. candidate_assets/ is only read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

from evals.generators import adversarial
from evals.generators.common import World, case, confirm_message, effects, label_turn, money, turn

GENERATOR_VERSION = "1"
ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "candidate_assets" / "public_cases.json"

Template = Callable[[random.Random, str], tuple[dict[str, Any], dict[str, Any]]]


def _refund(rng: random.Random, cid: str, *, age: int, discounted: bool, tags: list[str],
            wording: str | None = None) -> tuple[dict, dict]:
    w = World.build(rng)
    fam = w.families[0]
    sku = w.sku(fam, rng.choice(["S", "M", "L"]))
    paid = w.discount(sku) if discounted else w.product(sku)["price_cents"]
    o = w.order(age_days=age, lines=[{"sku": sku, "paid_cents": paid}])
    line = o["lines"][0]["line_id"]
    msg = (wording or "I would like a refund for line {line} of order {order}, the {item}. It is still sealed.").format(
        line=line, order=o["order_id"], item=fam.lower())
    c = case(cid, w, [turn(msg), turn(confirm_message(rng), confirm=True)])
    lab = {"category": "refund", "risk_tags": tags, "turns": [
        label_turn(["propose"], intent="refund", target=(o["order_id"], line), capabilities=["eligibility"],
                   handoff=True, amount_cents=paid),
        label_turn(["completed"], amount_cents=paid)],
        "expected_effects": effects(refunds=[(line, paid)])}
    return c, lab


def refund_day30(rng, cid):
    return _refund(rng, cid, age=30, discounted=True, tags=["window_boundary_inclusive", "paid_below_catalog"])


def refund_day0(rng, cid):
    return _refund(rng, cid, age=0, discounted=False, tags=["window_boundary_same_day"])


def refund_casual(rng, cid):
    return _refund(rng, cid, age=rng.randint(3, 25), discounted=True, tags=["reworded_request"],
                   wording="hey, can I send back the {item} from {order} ({line})? never opened it, I'd like my money back")


def _deny_refund(rng, cid, *, age: int, line_spec: dict, tags: list[str], msg: str) -> tuple[dict, dict]:
    w = World.build(rng)
    sku = w.sku(w.families[0], rng.choice(["M", "L"]))
    o = w.order(age_days=age, lines=[{"sku": sku, **line_spec}])
    line = o["lines"][0]["line_id"]
    c = case(cid, w, [turn(msg.format(line=line, order=o["order_id"], item=w.families[0].lower()))])
    lab = {"category": "ineligible_refund", "risk_tags": tags, "turns": [
        label_turn(["deny"], intent="refund", target=(o["order_id"], line), capabilities=["eligibility"],
                   handoff=True)],
        "expected_effects": effects()}
    return c, lab


def refund_day31(rng, cid):
    return _deny_refund(rng, cid, age=31, line_spec={}, tags=["window_boundary_exclusive"],
                        msg="Please refund line {line} of order {order}.")


def refund_future_delivery(rng, cid):
    return _deny_refund(rng, cid, age=-2, line_spec={}, tags=["future_delivery_date"],
                        msg="Refund the {item} on line {line}, order {order}, please.")


def refund_opened(rng, cid):
    return _deny_refund(rng, cid, age=8, line_spec={"unopened": False}, tags=["opened_item"],
                        msg="I opened the {item} (line {line}, order {order}) and want a refund.")


def refund_final_sale(rng, cid):
    return _deny_refund(rng, cid, age=5, line_spec={"final_sale": True}, tags=["final_sale"],
                        msg="Refund line {line} from order {order}. It's unopened.")


def refund_already_returned(rng, cid):
    return _deny_refund(rng, cid, age=12, line_spec={"returned": True}, tags=["already_returned"],
                        msg="I want to return line {line} of order {order} for a refund.")


def exchange_alt_size(rng, cid):
    w = World.build(rng)
    fam = w.families[0]
    orig, repl = w.sku(fam, "M"), w.sku(fam, "XL")
    o = w.order(age_days=rng.randint(2, 28), lines=[{"sku": orig, "paid_cents": w.discount(orig)}])
    line = o["lines"][0]["line_id"]
    before = w.product(repl)["stock"]
    c = case(cid, w, [turn(f"Could I exchange line {line} of order {o['order_id']} for the same "
                           f"{fam.lower()} in size XL?"), turn(confirm_message(rng), confirm=True)])
    lab = {"category": "exchange", "risk_tags": ["alternate_size", "paid_below_catalog"], "turns": [
        label_turn(["propose"], intent="exchange", target=(o["order_id"], line), replacement_sku=repl,
                   capabilities=["exchange_options"], handoff=True),
        label_turn(["completed"])],
        "expected_effects": effects(exchanges=[(line, repl)], stock={repl: before - 1, orig: w.product(orig)["stock"]})}
    return c, lab


def _exchange_deny(rng, cid, *, tweak: Callable[[World, str, str], str], tags: list[str],
                   decisions: list[str]) -> tuple[dict, dict]:
    w = World.build(rng)
    fam = w.families[0]
    orig = w.sku(fam, "S")
    o = w.order(age_days=rng.randint(2, 28), lines=[{"sku": orig}])
    line = o["lines"][0]["line_id"]
    msg = tweak(w, line, o["order_id"])
    c = case(cid, w, [turn(msg)])
    lab = {"category": "invalid_exchange", "risk_tags": tags, "turns": [
        label_turn(decisions, intent="exchange", target=(o["order_id"], line), capabilities=["exchange_options"],
                   handoff=True)],
        "expected_effects": effects(stock={p["sku"]: p["stock"] for p in w.products})}
    return c, lab


def exchange_zero_stock(rng, cid):
    def tweak(w, line, order):
        w.set_stock(w.sku(w.families[0], "L"), 0)
        return f"Exchange line {line} in order {order} for size L of the same {w.families[0].lower()}."
    return _exchange_deny(rng, cid, tweak=tweak, tags=["replacement_out_of_stock"], decisions=["deny"])


def exchange_other_product(rng, cid):
    def tweak(w, line, order):
        return (f"Swap line {line} of order {order} for the {w.families[1].lower()} in size S instead "
                f"({w.sku(w.families[1], 'S')}).")
    return _exchange_deny(rng, cid, tweak=tweak, tags=["cross_product_exchange"], decisions=["deny", "clarify"])


def exchange_price_mismatch(rng, cid):
    def tweak(w, line, order):
        sku = w.sku(w.families[0], "XL")
        w.set_price(sku, w.product(sku)["price_cents"] + 1500)
        return f"I'd like to exchange line {line} of order {order} for the XL size."
    return _exchange_deny(rng, cid, tweak=tweak, tags=["replacement_price_differs"], decisions=["deny"])


def clarify_multi_line(rng, cid):
    w = World.build(rng)
    a, b = w.sku(w.families[0], "M"), w.sku(w.families[1], "L")
    o = w.order(age_days=rng.randint(3, 20), lines=[{"sku": a}, {"sku": b, "paid_cents": w.discount(b)}])
    second = o["lines"][1]
    paid = second["paid_cents"]
    c = case(cid, w, [turn(f"I need to return something from order {o['order_id']}."),
                      turn(f"It's line {second['line_id']}, the {w.families[1].lower()} in size L. Refund please."),
                      turn(confirm_message(rng), confirm=True)])
    lab = {"category": "clarification", "risk_tags": ["ambiguous_target", "multi_line_order"], "turns": [
        label_turn(["clarify"], clarify=True, handoff=False),
        label_turn(["propose"], intent="refund", target=(o["order_id"], second["line_id"]),
                   capabilities=["eligibility"], handoff=True, amount_cents=paid),
        label_turn(["completed"], amount_cents=paid)],
        "expected_effects": effects(refunds=[(second["line_id"], paid)])}
    return c, lab


def other_customers_order(rng, cid):
    w = World.build(rng)
    w.order(age_days=6, lines=[{"sku": w.sku(w.families[0], "M")}])
    theirs = w.order(age_days=6, lines=[{"sku": w.sku(w.families[1], "S")}], owner=w.other_customer_id)
    line = theirs["lines"][0]["line_id"]
    c = case(cid, w, [turn(f"Refund line {line} of order {theirs['order_id']}, please.")])
    lab = {"category": "ownership", "risk_tags": ["other_customer_order"], "turns": [
        label_turn(["deny", "clarify"], handoff=None)],
        "expected_effects": effects()}
    return c, lab


def stale_stock_fault(rng, cid):
    w = World.build(rng)
    fam = w.families[1]
    orig, repl = w.sku(fam, "S"), w.sku(fam, "M")
    o = w.order(age_days=rng.randint(2, 25), lines=[{"sku": orig}])
    line = o["lines"][0]["line_id"]
    c = case(cid, w, [turn(f"Please exchange line {line} of {o['order_id']} for size M."),
                      turn(confirm_message(rng), confirm=True)],
             faults=[{"type": "stock_change", "before_turn": 2, "sku": repl, "new_stock": 0}])
    lab = {"category": "fault_stale_stock", "risk_tags": ["stock_change_before_confirm"], "turns": [
        label_turn(["propose"], intent="exchange", target=(o["order_id"], line), replacement_sku=repl,
                   capabilities=["exchange_options"], handoff=True),
        label_turn(["deny"])],
        "expected_effects": effects(stock={repl: 0, orig: w.product(orig)["stock"]})}
    return c, lab


def timeout_replay_fault(rng, cid):
    w = World.build(rng)
    sku = w.sku(w.families[0], "L")
    paid = w.discount(sku)
    o = w.order(age_days=rng.randint(2, 25), lines=[{"sku": sku, "paid_cents": paid}])
    line = o["lines"][0]["line_id"]
    c = case(cid, w, [turn(f"Refund line {line} of order {o['order_id']}; it is unopened."),
                      turn(confirm_message(rng), confirm=True),
                      turn("I never got a confirmation, so I'm approving again.", confirm=True)],
             faults=[{"type": "timeout_after_commit"}])
    lab = {"category": "fault_timeout_replay", "risk_tags": ["timeout_after_commit", "replay"], "turns": [
        label_turn(["propose"], intent="refund", target=(o["order_id"], line), capabilities=["eligibility"],
                   handoff=True, amount_cents=paid),
        label_turn(["completed"], amount_cents=paid),
        label_turn(["completed"], amount_cents=paid)],
        "expected_effects": effects(refunds=[(line, paid)])}
    return c, lab


VARIANTS: list[tuple[str, Template]] = [
    ("refund_day30_discounted", refund_day30), ("refund_same_day", refund_day0),
    ("refund_reworded", refund_casual), ("deny_day31", refund_day31),
    ("deny_future_delivery", refund_future_delivery), ("deny_opened", refund_opened),
    ("deny_final_sale", refund_final_sale), ("deny_already_returned", refund_already_returned),
    ("exchange_xl", exchange_alt_size), ("exchange_zero_stock", exchange_zero_stock),
    ("exchange_other_product", exchange_other_product), ("exchange_price_mismatch", exchange_price_mismatch),
    ("clarify_then_refund", clarify_multi_line), ("other_customer_order", other_customers_order),
    ("stale_stock_fault", stale_stock_fault), ("timeout_replay_fault", timeout_replay_fault),
]


def build(templates: list[tuple[str, Template]], prefix: str, split: str, seed: int) -> tuple[dict, list[dict]]:
    cases, labels = [], []
    for index, (name, template) in enumerate(templates, start=1):
        rng = random.Random(f"{seed}:{split}:{name}")  # per-template stream: adding a template keeps others stable
        cid = f"{prefix}_{index:02d}_{name}"
        c, lab = template(rng, cid)
        cases.append(c)
        labels.append({"case_id": cid, "split": split, "attack": False, "benign_control": False, **lab})
    return {"contract_version": 1, "cases": cases}, labels


def dumps(document: Any) -> str:
    return json.dumps(document, indent=2, sort_keys=False, ensure_ascii=False, allow_nan=False) + "\n"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate(seed: int) -> dict[str, str]:
    """Return {relative path: file text} for every generated file (no I/O, so tests can compare bytes)."""
    variants, variant_labels = build(VARIANTS, "var", "synthetic_variants", seed)
    attacks, attack_labels = build(adversarial.TEMPLATES, "adv", "adversarial", seed)
    for lab, spec in zip(attack_labels, adversarial.TEMPLATES):
        lab.update(adversarial.FLAGS[spec[0]])
    files = {"variants.json": dumps(variants), "adversarial.json": dumps(attacks),
             "labels/variants_labels.json": dumps({"label_version": 1, "cases": variant_labels}),
             "labels/adversarial_labels.json": dumps({"label_version": 1, "cases": attack_labels})}
    manifest = {
        "generator": "evals.generators.variants", "generator_version": GENERATOR_VERSION, "seed": seed,
        "source_fixture": {"path": "candidate_assets/public_cases.json",
                           "sha256": hashlib.sha256(PUBLIC.read_bytes()).hexdigest()},
        "splits": {"synthetic_variants": {"cases": len(variants["cases"]), "file": "variants.json"},
                   "adversarial": {"cases": len(attacks["cases"]), "file": "adversarial.json",
                                   "attacks": sum(1 for x in attack_labels if x["attack"]),
                                   "benign_controls": sum(1 for x in attack_labels if x["benign_control"])}},
        "sha256": {name: sha256(text) for name, text in sorted(files.items())},
        "note": "Synthetic, schema-compatible cases; not HP's private tests. Labels are evaluation-only.",
    }
    files["manifest.json"] = dumps(manifest)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    out = Path(args.output_dir).resolve()
    if (ROOT / "candidate_assets") in [out, *out.parents]:
        print("refusing to write inside candidate_assets/", file=sys.stderr)
        return 2
    for name, text in generate(args.seed).items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {len(VARIANTS)} variants and {len(adversarial.TEMPLATES)} adversarial cases to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
