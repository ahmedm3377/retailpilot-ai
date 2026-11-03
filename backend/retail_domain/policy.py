"""Pure, read-only return and exchange policy (CONTRACT.md "Disclosed business rules").

Every function takes a State and returns an immutable result with a stable
ReasonCode. Nothing here mutates state, reads the system clock or calls an LLM;
commits belong to a later step. Results carry plain values, never references to
the state's mutable model objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from retail_domain.models import Line, Order, Product, State

RETURN_WINDOW_DAYS = 30
DELIVERED_STATUS = "delivered"


class ReasonCode(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    CUSTOMER_MISMATCH = "CUSTOMER_MISMATCH"
    ORDER_NOT_FOUND = "ORDER_NOT_FOUND"
    LINE_NOT_FOUND = "LINE_NOT_FOUND"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    NOT_DELIVERED = "NOT_DELIVERED"
    DELIVERY_DATE_OUT_OF_RANGE = "DELIVERY_DATE_OUT_OF_RANGE"
    ITEM_OPENED = "ITEM_OPENED"
    FINAL_SALE = "FINAL_SALE"
    ALREADY_RETURNED = "ALREADY_RETURNED"
    SKU_NOT_FOUND = "SKU_NOT_FOUND"
    INVALID_REPLACEMENT = "INVALID_REPLACEMENT"
    REPLACEMENT_OUT_OF_STOCK = "REPLACEMENT_OUT_OF_STOCK"


@dataclass(frozen=True)
class LineResolution:
    ok: bool
    reason_code: ReasonCode
    order_id: str | None = None
    line_id: str | None = None
    sku: str | None = None
    # Lines the customer could mean when the target is not unique.
    candidate_line_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    reason_code: ReasonCode
    order_id: str | None = None
    line_id: str | None = None
    sku: str | None = None
    paid_cents: int | None = None
    delivery_age_days: int | None = None
    candidate_line_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RefundAssessment:
    eligible: bool
    reason_code: ReasonCode
    remedy: Literal["refund"] = "refund"
    order_id: str | None = None
    line_id: str | None = None
    sku: str | None = None
    refund_amount_cents: int | None = None
    candidate_line_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExchangeAssessment:
    eligible: bool
    reason_code: ReasonCode
    remedy: Literal["exchange"] = "exchange"
    order_id: str | None = None
    line_id: str | None = None
    original_sku: str | None = None
    replacement_sku: str | None = None
    price_cents: int | None = None
    replacement_stock: int | None = None
    candidate_line_ids: tuple[str, ...] = ()


def _find_order(state: State, order_id: str) -> Order | None:
    return next((o for o in state.orders if o.order_id == order_id), None)


def _find_product(state: State, sku: str) -> Product | None:
    return next((p for p in state.products if p.sku == sku), None)


def _locate(state: State, customer_id: str, order_id: str | None, line_id: str | None
            ) -> tuple[LineResolution, Order | None, Line | None]:
    if not order_id:
        return LineResolution(False, ReasonCode.CLARIFICATION_REQUIRED), None, None
    order = _find_order(state, order_id)
    if order is None:
        return LineResolution(False, ReasonCode.ORDER_NOT_FOUND, order_id=order_id), None, None
    if order.customer_id != customer_id:
        # Callers must not reveal another customer's order details.
        return LineResolution(False, ReasonCode.CUSTOMER_MISMATCH, order_id=order_id), None, None

    all_ids = tuple(line.line_id for line in order.lines)
    if not line_id:
        if len(order.lines) == 1:
            line = order.lines[0]
        else:
            return LineResolution(False, ReasonCode.CLARIFICATION_REQUIRED, order_id=order_id,
                                  candidate_line_ids=all_ids), order, None
    else:
        matches = [line for line in order.lines if line.line_id == line_id]
        if not matches:
            return LineResolution(False, ReasonCode.LINE_NOT_FOUND, order_id=order_id,
                                  line_id=line_id, candidate_line_ids=all_ids), order, None
        if len(matches) > 1:
            return LineResolution(False, ReasonCode.CLARIFICATION_REQUIRED, order_id=order_id,
                                  line_id=line_id, candidate_line_ids=all_ids), order, None
        line = matches[0]
    return LineResolution(True, ReasonCode.ELIGIBLE, order_id=order_id, line_id=line.line_id,
                          sku=line.sku), order, line


def resolve_order_line(state: State, customer_id: str, order_id: str | None,
                       line_id: str | None) -> LineResolution:
    """Uniquely identify one of the authenticated customer's order lines.

    A missing order ID, or a missing line ID on a multi-line order, needs
    clarification; the policy never picks a line arbitrarily.
    """
    return _locate(state, customer_id, order_id, line_id)[0]


def check_return_eligibility(state: State, customer_id: str, order_id: str | None,
                             line_id: str | None) -> EligibilityResult:
    """Whether the unit may be returned at all, regardless of remedy."""
    resolution, order, line = _locate(state, customer_id, order_id, line_id)
    if order is None or line is None:
        return EligibilityResult(False, resolution.reason_code, order_id=resolution.order_id,
                                 line_id=resolution.line_id,
                                 candidate_line_ids=resolution.candidate_line_ids)

    def result(code: ReasonCode, age: int | None = None) -> EligibilityResult:
        return EligibilityResult(code is ReasonCode.ELIGIBLE, code, order_id=order.order_id,
                                 line_id=line.line_id, sku=line.sku, paid_cents=line.paid_cents,
                                 delivery_age_days=age)

    if order.status != DELIVERED_STATUS or order.delivered_on is None:
        return result(ReasonCode.NOT_DELIVERED)
    age = (state.as_of - order.delivered_on).days
    if not 0 <= age <= RETURN_WINDOW_DAYS:
        return result(ReasonCode.DELIVERY_DATE_OUT_OF_RANGE, age)
    if not line.unopened:
        return result(ReasonCode.ITEM_OPENED, age)
    if line.final_sale:
        return result(ReasonCode.FINAL_SALE, age)
    if line.returned:
        return result(ReasonCode.ALREADY_RETURNED, age)
    return result(ReasonCode.ELIGIBLE, age)


def assess_refund(state: State, customer_id: str, order_id: str | None,
                  line_id: str | None) -> RefundAssessment:
    """Refund eligibility; the amount is exactly the line's paid_cents."""
    e = check_return_eligibility(state, customer_id, order_id, line_id)
    return RefundAssessment(e.eligible, e.reason_code, order_id=e.order_id, line_id=e.line_id,
                            sku=e.sku, refund_amount_cents=e.paid_cents if e.eligible else None,
                            candidate_line_ids=e.candidate_line_ids)


def assess_exchange(state: State, customer_id: str, order_id: str | None, line_id: str | None,
                    replacement_sku: str | None) -> ExchangeAssessment:
    """Same product_id, different size, same catalog price, replacement stock > 0."""
    e = check_return_eligibility(state, customer_id, order_id, line_id)

    def result(code: ReasonCode, replacement: Product | None = None,
               price: int | None = None) -> ExchangeAssessment:
        return ExchangeAssessment(
            code is ReasonCode.ELIGIBLE, code, order_id=e.order_id, line_id=e.line_id,
            original_sku=e.sku, replacement_sku=replacement_sku, price_cents=price,
            replacement_stock=replacement.stock if replacement else None,
            candidate_line_ids=e.candidate_line_ids)

    if not e.eligible:
        return result(e.reason_code)
    if not replacement_sku:
        return result(ReasonCode.CLARIFICATION_REQUIRED)
    original = _find_product(state, e.sku)
    replacement = _find_product(state, replacement_sku)
    if original is None or replacement is None:
        return result(ReasonCode.SKU_NOT_FOUND, replacement)
    if (replacement.sku == original.sku
            or replacement.product_id != original.product_id
            or replacement.size == original.size
            or replacement.price_cents != original.price_cents):
        return result(ReasonCode.INVALID_REPLACEMENT, replacement)
    if replacement.stock <= 0:
        return result(ReasonCode.REPLACEMENT_OUT_OF_STOCK, replacement)
    return result(ReasonCode.ELIGIBLE, replacement, replacement.price_cents)


def list_exchange_options(state: State, customer_id: str, order_id: str | None,
                          line_id: str | None) -> tuple[ExchangeAssessment, ...]:
    """Assess every other size of the line's product, in catalog order.

    Returns an empty tuple when the line itself cannot be resolved or returned,
    or its SKU is not in the catalog; use assess_exchange for the reason.
    """
    e = check_return_eligibility(state, customer_id, order_id, line_id)
    if not e.eligible:
        return ()
    original = _find_product(state, e.sku)
    if original is None:
        return ()
    return tuple(
        assess_exchange(state, customer_id, order_id, line_id, p.sku)
        for p in state.products
        if p.product_id == original.product_id and p.sku != original.sku
    )
