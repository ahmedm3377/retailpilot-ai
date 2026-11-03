from __future__ import annotations

import pytest

from app.domain.outcomes import WorkflowCode
from retail_domain.policy import ReasonCode
from app.services.transactions import commit_confirmed_proposal
from tests.conftest import count, make_line, make_state

pytestmark = pytest.mark.postgres


@pytest.mark.parametrize("remedy, replacement", [("refund", None), ("exchange", "TEE-L")])
def test_proposing_has_no_business_effects(pg_store, ops, load, remedy, replacement):
    load("public_02_exchange")
    before = pg_store.export_final_state()
    outcome = ops.create_proposal("C100", "REQ-1", "O100", "L1", remedy, replacement)
    assert outcome.kind == "proposed" and outcome.proposal.status == "pending"
    assert pg_store.export_final_state() == before
    assert count(pg_store, "proposals") == 1


def test_refund_amount_is_computed_by_the_service(pg_store, ops, load):
    load("public_01_refund_boundary")
    proposal = ops.create_proposal("C100", "REQ-1", "O100", "L1", "refund").proposal
    assert (proposal.amount_cents, proposal.paid_cents) == (3000, 3000)
    assert proposal.replacement_sku is None


def test_exchange_proposal_binds_skus_and_charges_nothing(pg_store, ops, load):
    load("public_02_exchange")
    proposal = ops.create_proposal("C100", "REQ-1", "O100", "L1", "exchange", "TEE-L").proposal
    assert (proposal.original_sku, proposal.replacement_sku) == ("TEE-M", "TEE-L")
    assert proposal.amount_cents == 0 and proposal.paid_cents == 3000


def test_ambiguous_target_yields_clarification_not_a_proposal(pg_store, ops, load):
    load("public_04_clarify")
    outcome = ops.create_proposal("C100", "REQ-1", "O100", None, "refund")
    assert outcome.kind == "clarify" and outcome.proposal is None
    assert outcome.candidate_line_ids == ("L1", "L2")
    assert count(pg_store, "proposals") == 0


@pytest.mark.parametrize("state_kwargs, customer, remedy, replacement, code", [
    ({"age_days": 31}, "CUST-9", "refund", None, ReasonCode.DELIVERY_DATE_OUT_OF_RANGE),
    ({"lines": [make_line(final_sale=True)]}, "CUST-9", "refund", None, ReasonCode.FINAL_SALE),
    ({}, "INTRUDER", "refund", None, ReasonCode.CUSTOMER_MISMATCH),
    ({}, "CUST-9", "exchange", "GADGET-B", ReasonCode.INVALID_REPLACEMENT),
    ({}, "CUST-9", "exchange", "WIDGET-D", ReasonCode.REPLACEMENT_OUT_OF_STOCK),
    ({}, "CUST-9", "store_credit", None, WorkflowCode.UNSUPPORTED_REMEDY),
])
def test_ineligible_requests_are_denied_without_a_proposal(pg_store, ops, load, state_kwargs, customer,
                                                           remedy, replacement, code):
    load(make_state(**state_kwargs))
    outcome = ops.create_proposal(customer, "REQ-1", "ORD-77", "LN-1", remedy, replacement)
    assert outcome.kind == "deny" and outcome.reason_code is code and outcome.proposal is None
    assert count(pg_store, "proposals") == 0


def test_completed_request_reports_existing_outcome(pg_store, ops, load):
    load("public_01_refund_boundary")
    proposal = ops.create_proposal("C100", "REQ-1", "O100", "L1", "refund").proposal
    committed = commit_confirmed_proposal(pg_store, "C100", "REQ-1", proposal.proposal_id)

    again = ops.create_proposal("C100", "REQ-1", "O100", "L1", "refund")
    assert again.kind == "already_completed" and again.proposal is None
    assert again.existing_result == committed
    assert count(pg_store, "proposals") == 1


def test_other_customer_cannot_see_completed_request(pg_store, ops, load):
    load("public_01_refund_boundary")
    proposal = ops.create_proposal("C100", "REQ-1", "O100", "L1", "refund").proposal
    commit_confirmed_proposal(pg_store, "C100", "REQ-1", proposal.proposal_id)
    outcome = ops.create_proposal("C999", "REQ-1", "O100", "L1", "refund")
    assert outcome.kind == "conflict" and outcome.existing_result is None


