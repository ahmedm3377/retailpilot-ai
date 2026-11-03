from __future__ import annotations

import pytest

from app.agents.messages import fallback_message
from app.agents.settings import AgentSettings
from app.guardrails.service import GuardrailService, consistency_problem

FACTS = {"decision": "propose", "proposal_id": "PROP-1", "remedy": "refund", "order_id": "O100",
         "line_id": "L1", "refund_amount_cents": 3000, "amount_text": "$30.00"}


def test_toxic_output_replaced_by_trusted_fallback(guards):
    fallback = fallback_message("propose", FACTS)
    out = guards.validate_customer_output("You stupid idiot, take your $30.00 and get lost.", FACTS, fallback)
    assert out.used_fallback and out.text == fallback
    assert any(e.validator == "ToxicLanguage" and e.outcome == "blocked" for e in out.events)
    assert "$30.00" in out.text and "PROP-1" in out.text  # fallback only restates trusted facts


def test_benign_output_passes_unchanged(guards):
    text = "I can refund line L1 of order O100 for $30.00. Please confirm the proposal."
    out = guards.validate_customer_output(text, FACTS, "FALLBACK")
    assert not out.used_fallback and out.text == text


def test_angry_customer_is_processed_not_refused(guards):
    result = guards.validate_and_sanitize_customer_input("Your service is crap and you are useless idiots. "
                                                         "Refund L1 of O100.")
    assert result.allowed
    assert any(e.validator == "ToxicLanguage" and e.outcome == "flagged" and e.reason_code == "ABUSIVE_TONE_NOTED"
               for e in result.events)


@pytest.mark.parametrize("text, problem", [
    ("Your refund has been processed.", "claims completion"),
    ("I can refund $40.00 for L1.", "not a trusted amount"),
    ("Please confirm proposal PROP-9.", "unknown proposal"),
])
def test_unsupported_claims_are_caught(text, problem):
    assert problem in consistency_problem(text, FACTS)


def test_completion_claim_allowed_after_commit():
    assert consistency_problem("Your refund of $30.00 has been processed.",
                               {**FACTS, "decision": "completed"}) is None


def test_fallback_for_every_decision_uses_only_trusted_values():
    deny = fallback_message("deny", {"reason_code": "DELIVERY_DATE_OUT_OF_RANGE"})
    assert "30-day" in deny and "$" not in deny
    assert "L1, L2" in fallback_message("clarify", {"candidate_line_ids": ["L1", "L2"]})
    assert "Nothing was changed" in fallback_message("error", {"reason_code": "MODEL_UNAVAILABLE"})
    exchange = fallback_message("completed", {"remedy": "exchange", "original_sku": "TEE-M",
                                              "replacement_sku": "TEE-L", "order_id": "O100", "line_id": "L1"})
    assert "TEE-M" in exchange and "TEE-L" in exchange


def test_output_fails_closed_when_toxicity_validator_unavailable(guards):
    broken = GuardrailService(AgentSettings(), jailbreak=guards.jailbreak, pii=guards.pii, load_models=False)
    broken.unavailable["ToxicLanguage"] = "simulated load failure"
    out = broken.validate_customer_output("Fine text.", FACTS, "FALLBACK")
    assert out.used_fallback and out.events[-1].outcome == "validator_unavailable"
    inp = broken.validate_and_sanitize_customer_input("Refund L1 of O100")
    assert inp.allowed  # input toxicity is advisory only
    broken.unavailable["DetectJailbreak"] = "simulated"
    broken.jailbreak = None
    assert broken.validate_and_sanitize_customer_input("Refund L1 of O100").reason_code == "VALIDATOR_UNAVAILABLE"
