# E3-confirm-template

**Hypothesis.** Confirmation turns spend ~1.2 s of ~1.7 s on one Customer-agent call that only rewords an already-stored result. Using the existing deterministic template (built from trusted Step 2 facts) on confirmation turns should save ~1.2 s per confirmation without changing any decision or business effect; proposal, deny and clarify turns keep LLM wording, so both agent roles still run on every proposal turn.

**Change.** AgentSettings.confirm_template (RETAIL_CONFIRM_TEMPLATE): when a turn carries a structured confirmation, the present node skips the explain LLM call and returns fallback_message(decision, facts). Control: E1 accepted state (pool on, template off); variant: pool on, template on.

**Control.** `evals/evidence/experiments/E1-pool/variant` (source `3e72efe9f5+6105ab4e`, config `24ba48140790`)

**Variant.** `evals/evidence/experiments/E3-confirm-template/variant` (source `3e72efe9f5+6105ab4e`, config `d6d3e38d35b5`)

**Result.** p50 8080 -> 5719 ms (29.2%), mean 20.7%, p95 22.8%; LLM calls/task 6 -> 4

**Decision.** accept: gate passed and latency improved

**Notes.** UX trade-off: confirmation replies are templated (e.g. 'Your refund of $30.00 for line L1 of order O100 is complete.'), which states the stored result exactly. Tests updated where they asserted the old call count; a new test covers the switch-off path.
