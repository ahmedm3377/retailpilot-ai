# E1-pool

**Hypothesis.** A new PostgreSQL connection costs ~22 ms here vs ~1 ms per query on an open one, and a task opens 25-35 connections, so pooling per schema should cut most of the ~0.8 s/task 'other' time without touching transaction boundaries.

**Change.** RetailStore.connect() yields a psycopg_pool ConnectionPool connection (one pool per database+schema, min 1, max 8, same autocommit/dict-row/search_path); pools close before their schema is dropped. Switch: RETAIL_DB_POOL (control 0, variant 1); RETAIL_CONFIRM_TEMPLATE=0 in both arms.

**Control.** `evals/evidence/experiments/E1-pool/control` (source `3e72efe9f5+d423ba92`, config `b9c594125b7b`)

**Variant.** `evals/evidence/experiments/E1-pool/variant` (source `3e72efe9f5+6105ab4e`, config `24ba48140790`)

**Result.** p50 8601 -> 8080 ms (6.1%), mean 4.6%, p95 0.9%; LLM calls/task 6 -> 6

**Decision.** accept: gate passed and latency improved

**Notes.** Control measured on AC power (decode probe 64.5 tok/s after) before the gate recorded AC status; the first variant attempt ran partly on battery and is kept as _invalid_variant_on_battery (see INVALID.md). Variant rerun on AC: decode 66.2 -> 63.6 tok/s.
