# Local observability stack

OpenTelemetry tracing and metrics for the returns desk. All of it runs on this machine: nothing goes to a cloud service. `app.api.serve` turns telemetry on; everything else keeps it **off** unless `RETAIL_OTEL_ENABLED=true`.

```
app (private TracerProvider + MeterProvider, OTLP/HTTP)
  -> hp-retail-otel-collector  127.0.0.1:14318
       -> traces  -> hp-retail-tempo       127.0.0.1:13200
       -> metrics -> :8889 <- hp-retail-prometheus 127.0.0.1:19090 (scrape every 5 s, 7-day retention)
                     hp-retail-grafana     127.0.0.1:13000 (anonymous Viewer, dashboards "Retail AI desk - overview" and "- request")
```

This is its own Compose project (`hp-retail-observability`), with its own ports, network and volumes. It doesn't touch any other collector or Grafana on this machine, and the app refuses to export to 4317/4318 (`app/observability/config.py`). Images are pinned by digest, and Grafana analytics, update checks and news are off.

## Run

```bash
docker compose -f observability/compose.yaml up -d
docker compose -f observability/compose.yaml ps
```

Then run anything with telemetry on: the adapter, the evaluation runner, the benchmark runner or the API.

```bash
uv run python -m mcp_server          # in its own terminal, first
RETAIL_OTEL_ENABLED=true uv run python adapter.py --input candidate_assets/public_cases.json --output results.json
RETAIL_OTEL_ENABLED=true uv run python -m app.api.serve
```

In PowerShell, run `$env:RETAIL_OTEL_ENABLED="true"` first. Open http://127.0.0.1:13000; the overview is the home page. Click a request in its tables to open the request dashboard. API turn responses carry an `X-Trace-Id` header, and the evaluation runner's raw records include `trace_id`: paste one into the request dashboard's Trace ID box.

To stop the stack (this keeps the data), run `docker compose -f observability/compose.yaml stop`.

## What a trace looks like

Every customer turn is one trace. A confirmation is a separate trace, linked to its chat turn by `retail.proposal_ref`, a salted hash and never the proposal ID.

```
retail.customer_turn  (turn kind, decision, LLM/tool call counts)
├─ retail.input.check  └─ retail.guardrail.input  └─ retail.guardrail.validate × validators   (always first)
├─ retail.context.load  (the business checks' read, get_business_context)
├─ invoke_agent customer_resolution  (task understand) ├─ chat gemma4:e2b ├─ execute_tool lookup_customer_orders
│                                                       └─ retail.guardrail.tool_context / agent_action
├─ retail.handoff.check  (the agent.handoff event: from customer_resolution, to retail_operations)
├─ invoke_agent retail_operations  ├─ chat gemma4:e2b ├─ execute_tool <tool> └─ retail.guardrail.tool_context / agent_action
├─ retail.proposal.create  (outcome, remedy, reason code)
├─ retail.transaction.commit  (confirm turns: committed / denied / replayed / reconciled / conflict)
│    └─ retail.transaction.reconcile  (only after an injected post-commit timeout; the commit span is marked error)
└─ retail.response  └─ invoke_agent customer_resolution (task explain) └─ retail.guardrail.output
```

A blocked message stops after `retail.input.check` and goes straight to `retail.response`. An off-topic message has no `execute_tool`. A confirmation turn has neither `retail.context.load` nor any agent, so it works with the MCP server down.

The API adds a server span, `POST /api/v1/turns` or `POST /api/v1/proposals/{proposal_id}/confirm`, which carries only the route template. Each graph step's span has `retail.step`. A turn's spans are sent together when it ends, because Tempo's search can miss spans of a trace that arrived in a second batch.

With content capture on, spans also carry:

| Attribute | On | Value |
|---|---|---|
| `retail.content.query`, `retail.content.answer` | The turn | The masked customer message and the final reply |
| `retail.content.state_in`, `retail.content.state_out` | Each step | The graph state it received and what it changed |
| `retail.content.input`, `retail.content.output` | Steps, `chat` and `execute_tool` spans | What it read and produced, such as the model's prompt and reply or the tool's arguments and result |

## Metrics (Prometheus names)

| Metric | Labels |
|---|---|
| `retail_turn_duration_seconds` (histogram) | `turn_kind`, `decision` |
| `retail_llm_call_duration_seconds` | `agent_role`, `outcome` |
| `retail_llm_tokens` | `agent_role`, `token_type` (only when the engine reported usage, so a missing value is never counted as 0) |
| `retail_llm_output_token_rate_per_second` | `agent_role`, `method="effective_call"`: output tokens divided by the whole call's wall time, not raw decode speed |
| `retail_step_duration_seconds` | `step`, `outcome` |
| `retail_tool_calls_total`, `retail_tool_duration_seconds` | `tool_name`, `outcome`, `transport` |
| `retail_guardrail_checks_total`, `retail_guardrail_duration_seconds` | `gate`, `validator`, `outcome` (`passed`, `sanitized`, `flagged`, `blocked`, `validation_error` or `validator_unavailable`). Abusive input is `flagged`: noted, never blocked |
| `retail_business_outcomes_total` | `remedy`, `outcome` (a replay or reconciliation is never counted as a second commit) |
| `retail_mcp_startup_duration_seconds` | none: connecting to the MCP server, once per request |
| `retail_system_cpu_percent`, `retail_system_memory_percent` | none: the whole machine |
| `retail_gpu_utilization_percent`, `retail_gpu_memory_mb` | none: the whole GPU from `nvidia-smi`, including other applications; absent without it |
| `retail_process_rss_mb` | `component`: `app`, `model_server` or `mcp_server` |

CPU, RAM and GPU are sampled each time metrics are exported (every 5 s), and only while a process with telemetry on is running. Their p50 and p95 are of those samples. Whole-range counts use `last_over_time` differences rather than `increase()`, which would miss the first sample of each series.

Every label value passes through an allowlist; anything else becomes `other`. `job` is `hp-retail/hp-retail-agentic-ai`, and `instance` is a random per-process ID.

## Privacy rules (enforced by code and tests)

- Metric labels never carry IDs or message text. Every value passes through an allowlist.
- Without content capture, spans never carry customer, order, line, request or proposal IDs or message text. Exception messages and stack traces are never recorded, with or without it; errors are recorded only as `error.type`.
- With `RETAIL_OTEL_CAPTURE_CONTENT=true`, which is the default only for `app.api.serve`, spans carry the `retail.content.*` attributes above. That includes IDs, order data and model prompts, cut to `RETAIL_OTEL_MAX_CONTENT_CHARS`. They stay in the local Tempo for 7 days, and the local Grafana shows them to anyone on this machine without a login.
- Customer text is captured only after the input guard has masked PII. Text the guard blocks is recorded as `[blocked by the input guard]`, never as written.
- The app uses **private** providers, never the global ones, so Guardrails, LiteLLM or LangChain internals can't emit spans through our exporter. Guardrails hub telemetry is forced off, even when `~/.guardrailsrc` enables it, because every `Guard()` re-reads that file.
- If the collector is down, exports fail quietly. The turn, its decision and the database are unaffected (tested).

## Verify

```bash
uv run pytest -q backend/tests/test_observability.py
uv run python -m app.tools.verify_observability
```

The first command runs the deterministic tests: in-memory exporters with scripted LLM replies and real PostgreSQL.

The second runs public cases 01, 02, 05 and 06 on the real model, then checks:
- each turn's trace in Tempo;
- that Prometheus counts for this process equal what the run observed;
- that every metric query on the overview returns data.

It writes `evals/evidence/observability/verification.json`.
