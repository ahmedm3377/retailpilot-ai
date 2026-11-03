# Local Agentic Retail Returns & Exchanges Desk

A fully local, agentic customer-service desk for retail **refunds** and **same-product size exchanges**. Two cooperating LLM agents talk to the customer and investigate orders, while deterministic Python and PostgreSQL decide what is allowed and move the money and stock. Everything runs on one machine, with no cloud APIs: the model is served by a local engine of your choice (Ollama by default), guardrails run locally, and the demo works with the internet switched off.

> **The core idea:** the LLMs decide *what to do*; deterministic code decides *whether it is allowed* and then does it, exactly once.

---

## Contents

- [Business use case](#business-use-case)
- [Features](#features)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing and evaluation](#testing-and-evaluation)
- [Performance](#performance)
- [Observability](#observability)
- [Security and safety model](#security-and-safety-model)
- [Troubleshooting](#troubleshooting)
- [Development notes](#development-notes)

---

## Business use case

A retailer's returns desk handles a high volume of repetitive requests: "refund this", "swap it for a larger size", "did my refund go through?". Automating them with an LLM is attractive, but the risks are concrete:
- refunding the wrong amount;
- refunding an item that isn't eligible;
- refunding the same item twice after a network timeout;
- selling stock that just ran out;
- being talked into a refund by a cleverly worded message.

This project shows how to get the conversational benefits of LLM agents **without letting the model own money, eligibility or inventory**.

**What the desk does**

| Customer intent | Outcome |
|---|---|
| Refund an eligible item | A proposal for exactly the amount paid for that line. After the customer explicitly confirms: a refund record, the line marked returned, the item quarantined, and an idempotency entry. |
| Exchange for another size of the same product | Checked for same product, different size, same catalog price and replacement in stock. After confirmation the replacement stock drops by one and the original is quarantined. |
| Ambiguous request ("return something from my order") | A clarifying question naming the candidate lines. |
| Ineligible request (expired, opened, final sale, already returned, another customer's order) | A clear denial, with no change to business state. In the UI, a customer who says the item was opened also has that recorded on the line. |
| Repeated confirmation, or a retry after a timeout | The original result, with no second refund or stock movement. |

**Business rules** (enforced only in deterministic code):
- The order must be `delivered`, belong to the authenticated customer, and be **0–30 calendar days** old, inclusive. Age is measured from the data's `as_of` date, never the system clock.
- The unit must be unopened, not final sale and not already returned.
- A refund is exactly the line's `paid_cents`, in integer cents, with discounts already allocated.
- An exchange must be the same `product_id` in a different size, at the same catalog price, with replacement stock above 0. The original never goes back to sellable stock; it is quarantined.
- **No business change happens without an explicit, structured confirmation** of a specific, immutable proposal. Typing "yes" never counts.
- Every request is idempotent by `request_id`.

---

## Features

**Agentic conversation**
- Two LLM roles on a LangGraph graph:
  - a **Customer Resolution Agent**, which calls its own tool to read the customer's orders through the MCP server, works out the intent (refund or exchange) and the item, asks for clarification and writes replies;
  - a **Retail Operations Agent**, which receives the structured request, investigates with read-only tools and recommends a remedy, or hands the request back when the customer must answer something.
- Model-agnostic through the LiteLLM SDK: the model and its engine (Ollama, vLLM or another local server) are set in `.env`. Schema-constrained JSON output and deterministic decoding. The results below use `gemma4:e2b` on Ollama.
- Multi-turn memory in PostgreSQL checkpoints.

**Deterministic business core**
- A pure policy engine with explicit reason codes (`ELIGIBLE`, `DELIVERY_DATE_OUT_OF_RANGE`, `INVALID_REPLACEMENT`, …).
- Immutable proposals, enforced by a database trigger.
- A trusted-only commit: one transaction, advisory lock per request, ordered row locks, policy re-checked on locked rows, exactly-once effects, and a full audit trail.
- Fault handling: stock changing between proposal and confirmation, and a timeout *after* a successful commit, which is reconciled by reading the result, never by committing again.
- A deterministic check that what the customer *named* (order, line, SKU) matches what the agent resolved, so the desk never silently substitutes a different item.
- A separate business layer (`backend/app/business/`) that holds every deterministic step and runs with no model at all. The graph only routes.

**Guardrails (all local)**
- Four gates: input, tool context, agent action schema, and customer output.
- Prompt-injection and jailbreak detection, toxicity, PII masking, and an output consistency check against trusted facts.
- They run first: a blocked message reaches no model and no tool.
- Fail-closed by default.

**Interfaces**
- A web UI (React + TypeScript) with request entry, a proposal card, explicit **Confirm / Cancel**, execution status, a readable audit timeline and actionable errors. An optional dropdown runs a public case or an uploaded JSON case on its own isolated data.
- A REST API (FastAPI) with isolated sessions and retry-safe actions (`client_action_id`).
- A batch adapter (`adapter.py`) that runs scenario files through the **same runtime** and writes a contract-shaped JSON result.

**Engineering**
- **Tools on an MCP server:** each agent's read-only tools, and the read behind the business checks, live on a [standalone MCP server](#standalone-mcp-server) over Streamable HTTP, which must be running before the app starts.
- **Local observability:** Grafana dashboards for latency, CPU, RAM, GPU, tokens, throughput, tool calls and failures, plus a step-by-step trace of each request. No cloud; on for the API and off elsewhere unless you enable it.
- **Measured performance work:** a benchmark harness with a quiet-machine gate and verified evidence.
- **Evaluation suite:** labeled synthetic and adversarial scenarios, an independent business oracle, agent-trajectory scoring, and real-browser tests.

---

## Architecture

```mermaid
flowchart TB
    Browser["Web UI<br/>React + TypeScript"] --> API["FastAPI"]
    Cases["Scenario JSON"] --> Adapter["adapter.py"]
    API --> Runtime["AgentRuntime<br/>one guarded turn"]
    Adapter --> Runtime

    subgraph Graph["LangGraph turn"]
        direction TB
        Guard["Input guardrails<br/>always first"]
        Load["load_context<br/>reads for the checks"]
        Customer["Customer Resolution Agent<br/>LLM"]
        Check["check_handoff<br/>deterministic"]
        Ops["Retail Operations Agent<br/>LLM"]
        Propose["create_proposal, commit<br/>deterministic"]
        Guard --> Load --> Customer --> Check --> Ops --> Propose
    end

    Runtime --> Graph
    Check --> Desk["BusinessDesk<br/>app/business: no model imports"]
    Propose --> Desk
    Customer --> Model["Local model server<br/>through LiteLLM"]
    Ops --> Model
    Load -->|"host-only read"| MCP["Standalone MCP server<br/>Streamable HTTP"]
    Customer -->|"its own tool"| MCP
    Ops -->|"its own tools"| MCP
    MCP -->|"read-only transactions"| DB[("PostgreSQL 17")]
    Desk -->|"one atomic transaction"| DB
    Graph -. "traces and metrics" .-> Obs["OpenTelemetry, Tempo,<br/>Prometheus, Grafana"]
```

Code paths are relative to `backend/`. A blocked message skips straight to the reply, and a confirmed proposal goes straight to `commit`. When the Operations Agent needs an answer from the customer, the reply asks for it, and the next turn updates the request and hands it back. PostgreSQL holds `retail_store` for sign-ins and one schema per case session or run, with products, orders, proposals, returns, refunds, exchanges, quarantine, idempotency and the append-only `audit_events`. The MCP server listens on 127.0.0.1:8765 with six agent tools and one host read. The MCP server is a separate process that must be running before the app starts. Every read before the write path goes through it, and the app has no in-process copy of those tools. The graph only routes: `load_context` reads through the server and hands the result to `app/business/` as data. That layer holds every business rule and write, imports nothing from the agents, and runs with no model at all.

**What stays a direct database read, and why:** the commit and the proposal's final check read the rows they then write, on the write path (exactly-once needs locked rows in one transaction); timeout recovery must work with the server down; and the UI's panels, the audit trail and the adapter's final state are reporting, not decisions.

### How the two agents work together

```mermaid
flowchart LR
    Msg["Customer message"] --> C["Customer agent:<br/>reads the orders with its tool,<br/>then intent and item"]
    C --> G["ground_target and resolve_handoff:<br/>keep only what the customer identified"]
    G -->|"something missing"| Ask["Ask the customer"]
    G -->|"complete request"| O["Operations agent:<br/>eligibility, exchange options"]
    O -->|"size missing"| Ask
    O --> Opened{"Opened or<br/>unopened?"}
    Opened -->|"not said yet, UI only"| Ask
    Opened -->|"opened"| Sorry["Apology, no proposal"]
    Opened -->|"unopened"| P["create_proposal:<br/>computes the amount itself"]
    P --> Confirm{"Structured<br/>confirmation?"}
    Confirm -->|"no: text like 'yes'"| Wait["Nothing commits"]
    Confirm -->|"yes"| Commit["Trusted commit"]
```

1. **Check, then collect.** Every message passes all the input guardrails first; a blocked message reaches no agent, tool or database table beyond its audit rows. Only then does `load_context` read what the business checks need, in one call to the MCP server.
2. **Understand.** On a new request the Customer Resolution Agent calls `lookup_customer_orders` itself, then works out the remedy and size. Off-topic messages read nothing. Deterministic code (`ground_target`) keeps only the order and line the customer actually identified, so an ID the model guessed or read from a tool is never used.
3. **Ask, if needed.** If the remedy or the item isn't clear, the desk asks, with fixed wording when it lists orders or lines so that no option can be dropped or invented. The partial request stays in the checkpoint.
4. **Hand off.** The complete request goes to the Retail Operations Agent as a state in the shape of a contract case: `case_id`, `customer_id`, `request_id`, `initial_state` (this customer's orders and the related products, read from the database), `turns`, plus `intent`, `order_id`, `line_id` and `requested_size`. Two deliberate differences from a case: `turns` holds decisions rather than wording, so customer text can't reach the Operations Agent, which has the wider set of tools, and `faults` is left out, because faults are never disclosed to an agent. The agent then checks eligibility and exchange options with read-only tools.
5. **Hand back, if needed.** If the customer must still answer something (today, which size, which is always their choice even with one left), the request goes back to the Customer Resolution Agent, which asks. The answer updates the kept request (`merge_pending_request`), and the updated request goes to the Operations Agent again.
6. **Propose and execute.** In the UI the desk first asks whether the item is opened or unopened. Deterministic code creates the proposal, and **Confirm** commits it.

Each handoff is an `agent_handoff` row in the audit, with the direction and the request, so the UI timeline and the adapter's `action_audit` both show it. An agent can ask for a handback, but deterministic code decides whether an answer is really missing: a guessed size isn't used, an ineligible item is denied rather than asked about, and a request with nothing missing goes ahead.

### Layers

| Layer | Responsibility | Code |
|---|---|---|
| Entry points | The web UI, REST API and batch adapter. Thin clients of one runtime; nothing scenario-specific. | `frontend/`, `backend/app/api/`, `adapter.py` |
| Orchestrator | One guarded turn. Routes to the agents, records handoffs, maps outcomes to a decision (`propose · clarify · deny · completed · error`). | `backend/app/services/agent_runtime.py`, `backend/app/agents/graph.py` |
| Agents | Two LLM roles that produce typed JSON actions and call their own tools. They never compute money or eligibility. | `backend/app/agents/customer_agent.py`, `operations_agent.py` |
| Tool layer | Read-only tools by role (below). Customer and request identity are bound from trusted state, never from model output. | `backend/app/agents/tool_registry.py`, `backend/app/mcp/` |
| Business layer | Every deterministic business step: checking a request, proposing, committing. No model, no agent imports. | `backend/app/business/` |
| Guardrails | Input, tool-context, action-schema and output gates. | `backend/app/guardrails/` |
| Domain core | Pure eligibility, refund and exchange rules with reason codes. The single source of business rules. | `backend/retail_domain/policy.py` |
| Trusted services | Proposals, the exactly-once commit, cancellation, reconciliation, fault injection. | `backend/app/services/` |
| State store | PostgreSQL with versioned migrations, constraints and triggers. | `backend/app/db/` |

### Which tools each agent gets

| Caller | Tools | How it's limited |
|---|---|---|
| Customer Resolution Agent (LLM) | `lookup_customer_orders` | The first step of a new request must call it, and code then ends the tool phase |
| Retail Operations Agent (LLM) | `check_return_eligibility`, `get_exchange_options`, `check_inventory`, `get_order_line`, `get_request_status` | Its own list and a step budget |
| Business checks (code, never a model) | `get_business_context` | Not in either agent's list or schema, so no model can name it |

A tool outside an agent's list is refused and audited. The Customer agent sees the customer's wording, but its only tool takes no arguments, so that wording can't choose what it reads. Its limits were measured on `gemma4:e2b`: left free it skipped the tool on 8 of 8 new requests, then with more freedom called tools until the budget ran out and never handed off, and a tool call on a short answer lost the answer. A larger model could be given more.

### Life of a request

```mermaid
sequenceDiagram
    actor U as Customer
    participant R as Runtime and guards
    participant CA as Customer agent
    participant OA as Operations agent
    participant M as MCP server
    participant T as Trusted services
    participant D as PostgreSQL

    U->>R: Refund line L1 of order O100
    R->>R: input guardrails first, masks PII
    R->>M: business checks' read
    M->>D: read-only query
    R->>CA: masked message
    CA->>M: lookup_customer_orders
    M-->>CA: the customer's orders
    CA-->>R: refund, O100, L1
    R->>R: check against what the customer identified
    R->>OA: handoff, case-shaped state
    OA->>M: check_return_eligibility
    M-->>OA: ELIGIBLE, 3000 cents
    OA-->>R: recommend refund
    R-->>U: Is the item opened or unopened?
    U->>R: Unopened
    R->>T: create_proposal
    T->>D: immutable proposal
    R-->>U: Proposal for $30.00, please confirm
    U->>R: Confirm button
    T->>D: lock, re-check policy, all effects, one transaction
    R-->>U: Your refund is complete
```

1. **Chat turn.** The input guard sanitizes the message, masks PII and blocks injections.
2. On a new request the Customer agent calls `lookup_customer_orders`, then extracts intent, order, line and size.
3. Deterministic checks resolve that target against the customer's real orders, and against what the customer explicitly named.
4. The Operations agent calls read-only tools (eligibility, exchange options, inventory) and recommends a remedy.
5. In the UI, the desk asks "opened or unopened?" if the customer hasn't said. "Opened" gets an apology and is recorded; nothing is proposed.
6. `create_proposal` computes the amount itself and stores an immutable proposal.
7. The Customer agent words the reply, and the output guard checks it against trusted facts. Questions that list options, refusals of a size and the apology for an opened item use fixed wording, with no model call.
8. **Confirmation.** The UI's Confirm button, or the adapter's structured `confirm_proposal`, sends the proposal ID through a trusted path. The commit re-checks everything under locks, writes all effects in one transaction, and records the idempotency entry and audit event.
9. **Timeout after commit.** The result was saved, so recovery *reads* it (`reconcile_request`) and reports `completed`. The audit trail shows that read as its own `reconcile_request` row, between the timeout and any replay, with no effects. A repeat confirmation returns the stored result.

What happens to a confirmed proposal, including the two injected faults:

```mermaid
flowchart TD
    Start["Confirmed proposal"] --> Lock["Lock rows, re-check policy"]
    Lock -->|"stock now 0"| Deny["Deny, no effects"]
    Lock -->|"still valid"| Tx["Commit in one transaction"]
    Tx --> Timeout{"Timeout after<br/>commit?"}
    Timeout -->|"no"| Done["Completed"]
    Timeout -->|"yes"| Read["Read the stored result<br/>never commit again"]
    Read --> Done
    Done --> Replay{"Same request<br/>confirmed again?"}
    Replay -->|"yes"| Stored["Return the original result<br/>no second refund"]
```

---

## Tech stack

| Area | Technology |
|---|---|
| Language and packaging | Python 3.13 managed by **uv** (`uv.lock`); Node.js + npm for the UI |
| LLM serving | Any local engine **(VLLM, Ollama)** through the **LiteLLM** Python SDK, set in `.env`. Default: **Ollama** with `gemma4:e2b` (Q4_K_M) |
| Agents | **LangGraph** `StateGraph` with a PostgreSQL checkpointer |
| Validation | **Pydantic v2** (strict models, `extra="forbid"`) |
| Guardrails | **Guardrails AI** with local validators: DetectJailbreak, ToxicLanguage, DetectPII (Presidio + spaCy), and a custom TaskInjection |
| Database | **PostgreSQL 17** (Docker, pinned by digest) via psycopg 3 and `psycopg_pool` |
| API | **FastAPI** on Uvicorn, bound to 127.0.0.1 |
| UI | **React 19 + TypeScript** (strict) on **Vite**; Vitest and React Testing Library; **Playwright** |
| Tool transport | Official **MCP Python SDK 2.x** over Streamable HTTP, on loopback |
| Observability (optional) | **OpenTelemetry** SDK → Collector → Tempo (traces) and Prometheus (metrics) → Grafana |

---

## Project structure

```text
├── adapter.py             # the contract's entry point: scenario JSON in, contract-shaped results out
├── candidate_assets/      # the supplied contract, public scenarios and validator (read-only)
├── backend/
│   ├── app/               # API, agents, business layer, guardrails, services, database, telemetry
│   ├── mcp_server/        # standalone read-only MCP server; imports nothing from app/
│   ├── retail_domain/     # pure contract models and rules, shared by app/ and mcp_server/
│   ├── tests/
│   └── docker/            # PostgreSQL first-run script
├── frontend/              # React + TypeScript UI, with component and browser tests
├── observability/         # local OpenTelemetry Collector, Tempo, Prometheus and Grafana
├── evals/                 # evaluation suite, benchmarks, oracle and evidence
│   └── evidence/          # current evaluation, older baseline-vs-final results and experiments, telemetry check
├── deploy/                # everything in Docker: compose files, backend and UI images, nginx
├── start_demo.bat         # Windows: the whole desk in Docker, with the demo store reset
├── pyproject.toml         # dependencies; also docker-compose.yml, .env.example
```

---

## Getting started

Every command runs from the **repository root** unless it says `cd frontend`. The order matters:

```text
Docker services (PostgreSQL, optional telemetry) ─► model server ─► seed the store ─► MCP server ─► API ─► UI
```

**In a hurry, on Windows?** Start Docker Desktop and run `start_demo.bat`. It runs everything in Docker, with no other setup, **resetting the store to the seed data** each time. The details are in [Quick start on Windows](#quick-start-on-windows).

### Prerequisites

| Tool | Notes |
|---|---|
| [uv](https://docs.astral.sh/uv/) | Installs Python 3.13 and all Python dependencies (the system Python is not used) |
| Docker Desktop | Runs PostgreSQL and the optional observability stack |
| A local model server | Default: [Ollama](https://ollama.com) with `gemma4:e2b`. See [Choosing the model](#choosing-the-model) for others |
| Node.js 22 + npm | For the web UI and Playwright |

Plan for about **15 GB of disk** (model, guardrail models, images) and **16 GB of RAM**. A small NVIDIA GPU (tested on a 6 GB GTX 1060) makes the model fast enough for interactive use.

### 1. One-time setup (needs internet once)

```bash
git clone https://github.com/ahmedm3377/retailpilot-ai.git && cd retailpilot-ai
uv sync --extra dev                                             # Python 3.13 environment from uv.lock
ollama pull gemma4:e2b                                          # the default model (~4.6 GB); skip if you use another
uv run python -m guardrails_ai.detect_jailbreak.post_install    # local guardrail models
uv run python -m guardrails_ai.toxic_language.post_install
cd frontend && npm install && npx playwright install chromium && cd ..
docker compose pull postgres                                    # PostgreSQL image, pinned by digest
docker compose -f observability/compose.yaml pull               # optional: the telemetry images
```

- Copy `.env.example` to `.env` only if you need non-default settings (another model, another `MCP_AUTH_TOKEN`, …). Without `.env`, the defaults in `.env.example` apply.
- After this step nothing needs the internet.

### 2. Start the Docker services

**PostgreSQL (required):**

```bash
docker compose up -d --wait postgres     # container hp-retail-postgres on 127.0.0.1:55432
docker compose ps                        # STATUS should say "healthy"
```

- It listens on **55432**, not 5432, so it doesn't collide with other installs. The credentials are `retail`/`retail`, for local development only.
- On first start it creates two databases: `retail` (the web UI's data) and `retail_test` (tests, the adapter and evaluations). The tables are created by the app, not here.
- The data lives in the `hp_retail_pgdata` volume and survives restarts and reboots.

**Observability stack (optional):** Grafana, Tempo, Prometheus and the OpenTelemetry Collector on 127.0.0.1. You don't have to start it yourself: the API starts it when it launches (step 5). To start it by hand:

```bash
docker compose -f observability/compose.yaml up -d     # Grafana on http://127.0.0.1:13000
```

See [Observability](#observability) for what it shows and how to run without it.

### 3. Start the model server

Make sure the model server named in `.env` is running and has the model. For the default Ollama, start the Ollama desktop app (or run `ollama serve` in its own terminal), then check:

```bash
ollama list                              # gemma4:e2b should be listed
```

The app never starts or stops the model server itself.

### 4. Seed, reset or clean the database

The web UI works on one persistent schema, `retail_store`, in the `retail` database. It is filled from `backend/app/db/demo_store_seed.json`, and commits stay in it, so a refunded item stays refunded across sign-outs and restarts.

| To… | Run | When |
|---|---|---|
| **Seed** the store (only if it's empty) | `uv run python -m app.db.demo_store` | Optional: the API does this by itself on its first start |
| **Reset** the store to the seed data | `uv run python -m app.db.demo_store --reset` | Before a fresh trial. Deletes every proposal, refund, exchange and return in the store |
| **Clean up** sessions | `uv run python -m app.api.cleanup --older-than-hours 0` | Ends every UI session and removes its conversation and case schemas |
| See what cleanup would remove | `uv run python -m app.api.cleanup --older-than-hours 0 --dry-run` | Any time |
| Clean only old sessions | `uv run python -m app.api.cleanup --older-than-hours 24` | Routine housekeeping |

**For a fresh trial**, stop the API first (Ctrl+C in its terminal), then run cleanup **before** reset:

```bash
uv run python -m app.api.cleanup --older-than-hours 0     # 1. end every session and remove its conversation
uv run python -m app.db.demo_store --reset                # 2. put the store back to the seed data
```

The order matters: the audit trail finds a session's events by their numbers, and those restart at the reset, so a session left open across a reset could show the wrong events. On a brand-new database, before the API has ever run, cleanup fails because the session registry doesn't exist yet; skip it then. The MCP server can keep running throughout.

What a reset changes:

| | What happens |
|---|---|
| Orders, stock, proposals, refunds, exchanges, returns, quarantine, and lines recorded as opened | Back to the seed data, so `C100` can return items again |
| Delivery dates | Counted back from today, so the seeded items are in the same position in the 30-day window |
| The store's audit log | Emptied, with its numbering restarted from 1 |
| Sign-in sessions and their conversations, case sessions and their schemas | Ended and removed (by the cleanup step) |
| `retail_ui_checkpoints`, `retail_ui_registry` | Stay, now with no active sessions. Ended sessions stay listed as ended, which is harmless |
| Grafana's traces and metrics | Stay, because they live in their own stack. Narrow the dashboard's time range to see only the new trial |

The adapter, tests and evaluations don't use this data: each case starts from its own initial state in a fresh schema in `retail_test`, which is dropped afterwards, so they need no seed or reset.

### 5. Start the application

Use a **separate terminal for each** of the three processes, started **in this order**. Each one must be up before the next.

**5.1 The MCP server (first).** The agents' read-only tools and the business checks' read live here, so the API, the benchmarks and the evaluation suite all refuse to start without it.

```bash
uv run python -m mcp_server              # http://127.0.0.1:8765/mcp
```

**5.2 The API.**

```bash
uv run python -m app.api.serve           # http://127.0.0.1:8080 (interactive docs at /docs)
```

- On start it checks the MCP server, seeds `retail_store` if it's empty, starts the observability stack if it isn't running, loads the guardrail models and warms the model. That takes 25–35 s.
- `--no-telemetry` runs it without telemetry; `--no-start-stack` never starts containers; `--port` changes the port.
- Check readiness with `curl http://127.0.0.1:8080/health/ready`.

**5.3 The web UI.**

```bash
cd frontend
npm run dev                              # http://127.0.0.1:5173
```

Open http://127.0.0.1:5173 and sign in as `C100` or `C200`. The header badge says when the desk is ready. The [Web UI walkthrough](#web-ui-walkthrough) explains what to try.

| Service | Address |
|---|---|
| Web UI | http://127.0.0.1:5173 |
| API (and `/docs`) | http://127.0.0.1:8080 |
| MCP server | http://127.0.0.1:8765/mcp |
| PostgreSQL | 127.0.0.1:55432 (`retail`/`retail`) |
| Grafana | http://127.0.0.1:13000 |

### Quick start on Windows

`start_demo.bat`, run from the project root, runs the whole desk **in Docker** ([Run everything in Docker](#run-everything-in-docker-another-machine)). It needs only Docker Desktop running: no uv, Node.js or Ollama install.

1. builds the images, using the GPU overlay when Docker has the NVIDIA runtime (`start_demo.bat --cpu` never uses it);
2. starts PostgreSQL and the model server, with the API and UI stopped, and pulls the model if it isn't there yet;
3. ends every session and **resets the store to the seed data**;
4. starts the MCP server, API, UI and observability stack, waits until the desk is ready, and opens the UI.

The first run needs the internet and takes a while (about 13 GB of images and the 4.6 GB model); later runs work offline. It reads the ports from `deploy/.env` if you set them there, and refuses to start if one is already taken: the native stack uses the same ports, so stop it first. Run again while the desk is up, it skips that check and gives a fresh trial: it rebuilds, stops the API and UI, and resets the store again. At the end it prints the exact `docker compose` command it used (with `-f deploy/compose.gpu.yaml` when it chose the GPU); stop the desk with that command plus `stop`.

Put `MCP_AUTH_TOKEN` in `deploy/.env` (copy `deploy/.env.example`). The token is fixed when the stack starts. A later `docker compose … run --rm tools …` from another terminal must resolve the same one, or the MCP server refuses it. A token that only lives in one terminal's environment (VS Code, for example, can load the root `.env` into new terminals) breaks this.

### 6. Check it runs offline

Disconnect the machine from the network, make sure PostgreSQL and the model server are running, then run:

```bash
python adapter.py --input candidate_assets/public_cases.json --output results.json
```

It runs the public scenarios end to end through the same agents and tools as the UI, and must exit with code `0`. It starts its own MCP server for the run if none is running. See [Batch adapter](#batch-adapter) for checking the results.

### 7. Stop everything

Press Ctrl+C in the UI, API and MCP server terminals (or close their windows), then:

```bash
uv run python -m app.api.cleanup --older-than-hours 0     # optional: end sessions and remove their conversations and case schemas
docker compose stop postgres                              # keeps the data volume
docker compose -f observability/compose.yaml stop         # keeps traces and metrics
```

Use `stop`, **never `docker compose down -v`**, which deletes the database volume.

### After a reboot

Nothing needs to be installed again. The observability containers restart with Docker by themselves (`restart: unless-stopped`); PostgreSQL does not:

```bash
docker compose up -d --no-build --wait postgres      # 1. database
                                                     # 2. start Ollama (desktop app or `ollama serve`)
uv run python -m mcp_server                          # 3. own terminal
uv run python -m app.api.serve                       # 4. own terminal
cd frontend && npm run dev                           # 5. own terminal
```

The store keeps its data. Reset it (step 4) only if you want a fresh trial.

### Run everything in Docker (another machine)

`deploy/compose.yaml` runs the whole desk in containers: PostgreSQL, Ollama and its model, the MCP server, the API, the web UI (built and served by nginx) and the observability stack. The machine needs only Docker (Compose v2.30 or later) and this repository; no Python, Node.js or Ollama install.

```bash
cp deploy/.env.example deploy/.env      # optional: change MCP_AUTH_TOKEN, the ports, the model
docker compose -f deploy/compose.yaml -f deploy/compose.gpu.yaml up -d --build    # with an NVIDIA GPU
docker compose -f deploy/compose.yaml up -d --build                               # CPU only (much slower)
```

- **First start needs the internet:** it builds the two images (the backend image is about 13 GB, mostly PyTorch's CUDA libraries from `uv.lock`, with the guardrail models baked in) and pulls `gemma4:e2b` (about 4.6 GB) into a volume. After that, `up` works offline.
- **Ready:** `up` returns once the API is healthy. Open http://127.0.0.1:5173 and sign in as `C100`. Grafana is on http://127.0.0.1:13000, and PostgreSQL on 127.0.0.1:55432 for host tools. The store is seeded on the API's first start.
- **GPU:** `compose.gpu.yaml` needs the NVIDIA driver plus the NVIDIA Container Toolkit (Linux), or Docker Desktop on WSL 2 (Windows). It also gives the API and `tools` containers `nvidia-smi`, so results report the GPU and VRAM. The guardrail models stay on the CPU.
- **Resources:** give Docker at least 8 GB of RAM and 4 CPUs (Docker Desktop: Settings → Resources). With 2 CPUs a case takes about twice as long as natively.

Run the adapter and admin commands in the same image, against the running stack. Use the same `-f` files the stack was started with (`start_demo.bat` prints them): leaving out `deploy/compose.gpu.yaml` makes Compose recreate Ollama without the GPU. Keep `MCP_AUTH_TOKEN` in `deploy/.env` so the token matches the running MCP server.

```bash
docker compose -f deploy/compose.yaml run --rm tools python adapter.py --input candidate_assets/public_cases.json --output out/results.json
docker compose -f deploy/compose.yaml run --rm tools python -m app.api.cleanup --older-than-hours 0
docker compose -f deploy/compose.yaml run --rm tools python -m app.db.demo_store --reset
```

`out/` on the host receives the results. Stop with `docker compose -f deploy/compose.yaml stop`; `down` without `-v` also keeps the data.

**How it stays local-only:** the app accepts only loopback for PostgreSQL, the model server, the MCP server and the collector. So those services, the API and nginx share one network namespace (the `net` service) and talk over 127.0.0.1 exactly as on a native run; no safety check is relaxed. Only nginx, PostgreSQL and Grafana, Tempo and Prometheus are published, on `BIND_ADDRESS` (127.0.0.1 by default). Setting it to `0.0.0.0` lets other machines reach the UI, which has no password.

---

## Usage

### Web UI walkthrough

The UI is a conversation, with the proposal, execution status, your orders and the audit trail beside it. Enter your **customer ID** (try `C100` or `C200`; small or capital letters both work) and press **Start**. The server looks the ID up in PostgreSQL and binds the session to that customer, so the chat only ever sees and acts on that person's orders, and nothing typed in the chat can change who you are. This is a demo sign-in without a password; a real deployment would use the store's login.

The desk doesn't choose for you when there is a real choice. Before it proposes anything, it needs:
- **the remedy:** a refund, or an exchange for a different size. A plain "I want to return it" means a refund, which you still see as a proposal and can **Cancel**. If the desk can't tell what you want, it asks;
- **the item, when it isn't unique:** you can use an order number, line number, SKU or product name ("my hoodie"). If you have several delivered orders or the order has several lines, the desk asks, and lists only the orders and lines that can really be returned; an order or line the model guesses is never used. If only one of your orders is eligible, it goes straight to that order's lines. If you enter a number that isn't yours or doesn't exist, it says "You entered a wrong number." and lists the eligible ones again. Naming one of your own orders that isn't eligible gets a refusal with the reason. If you have only one delivered order, or the order has only one line, that is the unique target (as the contract says) and needs no question, unless your message names an ID that doesn't exist;
- **the size, for an exchange:** always your choice, even if only one is left. The desk asks and compares your answer with the sizes left. If it doesn't match (no such size, no stock, another price, or the size you already have), the desk apologizes, says why, and lists the sizes left. The request stays open, so you can just name another size. If no other size is open, the refusal ends the request. Words such as "large" become the catalog's size, and a bare "l" is read by fixed rules if the model misses it.

**Opened or unopened comes first.** Before any proposal, the desk asks whether the item is opened or unopened, unless your message already said so, and only when an offer is really possible (an expired item is simply denied). Your answer is read by fixed rules, not the model, and an unclear one such as "yes" is asked again:
- **Opened:** the desk apologizes that it can't proceed, nothing is proposed, and the store records the line as opened, with an audit row. That lasts until the store is reset. Saying so while a proposal waits also retires the proposal.
- **Unopened:** you get the proposal. This can't make an item eligible: if the store says it's opened, the request is denied, and the commit re-checks.

The adapter works differently, because the contract's cases go straight from the request to the confirmation, and a question in between would leave turn 2 with nothing to confirm. It uses the same agents, models and tools, but the question is added to the proposal instead, and saying "opened" refuses without changing the store (the contract requires refusals to be no-ops). `ask_item_condition` on `AgentRuntime` is the one difference.

**Confirm** is the only way to execute a proposal: typing "yes" never commits. **Cancel** invalidates a pending proposal. While a proposal waits, the desk keeps its details: naming another size replaces it (only the newest can be confirmed), and a message that changes nothing shows it again. **My orders** stays collapsed until you open it or a change commits: it then opens by itself and shows each item's outcome from the database, such as "Refunded $30.00" or "Exchanged for TEE-L".

Off-topic messages and blocked jailbreak attempts get the same friendly reply, which explains what the desk can help with. The audit trail shows which it was: an out-of-scope row after a model call, or a blocked guardrail with no model call at all. A request being clarified stays open through an off-topic message, so you can carry on where you left off. Abusive wording is `flagged` in the trail and still processed, because an upset customer may still be owed a refund.

You can ask for several items of an order ("both", "L1 and L2", "the whole order"). One request is one transaction, so the desk starts with the first item that isn't already returned and tells you which to ask about next. An item already refunded or exchanged is always declined, on its own or as part of such a request, and nothing changes. In the chat, a message after a completed transaction automatically starts the customer's next request.

**Running a specific case.** Under the login field, the optional **Case** dropdown picks the data for the session:

| Choice | The session works on |
|---|---|
| None (default) | The main `retail_store` schema with all its data, and commits stay there |
| Case 1–6 | An isolated copy of that case from `candidate_assets/public_cases.json`, in its own schema, with the case's faults. The customer ID fills in by itself |
| New empty case | An isolated copy of a case from a JSON file you upload (up to 1 MB), in its own schema, with its faults. The file holds one case, or `{"cases": [...]}` like `public_cases.json`; if it has several, you choose one. A file that doesn't fit is refused with the reason |

The chat header shows which data you're on. A case's `turns` aren't played: you type the messages yourself. Each start makes a fresh copy, sign-out deletes it, and the main store is never changed. Faults count your turns in the session, so a `before_turn: 2` stock change happens before your second turn. For case 5, say "It is unopened." in your first message: otherwise your answer to the opened-or-unopened question is the second turn, and the stock drops before you ever see a proposal. The dropdown exists for demos and testing; a real deployment wouldn't offer it.

**Where the data comes from.** Without a case, the UI reads customers, orders and stock only from the persistent `retail_store` schema in PostgreSQL. It's filled once from `backend/app/db/demo_store_seed.json` the first time the API starts (delivery dates are counted back from that day), and commits stay in it, so a refunded item stays refunded across sign-outs and restarts. To go back to the seed data, see [Seed, reset or clean the database](#4-seed-reset-or-clean-the-database).

The seeded customer `C100` has orders to try each path: `O100` (a tee and a jacket; the jacket's other size is out of stock), `O101` (a hoodie; its L costs more, so it can't be exchanged to), `O090` (delivered too long ago) and `O102` (final sale). `candidate_assets/public_cases.json` is used by the adapter, the tests and the **Case** dropdown, and never changes the store.

### Contract conformance

How `candidate_assets/CONTRACT.md` is met, and where:

| Contract clause | How it's met |
|---|---|
| `python adapter.py --input … --output …` | Works from any Python: outside the project's environment, `adapter.py` relaunches itself in `.venv`, else through `uv`. It starts the MCP server for the run if none is running, and stops it afterwards |
| Same agents, models and tools as the UI | Both call `AgentRuntime.handle_customer_turn`. One step differs, `ask_item_condition`: the UI asks "opened or unopened?" first, which the contract's turn 2 confirmations leave no room for |
| Sequential cases, reset before each | A fresh database schema per case |
| Resolve `last`; no mutation without it | `backend/app/evaluation/runner.py` resolves it; only a confirmation turn can reach `commit` |
| Proposal binding, revalidation before commit | `backend/app/services/transactions.py`, on locked rows with `backend/retail_domain/policy.py` |
| Clarify if the target isn't unique | `ground_target` and `resolve_handoff` in `backend/app/business/request_checks.py` |
| Business rules | `backend/retail_domain/policy.py`, using the state's `as_of` |
| Records and idempotency | `returns`, `refunds`, `exchanges`, `quarantine`, `idempotency` in `final_state` |
| Faults at the tool boundary, recorded as external events | `backend/app/services/faults.py`; never disclosed to an agent |
| Audit: agent, tool, arguments, result/error, effects | `action_audit`, in the order the steps ran, with no prompts or reasoning |
| Include failed runs | A case that breaks stays in `results`, with `run_failed`, `error` and every turn listed. If nothing can run (MCP server, model or database down), every case is listed as failed. The exit code is then 3 |

The adapter needs PostgreSQL and the model server named in `.env` running; it starts the MCP server itself when needed. A server already running is used and left running. One choice goes beyond the contract: an unknown ID in a message gets a question rather than a substitute item. The contract doesn't define a plain "return", so the desk treats it as a refund; the customer still has to confirm the proposal.

### Choosing the model

The model and where it runs are set in `.env`; no code changes. If there's no `.env`, the defaults in `.env.example` apply. Copy that file to `.env` and change the `LLM_*` lines.

| Setting | Ollama (default) | vLLM |
|---|---|---|
| `LLM_MODEL` | `ollama_chat/gemma4:e2b` | `hosted_vllm/<model name>` |
| `LLM_API_BASE` | `http://127.0.0.1:11434` | `http://127.0.0.1:8000/v1` |
| `LLM_EXTRA_PARAMS` | `{"think": false}` | `{}` |
| `LLM_API_KEY` | not needed | only if your server requires one |

- **Routes:** any [LiteLLM route](https://docs.litellm.ai/docs/providers) works, as long as the model honours JSON schema output.
- **Local only:** the brief forbids hosted LLMs, so `LLM_API_BASE` must be loopback. To use your own server on another machine, add it to `LLM_ALLOWED_HOSTS`.
- **Declare the model:** the contract's output needs the model revision, quantization and engine version, and no neutral API reports them. Set `LLM_MODEL_REVISION`, `LLM_QUANTIZATION`, `LLM_ENGINE` and `LLM_ENGINE_VERSION`, and keep them accurate when you switch. Anything unset is reported as `unavailable`.
- **Benchmarks:** `LLM_SERVER_PROCESSES` names the server's processes for RAM measurement, and `LLM_UNLOAD_COMMAND` evicts the model for `--cold-start`.
- **Prompts and quality:** the agents' prompts and all the recorded results were tuned and measured on `gemma4:e2b`. Re-run the evaluation suite after switching model.

### Standalone MCP server

The agents' six read-only tools, and the read behind the business checks, live on a separate server that runs on its own, and the app has no other copy of them. Start it in its own terminal before the API, the benchmarks or the evaluation suite; each of them stops at startup with a message saying so if it can't reach the server. The adapter starts one for its own run if none is running, as the contract allows, and exits with code `3` only if it can't reach or start one.

```bash
uv run python -m mcp_server
```

- **Independent:** `backend/mcp_server/` imports nothing from `app/`. It reads PostgreSQL with its own SQL on read-only transactions, and applies the rules in `backend/retail_domain/`, the same code the app uses, so eligibility can't drift between them.
- **Read-only:** it writes nothing. The app records each tool call's audit row, so the call shows in `action_audit`.
- **Two kinds of tool:** six for the agents, each agent offered only [its own](#which-tools-each-agent-gets), and `get_business_context` for the app's business checks. That one returns the whole schema's state, so no model is ever offered it, and its audit row records the outcome only.
- **Scope:** one server serves every customer. The app sends the customer, request and schema in headers on every request, and they're validated before any query. They're never tool arguments, so a model can't set them.
- **Access:** it listens on `127.0.0.1` only, every request needs `MCP_AUTH_TOKEN`, and other Host headers are refused. Whoever holds the token can ask about any customer, so keep it private.
- **No fallback:** if the server stops while the app is running, the turn ends with a clear "tools unavailable" reply and an audit entry, and nothing is proposed. Confirming a proposal doesn't need the server, because proposals, commits and reconciliation are the app's trusted code, not tools. Once the server is back, the desk recovers by itself, with no API restart.
- **Databases:** it reads `MCP_DATABASE_URL` and `MCP_TEST_DATABASE_URL`, or `DATABASE_URL` and `RETAIL_TEST_DATABASE_URL` if those are unset.

### REST API

| Method | Route | Purpose |
|---|---|---|
| GET | `/health/live`, `/health/ready` | Liveness; readiness of PostgreSQL, the model server and the guardrails. The MCP server is checked once, when the API starts |
| GET | `/api/v1/demo/cases` | The cases offered by the sign-in dropdown |
| POST | `/api/v1/demo/sessions` | Demo sign-in `{customer_id}`, looked up in the persistent store; returns a token (send it as `X-Demo-Session`). Add `scenario_id` or `uploaded_case` for an isolated copy of one case |
| GET | `/api/v1/demo/session` | The current session: customer, orders, latest proposal and history |
| POST | `/api/v1/demo/sessions/{id}/end` | Sign out: ends the session and deletes what is only its own |
| POST | `/api/v1/turns` | A chat turn `{message, client_action_id}` (never commits) |
| POST | `/api/v1/proposals/{id}/confirm` | The **only** way to commit; retry-safe with the same `client_action_id` |
| POST | `/api/v1/proposals/{id}/cancel` | Invalidate a pending proposal |
| GET | `/api/v1/request-status`, `/api/v1/audit` | Persisted status and the sanitized audit timeline |

Business outcomes are not HTTP errors: they arrive in `decision`. Errors use `{error_code, message, correlation_id}`.

### Batch adapter

```bash
python adapter.py --input candidate_assets/public_cases.json --output results.json 
uv run python candidate_assets/validate_submission.py --input candidate_assets/public_cases.json --output results.json
uv run python -m evals.check_results --input candidate_assets/public_cases.json --output results.json
```

The supplied validator checks structure only. `evals.check_results` checks business correctness: it runs the independent oracle on every case and, for the public cases, compares the decisions with the expected ones.

- Each scenario runs in a fresh, isolated database schema from its own initial state, through the same agents, tools and guardrails as the UI. Confirmation is a structured flag in the input (`confirm_proposal: "last"`), never inferred from text.
- The output holds, per scenario: turns (decision, response, proposal), the action audit (both agents' tool calls included), the final state and metrics (latency, peak RAM/VRAM, tokens), plus hardware and model configuration.
- Exit codes: `0` ok, `2` bad input, `3` infrastructure (MCP server, database, model server or guardrail models unavailable); on `3` the results file still lists every case, with failed ones marked.
- For three clean repetitions with the independent oracle: `uv run python -m evals.benchmarks.run --variant mine --input candidate_assets/public_cases.json --repetitions 3 --output-dir out/bench`. The `out/` folder is git-ignored.
- To trace one scenario in the terminal, with the MCP server running: `uv run python -m app.tools.agent_demo --case public_02_exchange --trace`.

---

## Configuration

All settings are environment variables. `.env.example` lists every one with its default.

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `postgresql://retail:retail@127.0.0.1:55432/retail` | App database (UI sessions). Only local hosts are accepted. |
| `RETAIL_TEST_DATABASE_URL` | `…/retail_test` | Tests and evaluation runs. The name must end in `_test`. |
| `LLM_MODEL`, `LLM_API_BASE` | `ollama_chat/gemma4:e2b`, `http://127.0.0.1:11434` | The LiteLLM route and where it runs. See [Choosing the model](#choosing-the-model) |
| `LLM_TEMPERATURE`, `LLM_TOP_P`, `LLM_SEED`, `LLM_MAX_TOKENS` | `0`, `1`, `7`, `512` | Deterministic decoding |
| `LLM_TIMEOUT_SECONDS` | `75` | Limit per model call |
| `LLM_ALLOWED_HOSTS` | empty | Non-loopback model servers you allow |
| `AGENT_MAX_TOOL_STEPS` | `5` | The Operations Agent's tool steps per turn. The Customer agent's limit is fixed in code |
| `LLM_EXTRA_PARAMS` | `{"think": false}` | Engine-specific parameters, as JSON |
| `GUARDRAILS_*` | enabled, fail-closed | `GUARDRAILS_JAILBREAK_THRESHOLD` (`0.8`), `GUARDRAILS_TOXICITY_THRESHOLD` (`0.5`), `GUARDRAILS_MAX_INPUT_CHARS` and the on/off switches |
| `RETAIL_STORE_SCHEMA` | `retail_store` | The UI's persistent store |
| `RETAIL_DB_POOL` | `1` | Pooled PostgreSQL connections per schema |
| `RETAIL_CONFIRM_TEMPLATE` | `1` | Confirmation replies use a deterministic template (saves one LLM call) |
| `MCP_SERVER_URL`, `MCP_SERVER_PORT` | `http://127.0.0.1:8765/mcp`, `8765` | Where the standalone server is |
| `MCP_AUTH_TOKEN` | placeholder in `.env.example` | Shared by the server and the app. **Change it**: whoever holds it can read any customer's data through the tools |
| `RETAIL_OTEL_ENABLED` | `true` for the API server, `false` elsewhere | Local telemetry to the project's own collector (127.0.0.1:14318); an explicit value always wins |
| `RETAIL_OTEL_CAPTURE_CONTENT` | `true` for the API server, `false` elsewhere | Each step's input and output on its trace, in the local Tempo only |
| `RETAIL_OTEL_ENDPOINT`, `RETAIL_OTEL_MAX_CONTENT_CHARS` | see `.env.example` | Where telemetry goes (loopback only), and how much content a span keeps |

In PowerShell, set a variable with `$env:RETAIL_DB_POOL = '0'` before the command. In bash, prefix the command: `RETAIL_DB_POOL=0 uv run …`.

---

## Testing and evaluation

```bash
uv run pytest -m "not real_model" -q  # deterministic suite: domain, business layer, PostgreSQL, guardrails, graph, API, MCP (~8 min)
uv run pytest -m real_model -q        # the model named in .env, end to end (~6 min)
cd frontend
npm run test                          # Vitest unit tests
npm run build                         # strict TypeScript + production build
npm run e2e                           # Playwright against the real stack (starts its own API and UI)
```

Stop the demo API and UI before `npm run e2e`: Playwright starts its own on ports 8080 and 5173. It uses the MCP server if one is running, and otherwise starts one. The Python tests start their own.

**How correctness is checked**

```mermaid
flowchart LR
    Sets["Public, variants,<br/>adversarial"] --> Run["Same runtime<br/>as the UI"]
    Run --> Oracle["Independent oracle<br/>imports nothing from the app"]
    Run --> Agent["Audit-based scoring<br/>tools, handoffs, claims"]
    Run --> Validator["HP structure validator"]
    Oracle --> Report["Report and evidence manifest"]
    Agent --> Report
    Validator --> Report
```

| Layer | Evidence |
|---|---|
| Domain and transactions | Unit and PostgreSQL tests, including concurrent confirmations on separate connections (one commit), mid-write rollback, immutability and append-only triggers. |
| Separation | `test_business_separation.py` drives the business layer from check to commit and replay with no model and no graph, and asserts that it imports nothing from the agents, and that the agents can't reach it. |
| Agents and graph | A scripted fake model checks routing, permissions and failure handling. **It is not inference**; real-model tests are marked `real_model`. |
| Business outcomes | An **independent oracle** (`evals/oracle.py`) re-implements the rules on raw JSON and imports nothing from `app`. |
| Agent behavior | `evals/` scores intent, both agents' tool-call validity, capability coverage, handoffs and grounded claims from the persisted audit, with labeled data. Real-model tests check that the Customer agent really calls its tool. |
| Robustness | Adversarial scenarios (prompt injection, forged approval, tool/SQL spoofing, cross-customer requests, text-only "yes", control characters) and benign controls. |
| End to end | Playwright on the real stack: the six public cases, Confirm vs typed "yes", Cancel, double-click and retry safety, the opened-or-unopened flow, the case dropdown and a file upload, the audit trail's order, privacy (no token or IDs in URL, console or storage), and loopback-only network. |

**Latest evaluation results** (`evals/evidence/evaluation/report/evaluation.md`):

| Suite | Result |
|---|---|
| Public scenarios, 3 clean runs (oracle + expected decisions) | 18/18 |
| Synthetic variants (new IDs, prices, dates, wording, faults) | 16/16 |
| Adversarial set (9 attacks, 1 PII case, 4 benign controls) | 13/14. 0 prohibited effects, 0 bypasses; 1 benign request that quoted attack text was blocked (a known, accepted false positive) |
| Tool-call validity (both agents, each against its own list) / capability coverage / handoffs | 99/99 · 38/39 · 41/42 |
| Grounded claims in replies | 79/79 |
| Backend tests / real-model tests / UI unit tests | 617 · 15 · 27, all passing |
| Playwright on the real stack | 16/16 |

These suites run the adapter's path, so the opened-or-unopened question that the UI asks first is covered by the Playwright and pytest suites, not by these scores.

```bash
uv run python -m mcp_server          # in its own terminal, first
uv run python -m evals.runner --suite all --public-runs 3 --telemetry local --output-dir out/eval
uv run python -m evals.scoring.report --input-dir out/eval
```

`evals/README.md` describes the datasets, label rubric and scoring rules.

---

## Performance

> **These numbers are historical: they don't describe the current design.** Every result under `evals/evidence/final` and `evals/evidence/experiments` ran with the tools in-process and before the Customer agent's tool call, so they include neither the MCP server's cost nor that extra model call. The current evaluation's median is 5 LLM calls and about 9.3 s per case, but that run isn't a controlled benchmark. Re-run the commands below for current numbers, with heavy apps closed: the quiet-machine gate needs about 5.4 GB of free RAM before the model is loaded, and refuses to run otherwise.

Measured on one laptop (i7-8750H, GTX 1060 Max-Q 6 GB, 16 GB RAM, Windows 11, on AC power): every public scenario 3 times from a clean state, so 18 task runs per variant. Benchmarks only run when a quiet-machine gate passes: enough free RAM, idle GPU, AC power, and a stable decode rate. No other model client should be running.

Both variants ran the same code on the same test set. The **baseline** is the real, working system with two optimizations turned off (`RETAIL_DB_POOL=0`, `RETAIL_CONFIRM_TEMPLATE=0`); the **final** version has them on. Evidence is in `evals/evidence/final/`.

| | Baseline | Final | Change |
|---|---|---|---|
| Complete-task success | 18/18 | 18/18 | no regression |
| Failed tasks / failed model calls | 0 / 0 | 0 / 0 | |
| Post-commit timeouts injected / recovered | 3 / 3 | 3 / 3 | |
| Successful-task p50 | 9.74 s | 6.90 s | −29.1% |
| Successful-task p95 | 12.64 s | 11.65 s | −7.8% |
| LLM calls per task (median) | 6 | 4 | −2 |
| Output tokens per second of LLM call (median) | 32.4 | 34.9 | +7.7% |
| Completion tokens, all runs | 4,135 | 3,568 | −13.7% |
| Peak RAM (Python + Ollama) | 4,900 MB | 5,275 MB | +7.7% (worse) |
| Peak accelerator memory (Ollama model allocation) | 315.5 MB | 315.5 MB | none |
| Cold startup, 1 sample each (not a code effect, see below) | 55.1 s | 33.9 s | |

**Reading these numbers:**
- **Cold startup is separate from warm execution:** the task latencies above are all warm. The model was removed from memory before each variant, but the difference is not an improvement: neither optimization affects startup, and the baseline ran first, so the final variant probably benefited from the operating system's file cache.
- **p95** of 18 runs is close to the maximum, so treat it as a rough figure. No statistical significance is claimed.
- **Accelerator memory:** in this recorded run, Ollama's reported allocation for the model, not a per-process driver reading. New runs report whole-GPU use from `nvidia-smi` instead, so they won't match this row. Whole-GPU use was 3,045 MB in both, including other applications.

**Declared for every run** (full table in `evals/evidence/final/comparison/comparison.md`, values in each `environment.json`):
- **Seeds:** LLM seed 7, temperature 0, top_p 1.
- **Concurrency:** 1 case, 1 turn and 1 LLM request at a time.
- **Caches:** model evicted before the run and then kept loaded; Ollama's prompt cache not cleared; guardrail models loaded once per process; fresh database schemas for every case in every repetition; the OS file cache not controlled.
- **Timeouts:** 75 s per LLM call, 512 tokens, 5 agent tool steps, no overall task timeout.

**Per-run records:** `run_1.json` to `run_3.json` (contract results) and `runs.jsonl` (one row per task run) in each variant's folder.

**Reproduce** (with Chrome, VS Code and other heavy apps closed; PowerShell users set the variables with `$env:`):

```bash
uv run python -m mcp_server          # in its own terminal, first
RETAIL_DB_POOL=0 RETAIL_CONFIRM_TEMPLATE=0 uv run python -m evals.benchmarks.run --variant baseline --input candidate_assets/public_cases.json --repetitions 3 --output-dir evals/evidence/final/baseline --cold-start
RETAIL_DB_POOL=1 RETAIL_CONFIRM_TEMPLATE=1 uv run python -m evals.benchmarks.run --variant final --input candidate_assets/public_cases.json --repetitions 3 --output-dir evals/evidence/final/final --cold-start
uv run python -m evals.benchmarks.compare --baseline evals/evidence/final/baseline --optimized evals/evidence/final/final --output-dir evals/evidence/final/comparison
uv run python -m evals.benchmarks.verify evals/evidence/final
```

**Required behaviours and where each is shown:**

| Brief | Public case | Result in every run |
|---|---|---|
| Ambiguous request | `public_04_clarify` | Clarify, then propose, then completed |
| Policy rejection | `public_03_expired` | Denied, nothing changed |
| Stock changing before confirmation | `public_05_stale_stock` | Proposed, then denied at commit |
| Duplicate request | `public_06_timeout_replay` | The repeat returns the original result, one refund |
| Timeout after commit | `public_06_timeout_replay` | Reconciled by reading the result, not by committing again |

"Never invent a successful transaction": a turn reports `completed` only from a committed record in the database, and an independent oracle (`evals/oracle.py`) re-checks every result's final state.

The two optimizations, tested one at a time behind a correctness gate in `evals/evidence/experiments/`:
1. **A PostgreSQL connection pool per schema** (`RETAIL_DB_POOL`).
2. **A deterministic confirmation reply template** (`RETAIL_CONFIRM_TEMPLATE`), which saves one LLM call.

To benchmark and compare your own variants:

```bash
uv run python -m evals.benchmarks.run --variant mine --input candidate_assets/public_cases.json --repetitions 3 --output-dir out/bench 
uv run python -m evals.benchmarks.compare --baseline <dir A> --optimized <dir B> --output-dir out/compare
```

The committed evidence lives in `evals/evidence/`, and each part has a verifier:
- `uv run python -m evals.benchmarks.verify evals/evidence/final`
- `uv run python -m evals.verify evals/evidence/evaluation`

---

## Observability

Telemetry is **local**. The project runs its own OpenTelemetry Collector, Tempo, Prometheus and Grafana on 127.0.0.1 ports 14318, 13200, 19090 and 13000. Nothing is exported off the machine.

```mermaid
flowchart LR
    App["App<br/>private OTel providers"] -->|"OTLP/HTTP, loopback"| Col["Collector<br/>127.0.0.1:14318"]
    Col -->|"traces"| Tempo["Tempo"]
    Col -->|"metrics"| Prom["Prometheus"]
    Tempo --> Graf["Grafana<br/>127.0.0.1:13000"]
    Prom --> Graf
```

**The API turns it on at startup.** `uv run python -m app.api.serve` (with the MCP server already running) checks the collector and, if it isn't answering, runs `docker compose -f observability/compose.yaml up -d` (this project's stack only; the images are local, so it works offline). It prints one line such as `Telemetry: on - started the local stack; collector at http://127.0.0.1:14318; Grafana http://127.0.0.1:13000`. If Docker isn't available, the API keeps running without telemetry and says so. The containers use `restart: unless-stopped`, so after a reboot they come back with Docker.

| To… | Use |
|---|---|
| Run the API without telemetry | `--no-telemetry`, or `RETAIL_OTEL_ENABLED=false` |
| Only report a missing collector, never start containers | `--no-start-stack` |
| Stop the stack (keeps its data) | `docker compose -f observability/compose.yaml stop` |

The adapter, benchmarks and tests keep telemetry **off** unless `RETAIL_OTEL_ENABLED=true`, so measurements stay comparable. The evaluation runner captures spans in memory by default (`--telemetry local`), with no stack needed, because its handoff score reads them.

Grafana (http://127.0.0.1:13000) opens on the **overview** dashboard:

| Section | What it shows |
|---|---|
| Application | Turn p50 and p95, overall and by kind, and p50 and p95 for each step of a turn |
| CPU, RAM and GPU | p50 and p95 of whole-machine CPU and RAM and whole-GPU use and memory, plus RAM for the API, model server and MCP server |
| Tokens and usage | Input, output and total tokens, tokens per turn and per minute, by agent |
| Model throughput | Output tokens per second, LLM call p50 and p95, and LLM calls per turn |
| Tool calling | Calls, failures and refusals, p50 and p95 by tool, and guardrail verdicts (`flagged` means abusive tone noted, not blocked) |
| Failures | Error turns, model and tool failures, guardrail blocks and outages, recovered timeouts, and the requests that failed |
| Requests | Recent requests with the customer's message and the reply |

Click a request to open the **request** dashboard. It shows that request's steps in order with each step's input, output, state before and after, and time, then every model and tool call, then the timeline.

**What traces store.** The API records each step's input and output on its trace, including the customer's message, order data and model prompts. This stays in the local Tempo for 7 days. Customer text is captured only after PII masking, and text the input guard blocks is never captured. Set `RETAIL_OTEL_CAPTURE_CONTENT=false` to turn it off. It's off by default for the adapter, benchmarks, evaluations and tests.

Metric labels and the rest of the spans never carry IDs or message text. `observability/README.md` has the span tree, metric names and privacy rules.

---

## Security and safety model

- **Money and stock are never the model's decision.** Eligibility, amounts, proposal binding, commit and reconciliation are deterministic, trusted code. No agent can call `create_proposal`, `commit`, fault injection, reset or SQL.
- **Explicit confirmation only.** A commit needs a structured confirmation of a specific, immutable proposal bound to the customer, request, line, remedy and amount. Chat text, including "yes" or `{"approved": true}`, never commits.
- **Exactly once.** An advisory lock per request, ordered row locks, policy re-checked under the locks, and one transaction. Unique constraints back up every invariant, and replays return the stored result.
- **Trusted identity.** The customer and request come from the server-side session or the scenario, never from the browser or the model. Tool arguments are limited to order, line and SKU, and are checked against allowlists.
- **Least privilege for tools.** Each agent is offered only its own read-only tools, and a tool outside its list is refused and audited. The business checks' full-state read is in neither agent's list or schema.
- **A refusal rests on the customer's words.** "It was opened" is read by fixed rules from the customer's own message, never from a model's reading of a tool result, and nothing a customer says can mark an item unopened.
- **Guardrails on every boundary.** Input (sanitize, injection, jailbreak, toxicity, PII masking), tool results (treated as untrusted), agent actions (strict schema) and replies (consistency with trusted facts). The input checks run before anything else, and the guardrails fail closed. Abusive input is the one exception to blocking: it is flagged, never refused.
- **No silent substitution.** If the customer names an order, line or replacement that differs from what the agent resolved, the desk asks or denies instead of acting on something else.
- **Local only.**
  - Model, guardrails, MCP tool server, database and telemetry all run on the machine.
  - The model endpoint and model name are allowlisted.
  - Library telemetry is disabled.
- **Isolation.** Sign-ins share the persistent `retail_store` schema, bound to their own customer. Each case session, listed or uploaded, and each evaluation scenario gets its own schema, and an uploaded case can never touch the store. Code refuses non-local databases, unexpected schema names, and test runs against non-test databases.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Database connection errors | Start the container: `docker compose up -d postgres`. The port is **55432**. |
| UI stuck on "Loading local models…" | Make sure the model server named in `.env` is running and has the model. |
| A guardrail model is missing from the cache | Rerun the two `post_install` commands once while online. |
| "The MCP server is not available" at startup | Start it first: `uv run python -m mcp_server`. Both sides read the same `MCP_AUTH_TOKEN`. If the adapter reports it, something else on that port refused the token. |
| Benchmark exits with "machine not quiet" | Close large apps (browsers), connect AC power, and wait for other model clients to finish. |
| PowerShell rejects `VAR=value command` | That's bash syntax. Use `$env:VAR = 'value'`, then run the command. |
| `npm run e2e` fails to start servers | Ports 8080 or 5173 are in use. Stop the demo API and UI first. |
| `start_demo.bat` says a port is in use | The native stack (or another app) holds it. Stop it, or choose other ports in `deploy/.env`. |
| A case 5 exchange is denied before any proposal | Its stock drops before your second turn. Say "It is unopened." in your first message. |
| A chat request says "tools unavailable" | The MCP server has stopped. Start it again; the desk recovers without an API restart. |
| The Case dropdown shows only "None" and "New empty case" | The case list couldn't be read from the API. Check that it's running and ready. |

---

## Development notes

- **Running tests.** Run one test with `uv run pytest backend/tests/test_return_policy.py -q`. Database tests **fail** rather than skip when PostgreSQL is down. `-m "not postgres"` runs the pure tests only. Pytest starts its own MCP server on a free port, so you don't need one running.
- **Scenario fixtures.** `candidate_assets/` holds the scenario contract (`CONTRACT.md`), the public scenarios and the structural validator. These files are treated as read-only, and their exact bytes are hashed into every result. Use the scenarios as tests; don't special-case them (a test greps the runtime code for scenario IDs and phrases).
- **Adding business rules.** Eligibility is in `backend/retail_domain/policy.py`, the single source of truth, and the steps around it are in `backend/app/business/`. The database services call the policy on state built from locked rows, and the independent oracle must be updated separately.
- **Evidence.** `evals/evidence/` holds run records with SHA-256 manifests. Try new runs under `out/` first, and back up before regenerating evidence in place.
