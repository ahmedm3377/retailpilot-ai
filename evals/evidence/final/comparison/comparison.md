# Benchmark comparison: baseline versus final

Input SHA-256 `f3f9dc73ac2d846de3af5c8647afcccd58d0335b80cd06f416f4dd08a1ca6165` · model digest `b37049369adf` · baseline `e20a5fcfbf+8b392f0e` (config `d300992d6df0`) → optimized `e20a5fcfbf+8b392f0e` (config `2a768d7bf1af`)

**Comparable:** True
**Quality preserved:** True · **claims allowed:** True

## Correctness

| | Baseline | Optimized |
|---|---|---|
| Complete-task success | 18/18 | 18/18 |
| Failures by category | none | none |
| Error turns | 0 | 0 |
| Failed model calls | 0 | 0 |
| Injected / recovered post-commit timeouts | 3 / 3 | 3 / 3 |

## Latency of successful tasks (ms)

| Metric | Baseline | Optimized | Improvement |
|---|---|---|---|
| p50 | 9,736.2 | 6,902.6 | 29.1% |
| p95 | 12,636.7 | 11,648.8 | 7.8% |
| mean | 9,402.3 | 7,063.7 | 24.9% |
| n | 18 | 18 | |

| Case (median ms) | Baseline | Optimized | Improvement |
|---|---|---|---|
| public_01_refund_boundary | 8,140.5 | 5,520.8 | 32.2% |
| public_02_exchange | 10,276.0 | 7,206.9 | 29.9% |
| public_03_expired | 6,044.3 | 5,399.8 | 10.7% |
| public_04_clarify | 12,482.2 | 10,029.1 | 19.7% |
| public_05_stale_stock | 9,626.2 | 7,110.4 | 26.1% |
| public_06_timeout_replay | 9,736.2 | 6,250.4 | 35.8% |

## Where the time goes (median per task, ms)

| Component | Baseline | Optimized | Improvement |
|---|---|---|---|
| llm_ms | 7,053.2 | 5,591.4 | 20.7% |
| guard_ms | 861.5 | 647.1 | 24.9% |
| other_ms | 1,295.8 | 433.3 | 66.6% |
| LLM calls per task | 6 | 4 | 33.3% |

## Tokens, memory, cold start

| Metric | Baseline | Optimized | Change |
|---|---|---|---|
| Completion tokens (total) | 4135 | 3568 | 13.7% |
| Effective output tokens per call-second (median) | 32.4 | 34.9 | 7.7% (higher is better) |
| peak_ram_mb_max | 4,900.0 | 5,275.2 | -7.7% |
| peak_vram_mb_max | 315.5 | 315.5 | 0.0% |
| peak_gpu_device_used_mb_max | 3,045.0 | 3,045.0 | 0.0% |
| Cold start ms | 55,120.8 | 33,908.6 | 38.5% |

## Declared seeds, concurrency, caches and timeouts

| Group | Setting | Baseline | Optimized |
|---|---|---|---|
| seeds | llm_seed | 7 | 7 |
| seeds | temperature | 0.0 | 0.0 |
| seeds | top_p | 1.0 | 1.0 |
| seeds | note | the only random input to decisions; schema names and session IDs are random but unused by any decision | the only random input to decisions; schema names and session IDs are random but unused by any decision |
| concurrency | cases | 1 | 1 |
| concurrency | turns | 1 | 1 |
| concurrency | llm_requests | 1 | 1 |
| concurrency | ollama_num_parallel | server default | server default |
| concurrency | note | cases run one after another in one process | cases run one after another in one process |
| caches | model_load | evicted from Ollama before the run, so cold start includes the model load | evicted from Ollama before the run, so cold start includes the model load |
| caches | warm_runs | the model stays loaded between cases (one warm-up call before the first) | the model stays loaded between cases (one warm-up call before the first) |
| caches | ollama_prompt_cache | server default, not cleared between cases | server default, not cleared between cases |
| caches | guardrail_models | loaded once per process from local caches | loaded once per process from local caches |
| caches | database_connections | new connection per use | per-schema pool |
| caches | business_state | none carried over: fresh schemas for every case in every repetition | none carried over: fresh schemas for every case in every repetition |
| caches | os_file_cache | not controlled, so a later variant may read model files from memory | not controlled, so a later variant may read model files from memory |
| timeouts | llm_call_s | 75.0 | 75.0 |
| timeouts | llm_max_tokens | 512 | 512 |
| timeouts | agent_tool_steps | 5 | 5 |
| timeouts | tool_backend | direct | direct |
| timeouts | task | no overall task timeout; injected post-commit timeouts are counted separately | no overall task timeout; injected post-commit timeouts are counted separately |

## Experiments

| ID | Decision | Effect | Reason |
|---|---|---|---|
| E1-pool | accept | p50 8601 -> 8080 ms (6.1%), mean 4.6%, p95 0.9%; LLM calls/task 6 -> 6 | gate passed and latency improved |
| E3-confirm-template | accept | p50 8080 -> 5719 ms (29.2%), mean 20.7%, p95 22.8%; LLM calls/task 6 -> 4 | gate passed and latency improved |

## Method and limits

- Percentiles: nearest-rank: sorted value at 1-based rank ceil(q*n); no interpolation.
- Latency scope: first turn start to last turn end; successful runs only.
- Sample: 3 and 3 repetitions x 6 cases; n(successful) = 18 and 18; p95 of n<=20 is a near-maximum, no significance claimed.
- Formulas: improvement_pct = 100 * (baseline - optimized) / baseline; increase_pct = 100 * (optimized - baseline) / baseline.
- Memory scope: {'peak_ram_mb': 'adapter Python RSS + its child processes (Step 9 MCP servers) + ollama* RSS summed per sample (shared pages double-counted); runs before Step 9 did not sample children', 'peak_vram_mb': 'Ollama /api/ps size_vram (model allocation, not per-process driver data)', 'peak_gpu_device_used_mb': 'whole GPU incl. other applications'}.
- Throughput: Ollama-reported completion tokens / summed LLM call wall time (not pure decode time).
- Experiment effects were measured one change at a time against the then-accepted state; they are not additive. The headline numbers are a fresh baseline-versus-final measurement.
- Reproduce: see the README's Performance section (benchmark run, compare and verify commands).
