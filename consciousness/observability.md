# Observability

## Current state (2026-07-09)

Three separate places record state:
1. **`slowandsweet/state.db`** (SQLite) — one row per plan run.
   Persists cost + wall time. Home for `record_call`.
2. **`slm-router/feedback.jsonl`** (JSONL) — one row per router
   feedback POST. Persists cost + quality per plan.
3. **`slm-router/metrics.jsonl`** (JSONL) — one row per training
   run.

Plus:
- `slm-router` `GET /metrics` — live in-process counters.
- Structured JSON logs from `slm-router/server.py` to stdout.

Nothing correlates across the three stores. If you want to answer
"for the plan that hit p99 latency yesterday, what was the router
decision and did the auto-promote gate accept the next head?" —
you have to join by run_id and timestamp by hand.

## What is instrumented

| Signal                         | Where              | Type      | Retention        |
| ------------------------------ | ------------------ | --------- | ---------------- |
| Plan wall time                 | state.db `calls`   | int, ms   | forever          |
| SLM output tokens per plan     | state.db `calls`   | int       | forever          |
| Estimated solo tokens          | state.db `calls`   | int       | forever          |
| Router decision + confidence   | feedback.jsonl     | struct    | forever          |
| Router quality verdict         | feedback.jsonl     | enum      | forever          |
| Router /route latency (rolling)| in-memory counter  | histogram | last 1000 samples|
| Predictions by decision/policy | in-memory counter  | int       | since server up  |
| Auto-promote decision          | metrics.jsonl      | struct    | forever          |
| SQL schema version             | state.db meta      | int       | forever          |

## What is not

- **Encoder inference latency separately from head latency.** Both
  are folded into `/route` latency. If the encoder starts to
  regress we'd see /route go up but not know why.
- **RSS, VMS, CPU%.** Nothing captures process resource
  footprint. `psutil` would give us this in one place.
- **GPU utilization.** Currently unused, but if we move the
  encoder to Metal or CUDA we'd want to know saturation.
- **Ollama backend metrics.** Loaded models, VRAM, model swap
  count. All hidden inside `ollama serve`. `ollama ps` gives a
  partial view.
- **Router service uptime, restart count.** In-process counter
  only; wiped on restart.
- **Reviewer verdicts over time.** `slm-experiments` writes them
  per run to a JSON snapshot, but there's no rolling win-rate.
- **Cross-component correlation ID.** No trace context flows
  from a caller through router → queue → planner → feedback. If
  a user reports "my last request was slow", we grep by timestamp
  and hope.

## What we're guessing about

- **Real production traffic shape.** Every prompt we've routed
  came from tests. Distribution of lengths, categories, and
  quality expectations is unknown. Everything the router has
  learned so far is from curated bench prompts + synthetic
  anchors. Unknown-unknown until we plug this into the plugin.
- **Cache hit rate for repeated prompts.** No dedup, no cache.
  We don't know if 30 % of user prompts are near-duplicates
  (in which case caching would dominate any router improvement).

## Log

### 2026-07-08 — Router logs are useful, but ephemeral
The router's structured JSON stdout is exactly the right shape
for downstream tooling (`jq | tee`, ship to Loki, etc.). Nobody
is doing that yet. On the test box the logs live in `/tmp/router.log`
and vanish. First follow-up when we deploy for real.

### 2026-07-08 — n=20 is the whole history
`state.db` has 20 rows total across all test runs. That's low
enough that any dashboard would show noise, not signal. Priority:
dogfood at least a hundred plan runs (from bench + experiments)
before drawing conclusions from these fields.
