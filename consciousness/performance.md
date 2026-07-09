# Performance

## Current state (2026-07-09)

We're not fast. A plan run through the queue takes ~10 s wall clock on
average and can hit 30 s for a 6-node DAG. The `/route` call adds
~290 ms of MiniLM inference in front of every plan submission. Almost
all latency is model-time — Python overhead is a rounding error.

No component is memory-constrained on the test box. GPU is unused;
Ollama runs on CPU/Metal, MiniLM runs on CPU. On my M-series Mac each
Ollama model swap costs a few seconds if it's not the last-used one.

## Metrics we track

- **E2E plan latency (wall_ms)** — recorded per plan by
  `slowandsweet/state.py::record_call`. Current stats (n=20):
  - avg wall: **9 855 ms**
  - avg SLM output tokens: **378**
  - avg estimated solo output: **407**
- **/route latency (ms)** — logged per request by the router server
  (structured JSON to stdout). Rolling p50/p99 available at
  `GET /metrics`. Observed p50 so far: **~290 ms**, dominated by
  MiniLM encode.
- **Model load time** — MiniLM: ~10 s cold. `all-MiniLM-L6-v2` weights
  are ~90 MB, loaded once per process.
- **Resource footprint** — not instrumented. `/metrics` doesn't expose
  RSS or CPU. This is a gap.

## Budgets (aspirational, not measured)

| Path                     | Target p50 | Target p99 | Where we are |
| ------------------------ | ---------: | ---------: | -----------: |
| `/route`                 |     100 ms |     250 ms | ~290 ms p50  |
| Full plan run (6 nodes)  |       10 s |       30 s | at target    |
| First-run server startup |       15 s |       30 s | ~15 s        |

## Known bottlenecks

- **MiniLM encode is CPU-bound.** ~290 ms/call is the encoder itself,
  not the head. On a warm cache the head predict is <1 ms. Options:
  ONNX quantized model (fastembed) to get to <50 ms; or batch requests
  and share a single encode across concurrent /route calls.
- **Model swapping in Ollama.** If the queue hits `qwen2.5:3b` after
  `llama3.2:3b` and Ollama has to unload/reload, the first request
  eats ~3 s. `OLLAMA_NUM_PARALLEL` mitigates this by keeping both hot;
  cost is RAM.
- **Serial DAG execution when it doesn't need to be.** The planner
  runs ready-nodes as they become available, but the current merge-
  intervals plan has three independent trace nodes that could parallelise
  and don't (single worker per model).

## Log

### 2026-07-08 — First real router latency observed: 290 ms
End-to-end plan run through the wired queue reported the router took
290.309 ms on the first `/route` call. That's the MiniLM encode
dominating. `Head.predict` on the loaded encoding is negligible
(<1 ms). If we want <100 ms p50, the encoder has to shrink or go
ONNX.

### 2026-07-08 — Wall-time counterfactual shows mixture was slower
The plan run reported `wall_ms_saved_vs_solo: -4343` — the SLM DAG
was 4.3 s **slower** than the estimated solo path. For the small
merge-intervals plan, solo wins on latency too, not just tokens.
This is the router's job to learn.

### 2026-07-04 — Wall-time estimate for the solo counterfactual
Commit `00af84a` added a rough estimate of what the solo call would
have cost by assuming ~40 tok/s frontier output rate (25 ms/token).
Not calibrated to a real Anthropic API run yet. Worth measuring once
someone runs the `slm-experiments` harness with a real key.
