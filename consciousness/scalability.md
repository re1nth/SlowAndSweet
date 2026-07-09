# Scalability

## Current state (2026-07-09)

Single-user, single-box. Nothing in the system is designed for
concurrent tenants. There is one process per component and one
port per process. The queue and router hold state in memory (in
addition to disk); scaling either horizontally would need a shared
store first.

## Metrics we track

- **Concurrent plans in flight** — the queue's `/status` reports
  `queue_depths` per model. Not persisted. Peak observed: 4 (one
  worker per model, all busy).
- **Feedback records per day** — `wc -l feedback.jsonl`. No
  rotation yet; will grow forever.
- **Head file size** — currently a few KB. No concern until we move
  to a bigger head architecture.

## Where each component breaks

### `slm-queue`
- **~100 QPS in-process ceiling.** The queue is stdlib
  `ThreadingHTTPServer` with one thread per worker slot from
  `slms.yaml`. Above that many concurrent plans queue up.
- **Per-model FIFO.** No priority, no fairness across users. Fine
  for one user, hostile for many.
- **In-memory task/run registry.** Restarting the queue loses
  in-flight work. No persistence layer.

### `slm-router`
- **One MiniLM instance per process.** Encoder is thread-safe under
  torch's GIL for our workload but not batched across concurrent
  requests. Above ~5-10 QPS on this box, `/route` latency inflates
  linearly. Fix: batch a few ms of requests into a single encode.
- **`feedback.jsonl` is append-only, single-writer.** POSIX safe
  under one process; would need rotation + coordination if we
  scaled router replicas.
- **`heads/HEAD` file pointer.** Fine for one process. If two
  routers ran on the same volume, they'd race on the atomic
  rename during `train.py`. Design says to run one; enforce it
  or the auto-promote gate becomes unreliable.

### `slm-experiments`
- **Sequential cases by default (`--parallel 1`).** Fine because
  the harness is a batch tool, not a service. Scales linearly
  with cases at whatever parallelism is set.

### Ollama
- **Model swap thrash if the working set > VRAM/RAM.** Four SLMs
  at 2-3 GB each = 8-12 GB. On the test box (M4 Pro w/ 24 GB
  unified) all four fit hot. Below 16 GB it becomes a swap
  festival.

## The "10× traffic" thought experiment

If tomorrow we had to serve 10 plans/second instead of 1 per minute:

1. **Router encoder** — needs batching or ONNX. First to fall.
2. **Ollama** — `OLLAMA_NUM_PARALLEL` and more workers per model
   in `slms.yaml`. Cost: linear RAM.
3. **Queue** — swap in-memory dispatch for Redis or a real queue.
4. **feedback.jsonl** — rotate at 100 MB (already noted as a TODO),
   promote hot pages to Parquet for training.
5. **State DB** — SQLite at ~1 QPS/writer is fine. At 100 QPS it's
   done; move to PostgreSQL.

## Log

### 2026-07-08 — Two queues, two routers, one shared feedback.jsonl
During end-to-end testing I ran a second `slm-queue` on 8081 next
to the running instance on 8080, both pointing at the same router.
It worked cleanly for cost feedback because the router serialises
writes. But — both queues share the same `state.db` at
`~/.slowandsweet/state.db` via the SQLite-backed `record_call`, and
SQLite is single-writer per file. At meaningful concurrency the two
queues would contend on that lock. Something to remember when
someone stands up a second queue "for testing" in prod.

### 2026-07-04 — Port allocation is entirely ad hoc
`slm-queue/server.py` picks 8080, `slm-queue/mcp_server.py` picks
8090, `slm-router/server.py` picked 8090 (colliding) and got moved
to 8092. There is no ports table. Scalability implication: when
we go to compose these as containers, the port map is discovered by
grep. That's fine at 3 components; painful at 10.
