# SLM Queue Server

A local prototype of a task queue + worker pool for small language models.
Tasks (natural-language prompts) are submitted over HTTP, routed to a model
by simple heuristics, dispatched to a pool of worker threads, and answered
by a shared local [Ollama](https://ollama.com/) backend.

The worker pool topology is read directly from
[`../slm-deploy/slms.yaml`](../slm-deploy/slms.yaml) — the same spec that
[`slm-deploy/validate.py`](../slm-deploy/validate.py) checks for node fit.

## Architecture

```
                                         per-model queue.Queue
                                       ┌───────────────────────┐
                                       │ smollm2:1.7b  ──► W#0 │
   client ──► POST /tasks ──► router ──┤              ──► W#1 │
                                       │ gemma2:2b     ──► W#0 │──► ollama serve
                                       │ llama3.2:3b   ──► W#0 │     (one process,
                                       │ qwen2.5:3b    ──► W#0 │      one weight copy
                                       └───────────────────────┘      per model)
                                                 │
   client ──► GET /tasks/<id> ◄── in-memory results table ◄────────────┘
```

- **One queue per declared model** (`queue.Queue`, thread-safe).
- **One worker thread per replica** (`spec.replicas` in the YAML).
- **Single Ollama backend** — each unique model is loaded into memory once.
  Replicas are concurrent client workers, not duplicated weight copies.
  Set `OLLAMA_NUM_PARALLEL=N` when starting `ollama serve` if you want real
  parallel generation against the same model.

## Files

| File         | Purpose                                                    |
| ------------ | ---------------------------------------------------------- |
| `server.py`  | HTTP server, dispatcher, per-model queues, worker threads. |
| `router.py`  | `choose_model(prompt, available)` — rule-based routing.    |
| `client.py`  | Demo client; submits a batch and polls until done.         |

## Endpoints

| Method | Path           | Body / Response                                                |
| ------ | -------------- | -------------------------------------------------------------- |
| POST   | `/tasks`       | `{"prompt": "..."}` → `202 {"task_id", "model"}`               |
| GET    | `/tasks/<id>`  | task state: `status`, `model`, `worker`, `result`, timestamps  |
| GET    | `/status`      | queue depths, task counters, worker count                      |

`status` transitions: `pending` → `running` → `done` (or `error`).

## Routing

`router.choose_model` picks among the available deployed models in this
order:

| Trigger                                                       | Model      |
| ------------------------------------------------------------- | ---------- |
| Code-ish: `code`, `function`, `python`, `sql`, `regex`, ` ``` ` | `qwen2.5`  |
| Math/reasoning: `calculate`, `solve`, `prove`, `equation`, …  | `llama3.2` |
| Long prompt (`len(prompt) > 500`)                             | `gemma2`   |
| Default (fastest)                                             | `smollm2`  |

If a preferred model isn't deployed, the next rule wins; ultimate fallback
is the first deployed model.

## Run

```sh
# Terminal 1 — start ollama (set NUM_PARALLEL for replica parallelism)
OLLAMA_NUM_PARALLEL=2 ollama serve

# Terminal 2 — start the queue server
python3 slm-queue/server.py --port 8080

# Terminal 3 — submit a batch
python3 slm-queue/client.py --base http://127.0.0.1:8080
```

Or hit it directly with `curl`:

```sh
curl -s -X POST http://127.0.0.1:8080/tasks \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Write a Python function for the nth Fibonacci number."}'
# {"task_id": "bdc279732301", "model": "qwen2.5:3b"}

curl -s http://127.0.0.1:8080/tasks/bdc279732301
curl -s http://127.0.0.1:8080/status
```

## Experiment

10 prompts submitted as a batch against the default deployment
(`smollm2:1.7b`×2, `gemma2:2b`×1, `llama3.2:3b`×1, `qwen2.5:3b`×1 — 5
workers total). Ollama at default settings (single-stream per model).

### Routing — which model received each prompt

| Prompt (truncated)                                   | Routed to      | Why          |
| ---------------------------------------------------- | -------------- | ------------ |
| Write a Python function for nth Fibonacci…           | `qwen2.5:3b`   | code         |
| Write a regex that matches a US zip code.            | `qwen2.5:3b`   | code         |
| Give me a SQL query to find the top 3 customers…    | `qwen2.5:3b`   | code         |
| Calculate 17 * 23 and show your reasoning…          | `llama3.2:3b`  | math         |
| Solve for x: 3x + 5 = 26.                            | `llama3.2:3b`  | math         |
| Prove that the sum of two even integers is even.    | `llama3.2:3b`  | math         |
| Summarize photosynthesis in one sentence.           | `smollm2:1.7b` | default      |
| Name three jazz musicians active in the 1950s.      | `smollm2:1.7b` | default      |
| What is the capital of Iceland?                     | `smollm2:1.7b` | default      |
| (700-char story prompt about a lighthouse keeper)   | `gemma2:2b`    | long prompt  |

### Timing — per task

`queue` is the wait between submission and a worker picking it up;
`gen` is the actual Ollama call.

| task_id        | worker             | queue (s) | gen (s) | tokens |
| -------------- | ------------------ | --------: | ------: | -----: |
| bdc279732301   | `qwen2.5:3b#0`     |      0.00 |    8.34 |    256 |
| 6471576ca299   | `qwen2.5:3b#0`     |      8.34 |   12.07 |    256 |
| 1f541038096a   | `qwen2.5:3b#0`     |     20.42 |    6.17 |    256 |
| e4e3818440be   | `llama3.2:3b#0`    |      0.00 |    5.00 |    106 |
| ce7da52b8a4b   | `llama3.2:3b#0`    |      5.00 |   13.32 |     83 |
| e1442a34f715   | `llama3.2:3b#0`    |     18.32 |    4.09 |    130 |
| 5f42c35d9ede   | `smollm2:1.7b#1`   |      0.00 |    2.62 |     42 |
| bf4d6791ddc6   | `smollm2:1.7b#0`   |      0.00 |    5.79 |    142 |
| 109716e87c86   | `smollm2:1.7b#1`   |      2.62 |    6.32 |     25 |
| 9df717c37630   | `gemma2:2b#0`      |      0.00 |   13.44 |    256 |

### Aggregate

| Model           | tasks | total gen | avg gen |
| --------------- | ----: | --------: | ------: |
| `qwen2.5:3b`    |     3 |    26.59s |   8.86s |
| `llama3.2:3b`   |     3 |    22.42s |   7.47s |
| `smollm2:1.7b`  |     3 |    14.73s |   4.91s |
| `gemma2:2b`     |     1 |    13.44s |  13.44s |

**Wall time end-to-end: 26.89s** — vs ~77s if everything had been serialized
on one worker. The speedup is bounded by the slowest *per-model* queue
(qwen2.5 with three code prompts).

### Observations

- **Replicas worked.** Two smollm2 tasks were picked up immediately by
  workers `#0` and `#1` in parallel (both at `queue=0.00s`); the third
  smollm2 task only waited 2.62s, getting `#1` as soon as it freed up.
- **Single-replica models serialized.** qwen2.5 received 3 prompts; tasks
  2 and 3 waited ~8s and ~20s respectively in the queue.
- **Routing held.** Every prompt landed on the model the heuristics
  predicted. Long-prompt routing only triggered above 500 chars (an
  earlier 367-char "long story" prompt fell through to smollm2; padding
  to 700 chars correctly routed to gemma2).
- **Ollama serialization caveat.** With default `OLLAMA_NUM_PARALLEL=1`,
  two smollm2 worker threads still hit a serialized backend even though
  they pulled from the queue independently. To see *generation*
  parallelism (not just dispatch parallelism), restart ollama with
  `OLLAMA_NUM_PARALLEL` >= the maximum `replicas` in `slms.yaml`.

## Limitations

This is a prototype, not a production queue. Not implemented: persistence
(restart loses queue + results), backpressure / max queue depth, retries,
auth, streaming responses, distributed workers, cancellation,
result-eviction (results table grows unbounded).
