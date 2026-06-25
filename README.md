# SlowAndSweet

Delegate the leaf work of a frontier-model task to a pool of small local LMs.

## The idea

When a frontier-model assistant (e.g. Claude Code) handles a user prompt,
a lot of the work inside that response is mechanical — listing facts,
drafting boilerplate paragraphs, summarizing chunks, expanding bullet
points. The frontier model is overkill for those steps.

SlowAndSweet is a local-first system that lets the frontier model
**decompose** a prompt into a small DAG of subtasks and **delegate** the
leaves to a pool of [Ollama](https://ollama.com/)-hosted SLMs. The
frontier model keeps the planning, the quality judgment, and the final
synthesis.

The trade is exactly what the name says: each token is *slower* than a
frontier model, but tokens are local, free, and parallel — and the
parts you keep on the frontier model are the parts that actually
benefit from it.

## End-to-end flow

```
   ┌─────────────────────────────────────────────────────────────────┐
   │  Claude Code session                                            │
   │   1. user prompt arrives                                        │
   │   2. Claude authors a DAG plan (which leaves are mechanical?)   │
   │   3. Claude calls MCP tool   slm_submit_plan(plan)              │
   │   6. Claude reads node results, synthesizes the final answer    │
   └────────────┬───────────────────────────────▲────────────────────┘
                │ slm_submit_plan                │ slm_wait_plan
                ▼                                │
   ┌─────────────────────────────────────────────┴────────────────────┐
   │  slm-queue/mcp_server.py     (MCP, streamable HTTP)              │
   └────────────┬───────────────────────────────▲────────────────────┘
                │ POST /plans                    │ GET /plans/<id>
                ▼                                │
   ┌─────────────────────────────────────────────┴────────────────────┐
   │  slm-queue/server.py                                             │
   │   4. PlanRunner walks the DAG; submits ready nodes to            │
   │      per-model queue.Queue                                       │
   │   5. Worker threads (one per replica) pull tasks, call Ollama,   │
   │      write results back; downstream prompts get parent results   │
   │      substituted in via {{node.result}}                          │
   └────────────┬───────────────────────────────▲────────────────────┘
                │                                │
                ▼                                │
   ┌─────────────────────────────────────────────┴────────────────────┐
   │  ollama serve     (one process, one weight copy per model)       │
   │   smollm2:1.7b    gemma2:2b    llama3.2:3b    qwen2.5:3b    ...  │
   └──────────────────────────────────────────────────────────────────┘
```

## Components

The repo is a chain of small artifacts that build up to that flow:

| Where | Question it answers |
| ----- | ------------------- |
| [`slm-bench/`](slm-bench/) | Which local SLMs are actually fast enough that the trade is worth making? |
| [`slm-deploy/`](slm-deploy/) | How do you declare which models live on the host and how many copies of each? |
| [`slm-queue/`](slm-queue/) | How do you submit work, route it to a model, and run a multi-step DAG of prompts? |
| [`slm-queue/mcp_server.py`](slm-queue/mcp_server.py) | How does the frontier model actually call into the pool? |

### `slm-bench/` — pick the SLMs

[`slm-bench/slm_bench.py`](slm-bench/slm_bench.py) sends a fixed prompt
(`"Explain why the sky appears blue in 3 sentences, then give one
counterexample."`) to each model with `seed=42`, `temperature=0.7`,
`num_predict=256`. One warmup call, then a measured run.

| Model        | Tokens | Eval (s) | Tokens/sec | TTFT (s) | Wall (s) |
| ------------ | -----: | -------: | ---------: | -------: | -------: |
| smollm2:1.7b |    135 |     1.99 |       67.9 |    0.047 |     2.04 |
| gemma2:2b    |     99 |     1.64 |       60.3 |    0.080 |     1.73 |
| llama3.2:3b  |    174 |     3.26 |       53.3 |    0.102 |     3.37 |
| qwen2.5:3b   |    163 |     3.22 |       50.6 |    0.088 |     3.32 |
| phi3:mini    |    256 |     5.05 |       50.7 |    0.083 |     5.14 |

- **Tokens/sec** — `eval_count / eval_duration`.
- **TTFT** — `load_duration + prompt_eval_duration`.
- **Wall** — end-to-end request latency from the client.

Raw measurements (with full responses) are in
[`slm-bench/slm_bench_results.json`](slm-bench/slm_bench_results.json).
Compute and memory consumption across 10 prompts per model is captured
in [`resources.md`](resources.md), produced by
[`slm-bench/resources_bench.py`](slm-bench/resources_bench.py).

```sh
ollama serve   # in another terminal
for m in smollm2:1.7b gemma2:2b llama3.2:3b qwen2.5:3b phi3:mini; do ollama pull "$m"; done
python3 slm-bench/slm_bench.py
```

These numbers set the bar: a frontier model is roughly an order of
magnitude slower per token than the fastest of these, so we should only
delegate leaves that the smaller models can plausibly handle.

### `slm-deploy/` — declare the pool

A Kubernetes-inspired YAML schema for declaring which SLMs live on a
host and how many replicas of each, plus a validator that checks the
total against the node's resource budget.

- [`slm-deploy/node.yaml`](slm-deploy/node.yaml) — `kind: Node`,
  declares `capacity`, `allocatable`, and `gpu.shareable: true` for
  Apple Silicon's single shared GPU.
- [`slm-deploy/slms.yaml`](slm-deploy/slms.yaml) — one
  `kind: SLMDeployment` doc per model, with `replicas` and
  `resources.requests` / `resources.limits`.
- [`slm-deploy/validate.py`](slm-deploy/validate.py) — sums
  `replicas × requests` and compares to `allocatable`. Stdlib only.

```sh
python3 slm-deploy/validate.py slm-deploy/node.yaml slm-deploy/slms.yaml
```

The queue server reads `slms.yaml` directly — the spec is what
determines how many worker threads exist per model.

### `slm-queue/` — run the work

The orchestration core. Two surfaces:

**Task surface** — submit one prompt, get one result. A router picks
the model (code-ish → qwen2.5, math → llama3.2, long → gemma2,
default → smollm2), the task lands on that model's `queue.Queue`, and
a worker thread (one per replica) picks it up and calls Ollama.

| Method | Path           | Body / Response                                   |
| ------ | -------------- | ------------------------------------------------- |
| POST   | `/tasks`       | `{"prompt": "..."}` → `{"task_id", "model"}`      |
| GET    | `/tasks/<id>`  | task state: `status`, `model`, `worker`, `result` |
| GET    | `/status`      | queue depths, task counters, worker count         |

**Plan surface** — submit a whole DAG, get a structured run. Nodes
carry `depends_on` lists; their prompts may reference upstream outputs
via `{{node_id.result}}`. A background `PlanRunner` walks the DAG,
hands each ready node to the task surface, mirrors task state into
node state, and unlocks dependents. `GET /plans/<run_id>/ui` is an
auto-refreshing HTML page with a Mermaid diagram of the DAG (colored
by node state) and a status table.

| Method | Path                          | Purpose                                           |
| ------ | ----------------------------- | ------------------------------------------------- |
| POST   | `/plans`                      | submit a DAG plan; returns `run_id`               |
| GET    | `/plans/<run_id>`             | full plan run snapshot as JSON                    |
| GET    | `/plans/<run_id>/ui`          | live HTML status page (auto-refresh while running)|
| GET    | `/ui`                         | landing page listing plan files and runs          |

![DAG status page after a completed run](slm-queue/screenshots/dag_run.png)

```sh
OLLAMA_NUM_PARALLEL=2 ollama serve
python3 slm-queue/server.py --port 8080
# open http://127.0.0.1:8080/ui in a browser
```

See [`slm-queue/README.md`](slm-queue/README.md) for the architecture
in depth, the plan schema, routing rules, and per-component
experiments.

### `slm-queue/mcp_server.py` — let the frontier model call in

A thin [Model Context Protocol](https://modelcontextprotocol.io/)
adapter over the plan surface, exposing two tools to a Claude Code
session:

- `slm_submit_plan(plan)` → `{"run_id", "plan_id"}`
- `slm_wait_plan(run_id, timeout_s?)` → full plan snapshot

Setup (one-time):

```sh
python3 -m venv .venv
.venv/bin/pip install -r slm-queue/requirements.txt
```

Run alongside the queue:

```sh
python3 slm-queue/server.py --port 8080      &
.venv/bin/python slm-queue/mcp_server.py --port 8090 &
```

Register in `.mcp.json` (project root or `~/.claude/mcp.json`):

```json
{
  "mcpServers": {
    "slm-queue": {
      "type": "http",
      "url": "http://127.0.0.1:8090/mcp"
    }
  }
}
```

Restart Claude Code. The two tools show up in the tool list alongside
`Read` and `Bash`; the assistant can now decompose a prompt, dispatch
to local SLMs, and compose the answer from what comes back.

## What's not in scope

This is a prototype: in-memory state (restart loses queues and run
history), no auth, no retries, no streaming responses, no distributed
workers, no cancellation, no result-table eviction. The deploy
validator and the queue both run on one host. The point of the repo
is the shape of the delegation pattern, not a production-ready
implementation.
