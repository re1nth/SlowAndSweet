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
   │   1a. UserPromptSubmit hook -> slm-router /decompose            │
   │       - rule matches (multi-variant / for-each / summarize)     │
   │         -> hook injects `[autoroute:mixture]` + ready-made DAG  │
   │       - no match, router down, or /no-delegate prefix           │
   │         -> hook stays silent (frontier answers solo)            │
   │   2. Claude uses the injected DAG (or authors one manually)     │
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

## Setup

End-to-end install, macOS or Linux:

```sh
# 1. One-time system deps
#    macOS:  brew install ollama
#    Linux:  curl -fsSL https://ollama.com/install.sh | sh
ollama serve &                           # in another terminal is fine

# 2. Install the daemon CLI. Handles the state dir, the bearer token,
#    the SQLite metrics DB, and pulls the four SLMs.
pipx install ./slowandsweet
slowandsweet init

# 3. Install the Claude Code plugin. Reads the token from ~/.slowandsweet/
#    and templates .mcp.json into ~/.claude/plugins/slowandsweet/.
sh plugin/install.sh

# 4. Start the servers. (These will move under one process in a later
#    revision — for now they're separate so each stays inspectable.)
python3 slm-queue/server.py    --port 8080 &
python3 slm-queue/mcp_server.py --port 8090 &
python3 slm-router/server.py   --port 8092 &    # required for auto-route

# 5. Restart Claude Code. You now have:
#    - the slm-batch subagent (Claude invokes it automatically on 3+ homogeneous items)
#    - auto-route: UserPromptSubmit hook -> slm-router /decompose -> DAG hint
#    - /delegate <task>, /slm-doctor, /slm-stats slash commands
#    - a PostToolUse hook logging delegations to ~/.slowandsweet/calls.jsonl
```

Verify with `slowandsweet doctor` — every check should be `OK`, with `queue
server reachable` and `MCP server reachable` reflecting the running servers.

**Kill switches.**

- `slowandsweet disable` writes `~/.slowandsweet/disabled`; the MCP tool
  refuses plans **and** the auto-route hook goes quiet. Re-enable with
  `slowandsweet enable`. No Claude restart needed.
- `slowandsweet disable autoroute` silences only the auto-route hook;
  manual `/delegate` still works. Re-enable with `slowandsweet enable autoroute`.
- Prefix any single prompt with `/no-delegate ` to force solo for that turn.

`slowandsweet stats` reports today's delegation totals.

## Components

The repo is a chain of small artifacts that build up to that flow:

| Where | Question it answers |
| ----- | ------------------- |
| [`slm-bench/`](slm-bench/) | Which local SLMs are actually fast enough that the trade is worth making? |
| [`slm-deploy/`](slm-deploy/) | How do you declare which models live on the host and how many copies of each? |
| [`slm-queue/`](slm-queue/) | How do you submit work, route it to a model, and run a multi-step DAG of prompts? |
| [`slm-queue/mcp_server.py`](slm-queue/mcp_server.py) | How does the frontier model actually call into the pool? |
| [`slowandsweet/`](slowandsweet/) | User-facing CLI (`init`, `doctor`, `stats`, `disable`/`enable`) that owns `~/.slowandsweet/` — token, SQLite metrics, kill switch. |
| [`plugin/`](plugin/) | Claude Code plugin: `slm-batch` subagent, `/delegate` + `/slm-doctor` + `/slm-stats` commands, MCP registration with bearer auth, PostToolUse hook. |
| [`slm-experiments/`](slm-experiments/) | Does the mixture actually beat frontier-only? SxS harness with a blind reviewer over 10 cases. |

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

Gated by a shared bearer token read from `~/.slowandsweet/token`. If the
token file is absent the server runs unauthenticated (dev fallback) and
prints a warning; when present, requests without a matching
`Authorization: Bearer <token>` header are rejected with 401.

Don't wire this up by hand — use the [Setup](#setup) flow above. The
`slowandsweet` CLI generates the token, and `plugin/install.sh` renders
the plugin's `.mcp.json` with it substituted. If you're just poking at
the raw MCP surface for dev, start the server directly:

```sh
python3 slm-queue/server.py --port 8080 &
python3 slm-queue/mcp_server.py --port 8090 &
```

### `slowandsweet/` — the user-facing CLI

Installable via `pipx install ./slowandsweet`. Stdlib + PyYAML only, so
the wheel stays thin.

| Command                    | What it does |
| -------------------------- | ------------ |
| `slowandsweet init`        | Create `~/.slowandsweet/`, generate a 32-byte hex token (mode 0600), init the SQLite metrics DB, `ollama pull` each model in `slm-deploy/slms.yaml`. Idempotent. Does not auto-install Ollama — prints instructions and exits non-zero if missing. |
| `slowandsweet doctor`      | Ten-item health checklist (state dir, token mode, DB schema, Ollama reachable, each required model present, queue + MCP reachable, kill-switch state). `--json` for machine output. |
| `slowandsweet stats`       | Today's `delegated / abstained / failed` totals from `~/.slowandsweet/state.db`, plus top abstain reasons. |
| `slowandsweet disable`     | Create `~/.slowandsweet/disabled`. The MCP tool checks for it and refuses plans until re-enabled. |
| `slowandsweet enable`      | Remove the flag. |

`PlanRunner` in `slm-queue/planner.py` writes one row to the `calls`
table (and rolls up `metrics_daily`) on every terminal plan transition,
via a lazy import — the queue keeps running if the wheel isn't
installed.

### `plugin/` — the Claude Code plugin

Symlinked into `~/.claude/plugins/slowandsweet/` by `plugin/install.sh`,
so you can edit files in-repo and see them live. The exception is
`.mcp.json`: `install.sh` renders it from `.mcp.json.template` with the
real token substituted, then writes it as a real file (not a symlink)
in the install dir. This is why the install order is `slowandsweet init`
before `sh plugin/install.sh` — the token has to exist first.

The load-bearing surfaces:

- `agents/slm-batch.md` — the delegation subagent. Its `description`
  field is what Claude reads to decide when to invoke it (≥3 homogeneous
  mechanical items; abstain otherwise). Body handles the DAG build,
  submit/wait, and synthesis.
- `commands/delegate.md` — `/delegate <task>` force-invokes the
  subagent, bypassing the heuristic.
- `commands/slm-doctor.md`, `commands/slm-stats.md` — shell to the CLI.
- `hooks/hooks.json` — a `PostToolUse` hook scoped to
  `mcp__slm-queue__slm_*` only (not `.*` — that would fire on every
  tool call). Appends one JSONL line per delegation to
  `~/.slowandsweet/calls.jsonl`.

There is intentionally no `CLAUDE.md` at the plugin root — Claude Code
plugins don't auto-load a root `CLAUDE.md`, so any prose there would be
invisible. The subagent's `description` carries the load-bearing
"when to invoke" and "silently fall through on abstain" contract.

## What's not in scope

Still a prototype. What's there now:

- Auth on the MCP surface (shared bearer token, gated by
  `~/.slowandsweet/token`).
- SQLite-backed metrics for delegations (`~/.slowandsweet/state.db`) so
  `slowandsweet stats` reflects real runs.
- A cross-process kill switch (`slowandsweet disable`).
- A one-command plugin install.

What's still out:

- Queue/plan-run state is in memory — restarting `slm-queue/server.py`
  loses in-flight runs and history.
- No retries, no streaming responses, no distributed workers, no
  cancellation, no result-table eviction, no per-node timeouts.
- The router in `slm-queue/router.py` is regex over keywords, not a
  learned classifier.
- The deploy validator and both servers run on one host.

The point of the repo is the shape of the delegation pattern, not a
production-ready implementation.
