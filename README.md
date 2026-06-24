# SlowAndSweet

Benchmarks for small language models (SLMs) running locally via [Ollama](https://ollama.com/).

## SLM Benchmark

The script in [`slm-bench/slm_bench.py`](slm-bench/slm_bench.py) sends a fixed prompt
(`"Explain why the sky appears blue in 3 sentences, then give one counterexample."`)
to each model with `seed=42`, `temperature=0.7`, `num_predict=256`. Each model gets one
warmup call before the measured run.

### Results

| Model        | Tokens | Eval (s) | Tokens/sec | TTFT (s) | Wall (s) |
| ------------ | -----: | -------: | ---------: | -------: | -------: |
| smollm2:1.7b |    135 |     1.99 |       67.9 |    0.047 |     2.04 |
| gemma2:2b    |     99 |     1.64 |       60.3 |    0.080 |     1.73 |
| llama3.2:3b  |    174 |     3.26 |       53.3 |    0.102 |     3.37 |
| qwen2.5:3b   |    163 |     3.22 |       50.6 |    0.088 |     3.32 |
| phi3:mini    |    256 |     5.05 |       50.7 |    0.083 |     5.14 |

- **Tokens/sec** — generation throughput (`eval_count / eval_duration`).
- **TTFT** — time to first token, approximated as `load_duration + prompt_eval_duration`.
- **Wall** — end-to-end request latency from the client.

Raw measurements (including full responses) are in
[`slm-bench/slm_bench_results.json`](slm-bench/slm_bench_results.json).

### Reproducing

```sh
ollama serve  # in another terminal
for m in smollm2:1.7b gemma2:2b llama3.2:3b qwen2.5:3b phi3:mini; do ollama pull "$m"; done
python3 slm-bench/slm_bench.py
```

Compute and memory consumption for 10 prompts per model is captured in
[`resources.md`](resources.md), produced by
[`slm-bench/resources_bench.py`](slm-bench/resources_bench.py).

## SLM Deployment Spec

[`slm-deploy/`](slm-deploy/) defines a Kubernetes-inspired YAML schema for declaring
which SLMs should be pre-loaded on a host and validating that they fit in the
node's resource budget.

- [`slm-deploy/node.yaml`](slm-deploy/node.yaml) — `kind: Node`, declares `capacity`
  and `allocatable` (memory / cpu / gpu) plus `gpu.shareable: true` for Apple
  Silicon's single shared GPU.
- [`slm-deploy/slms.yaml`](slm-deploy/slms.yaml) — one `kind: SLMDeployment` document
  per model, with `replicas` and K8s-style `resources.requests` / `resources.limits`.
- [`slm-deploy/validate.py`](slm-deploy/validate.py) — sums `replicas × requests`
  across all deployments and compares to `allocatable`. Prints a fit table; exits
  non-zero on over-commit. Stdlib only (no PyYAML dependency).

```sh
python3 slm-deploy/validate.py slm-deploy/node.yaml slm-deploy/slms.yaml
```

## SLM Queue Server

[`slm-queue/`](slm-queue/) is a local prototype of a queue + worker pool that
consumes the SLM deployment spec. The HTTP server reads `slm-deploy/slms.yaml`,
spawns `replicas` worker threads per declared model, and routes each incoming
prompt to a model via simple rule-based heuristics. It also accepts whole
**DAG plans** on the producer side (JSON in [`slm-queue/plans/`](slm-queue/plans/)):
nodes can reference upstream outputs via `{{node.result}}`, a background runner
walks the topology, and `GET /plans/<run_id>/ui` renders a live Mermaid diagram
+ status table. See [`slm-queue/README.md`](slm-queue/README.md) for the
architecture, routing rules, and experiments.

- [`slm-queue/router.py`](slm-queue/router.py) — `choose_model(prompt, available)`:
  code-ish prompts → qwen2.5, math/reasoning → llama3.2, long prompts → gemma2,
  default → smollm2.
- [`slm-queue/server.py`](slm-queue/server.py) — `ThreadingHTTPServer` with an
  in-memory `queue.Queue` per model and one thread per replica. All workers call
  the local `ollama serve` HTTP API.
- [`slm-queue/client.py`](slm-queue/client.py) — demo client; submits a batch of
  prompts and polls until each completes.

Endpoints:

| Method | Path           | Body / Response                                    |
| ------ | -------------- | -------------------------------------------------- |
| POST   | `/tasks`       | `{"prompt": "..."}` → `{"task_id", "model"}`       |
| GET    | `/tasks/<id>`  | task state: `status`, `model`, `worker`, `result`  |
| GET    | `/status`      | queue depths + task counters + worker count       |

```sh
ollama serve  # in another terminal; set OLLAMA_NUM_PARALLEL=N for real
              # concurrent generation when any model has replicas > 1
python3 slm-queue/server.py --port 8080
python3 slm-queue/client.py --base http://127.0.0.1:8080
```
