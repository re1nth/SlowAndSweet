# `prism/` — split tasks, route each to LLM or SLM

In the original mixture pattern *every* leaf of the DAG goes to the
SLM pool. That's a blunt instrument: classification, accuracy-sensitive
translation, and entity extraction often need the frontier's
discrimination, but git-command boilerplate, single-paragraph
summarization, and template formatting don't.

prism is the lens that splits a task graph into "SLM-eligible" and
"LLM-only" nodes and dispatches each to the right backend.

```
            ┌──────────────────┐
            │  user prompt     │
            └────────┬─────────┘
                     ▼
           ┌────────────────────┐
           │  decomposer (LLM)  │  produces annotated DAG
           └────────┬───────────┘
                    ▼
   ┌──────────────────────────────────────────┐
   │ classifier (SLM, smollm2)                │  fills in `type` for any
   │   only consulted for untagged nodes      │  node the decomposer didn't tag
   └────────┬─────────────────────────────────┘
            ▼
   ┌──────────────────────────────────────────┐
   │ executor                                 │
   │   per-node backend = override            │
   │                    | policy[type]        │
   │                    | default_unknown     │
   │                                          │
   │   SLM nodes  ──►  slm-queue /tasks       │
   │   LLM nodes  ──►  frontier.complete()    │
   │   parents' results substituted into      │
   │   children via {{node.result}}           │
   └──────────────────────────────────────────┘
```

## Files

| File | Responsibility |
| ---- | -------------- |
| [`policies.yaml`](policies.yaml) | Task-type vocabulary and the type → backend mapping. The taxonomy is small on purpose (12 entries). |
| [`policy.py`](policy.py) | Loads `policies.yaml` and answers `backend_for(type)`. |
| [`dag.py`](dag.py) | `NodeSpec`, `NodeResult`, `PrismRun` dataclasses shared across the module. |
| [`classifier.py`](classifier.py) | One-of-N SLM classifier (default `smollm2:1.7b`). Memoized per `(prompt, model)`. |
| [`decomposer.py`](decomposer.py) | Frontier call that turns a free-text prompt into an annotated DAG. Strict JSON schema, system prompt marked for caching. |
| [`executor.py`](executor.py) | Walks the DAG, routes each node, parallelizes ready waves, substitutes upstream results. |
| [`cli.py`](cli.py) | `python prism/cli.py run "<prompt>"` or `--plan plan.yaml`. |

## Routing precedence

For every node, prism picks a backend in this order:

1. **Explicit override** — the node's `backend: slm | llm` field (if set).
2. **Policy lookup** — the node's `type:` field looked up in
   `policies.yaml`.
3. **Classifier fallback** — if `type:` is missing, the SLM classifier
   picks one of the policy types. If it returns `unknown`, prism
   routes to `default_unknown` (currently `llm`, to fail safe).

The classifier is consulted as little as possible: it's only called
when the upstream decomposer left a node untagged. In practice this
means classifier cost is bounded by the number of nodes prism didn't
already understand.

## Quick start

Prereqs are the same as `slm-experiments/`:

```sh
ollama serve &                                            # SLM workers
python3 slm-queue/server.py --port 8080 &                 # routing surface
.venv/bin/pip install -r slm-experiments/requirements.txt # also covers prism
export ANTHROPIC_API_KEY=sk-ant-...
```

Then either of:

```sh
# Decompose + run a free-text prompt.
.venv/bin/python prism/cli.py run "Compare e-scooters and e-bikes for urban commuting."

# Execute a pre-authored annotated DAG.
.venv/bin/python prism/cli.py run --plan path/to/plan.yaml

# Show the structured run snapshot.
.venv/bin/python prism/cli.py run --plan path/to/plan.yaml --json
```

Useful flags:

- `--local-frontier` — use a local Ollama model in place of Anthropic.
- `--no-classifier` — skip the SLM classifier; route untagged nodes
  straight to `default_unknown`.
- `--classifier-model` — pick a different Ollama model for the
  classifier (default `smollm2:1.7b`).
- `--parallel N` — max concurrent nodes in a wave (default 4).

## Plan schema

```yaml
plan_id: short-slug
description: human-readable one-liner
nodes:
  - id: a                # required, unique
    type: extract        # optional — skips classifier when set
    backend: slm         # optional override — beats policy + classifier
    model: smollm2:1.7b  # optional model pin (SLM model id)
    depends_on: []
    prompt: |
      Pull out fields x, y, z from this text:
      ...
  - id: b
    type: compose
    depends_on: [a]
    prompt: |
      Write the final answer using the extracted data: {{a.result}}
```

The terminal node (no children) is what the executor returns as the
final answer. Conventionally it's `type: compose` and ends up on the
LLM, but nothing stops you from making it an SLM node.

## Extending the taxonomy

`policies.yaml` is the source of truth. Add a new entry, give it a
backend, and the decomposer and classifier will both start using it
(the type list is read into both prompts at runtime).

Keep the vocabulary small. A classifier with 12 labels is much more
reliable than one with 40, and our SLMs lose calibration fast as
vocabularies grow.
