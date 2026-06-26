# `slm-experiments/` — SxS experimentation harness

Compares two ways to handle a prompt:

| Arm     | What it does                                                            |
| ------- | ----------------------------------------------------------------------- |
| **solo**    | one frontier-model call answers the whole prompt                        |
| **mixture** | local SLM pool runs a pre-authored DAG of leaves; one frontier call composes the final answer |

A **blind pairwise reviewer** (also a frontier call) sees both outputs
labeled A/B in a randomized order and picks a winner against a fixed
rubric (accuracy, completeness, clarity).

Per case the harness records: frontier input/output tokens, SLM output
tokens (Ollama `eval_count`), wall time per arm, the reviewer verdict,
and a markdown report.

## Layout

```
slm-experiments/
├── experiments.py            # CLI: list | run | report
├── runner.py                 # orchestrates arms + reviewer per case
├── reviewer.py               # blind pairwise SxS judge (cached system prompt)
├── report.py                 # snapshot -> markdown
├── metrics.py                # dataclasses for usage + verdicts
├── adapters/
│   ├── frontier.py           # AnthropicAdapter, OllamaFrontierAdapter, build_default
│   └── slm.py                # talks to slm-queue/server.py over HTTP
├── arms/
│   ├── solo.py               # Arm A
│   └── mixture.py            # Arm B (consumes slm-queue plan results)
├── cases/                    # 10 curated YAML cases, one per file
└── results/                  # JSON snapshots + rendered markdown
```

## Quick start

Run these from the repo root (`SlowAndSweet/`). Steps 1-4 are one-time
setup; step 5 is the actual experiment.

```sh
# 1. Make sure Ollama is up and the four SLMs are pulled.
ollama serve &                          # in its own terminal is fine
for m in smollm2:1.7b gemma2:2b llama3.2:3b qwen2.5:3b; do
  ollama pull "$m"
done

# 2. Start the slm-queue server (Arm B routes leaf prompts through it).
python3 slm-queue/server.py --port 8080 &

# 3. Install the harness's Python deps into the repo's venv.
python3 -m venv .venv                   # only the first time
.venv/bin/pip install -r slm-experiments/requirements.txt

# 4. Export an Anthropic API key.
export ANTHROPIC_API_KEY=sk-ant-...
# optional overrides (defaults shown):
# export EXPERIMENTS_FRONTIER_MODEL=claude-sonnet-4-6
# export EXPERIMENTS_LOCAL_FRONTIER=llama3.2:3b   # used when --local-frontier

# 5. Run the experiment.
.venv/bin/python slm-experiments/experiments.py list
.venv/bin/python slm-experiments/experiments.py run
```

The run writes `slm-experiments/results/run-<timestamp>.json` and a
sibling `.md` report, and prints the report to stdout.

## CLI reference

```sh
# List the cases that will run.
.venv/bin/python slm-experiments/experiments.py list

# Run all 10 cases against the Anthropic frontier.
.venv/bin/python slm-experiments/experiments.py run

# Run a named subset (comma-separated case ids).
.venv/bin/python slm-experiments/experiments.py run \
    --cases multi-doc-summary,code-explain,fact-recall

# Force the local Ollama frontier stand-in (skip Anthropic even if a key is set).
.venv/bin/python slm-experiments/experiments.py run --local-frontier

# Run multiple cases concurrently (default 1; raise carefully to avoid rate limits).
.venv/bin/python slm-experiments/experiments.py run --parallel 3

# Pin a run label and seed (seed controls the reviewer's A/B order randomization).
.venv/bin/python slm-experiments/experiments.py run --label baseline --seed 7

# Use a non-default queue server URL.
.venv/bin/python slm-experiments/experiments.py run --queue-url http://127.0.0.1:8080

# Re-render the markdown report for an existing run.
.venv/bin/python slm-experiments/experiments.py report --run run-1782456789
.venv/bin/python slm-experiments/experiments.py report --run slm-experiments/results/run-1782456789.json
```

### Local-frontier mode (no API key)

Without `ANTHROPIC_API_KEY` set, or with `--local-frontier`, the harness
uses a local Ollama model (default `llama3.2:3b`) as the frontier
stand-in for *all three* roles — solo arm, mixture composer, and
reviewer. The plumbing works end-to-end but the numbers are not a
Claude comparison; treat this as smoke-test mode only.

### Troubleshooting

- `ANTHROPIC_API_KEY not set` — export the key in the same shell that
  runs `experiments.py`.
- `urllib.error.URLError: Connection refused` to `:8080` — the queue
  server isn't running. Start `python3 slm-queue/server.py --port 8080`.
- `urllib.error.URLError: Connection refused` to `:11434` — `ollama
  serve` isn't running.
- `TimeoutError: plan <id> not done after 300s` — an SLM call hung.
  Check `http://127.0.0.1:8080/plans/<id>/ui` for which node is stuck.
- The reviewer returns `winner: tie` with `_parse_error` — the
  reviewer's JSON output couldn't be parsed; the raw text is preserved
  in `review.reasoning`. Usually means `max_tokens` was too low.

## Baseline results

First real run: [`results/sxs-real-2.md`](results/sxs-real-2.md) (and
the corresponding `.json` snapshot). Frontier was
`claude-sonnet-4-6`; reviewer the same. Headline:

| Arm     | Frontier in | Frontier out | SLM out | Σ arm time |
| ------- | ----------: | -----------: | ------: | ---------: |
| solo    | 1,073       | 2,327        | —       | 56.1s      |
| mixture | 2,800       | 2,293        | 1,865   | 85.6s      |
| reviewer | 9,022      | 1,728        | —       | —          |

- Cases: 10. **Solo won 8, mixture won 1 (`plan-steps`), tie 1
  (`fact-recall`).** Wall clock for the full run: 129.6s.
- On this case set the mixture pattern costs ~50% **more** frontier
  tokens than solo, because the composer prompt has to swallow every
  leaf result. The harness's job is to surface that.
- Where mixture wins is informative: in `plan-steps`, SLM-gathered
  tips *constrained* the composer toward the right answer (cool-season
  crops for early spring). Where mixture loses, the leaves were
  *replacing* judgment the frontier should have done itself
  (classification, extraction, fact-sensitive translation).

This is one snapshot, not a verdict on the pattern — re-run with
different cases, models, or prompts and the numbers move. The point is
the harness now produces evidence rather than vibes.

## Case schema

Each case is a YAML doc:

```yaml
id: short-slug                    # unique across cases/
name: Human-readable title
category: summarization | classification | extraction | code |
          math | translation | brainstorming | fact-recall |
          planning | analysis
notes: free-form
max_tokens: 600                   # caps each frontier call in this case
temperature: 0.0                  # applied to both arms

solo_prompt: |                    # Arm A input (the whole task)
  ...

solo_system: |                    # optional system prompt for Arm A
  ...

mixture_plan:                     # Arm B leaf DAG (slm-queue plan schema)
  plan_id: ...
  description: ...
  nodes:
    - { id: a, depends_on: [], prompt: "..." }
    - { id: b, depends_on: [], prompt: "..." }

mixture_composer_prompt: |        # Arm B final frontier call
  ... {leaf:a} ... {leaf:b} ...

mixture_composer_system: |        # optional system prompt for composer
  ...
```

`{leaf:NODE_ID}` is substituted with the corresponding plan node's
generated text right before the composer call. Add or edit YAMLs in
`cases/` — `experiments.py list` will pick them up automatically.

## Adding a new arm

The arms are just callables with the shape

```python
def run_my_arm(case: dict, frontier: FrontierAdapter, ...) -> ArmResult: ...
```

Wire it into `runner._run_one_case` and into the reviewer's pairwise
prompt (or extend the reviewer to a multi-way ranking).

## Caveats

- The slm-queue server only exposes Ollama's `eval_count` (output
  tokens) per node; prompt-eval tokens aren't tracked yet, so SLM
  input tokens are not in the report. Frontier input/output tokens
  are accurate.
- Arm B uses a *pre-authored* DAG per case so the comparison isolates
  "what the frontier offloads" rather than "how good is the
  decomposition." A future arm should generate the DAG dynamically
  from the user prompt.
- The reviewer is itself a frontier call. It's biased toward
  Anthropic's family of models — a second-opinion arm (e.g. an
  OpenAI judge) would be a worthwhile addition.
