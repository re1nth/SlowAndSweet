# `slm-router` — design doc

Status: **draft**, no implementation yet.
Owner: unassigned.
Related components: [`slm-queue/`](../slm-queue/README.md),
[`slm-experiments/`](../slm-experiments/README.md),
[`slowandsweet/state.py`](../slowandsweet/slowandsweet/state.py),
[`slm-experiments/results/lc_frontier_bench/`](../slm-experiments/results/lc_frontier_bench/).

## 1. Motivation

The `SlowAndSweet` pipeline can serve a prompt one of two ways:

- **Solo** — send the whole prompt to Claude Code and return its answer.
- **Mixture** — decompose the prompt into a DAG, let local SLMs handle the
  mechanical leaves, feed distilled outputs into Claude Code as a
  composer prompt.

The [LC frontier bench](../slm-experiments/results/lc_frontier_bench/)
shows the mixture pattern saves an average **30.7 %** of Claude Code's input
tokens across 10 LeetCode problems — but the win varies from **+56.9 %**
(long, wordy problems) to **−7.4 %** (tiny problems where the SLM sketch
is longer than the raw statement). A one-size-fits-all decision is
demonstrably wrong.

The current router — `slm-queue/router.py::choose_model` — is a fixed
keyword classifier that picks *which SLM* handles a leaf, not *whether
the mixture pattern is worth invoking at all*. We need a component that
sits **before** the queue and answers a different question:

> Given this incoming prompt, will the mixture path save Claude Code
> enough tokens to justify the extra latency, without meaningfully
> degrading answer quality?

`slm-router` is that component. It's a small learned model — MiniLM
embedding plus a linear head — served as a sidecar to `slm-queue`,
continually retrained from the outcomes of the decisions it (and the
system as a whole) has already made.

## 2. Goals and non-goals

### 2.1 Goals

- **G1.** Answer `POST /route` with a real-valued `predicted_reduction_pct`
  and an actionable `decision ∈ {solo, mixture, unsure}` in under 20 ms
  p99 on CPU.
- **G2.** Continually retrain from a `feedback.jsonl` log that grows one
  record per plan run in the queue, without operator intervention beyond
  configuring a cadence.
- **G3.** Track *two* outcome axes: **cost** (input-token savings) and
  **quality** (composer-vs-solo answer preference). Refuse to route into
  the mixture path when either signal is under-informed.
- **G4.** Version and roll back the trained head trivially — a single
  `head.joblib` file per version, plus a `HEAD` symlink or pointer.
- **G5.** Cold-start gracefully. On day one the router falls back to a
  hand-tuned heuristic that mirrors `slm-queue/router.py`'s style and
  logs everything so a learned head can supersede it later.

### 2.2 Non-goals

- **N1.** Rewriting `slm-queue/router.py::choose_model`. That routes leaves
  between SLMs (`smollm2` vs `qwen2.5` etc.); `slm-router` routes between
  *arms* (solo vs mixture). Both routers coexist.
- **N2.** Building a new reviewer. Quality feedback reuses
  [`slm-experiments/reviewer.py`](../slm-experiments/reviewer.py).
- **N3.** Fine-tuning the MiniLM encoder. The head is what we train;
  the encoder is frozen. End-to-end fine-tuning is a >10 k-label regime.
- **N4.** Serving multiple tenants. Single-user local deployment first.
- **N5.** Online SGD. Batch retraining on cadence is simpler to reason
  about and reversible; we'll only reach for `SGDClassifier` if the
  latency profile forces us to.

## 3. Architecture

```
   ┌──────────┐   POST /route         ┌────────────────────────────┐
   │ caller   │──────────────────────►│  slm-router (port 8092)    │
   │ (slm-    │◄──────────────────────│                            │
   │  queue,  │  {decision,           │  MiniLM encoder (frozen)   │
   │  plugin, │   predicted_          │      │                     │
   │  ...)    │   reduction,          │      ▼                     │
   └────┬─────┘   confidence,         │  head.joblib (versioned)   │
        │         decision_id}        │      │                     │
        │                             │      ▼                     │
        │                             │  cost head   quality head  │
        │                             └──────┬─────────────┬───────┘
        │                                    │             │
        │  plan runs                         │             │
        ▼                                    ▼             ▼
   ┌──────────────┐  after each run   ┌─────────────────────────────┐
   │ slm-queue    │──────────────────►│  POST /feedback             │
   │ planner.py   │  {decision_id,    │  appends to feedback.jsonl  │
   └──────┬───────┘   observed_       └──────────────┬──────────────┘
          │           reduction,                     │
          │           quality?,                      │
          │           ...}                           │
          ▼                                          │
   ┌──────────────────┐                              │
   │ state.py SQLite  │                              │
   │ (counterfactual  │                              │
   │  cost signal)    │                              ▼
   └──────────────────┘                     ┌──────────────────────┐
                                            │ train.py (cron)      │
                                            │  reads feedback.jsonl│
                                            │  writes head.joblib  │
                                            └──────────────────────┘
```

The router is a **single Python process** exposing three endpoints, holding
one embedding model in memory and one small classifier head on disk. It
does *not* own a database of its own — feedback logs are append-only
JSONL for durability and easy inspection.

## 4. Core concepts

### 4.1 Decision

A tuple returned by `/route`:

```
{
  "decision_id": "d_a1b2c3",           // opaque handle for later feedback
  "decision": "solo" | "mixture" | "unsure",
  "predicted_reduction_pct": 34.2,     // regression head output
  "predicted_quality_ok_prob": 0.87,   // quality-preservation prob
  "confidence": 0.72,                  // width-of-interval or ensemble std
  "head_version": "v42",               // which model made the call
  "policy": "learned" | "heuristic" | "explore"
}
```

`unsure` is a first-class outcome. If either head's confidence is below a
configured threshold the caller is expected to fall back to solo (safe
default), and the router logs the fact so exploration can trigger.

### 4.2 Feedback record

One JSONL line per plan run, appended by the queue after the run
terminates:

```json
{
  "decision_id": "d_a1b2c3",
  "timestamp": "2026-07-06T22:41:12Z",
  "prompt_hash": "sha256:...",             // dedup key
  "prompt_len_tok": 289,
  "decision_made": "mixture",
  "policy": "learned",
  "head_version": "v42",
  "outcome": {
    "solo_tokens_estimated": 289,
    "composer_tokens_actual": 189,
    "observed_reduction_pct": 34.6,
    "wall_ms_slm_dag": 6100,
    "wall_ms_saved_vs_solo": -1800,        // NEGATIVE — mixture was slower
    "quality_verdict": "tie",              // A|B|tie|null (null = no judge run)
    "quality_source": "reviewer_shadow"    // reviewer_shadow | user_thumbs | none
  }
}
```

Rationale for JSONL: append-only, single-writer safe on POSIX, trivial to
`tail -f`, trivial to `jq`, no schema migrations. If we outgrow it,
promote to Parquet later.

### 4.3 Head

The head is the *only* trainable object in the router. Two sklearn
estimators pickled together into `head.joblib`:

- **`cost_regressor`**: `LinearRegression` on the MiniLM 384-dim vector,
  predicting `observed_reduction_pct`.
- **`quality_classifier`**: `LogisticRegression` predicting
  `P(mixture is not worse than solo)` — i.e. `P(quality_verdict ∈ {B, tie} | decision_made=mixture)`.

Both are tiny (kilobytes). Fast to retrain. Trivial to interpret via
their coefficients. This is the smallest thing that supports the two-axis
decision the design requires.

### 4.4 Decision policy

Given the two heads, the policy that turns a prediction into a decision:

```
if quality_classifier.predict_proba(x)[1] < QUALITY_FLOOR:
    return "solo"                 # never route if we expect quality drop
if cost_regressor.predict(x) < COST_FLOOR_PCT:
    return "solo"                 # not enough savings to be worth it
if confidence(x) < CONFIDENCE_FLOOR:
    if random.random() < EPSILON:
        return "explore"          # dual-run, log both, learn
    return "solo"
return "mixture"
```

Sensible defaults: `QUALITY_FLOOR = 0.7`, `COST_FLOOR_PCT = 20`,
`CONFIDENCE_FLOOR = 0.5`, `EPSILON = 0.10`. All read from a YAML config,
hot-reloadable.

## 5. API

### 5.1 `POST /route`

```
Request:
  {"prompt": "...", "context": {...optional...}}

Response 200:
  { see §4.1 Decision }
```

Sub-20 ms p99 CPU target. The encoder inference is the bottleneck (~5-10 ms
for 384-dim MiniLM); the head is negligible. Encoder runs on 1 CPU
thread with `SentenceTransformer.encode(..., convert_to_numpy=True)` and
no batching — routing is one-request-at-a-time by shape.

### 5.2 `POST /feedback`

```
Request:
  { see §4.2 Feedback record }

Response 200:
  {"accepted": true}
```

Idempotent by `decision_id` — resubmitting overwrites the previous
record. The queue writes here at plan-terminal-status time.

### 5.3 `GET /metrics`

Returns Prometheus-formatted or JSON metrics:

- `router_predictions_total{decision="solo|mixture|unsure",policy="..."}`
- `router_prediction_latency_seconds` (histogram)
- `router_head_version` (gauge)
- `router_feedback_records_total`
- `router_train_last_success_timestamp`
- `router_train_last_error_timestamp`
- `router_regressor_pearson_r_holdout` (updated after training)
- `router_classifier_accuracy_holdout` (updated after training)

### 5.4 `POST /reload`

Trigger the router to reload `head.joblib` from disk without restarting
the process. Called by `train.py` after a new head is written.

### 5.5 `POST /train` (dev only)

Debug endpoint to force an immediate retrain synchronously. Disabled in
production — retraining is scheduled, not RPC-driven, to keep the serving
process purely inference.

## 6. Files and layout

```
slm-router/
├── DESIGN.md                    ← this doc
├── README.md                    ← quick start once code lands
├── requirements.txt             ← sentence-transformers, scikit-learn, joblib
├── config.yaml                  ← thresholds, cadence, model choices
│
├── server.py                    ← HTTP server (stdlib http.server + threading)
├── model.py                     ← Encoder + Head classes; disk I/O
├── policy.py                    ← §4.4 decision policy, pure function
├── heuristic_fallback.py        ← rule-based baseline for cold start
│
├── train.py                     ← reads feedback.jsonl, writes new head
├── bootstrap.py                 ← seed head from existing bench+experiments
├── explore.py                   ← ε-greedy dual-run helper for the queue
│
├── heads/                       ← versioned head artifacts
│   ├── v0.joblib                ← bootstrap head
│   ├── v1.joblib
│   ├── ...
│   └── HEAD                     ← text file naming the active version
│
├── feedback.jsonl               ← append-only outcome log (rotated at 100 MB)
└── metrics.jsonl                ← per-training-run metrics (holdout MAE, etc.)
```

## 7. Component detail

### 7.1 `server.py`

Single-threaded HTTP server (stdlib) with a background thread pool for
inference. Keeps the encoder loaded once. Watches
`heads/HEAD` for changes and reloads on the next inference so `/reload`
is a fallback rather than a requirement.

Auth: bearer token from `SLM_ROUTER_TOKEN` env var if set; otherwise
open on `127.0.0.1`. Mirrors the existing pattern in
[`slm-queue/mcp_server.py`](../slm-queue/mcp_server.py).

### 7.2 `model.py`

```python
class Encoder:
    """Wraps SentenceTransformer('all-MiniLM-L6-v2'). Frozen."""
    def encode(self, prompt: str) -> np.ndarray: ...  # (384,)

class Head:
    """Holds cost_regressor + quality_classifier. Persist as one joblib."""
    def predict(self, x: np.ndarray) -> Decision: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Head": ...
```

Rationale for keeping both heads in one file: they must be trained on the
same feature space and versioned together. Splitting them tempts drift.

### 7.3 `policy.py`

Pure function `decide(head_output, config) -> Decision`. No I/O. Trivially
unit-testable. This is where the exploration / floor thresholds live so
they can be tweaked without touching the server.

### 7.4 `heuristic_fallback.py`

Cold-start policy used when `heads/v0.joblib` doesn't exist yet or when
the trained head's confidence is systematically underwater. Roughly:

```python
def heuristic(prompt: str) -> Decision:
    tok = tiktoken_len(prompt)
    if tok < 120:                       # very short — solo
        return Decision("solo", reduction=0, confidence=0.9)
    has_examples = "Example" in prompt or "```" in prompt
    has_constraints = "Constraints" in prompt
    verbose = tok > 250 and has_examples
    if verbose:
        return Decision("mixture", reduction=30, confidence=0.6)
    return Decision("solo", reduction=5, confidence=0.4)
```

Direct descendant of the bench's own finding: the length+markers baseline
already scores `Pearson r = 0.62` on the LC bench. Good enough to bootstrap.

### 7.5 `train.py`

Command-line entrypoint, runs in cron:

```
python train.py
  --feedback feedback.jsonl
  --out heads/v{N+1}.joblib
  --holdout 0.2
  --min-records 40
```

Flow:

1. Load all feedback records; drop records with `outcome.observed_reduction_pct == null`.
2. Group by `prompt_hash`; keep the most recent.
3. Encode all prompts with `Encoder`.
4. Random 80/20 split (seeded).
5. Fit `LinearRegression` on `observed_reduction_pct`.
6. Fit `LogisticRegression` on `quality_verdict != "A"` (mixture not
   worse than solo) — but *only over records where quality_verdict is not
   null*. Falls back to a uniform 0.5 prior if fewer than
   `--min-quality-labels 20` records exist.
7. Compute holdout MAE + Pearson r for the regressor and accuracy for
   the classifier. Compare to the currently-active head.
8. **Auto-promote gate**: only bump `HEAD` if the new head strictly
   beats the current head on both metrics *and* has ≥ `--min-records`
   examples. Otherwise write the head file but leave `HEAD` alone.
9. Append a line to `metrics.jsonl` with the training summary.

The gate is the single most important safety property. A bad
`feedback.jsonl` (poisoned by a runaway plan) should never silently take
over the router.

### 7.6 `explore.py`

Helper that lives inside `slm-queue`. On any request tagged
`decision.policy == "explore"` or with probability `EPSILON`, the queue
runs *both* paths and reports the true (solo_tokens, composer_tokens,
quality_verdict) tuple back via `/feedback`.

Cost: ε × (extra solo call). At ε=0.10 with 100 requests/day, we pay 10
extra frontier calls per day for unbiased training data.

### 7.7 `bootstrap.py`

Reads:

- `slm-experiments/results/lc_frontier_bench/comparison.json` — 10 cost labels
- `slm-experiments/results/lc_frontier_bench/slm_results.json` — prompt text
- `slm-experiments/results/sxs-real-2.json` — 10 quality verdicts from
  the harness
- `slowandsweet/*.sqlite` `record_call` rows — plan-run cost outcomes

Produces `heads/v0.joblib` — the seed head. Roughly 30 records of
cost signal, ~10 of quality signal. Not enough for confident routing;
enough to bias the cold policy away from the random-init baseline.

## 8. The counterfactual problem

### 8.1 Statement

Once the router is deployed, every routing decision is *action-conditional*:
we observe the outcome of the arm we picked, never the other. If the router
learns from its own logs it will confirm its own biases (the standard
"logged data isn't IID" issue in contextual bandits).

### 8.2 Options considered

1. **Do nothing.** Log only chosen-arm outcomes; retrain naively.
   *Rejected.* The failure mode is silent and compounds: an early bias
   toward "solo" starves the mixture head of positive examples until it
   only fires on obvious cases.
2. **Inverse propensity scoring.** Reweight logged outcomes by
   `1 / P(action | context)`. Correct but requires storing propensities
   and adds variance.
3. **ε-greedy dual-inference.** For ε of traffic, run both arms and log
   both outcomes.  *Chosen.* Simplest to implement, gives clean IID
   training data on the explored slice, cost is bounded and adjustable.
4. **Full shadow mode.** Always run both arms, always log both.
   *Rejected.* Doubles frontier cost.

### 8.3 Exploration strategy

- Start with `EPSILON = 0.20` at cold start; decay to `0.05` after the
  first 200 feedback records.
- Bias exploration toward *uncertain* prompts: sample explore-mode
  with probability `min(1, EPSILON × (1 + confidence_gap))`. This is
  a poor-man's Thompson sampling — cheap, biased toward informative
  exploration.
- Explore records get a `policy: "explore"` tag in the feedback log and
  are weighted higher in training (they carry unbiased signal).

## 9. Quality signal

Quality is the harder half of the signal. The design integrates the
existing `slm-experiments/reviewer.py` in two modes:

**Shadow reviewer.** On ε% of explore traffic, run both solo and
mixture, then invoke the reviewer on the pair. This is the same
plumbing `slm-experiments/runner.py` already uses per case.
Cost: an extra frontier call per explore sample.

**User signal.** If a downstream caller (the Claude Code plugin) can
capture implicit signal — thumbs up/down, "regenerate" clicks,
re-asking a similar prompt — feed those in as `quality_source:
user_thumbs`. Weight them lower than reviewer verdicts.

**Never trust a single quality label.** The classifier requires
`--min-quality-labels 20` before it graduates from prior. Below that,
`P(quality_ok) = 0.5` and the policy defaults to solo unless cost signal
is *very* strong (`predicted_reduction_pct > 40 AND confidence > 0.8`).

## 10. Cold start

Day-zero deployment order:

1. `bootstrap.py` produces `heads/v0.joblib` from existing artifacts.
2. `HEAD` points to `v0`.
3. Server starts. All predictions are logged with `head_version: v0`.
4. `EPSILON` starts at 0.20. The queue exercises both arms on 20 % of
   traffic.
5. After 200 records of cost signal, `train.py` produces `v1` and the
   auto-promote gate accepts if it beats `v0` on holdout MAE.
6. After 20 quality-labeled records, the quality classifier trains for
   real; before that, the prior is used.
7. `EPSILON` decays as calibration improves.

## 11. Model versioning and rollback

- Each `train.py` run writes a numbered head file (`heads/vN.joblib`).
- `HEAD` (a text file, one line) names the active version.
- `POST /reload` swaps in whatever `HEAD` names.
- Rolling back: overwrite `HEAD` with the previous version number and
  `POST /reload`. That's the whole procedure.
- `metrics.jsonl` retains one line per training run with holdout scores
  so drift can be inspected.

Retain the last 30 head files; delete older ones via a nightly janitor
step. Head files are small (kilobytes), so this is generous.

## 12. Observability

- Log every decision as JSON to stdout. Log level configurable.
- `/metrics` (§5.3) exposes counters and gauges.
- Nightly a `router_report.md` is regenerated summarising:
  - decisions made per policy
  - decisions the auto-promote gate rejected
  - drift alerts (regressor MAE up >30 % week-over-week)
  - top-10 prompts where policy disagreed with heuristic fallback

The last one is the most useful in practice: it surfaces where the
learned head is diverging from the rule-based expectation. Either the
head has learned something legitimate, or a bug is being masked.

## 13. Failure modes and safeguards

| Mode                                                     | Safeguard                                                    |
| -------------------------------------------------------- | ------------------------------------------------------------ |
| Feedback log corrupted by a runaway plan                 | Auto-promote gate; anomaly detection on outcome fields       |
| Encoder OOMs the process                                 | Encoder pinned to one thread, no batching, 512-token cap     |
| `head.joblib` file gets corrupted mid-write              | Atomic rename: write to `head.joblib.tmp`, `fsync`, rename   |
| Router service down                                      | Queue falls back to heuristic policy; timeout=200 ms         |
| Model drift after upstream prompt distribution changes   | Weekly holdout MAE alert                                     |
| Quality reviewer is systematically biased                | Track solo-win rate over time; if >0.8, reviewer needs audit |
| Someone deletes `HEAD`                                   | Server falls back to heuristic; alert                        |

## 14. Deployment

**Phase 1: same-box, subprocess-managed.** The router is a Python
process launched next to `slm-queue/server.py`. Both processes read the
same `slms.yaml`. Router listens on `127.0.0.1:8092` — 8080 is the queue's HTTP server, 8090 is its MCP server. The queue's
router integration is behind an env var (`SLM_ROUTER_URL`), falling
back to the current keyword classifier when unset. This is the shape
that ships first.

**Phase 2: containerized sidecar.** Ship the router as a container image
built from the same base as the queue plugin (see `plugin/`). Compose
the two services with a `docker-compose.yml` alongside Ollama. Persist
`feedback.jsonl` and `heads/` on a mounted volume.

**Phase 3: multi-worker HTTP.** Only if latency budget requires. The
current design is single-worker, single-threaded on inference; MiniLM
encoding is fast enough for the local deployment scale.

## 15. Integration touchpoints

### 15.1 `slm-queue`

Minimal changes:

- `router.py`: when `SLM_ROUTER_URL` is set, call `POST /route` with the
  raw prompt before returning a model choice. If the router says
  `solo`, return `None` and let the queue caller degrade to the direct-
  frontier path.
- `planner.py`: after `_record_run_metric`, also `POST /feedback` with
  the outcome record. Include `estimated_solo_tokens` (already
  captured) as `outcome.solo_tokens_estimated`.

Neither change removes existing behavior; both are gated on the env
var. The queue with no router configured behaves exactly as today.

### 15.2 `slm-experiments`

Two touchpoints:

- `bootstrap.py` reads `results/sxs-real-2.json` for its seed quality
  labels.
- Each `runner.py` run appends its per-case verdicts to
  `slm-router/feedback.jsonl` so experiments continually feed the router.

### 15.3 `slowandsweet/state.py`

No changes required — the router reads the SQLite via a read-only
connection to enrich cost outcomes with wall-time data. The
counterfactual fields added by `00af84a` are exactly what the router
consumes.

### 15.4 `plugin/`

The Claude Code plugin's kill switch already exists (see commit
`dc9a520`). No new integration required; the router lives beneath the
plugin's abstraction, and the plugin can still cut off all mixture
routing regardless of router opinion.

## 16. Milestones

| Milestone | Scope                                                                 | Blocker for next                          |
| --------- | --------------------------------------------------------------------- | ----------------------------------------- |
| M0        | This design doc merged                                                | Alignment on 2-axis (cost+quality)        |
| M1        | `server.py` + `model.py` + `heuristic_fallback.py`; heuristic-only    | Wire into queue behind env var            |
| M2        | `bootstrap.py` → `heads/v0.joblib`; `/route` uses learned head        | Log feedback back from queue              |
| M3        | `slm-queue` writes feedback; `train.py` retrains on cadence           | ≥ 200 feedback records                    |
| M4        | Auto-promote gate + `metrics.jsonl` + nightly `router_report.md`      | ε-exploration integration in queue        |
| M5        | Quality classifier live (behind `--min-quality-labels` guard)         | 20 quality-labeled records                |
| M6        | Containerized deployment; multi-service `docker-compose.yml`          | —                                         |

## 17. Alternatives considered

- **Just extend `slm-queue/router.py::choose_model`.** Considered but
  rejected: it already has a clear responsibility (per-leaf routing) and
  no state. The arm decision is orthogonal and deserves its own service.
- **Push routing into the plugin.** Rejected: routing needs the SLM
  outcome log, which is a queue-side concept. Plugin doesn't have it.
- **Use RouteLLM instead of building our own.** RouteLLM's pretrained
  routers assume a strong / weak *LLM* pair, not a solo / mixture *arm*
  pair. Different label space. Worth revisiting once we have >1k
  labels; until then a custom head is smaller and faster.
- **Use classification only, drop regression.** Rejected: the LC bench
  showed classification loses to the constant baseline at n=10 while
  regression preserves rank correlation. Regression is the safer
  primary signal.

## 18. Open questions

- **Q1.** How do we bias the router without an actual production traffic
  distribution? All bench + experiment prompts are curated. Real
  traffic might be shorter, chattier, more repetitive. Plan: ship
  behind a plugin feature flag, gather two weeks of real logs, then
  enable learning.
- **Q2.** Should the router condition on user identity or session?
  Currently no — single-user local first. Deferred.
- **Q3.** How to model *partial* decompositions? Today it's solo vs
  full mixture DAG. In principle the router could pick which leaves
  to keep. Explicit non-goal for M1-M6.
- **Q4.** What's the right default `EPSILON` decay? Numeric choice
  here is a placeholder; needs calibration once bootstrap data
  arrives.
- **Q5.** Do we need per-domain heads (code / math / prose)? The LC
  bench is code-only; a mixed corpus might reveal that one head
  underfits. Deferred until we see the failure.
