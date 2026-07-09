# Feedback loops

## Current state (2026-07-09)

The system has one closed feedback loop and two half-open ones.

**Closed**: Router → queue → planner → `feedback.jsonl` → `train.py` →
`heads/vN.joblib` → router. Every plan submission produces a
learning event. Verified end-to-end 2026-07-08.

**Half-open (quality)**: `slm-experiments/runner.py` posts
reviewer verdicts to the same `feedback.jsonl` when the harness
runs. But the harness runs only when someone manually invokes
`experiments.py run` — no scheduled run, no continuous verdict
flow.

**Half-open (correction)**: `metrics.jsonl` records auto-promote
gate outcomes, but nothing consumes it. A demoted head just sits
in `heads/` waiting for a follow-up training run to replace it.
There's no "roll back" trigger from live signal.

## Loop diagram

```
  incoming        ┌───────────────┐         ┌────────────────┐
  prompts ──────► │ slm-router    │────────►│ Decision       │
                  │  /route       │         │  {solo,mix,    │
                  └───────────────┘         │   unsure}      │
                        ▲                    └──────┬─────────┘
                        │                            │
                        │                            ▼
                        │            ┌──────────────────────────┐
              head      │            │ slm-queue                │
              reload    │            │  /plans → runs → done    │
                        │            └──────┬───────────────────┘
                        │                    │
              ┌─────────┴──────┐              │
              │ train.py       │              │
              │  auto-promote  │              │
              │  gate          │              │
              └────────▲───────┘              │
                       │                       │ POST /feedback
                       │                       │ (cost + quality)
                       │  read              │
                       │                       ▼
                  ┌────┴────────────────────────────┐
                  │  feedback.jsonl                 │
                  │  cost from queue,               │
                  │  quality from slm-experiments   │
                  └─────────────────────────────────┘
```

## Signals we already learn from

| Signal              | Source                       | Emit rate       |
| ------------------- | ---------------------------- | --------------- |
| observed_reduction_pct | queue planner post-run    | 1 per plan      |
| wall_ms_saved_vs_solo  | queue planner post-run    | 1 per plan      |
| quality_verdict     | slm-experiments reviewer     | 1 per case run  |
| explored decision   | router policy (ε-greedy)     | ε × plan submits |

## Signals we could learn from and don't

- **User-facing "regenerate" click** in the plugin — implicit
  thumbs-down. Zero cost signal. Nobody wires this yet.
- **Session outcome** — did the user keep the mixture answer
  or throw it away? Hardest to capture cleanly; noisiest but
  potentially most honest.
- **Follow-up prompt similarity** — if the user asks a very
  similar prompt right after, the previous answer probably
  wasn't sufficient. Requires embedding-space similarity that
  we already have from MiniLM.
- **Latency SLA violations** — a plan that took 60 s when the
  budget was 10 s is a signal even if the answer is correct.

## The counterfactual problem

Every routing decision is action-conditional: we see the outcome
of the arm we picked, never the other. The design (§8) picks
ε-greedy dual-inference as the correction. The **queue currently
runs mixture unconditionally**, so all plan runs are implicitly
mixture-arm samples — no counterfactual signal at all from the
queue side. That's fine for cost signal (we can compare mixture
tokens to estimated solo tokens) but bad for quality signal (we
never see what solo would have produced).

Quality counterfactual currently comes only from
`slm-experiments`, which runs both arms per case. That signal
needs to be volume-multiplied to matter. Options:
- Wire the plugin to shadow-run the solo arm on ε of traffic,
  post the verdict.
- Have the queue itself run a solo pass on ε of traffic when
  a reviewer is available.
- Sample from `slm-experiments/cases/` in a nightly cron and
  post verdicts to feedback.jsonl continuously.

## Log

### 2026-07-08 — First closed loop
End-to-end verified: `router.choose_arm` at plan submit →
Decision stashed → plan runs → planner posts feedback →
`train.py` on 46 records produces v1 → auto-promote gate
accepts (v0 had null metrics) → new head active. First
"machine learned from itself" moment.

### 2026-07-08 — Quality classifier at accuracy 1.0 is a lie
`train.py`'s output: `qacc=1.000`. Fully lying. 10 quality
labels with a clean short-prompt / long-prompt split means
the head memorised the separation. Real accuracy under
distribution shift will collapse. Track: quality holdout
accuracy over time; when it stops being 1.0, the head has
started to see real prompts.
