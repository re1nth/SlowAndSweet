# Technical debt

## Current state (2026-07-09)

Every fast decision we've taken is here. Nothing is a fire yet;
several will be within 3-6 months if the system sees real
traffic.

## The ledger

Ordered by roughly-when-it-will-hurt.

### Composer token approximation is heuristic
- **Debt**: `planner._composer_prompt_approx` estimates the
  composer input by concatenating terminal-node results under
  a fixed wrapper. Real callers may build wildly different
  composer prompts.
- **Why we took it**: no formal "composer" step exists in the
  queue; needed *something* better than the same-value proxy
  the earlier version had.
- **Will hurt when**: the plugin lands a real composer with a
  non-trivial wrapper. Every reduction% before then is
  approximate.
- **Expiry**: as soon as the plugin's real composer prompt
  can be introspected, tokenise the actual prompt.

### `prompt_text` in feedback records
- **Debt**: We emit the full raw prompt into `feedback.jsonl`
  because `train.py` needs it to embed. Design doc doesn't
  say we should.
- **Why we took it**: the alternative — running the encoder
  at emit time and storing embeddings — is a lot more work
  for a local prototype.
- **Will hurt when**: this becomes a multi-user service. Raw
  user prompts on disk is a privacy story we don't have.
- **Expiry**: before any deployment outside a single dev's
  box. Fix: emit embeddings, not text.

### No cron for `train.py`
- **Debt**: Retraining is manual. `metrics.jsonl` only grows
  when someone runs the script.
- **Why we took it**: cron adds a scheduler dependency; the
  design lists this as M4 milestone.
- **Will hurt when**: we forget to retrain for a week and the
  head drifts.
- **Expiry**: as soon as feedback volume > 40/day for a week.

### Router's `_decision_stash` has no TTL
- **Debt**: `slm-queue/router.py` stashes decisions keyed by
  prompt hash, pop-on-read. A plan that never terminates leaks
  one entry per submission.
- **Why we took it**: simplicity, pop-on-read felt clean.
- **Will hurt when**: a bad Ollama run leaves plans stuck in
  `running` forever. At 1000/day and a 1 % stuck rate, 10
  leaks/day, ~50 KB/day.
- **Expiry**: add a TTL sweep whenever the plugin or a UI
  surfaces "your prompt is stuck."

### `state.record_call` doesn't record composer_tokens_actual
- **Debt**: We now compute a real `composer_tokens_actual`
  in `planner._report_router_feedback`, but the same number
  never lands in `state.db`. `estimated_solo_tokens` there
  is still the old counterfactual output estimate.
- **Why we took it**: schema migration would need version
  bumping and there's no consumer yet.
- **Will hurt when**: someone builds a dashboard off
  `state.db` and can't answer "how many input tokens did
  we save this week" without joining to `feedback.jsonl`.
- **Expiry**: at first dashboard build.

### No config for the composer wrapper text
- **Debt**: `_COMPOSER_WRAPPER` in `planner.py` is hard-coded.
- **Why we took it**: deterministic + short. Fine.
- **Will hurt when**: a caller uses a substantially different
  wrapper and our estimate is off by more than 50 tokens.
- **Expiry**: when a real composer wrapper diverges from the
  hard-coded one by >20 %.

### `slm-experiments/runner.py` sends `head_version: experiment`
- **Debt**: Records tagged `head_version: experiment` are
  intermixed with real head-version records in `feedback.jsonl`.
  Training uses both indiscriminately.
- **Why we took it**: simple, works.
- **Will hurt when**: we want to compute per-head calibration
  metrics — the experiment rows pollute the head-specific
  buckets.
- **Expiry**: at first "per-head calibration" analysis.

### Two feedback record schemas coexist (with/without `prompt_text`)
- **Debt**: `bootstrap.py` reads three different source
  schemas (LC bench, sxs-real-2, synthetic). Runtime feedback
  emits a fourth. `train.py` normalises them but silently
  drops anything missing required fields.
- **Why we took it**: bootstrapping from existing artifacts
  was faster than defining one schema up front.
- **Will hurt when**: a schema drift causes silent record
  dropping. See [reliability.md](reliability.md).
- **Expiry**: when the training MAE plateaus and we suspect
  data volume is the issue.

### Bare `except: pass` in metric-recording paths
- **Debt**: `_record_run_metric` and `_report_router_feedback`
  swallow all exceptions. Necessary but blinding.
- **Why we took it**: metrics must never break a plan run.
- **Will hurt when**: metrics stop working and nobody notices
  for weeks. Detection: n-of-records grows more slowly than
  n-of-plan-runs.
- **Expiry**: add a Prometheus counter for the swallow rate.

## Log

### 2026-07-08 — This file was created
Every debt above was known at the time it was taken. Writing
them down here so we don't rediscover them at 2 AM.
