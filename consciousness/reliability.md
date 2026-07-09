# Reliability

## Current state (2026-07-09)

The design is aggressively defensive: every optional integration
degrades to a working default when the dependency is missing.
Router down? Queue behaves as before. slowandsweet wheel not
installed? Metrics silently skipped. `heads/HEAD` missing? Router
falls back to `heuristic_fallback.py`. This is deliberate — we
want partial systems to work — and it's the single strongest
reliability property we have.

The cost: silent failures are hard to distinguish from correct
behavior. Every `except: return` is a place a user won't know
something didn't work.

## Metrics we track

- **Plan-run terminal status** — `done | error` recorded per run
  in `state.db`. Currently n=20, all `delegated` (i.e., success).
  Once we accumulate errors, error rate becomes a top-line SLO.
- **Router feedback POST failures** — logged to stderr in
  `slm-router/explore.py`. Not counted. Should be.
- **Encoder OOM / load failure** — server logs a warning and
  drops to heuristic. Not counted.

## Failure modes we've catalogued

| Mode                                       | Blast radius            | Detection            | Recovery              |
| ------------------------------------------ | ----------------------- | -------------------- | --------------------- |
| Router service unreachable                 | choose_arm → "mixture"  | stderr warning       | none; degrades OK     |
| Ollama unreachable                         | plan node → error       | node.error populated | manual restart        |
| `heads/HEAD` missing                       | policy → heuristic      | server startup log   | run `bootstrap.py`    |
| `feedback.jsonl` corrupted mid-line        | train.py skips bad line | try/except JSON parse| none; silent          |
| Poisoned feedback (runaway plan)           | v_new fails gate        | auto-promote gate    | manual rollback HEAD  |
| `slowandsweet` wheel missing               | SQLite metrics skipped  | none                 | none; silent          |
| Encoder OOM                                | 500 on /route            | server error handler | restart server        |
| Port collision at startup                  | crash on bind           | traceback            | manual: pick new port |
| Duplicate node ids in plan                 | PlanError               | 400 response         | user must fix plan    |
| Cycle in plan DAG                          | PlanError               | 400 response         | user must fix plan    |

## What we can't yet detect

- **A silently-drifting learned head.** If v1 has better holdout
  metrics than v0 but worse real-world quality (Goodhart's law),
  we won't notice. Ideas: canary shadow the previous head for X%
  of traffic, compare live outcomes.
- **A miscounted feedback record.** `_load_feedback` drops records
  missing required fields (`prompt_text`, `prompt_hash`,
  `observed_reduction_pct`) silently. A schema drift on the
  emitter side would look like "not enough training data" — a
  benign-sounding message.
- **A slow leak in the router's decision cache.** Capped at
  10 000 entries with FIFO eviction, but the queue's
  `_decision_stash` uses pop-on-read semantics — a plan that
  never terminates leaks one entry per submission. Not
  catastrophic; would need thousands of orphaned plans to
  matter.

## Log

### 2026-07-08 — Auto-promote gate saved us from a garbage head
Ran `train.py` twice on the same feedback set. Second run
produced identical metrics; the gate refused to promote
(demands strict improvement). That's the exact behavior we
want — the gate is the safety net for the whole learning loop.

### 2026-07-08 — Router service was down; queue worked
When the router service crashed on the 8090 collision, the
queue kept accepting plans and running them; `choose_arm`
returned "mixture" via the fallback path. Zero user impact
except that no feedback was logged during the outage. Exactly
what the design called for.

### 2026-07-07 — Every planner integration is best-effort
`_record_run_metric` and `_report_router_feedback` both wrap
their whole body in try/except and swallow all exceptions.
Necessary because metrics must never break a plan run, but
we've turned "metrics stopped working" into a silent failure
mode. Track: if `n=20` today and `n=20` tomorrow with plans
having run, something's broken.
