# `consciousness/` — the software's persistent mind

This folder is a running architectural log. Not a spec, not a roadmap. A log —
what an engineer's brain does between the moment they finish a change and the
moment, months later, they need to remember *why*.

Each file tracks one axis of concern (performance, cost, reliability, …) with
three sections:

- **Current state** — a paragraph that gets rewritten in-place as reality
  changes. If you read only this you should know where we are.
- **Metrics we track** — what is measurable and where it lives on disk.
- **Log** — dated entries, latest at top. Observations, decisions, dead ends.
  Never delete an entry; correct it with a newer one that supersedes it.

## Why a folder, not a database

Because the folder gets committed. `git blame` on a log line tells you when
an assumption changed and who changed it. A database wouldn't.

## Cadence

Update the theme file that's most affected by a change *in the same commit
that makes the change*. If you can't figure out which theme is affected, add
an entry to `evolution.md` and let a future entry migrate it.

If a whole month has passed without a single entry, something is wrong — either
we've stopped changing anything, or we've stopped thinking about it. Both are
red flags.

## The themes

| File                     | What it holds                                                   |
| ------------------------ | --------------------------------------------------------------- |
| `ease_of_use.md`         | Install friction, docs, error messages, first-run experience    |
| `performance.md`         | E2E latency budgets, throughput, CPU / RAM / GPU footprint      |
| `scalability.md`         | Where each component saturates; the shape of "10× more traffic" |
| `reliability.md`         | Failure modes, degradation paths, recovery                      |
| `observability.md`       | What we can see, what we can't, what we're guessing about       |
| `security_privacy.md`    | Attack surface, data handling, auth, prompt leakage             |
| `cost.md`                | Token cost per pattern, infra cost, opportunity cost            |
| `quality.md`             | Output quality, decision quality, learned-head calibration      |
| `extensibility.md`       | How hard is it to add a new SLM, arm, router, plugin surface    |
| `testing.md`             | Coverage, gaps, the tests we wish existed                       |
| `technical_debt.md`      | Shortcuts we took and when they'll come due                     |
| `feedback_loops.md`      | How the system learns from itself — router, reviewer, metrics   |
| `ethics.md`              | Value judgments baked into the code that deserve to be surfaced |
| `evolution.md`           | Where the software is going next, and why                       |

Themes without a file yet are okay — add one when you have something to say.

## How to write a log entry

Bad:
> Fixed the router.

Good:
> **2026-07-08 — Router feedback loop closed end-to-end.** The `/plans` handler
> now calls `choose_arm`; the planner posts feedback after each run. First
> real record showed `observed_reduction_pct = -184%` on the merge-intervals
> plan — mixture cost more input than solo for that plan's short description.
> Real signal, first time. Files: `slm-queue/planner.py`, `slm-queue/server.py`.

Cite files. Cite numbers. Explain what you learned, not what you did — the
diff already says what you did.
