# Evolution

## Current state (2026-07-09)

The software is at the "M3-ish" point on the DESIGN.md roadmap:
foundation, server, training, and queue wiring all landed;
feedback loop closed end-to-end; no cron, no quality-signal
production traffic, no docker.

The natural next moves in rough priority order:

1. **Dogfood volume.** Nothing matters until we have 100+ real
   plan runs from actual users. Bench + experiments give us
   maybe 20. Everything below this line is theoretical until
   volume arrives.
2. **Wire the plugin's mixture arm.** The plugin currently only
   uses the queue's task path (single SLM call), not the plan
   path (mixture DAG). Until it does, no router decisions ever
   fire in production.
3. **Cron `train.py`.** Every hour or every N feedback records.
   M4 milestone.
4. **Reviewer shadow mode.** Nightly job that pulls a sample
   from live plans, runs solo counterfactual + reviewer,
   posts verdicts to `feedback.jsonl`. Closes the quality
   loop.
5. **Composer path in the queue.** So `composer_tokens_actual`
   stops being an approximation and starts being measured.
6. **Docker Compose for the three services.** Router + queue +
   Ollama in one `up`. Cuts first-run time from ~30 min to
   maybe 3.
7. **ONNX quantised MiniLM.** If `/route` p50 stays at 290 ms
   we can't afford to gate every submission on it.

## Rejected paths and why

- **RouteLLM drop-in.** Considered in the DESIGN doc §17. Its
  routers are trained on strong-vs-weak LLM comparisons, not
  solo-vs-mixture arm comparisons. Different label space.
  Worth revisiting when we have >1 k labels.
- **Fine-tune MiniLM end-to-end.** Requires the >1 k label
  regime we don't have. Frozen encoder + linear head is the
  right choice for now.
- **Multi-tenant router.** Deferred until we have a second
  user. Explicit non-goal.
- **Per-domain heads (code / math / prose).** Deferred until
  we see the failure mode.

## Milestone tracker

Rough alignment with DESIGN.md §16 milestones.

| M   | Scope                                     | Status                    | Delta since design |
| --- | ----------------------------------------- | ------------------------- | ------------------ |
| M0  | Design doc merged                         | done (2026-07-06)         | —                  |
| M1  | Foundation + server + heuristic           | done (2026-07-07)         | —                  |
| M2  | Bootstrap head + learned inference        | done (2026-07-07)         | —                  |
| M3  | Queue writes feedback + training loop     | done (2026-07-08)         | wired /plans handler that wasn't in the design |
| M4  | Auto-promote + metrics + nightly report   | partial                   | auto-promote done; no cron, no report yet |
| M5  | Quality classifier live                   | partial                   | classifier trains when data exists; needs data |
| M6  | Containerised deployment                  | not started               | —                  |

## The next 3 months (if we can dogfood)

- **Month 1**: dogfood + observability. Add
  `router_report.md` nightly generation. Wire plugin to plan
  path. Aim for 1000 real plan runs.
- **Month 2**: quality signal at scale. Shadow reviewer job.
  Fix the composer approximation with a real composer step.
  Retire the "always solo" default and start trusting
  learned decisions.
- **Month 3**: harden. Docker compose. Auth by default.
  Rotate feedback log at 100 MB. Add a canary head that
  runs 5 % of traffic for regression detection.

## Log

### 2026-07-08 — The wire that wasn't in the design
Discovered that `slm-queue/server.py::POST /plans` needs to
call `choose_arm` — the design assumed it but didn't name it
as a discrete change. Added in commit `8e28225`. First
"design gap surfaced by implementation" moment.

### 2026-07-07 — Foundation split cleanly across four agents
The four-agent fan-out (foundation, server, training, queue
integration) worked because the design was detailed enough to
serve as a contract. Model to remember: docs first, agents
second, interface friction reported back into docs.

### 2026-07-06 — Design doc written before code
610 lines of DESIGN.md before any file existed. Paid off:
implementation didn't wander. When the model docstring drifted
from the design, we updated the code, not the design.

### 2026-07-04 — Bench came before router
`slm-experiments/results/lc_frontier_bench/` measured token
savings *before* we built anything that would exploit them.
Every design decision downstream cites that bench. Order
mattered.
