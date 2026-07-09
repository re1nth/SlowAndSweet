# Quality

## Current state (2026-07-09)

Quality is the harder half of the problem. Cost is easy to measure
(count tokens); quality needs a judge and the judge is either
expensive (frontier API call) or unreliable (a smaller model).

Today the router's quality classifier is trained on **10 verdicts**
from the `sxs-real-2.json` snapshot plus zero live signal. The
`_post_router_feedback` wire in `slm-experiments/runner.py` will
grow that number every time someone runs the harness, but no
production traffic feeds it. The classifier's holdout accuracy
was 1.0 on the synthetic seed set — a distinctly false confidence
signal that will collapse the moment a real prompt comes in.

## Metrics we track

- **Reviewer winner per case** — solo | mixture | tie, from
  `slm-experiments/reviewer.py::review_nway`. Stored in
  results snapshots.
- **Solo win rate** — computed by
  `slm-experiments/report.py`. From `sxs-real-2`: solo won 8/10,
  mixture won 1 (plan-steps), tie 1 (fact-recall).
- **Quality classifier holdout accuracy** — per training run,
  stored in `slm-router/metrics.jsonl`.
- **Correspondence between predicted quality prob and observed
  outcomes** — not tracked. Once we have >50 (verdict, prediction)
  pairs, plotting a calibration curve is the first thing to look at.

## What we currently know

- **Solo wins the majority of curated cases** in the 10-case
  bench. Mixture wins where the SLMs *constrained* the composer
  toward the right answer (e.g., `plan-steps`); mixture loses
  where the SLMs *replaced* judgment the frontier should have
  done itself (classification, extraction, fact-sensitive
  translation).
- **The bootstrap v0 head is severely overfit.** LOOCV on the LC
  bench showed no classifier beats the "always decompose"
  constant baseline (80 % acc) because the class is 8:2. The
  policy layer defends against this via the quality floor: v0
  predicts quality_prob ≈ 0.19 for the merge-intervals prompt
  (way under the 0.7 floor), so the router routes to solo. Good
  behavior from a bad head, thanks to the safety layer.
- **The reviewer itself is a frontier call.** It's biased toward
  Anthropic's own family. A second opinion (e.g., an OpenAI
  judge) would strengthen the ground-truth. Deferred.

## Where quality can silently drop

- **Composer prompt approximation drift.** `planner._composer_prompt_approx`
  concatenates terminal-node results with a fixed wrapper. If a
  future plan puts non-terminal content in a place a real composer
  would ingest (e.g., a middle "reasoning" node), our
  `composer_tokens_actual` under-counts and the router
  under-values that plan. Detection: compare the predicted
  reduction to a hand-run once we have plans with different
  shapes.
- **Reviewer verdict noise.** `winner=tie` is treated as
  "mixture not worse" (label 1) in training. Justifiable but
  arguable — a tie could equally mean "mixture is not better."
  If the training data skews tie-heavy, the quality head
  drifts optimistic.
- **Selection bias in explore mode.** Once the router starts
  routing "obviously solo" prompts away from mixture, the
  training data over-represents borderline cases. ε-explore
  is the safety net but only if it's actually enabled in the
  queue (currently not — the queue always runs mixture
  regardless of decision, so exploration is implicit).

## Log

### 2026-07-08 — Overfit head produced predictable "solo" defaults
Every real /route call so far has come back "solo" because v0's
quality prob was under the floor. That's the safety layer doing
its job — but it also means the router is currently equivalent
to "always solo, log everything" until the head sees more
signal.

### 2026-07-04 — Baseline SxS shows solo wins on curated cases
Baseline run recorded in [sxs-real-2.md](../slm-experiments/results/sxs-real-2.md).
Solo 8, mixture 1, tie 1. Mixture pattern costs ~50 % more
frontier tokens than solo across this case set because the
composer prompt swallows every leaf result. Very case-dependent.
