# Ethics and judgment calls

## Current state (2026-07-09)

Every routing decision is a value judgment. "Route this to the
cheap path" is quietly a decision about who bears which cost —
the user (worse answer), the operator (bigger bill), the ML
budget (more training runs), the reviewer LLM (more moderation
work).

The current system is optimising for *token savings*, gated by a
*quality floor*. That's a reasonable starting objective but a
partial one. Below is what we're implicitly assuming.

## Assumptions baked in

- **Frontier tokens are the scarce resource.** True today
  because the Claude bill is the largest single line item.
  Not true forever: at scale, latency or reviewer capacity
  may dominate.
- **Answer quality is measurable via a blind pairwise judge.**
  The reviewer LLM is treated as ground truth. It isn't —
  it's an opinionated judge with its own biases.
- **A tie is closer to "mixture is fine" than "mixture is
  worse".** Tie is labeled 1 (mixture not strictly worse) in
  training. Justified by cost-savings when quality is
  equivalent, but you could argue the other way when the
  cost of a bad answer >> cost of extra tokens.
- **Latency doesn't matter for correctness.** A slow correct
  answer scores the same as a fast correct one. In some
  domains (chatbot UX), latency IS quality.
- **Local SLM outputs are not sensitive.** We freely persist
  raw prompts and SLM responses. Fine in the current
  single-user prototype; a policy choice we should make
  explicit before that changes.

## Judgment calls we made explicitly

| Decision                                     | Rationale                                    |
| -------------------------------------------- | -------------------------------------------- |
| Router defaults to "solo" when unsure        | Safety over savings — a wrong "solo" costs some tokens; a wrong "mixture" can cost quality. |
| Quality classifier defaults to `P=0.5` prior | With fewer than 20 quality labels, don't route on it. |
| Auto-promote gate requires *strict* improvement | Prefer stability to churn. |
| ε-greedy exploration decays from 20 % to 5 % | Explore aggressively early, then trust learned decisions more. |
| Reviewer is a frontier LLM (same family)     | Cheap to build. Biased judge; accepted for now. |
| Feedback log is append-only, no deletions    | Immutability is worth more than storage. |
| No feature flags on the router               | Kill switch lives at the plugin layer instead. |

## Judgment calls we made implicitly

These weren't debated; they just happened. Worth surfacing so
we can push back if needed.

- **Language coverage.** Everything is English. The MiniLM
  encoder is multilingual-capable but the corpus isn't.
- **Domain coverage.** LC problems + curated cases + synthetic
  short prompts. Very unrepresentative of real user asks.
- **Fairness.** No per-user or per-domain policy differences.
  If some prompts are systematically under-served by the SLM
  path, we won't notice until someone complains.
- **Reversibility of learning.** Once a poison feedback record
  lands, it's in the JSONL. `train.py` re-reads the whole
  file every run. No "quarantine this record" mechanism.

## Where we should be uncomfortable

- **The router's confidence signal is uncalibrated.**
  `confidence = |P(quality) - 0.5| × 2` is a heuristic, not a
  probability. Presenting it to callers or users as
  "confidence" implies more than it delivers. Consider
  renaming or spelling out its limits in the API response.
- **We're training on synthetic examples we invented.** The
  seven "obvious solo" synthetic anchors in `bootstrap.py`
  are our best guess at what a short prompt looks like.
  Real short prompts may be structured differently. We
  should audit whether the model has learned "short = solo"
  vs "these specific short shapes = solo".
- **Reviewer verdict is the only ground truth.** If the
  reviewer LLM is systematically wrong (say, biased toward
  longer answers), our whole quality classifier learns the
  wrong function. There is no independent audit.

## Log

### 2026-07-08 — Quality classifier is confidently wrong
Fresh v1 head reports `quality_acc = 1.000` on holdout after
training on 46 records. Fully lying. See
[quality.md](quality.md) — this is where the ethics of
publishing a confident metric matter: the number is
technically correct on the data, and materially misleading
about the system.
