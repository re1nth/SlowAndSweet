# Cost

## Current state (2026-07-09)

The whole project exists because Claude token costs matter more
than local compute costs, and SLM decomposition can shave a
meaningful fraction off frontier input tokens *when the prompt
shape suits it*. The LC frontier bench (2026-07-04) put a number
on it:

- 10 LeetCode problems
- Full statement → Claude: **2 337 tokens** input
- SLM-distilled sketch → Claude: **1 620 tokens** input
- **30.7 % reduction, 9/10 problems benefited**

But — that's a specific corpus (wordy problem statements). The
one plan we've actually run through the wired queue (merge-
intervals, 32-token description) showed a **negative** reduction:
composer=126, solo=32, `observed_reduction_pct = -184 %`. The
mixture pattern is a bad fit when the input is already small,
and the router should learn to route those to solo.

## Metrics we track

| Signal                                | Where                              |
| ------------------------------------- | ---------------------------------- |
| Solo tokens (estimated)               | feedback.jsonl `outcome.solo_tokens_estimated` |
| Composer tokens (actual)              | feedback.jsonl `outcome.composer_tokens_actual` |
| Observed reduction %                  | feedback.jsonl `outcome.observed_reduction_pct` |
| Estimated solo cost (output tokens)   | state.db `estimated_solo_tokens`   |
| Total tokens saved (cumulative)       | SQL rollup on state.db             |

Current cumulative: **0 tokens saved** across 20 plan runs — the
counterfactual estimate matches actual, meaning the SQL rollup
isn't using the new tiktoken-based composer count yet. Follow-up:
also record `composer_tokens_actual` in `state.db` so the same
number lives in one place.

## Break-even analysis

Every mixture-arm plan pays:
- **Extra Claude latency**: ~290 ms router hop + composer round trip
- **Local SLM output tokens**: measured in `state.db.slm_tokens_out`
  (currently avg 378/plan). On local hardware these are ~free
  (electricity), but they eat wall time.
- **Extra dev complexity**: worth more than the tokens when the
  savings are small.

Roughly: mixture is worth it when saved input tokens × Claude
input price > (extra wall time in ms × opportunity cost). At
current Sonnet input pricing (~$3/M) and a 30 % reduction on
a 2 000-token prompt, that's ~$0.002 saved per request. You need
volume before that adds up, and the router needs to be right
more often than wrong on which requests qualify.

## Log

### 2026-07-08 — Router itself has a cost
The router costs ~290 ms and one MiniLM inference per decision.
On the merge-intervals plan (32-token description) the router
spent more compute than it saved. This is why the confidence
floor + heuristic fallback matter: a router that runs on every
tiny request is a net negative. Fix path: skip the router
altogether for prompts under a length threshold (do the
heuristic inline in the caller for very-short prompts).

### 2026-07-04 — First real cost-savings measurement
LC bench totals: 2 337 → 1 620 (−30.7 %). See
[`slm-experiments/results/lc_frontier_bench/README.md`](../slm-experiments/results/lc_frontier_bench/README.md).
The single most cited fact in the project. Note: this is input-
side savings only. Output-side savings are usually smaller
because Claude has to produce roughly the same answer either way.
