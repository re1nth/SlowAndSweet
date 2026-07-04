# LeetCode frontier-token bench

Measures **tokens sent TO Claude Code** in two patterns for the same 10 LeetCode
problems. The SLM's own input/output cost is intentionally excluded — the point
is to see how much the frontier's ingestion shrinks when a local SLM pool does
the up-front distillation.

## Patterns

| Case | What the frontier sees                                             |
| ---- | ------------------------------------------------------------------ |
| 1    | Full problem statement + solve instruction                         |
| 2    | Algorithm sketch produced by an SLM DAG (`sketch → solve` on the queue), passed as the composer input. Statement itself is *not* re-sent. |

Both counted with `tiktoken cl100k_base` as a shared proxy for Claude token counts.

## Files

| File                    | Purpose                                                            |
| ----------------------- | ------------------------------------------------------------------ |
| `problems.json`         | The 10 scraped LeetCode problem statements + metadata              |
| `slm_results.json`      | Per-problem SLM DAG run snapshot (nodes with prompts + results)    |
| `claude_solutions.json` | Reference "what Claude would return" solutions used for output-token accounting in the earlier variant |
| `frontier_compare.py`   | Produces the SxS table (Case 1 vs Case 2) and `comparison.json`    |
| `persist_state.py`      | One-shot: refreshes `slm_results.json` from the live `slm-queue`   |
| `comparison.json`       | Latest run of `frontier_compare.py`                                |
| `run_slm.py`            | Original script that generated `slm_results.json` (submits to `slm-queue`) |
| `scrape.py`             | Original script that fetched `problems.json` via LeetCode GraphQL  |

## Reproduce

```sh
# 1. slm-queue must be running at :8080 (see slm-queue/README.md)
# 2. Refresh (or generate) the SLM run snapshot:
.venv/bin/python slm-bench/lc_frontier_bench/run_slm.py       # only if you want fresh runs
.venv/bin/python slm-bench/lc_frontier_bench/persist_state.py # only if the queue still has the old runs in memory

# 3. Rebuild the comparison table:
.venv/bin/python slm-bench/lc_frontier_bench/frontier_compare.py
```

## Latest result

| Problem                         | Diff   | Case 1 in | Case 2 in |     Δ | Reduction |
| ------------------------------- | ------ | --------: | --------: | ----: | --------: |
| Valid Parentheses               | Easy   |       242 |       168 |  + 74 |    30.6 % |
| Merge Two Sorted Lists          | Easy   |       237 |       163 |  + 74 |    31.2 % |
| Maximum Subarray                | Medium |       268 |       170 |  + 98 |    36.6 % |
| Climbing Stairs                 | Easy   |       207 |       191 |  + 16 |     7.7 % |
| Best Time to Buy and Sell Stock | Easy   |       289 |       189 | + 100 |    34.6 % |
| Contains Duplicate              | Easy   |       211 |       122 |  + 89 |    42.2 % |
| Product of Array Except Self    | Medium |       297 |       128 | + 169 |    56.9 % |
| Single Number                   | Easy   |       210 |       137 |  + 73 |    34.8 % |
| Move Zeroes                     | Easy   |       187 |       149 |  + 38 |    20.3 % |
| Reverse Linked List             | Easy   |       189 |       203 |  − 14 |   − 7.4 % |

**Total:** case 1 = 2 337, case 2 = 1 620 → **30.7 % fewer input tokens to Claude Code**.

## Caveats

- The sketch quality matters. `Reverse Linked List` regressed (−7.4 %) because
  the SLM sketch was verbose relative to the tiny problem statement — sometimes
  the distillation is longer than the source.
- This measures *input* only. The composer's *output* is essentially the same
  code either way, so the token asymmetry lives on the input side.
- Whether the SLM sketch is *sufficient* for Claude to produce correct code is
  a separate question this bench does not answer. To judge quality, plug these
  cases into `slm-experiments/` with an API key and let its blind reviewer
  score them.
