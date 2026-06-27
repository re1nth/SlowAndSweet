# Experiment run `prism-smoke`

- frontier: `anthropic:claude-sonnet-4-6`
- cases: 1
- wall time: 12.6s
- seed: 7

## Verdicts

- solo: **1** · mixture: **0** · prism: **0** · tie: 0 · skipped: 0

## Aggregate usage

| Arm     | Frontier in | Frontier out | SLM out | Σ wall (s) |
| ------- | ----------: | -----------: | ------: | ---------: |
| solo    | 227 | 199 | — | 5.2 |
| mixture | 193 | 242 | 119 | 8.5 |
| prism   | 181 | 226 | 119 | 8.7 |
| reviewer | 1197 | 182 | — | — |

## Per-case

| Case | Winner | solo tokens (front in/out · slm out) | solo wall | mixture tokens (front in/out · slm out) | mixture wall | prism tokens (front in/out · slm out) | prism wall |
| ---- | ------ | ----: | ----: | ----: | ----: | ----: | ----: |
| multi-doc-summary | solo | 227/199 · 0 | 5.2s | 193/242 · 119 | 8.5s | 181/226 · 119 | 8.7s |

## Reviewer notes

- **multi-doc-summary** → winner **solo** (conf 0.55). scores solo={'accuracy': 5, 'completeness': 5, 'clarity': 5}, mixture={'accuracy': 5, 'completeness': 5, 'clarity': 4}, prism={'accuracy': 5, 'completeness': 5, 'clarity': 4}. All three answers are accurate and complete, covering all three articles faithfully. Answer B is slightly more concise and flows more naturally without the somewhat overwrought framing of A and C, making it the clearest and most readable of the three. The differences are minor, but B's tighter prose edges it ahead.
