# Experiment run `sxs-real-2`

- frontier: `anthropic:claude-sonnet-4-6`
- cases: 10
- wall time: 129.6s
- seed: 7

## Verdicts

- solo wins: **8** · mixture wins: **1** · tie: 1 · skipped: 0

## Aggregate usage

| Arm     | Frontier in | Frontier out | SLM out | Wall (s) |
| ------- | ----------: | -----------: | ------: | -------: |
| solo    | 1073 | 2327 | — | 56.1 |
| mixture | 2800 | 2293 | 1865 | 85.6 |
| reviewer | 9022 | 1728 | — | — |

## Per-case

| Case | Winner | Solo tokens (in/out) | Solo wall | Mix tokens (front in/out · SLM out) | Mix wall |
| ---- | ------ | -------------------: | --------: | -----------------------------------: | -------: |
| multi-doc-summary | solo | 227/218 | 5.5s | 193/239 · 119 | 9.7s |
| classify-reviews | solo | 141/110 | 3.7s | 221/167 · 140 | 6.4s |
| extract-entities | solo | 133/266 | 5.4s | 206/153 · 117 | 4.1s |
| code-explain | solo | 120/424 | 9.2s | 190/215 · 120 | 10.5s |
| math-multistep | solo | 110/164 | 3.4s | 272/291 · 136 | 8.1s |
| translate-batch | solo | 70/92 | 2.3s | 168/99 · 114 | 4.9s |
| brainstorm-ideas | solo | 49/254 | 6.9s | 569/260 · 453 | 12.7s |
| fact-recall | tie | 89/54 | 1.5s | 117/51 · 37 | 2.9s |
| plan-steps | mixture | 57/411 | 9.0s | 436/544 · 304 | 14.9s |
| compare-contrast | solo | 77/334 | 9.2s | 428/274 · 325 | 11.4s |

## Reviewer notes

- **multi-doc-summary** → winner **solo** (conf 0.65). scores solo={'accuracy': 5, 'completeness': 5, 'clarity': 5}, mixture={'accuracy': 5, 'completeness': 5, 'clarity': 4}. Both answers are accurate and complete, covering all three articles with no factual errors. Answer B is more concise and flows more naturally as a unified paragraph, while Answer A is slightly more verbose with a longer framing sentence and closing. B's structure is tighter and easier to read, giving it a slight edge on clarity.
- **classify-reviews** → winner **solo** (conf 0.97). scores solo={'accuracy': 5, 'completeness': 5, 'clarity': 5}, mixture={'accuracy': 2, 'completeness': 4, 'clarity': 3}. Answer A misclassifies review 5 as MIXED, incorrectly interpreting 'returned it after a week' as a positive, when returning a product due to a defect is clearly NEGATIVE. Answer B correctly labels all five reviews and provides concise, accurate justifications for each. Answer B is superior in accuracy, completeness, and clarity.
- **extract-entities** → winner **solo** (conf 0.95). scores solo={'accuracy': 5, 'completeness': 5, 'clarity': 5}, mixture={'accuracy': 2, 'completeness': 3, 'clarity': 4}. Answer B makes two clear errors: it sets organization to null for Bio 1 (Stanford Medical Center is explicitly named) and sets location to null for Bio 2 (Berlin is explicitly stated). Answer A correctly extracts all available named entities and uses null only where information is genuinely absent, with helpful notes explaining each decision.
- **code-explain** → winner **solo** (conf 0.72). scores solo={'accuracy': 5, 'completeness': 5, 'clarity': 5}, mixture={'accuracy': 5, 'completeness': 4, 'clarity': 5}. Both answers are accurate and well-written. Answer B more explicitly addresses each requested component (initial pass, hash map, final pass) as distinct labeled blocks, and includes a space complexity breakdown alongside time complexity. Answer A is slightly less complete in that it doesn't separately call out the 'final pass' (return None) block, though it is more concise and flows naturally as prose.
- **math-multistep** → winner **solo** (conf 0.60). scores solo={'accuracy': 5, 'completeness': 5, 'clarity': 5}, mixture={'accuracy': 5, 'completeness': 5, 'clarity': 4}. Both answers arrive at the correct answer of $303 with identical calculations. Answer B includes an unnecessary and confusing note about 'extracted facts' and a 'cost_per_novel: 75' that has no basis in the prompt, which slightly reduces clarity. Answer A is cleaner and more concise without any extraneous content.
- **translate-batch** → winner **solo** (conf 0.82). scores solo={'accuracy': 5, 'completeness': 5, 'clarity': 5}, mixture={'accuracy': 3, 'completeness': 5, 'clarity': 4}. Answer B provides more accurate and natural translations across all three languages. Answer A has issues: the Spanish uses 'de extraordinario calor' (awkward phrasing) and omits 'del año' (of the year), while the French uses 'extrêmement' (extremely) instead of 'inhabituellement' (unusually), changing the meaning. Answer B faithfully preserves 'unusually' in all three languages and uses idiomatic phrasing throughout.
- **brainstorm-ideas** → winner **solo** (conf 0.72). scores solo={'accuracy': 5, 'completeness': 5, 'clarity': 5}, mixture={'accuracy': 5, 'completeness': 5, 'clarity': 4}. Both answers are accurate and fully address the prompt with 5 distinct, actionable ideas. Answer B edges out A on clarity and specificity—its ideas are more concrete and immediately actionable (e.g., specific pricing, a named weekly event, a real-time Wi-Fi display), whereas A's ideas like 'quarterly speaker series' and 'monthly workshops' are less tightly focused on the weekday afternoon draw. B also avoids the unnecessary preamble about 'deduplicated ideas.'
- **fact-recall** → winner **tie** (conf 0.95). scores solo={'accuracy': 5, 'completeness': 5, 'clarity': 5}, mixture={'accuracy': 5, 'completeness': 5, 'clarity': 5}. Both answers are factually correct, follow the required format exactly, and are equally clear. The only difference is B adds the specific date 'July 20' in Q3, which is accurate but neither adds nor detracts meaningfully from the response quality.
- **plan-steps** → winner **mixture** (conf 0.72). scores solo={'accuracy': 4, 'completeness': 4, 'clarity': 5}, mixture={'accuracy': 5, 'completeness': 5, 'clarity': 5}. Answer A is more accurate for an early spring context, specifically recommending cool-season crops (broccoli, spinach, kale, peas) suited to that season, whereas Answer B suggests tomatoes, zucchini, and green beans which are warm-season crops not appropriate for early spring planting. Answer A also provides more actionable detail in each step (specific soil pH ranges, mulch depths, irrigation methods). Both answers are well-structured and clear, but A's seasonal accuracy and depth give it the edge.
- **compare-contrast** → winner **solo** (conf 0.92). scores solo={'accuracy': 5, 'completeness': 5, 'clarity': 5}, mixture={'accuracy': 2, 'completeness': 4, 'clarity': 5}. Answer A contains a significant factual error, stating e-scooters have a range of only 1–3 miles when typical electric scooters achieve 15–30 miles per charge. Answer B provides accurate range figures for both vehicles and more precise regulatory details about the federal three-tier e-bike classification system. Both answers are well-structured and clear, but B's factual accuracy and greater depth make it the clear winner.
