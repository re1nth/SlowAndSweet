# Tier-2 router classifier prototype

Goal: predict, from prompt text alone, whether SLM decomposition will save
the frontier enough input tokens to be worth the extra latency. This is the
gate that would sit *in front* of the SLM queue in a real deployment.

Model: `sentence-transformers/all-MiniLM-L6-v2` (22M params, ~90 MB, <10 ms
CPU per query) + a `LogisticRegression` / `LinearRegression` head. That's
tier 2 in the [routing ladder](../../../../slm-experiments/README.md) —
about 80× smaller than `smollm2:1.7b`, the queue's smallest SLM.

## Method

- Ground truth = `reduction_pct` per problem from `../comparison.json`
- Binary label = `reduction_pct > 15%` (worth-decomposing)
- Leave-one-out CV over the 10 real problems, plus 7 synthetic short prompts
  as constant negative anchors (train-only)
- Baselines:
  - `constant` — always predict "decompose" (majority class)
  - `length` — LinearRegression on tiktoken statement length alone
  - `hand` — 6 hand features (length, code fences, question marks, "Example",
    "Constraints", word count)
  - `minilm` — MiniLM embedding + linear head

## Result

| Method                      | MAE (pp) | Pearson r | Binary acc |
| --------------------------- | -------: | --------: | ---------: |
| Constant (predict mean)     |    13.13 |       n/a |     0.80   |
| Length only (1 feature)     |    12.69 |    +0.500 |     0.70   |
| Hand features (6 features)  |    12.74 |    +0.622 |     0.60   |
| MiniLM 384-dim (22M params) |    11.88 |    +0.514 |     0.50   |

Class balance: 8 decompose / 2 solo.

## What this tells us

1. **The signal is real but weak** at n=10. Every learned model has Pearson r ≈
   0.5-0.6 with the actual reduction — meaningfully positive, not zero. So
   there *is* something learnable in the prompt text about how much
   decomposition helps.
2. **MiniLM barely beats length-only** on regression MAE. Given the class
   imbalance, a 22M-param encoder is overkill on this dataset. A one-feature
   linear regression captures most of the "wordy prompts benefit more" signal.
3. **No classifier beats "always decompose" on binary accuracy.** With 8/10
   positives, a constant predictor scores 0.80. Classifier probabilities
   cluster around 0.5 (see `predictions.json`) because there's not enough
   negative data to sharpen the boundary. Any real router needs
   O(hundreds-thousands) of labeled negatives.
4. **The right output is continuous, not binary.** All three learned models
   *rank* prompts correctly enough (positive r) that a threshold like
   `predicted_reduction > 25%` would be defensible. Binary "yes/no" at n=10
   isn't.

## Practical recommendations

- **For this codebase, extend `slm-queue/router.py` first.** The Pearson r=0.62
  from hand features says a well-tuned keyword+length heuristic will
  capture most of the routing signal — no ML dependency needed.
- **Graduate to MiniLM once you have >200 labeled prompts.** With more
  training data the classifier can produce reliable probability scores,
  which lets you set the routing threshold based on your own latency /
  cost tradeoff (e.g., "route only if `P(worth) > 0.7` and `predicted
  savings > 30%`").
- **Don't fine-tune MiniLM end-to-end below ~1k labels.** The
  frozen-embedding + linear-head recipe is the standard pattern for
  low-data classifier bootstrapping; end-to-end fine-tuning is a
  higher-data regime.

## Files

| File                    | Purpose                                                  |
| ----------------------- | -------------------------------------------------------- |
| `router_prototype.py`   | MiniLM + LOOCV, writes `predictions.json` + `summary.json` |
| `baseline_compare.py`   | Sanity baselines (constant, length, hand), writes `baseline_compare.json` |
| `predictions.json`      | Per-problem MiniLM predictions (LOOCV)                   |
| `summary.json`          | Aggregate metrics for MiniLM                             |
| `baseline_compare.json` | Aggregate metrics for all four methods                   |

## Reproduce

```sh
# From repo root (assumes .venv exists and has sentence-transformers + scikit-learn):
.venv/bin/python slm-experiments/results/lc_frontier_bench/classifier/router_prototype.py
.venv/bin/python slm-experiments/results/lc_frontier_bench/classifier/baseline_compare.py
```

MiniLM downloads once from HF (~90 MB) into your local cache.
