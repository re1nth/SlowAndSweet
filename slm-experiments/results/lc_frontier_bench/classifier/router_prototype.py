"""Tier-2 router classifier prototype: MiniLM embedding + linear head.

Question: from *only* the raw prompt text, can we predict whether SLM
decomposition will meaningfully reduce Claude Code's input tokens?

Signal (from ../comparison.json): reduction_pct per problem.

Setup:
- Embed each problem's statement with sentence-transformers/all-MiniLM-L6-v2 (22M).
- Two heads in parallel:
    * LinearRegression  -> predicts continuous reduction%
    * LogisticRegression -> predicts binary (reduction > threshold)
- Leave-one-out cross-validation (n=10 is tiny; LOOCV is the honest choice).
- To ground the classifier on the "obvious solo" extreme, add a small set of
  synthetic simple prompts labeled with reduction ~= 0. Without them, the
  model has no negative anchors to learn from (bench is 9-of-10 positive).

Outputs:
- predictions.json    per-problem prediction table
- summary.json        LOOCV aggregate metrics
- prints markdown to stdout
"""
import json, pathlib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score

ROOT = pathlib.Path(__file__).parent
BENCH = ROOT.parent
THRESHOLD_PCT = 15.0  # "meaningful" savings cutoff for the binary label

# --- Real bench data --------------------------------------------------------
problems = json.loads((BENCH / "problems.json").read_text())
comparison = json.loads((BENCH / "comparison.json").read_text())
red_by_slug = {r["slug"]: r["reduction_pct"] for r in comparison["rows"]}

real_examples = [
    {
        "text": p["statement"],
        "reduction": red_by_slug[p["slug"]],
        "label": "lc:" + p["slug"],
        "source": "bench",
    }
    for p in problems
]

# --- Synthetic negative anchors --------------------------------------------
# Short one-liners where SLM decomposition has nothing to distill.
# We estimate reduction ~= 0 to -20% (the sketch would be at least as long).
synthetic_anchors = [
    ("What is 2 + 2?", -10.0),
    ("Return True if n is even.", -5.0),
    ("Reverse a string in Python.", -8.0),
    ("What year did WW2 end?", -12.0),
    ("Print hello world.", -15.0),
    ("Is 7 a prime number?", -5.0),
    ("Sort this list: [3, 1, 2].", -8.0),
]

synth_examples = [
    {"text": t, "reduction": r, "label": f"synth:{i}", "source": "synthetic"}
    for i, (t, r) in enumerate(synthetic_anchors)
]

all_examples = real_examples + synth_examples
print(f"Loaded {len(real_examples)} bench + {len(synth_examples)} synthetic examples")

# --- Embed ------------------------------------------------------------------
print("Loading MiniLM (all-MiniLM-L6-v2, 22M params)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
texts = [ex["text"] for ex in all_examples]
X = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
y_reg = np.array([ex["reduction"] for ex in all_examples])
y_cls = (y_reg > THRESHOLD_PCT).astype(int)

# --- LOOCV over the 10 real bench examples ---------------------------------
real_idx = [i for i, ex in enumerate(all_examples) if ex["source"] == "bench"]

preds = []
for held in real_idx:
    train_idx = [i for i in range(len(all_examples)) if i != held]
    X_train, y_train_reg, y_train_cls = X[train_idx], y_reg[train_idx], y_cls[train_idx]

    reg = LinearRegression().fit(X_train, y_train_reg)
    pred_reduction = float(reg.predict(X[held:held + 1])[0])

    if len(set(y_train_cls)) < 2:
        pred_cls, pred_prob = None, None
    else:
        clf = LogisticRegression(max_iter=1000, C=0.5).fit(X_train, y_train_cls)
        pred_prob = float(clf.predict_proba(X[held:held + 1])[0, 1])
        pred_cls = int(pred_prob >= 0.5)

    ex = all_examples[held]
    preds.append({
        "label": ex["label"],
        "actual_reduction_pct": ex["reduction"],
        "actual_class": int(y_cls[held]),
        "predicted_reduction_pct": round(pred_reduction, 1),
        "predicted_class": pred_cls,
        "predicted_prob_worth_decomposing": round(pred_prob, 3) if pred_prob is not None else None,
        "reg_correct_direction": (ex["reduction"] > THRESHOLD_PCT) == (pred_reduction > THRESHOLD_PCT),
        "cls_correct": pred_cls == int(y_cls[held]) if pred_cls is not None else None,
    })

# --- Metrics ---------------------------------------------------------------
actuals = np.array([p["actual_reduction_pct"] for p in preds])
pred_regs = np.array([p["predicted_reduction_pct"] for p in preds])
mae = float(np.mean(np.abs(actuals - pred_regs)))
corr = float(np.corrcoef(actuals, pred_regs)[0, 1]) if len(actuals) > 1 else float("nan")

cls_actuals = [p["actual_class"] for p in preds if p["cls_correct"] is not None]
cls_preds = [p["predicted_class"] for p in preds if p["cls_correct"] is not None]
cls_acc = accuracy_score(cls_actuals, cls_preds) if cls_actuals else float("nan")

# --- Print + persist -------------------------------------------------------
print()
print("| Problem                            | Actual Δ% | Pred Δ% | P(worth) | Actual cls | Pred cls | ✓ |")
print("| ---------------------------------- | --------: | ------: | -------: | ---------- | -------- | - |")
for p in preds:
    ok = "✓" if p["cls_correct"] else "✗"
    slug_short = p["label"].replace("lc:", "")[:34]
    ac = "decomp" if p["actual_class"] else "solo"
    pc = "decomp" if p["predicted_class"] else "solo"
    print(f"| {slug_short:34} | {p['actual_reduction_pct']:>+7.1f}% | "
          f"{p['predicted_reduction_pct']:>+5.1f}% | {p['predicted_prob_worth_decomposing']:>6.3f}   | "
          f"{ac:10} | {pc:8} | {ok} |")

print()
print(f"Regression   : MAE = {mae:.1f} pp, Pearson r = {corr:.3f}")
print(f"Classification: accuracy = {cls_acc:.2f}  (n={len(cls_actuals)}, threshold = {THRESHOLD_PCT}% reduction)")

(ROOT / "predictions.json").write_text(json.dumps(preds, indent=2))
(ROOT / "summary.json").write_text(json.dumps({
    "n_real": len(real_examples),
    "n_synthetic_anchors": len(synth_examples),
    "threshold_pct": THRESHOLD_PCT,
    "regression_mae_pp": round(mae, 2),
    "regression_pearson_r": round(corr, 3),
    "classification_accuracy": round(cls_acc, 3),
    "model": "sentence-transformers/all-MiniLM-L6-v2 (22M params)",
    "head": "sklearn LinearRegression + LogisticRegression",
    "cv": "leave-one-out on bench examples; synthetic anchors always in train",
}, indent=2))
