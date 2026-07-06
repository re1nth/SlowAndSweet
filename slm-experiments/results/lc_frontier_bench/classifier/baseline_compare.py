"""Sanity baselines vs the MiniLM router.

Question: does the 22M-param embedding model actually add signal beyond
'just use the prompt length'? On a 10-example bench, we need to check.

Baselines evaluated with the same LOOCV protocol as router_prototype.py:
  1. Constant     — always predict 'decompose'. Trivial upper bound if the
                    class is imbalanced (as it is here: 8 of 10).
  2. Length only  — LinearRegression on tiktoken statement length alone.
  3. Length + code_marker + question_marker — hand features (3-dim).
  4. MiniLM (from router_prototype)             — for reference.

Reports MAE, Pearson r, and binary accuracy at threshold=15%.
"""
import json, pathlib
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
import tiktoken

ROOT = pathlib.Path(__file__).parent
BENCH = ROOT.parent
THRESHOLD_PCT = 15.0

problems = json.loads((BENCH / "problems.json").read_text())
comparison = json.loads((BENCH / "comparison.json").read_text())
red_by_slug = {r["slug"]: r["reduction_pct"] for r in comparison["rows"]}
enc = tiktoken.get_encoding("cl100k_base")


def hand_features(text: str) -> list[float]:
    return [
        len(enc.encode(text)),                       # length in tokens
        text.count("`"),                             # code fences
        text.count("?"),                             # question marks
        text.count("Example"),                       # examples in statement
        text.count("Constraints"),                   # constraints block?
        len(text.split()),                           # word count
    ]


records = []
for p in problems:
    records.append({
        "slug": p["slug"],
        "reduction": red_by_slug[p["slug"]],
        "text": p["statement"],
        "features": hand_features(p["statement"]),
    })

y = np.array([r["reduction"] for r in records])
y_cls = (y > THRESHOLD_PCT).astype(int)


def loocv_regression(X: np.ndarray) -> tuple[float, float]:
    """Return (MAE, pearson_r) via LOOCV LinearRegression."""
    preds = []
    for i in range(len(X)):
        train = [j for j in range(len(X)) if j != i]
        reg = LinearRegression().fit(X[train], y[train])
        preds.append(float(reg.predict(X[i:i + 1])[0]))
    preds = np.array(preds)
    mae = float(np.mean(np.abs(y - preds)))
    r = float(np.corrcoef(y, preds)[0, 1])
    return mae, r, preds


def loocv_binary(X: np.ndarray) -> tuple[float, list[int]]:
    """Return (accuracy, per-fold predictions) via LOOCV LogisticRegression."""
    preds = []
    for i in range(len(X)):
        train = [j for j in range(len(X)) if j != i]
        if len(set(y_cls[train])) < 2:
            preds.append(int(y_cls[train].mean() >= 0.5))
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(X[train], y_cls[train])
        preds.append(int(clf.predict(X[i:i + 1])[0]))
    return accuracy_score(y_cls, preds), preds


# 1. Constant baseline
constant_pred = np.full(len(y), y.mean())
constant_mae = float(np.mean(np.abs(y - constant_pred)))
constant_cls_pred = [int(y_cls.mean() >= 0.5)] * len(y_cls)
constant_acc = accuracy_score(y_cls, constant_cls_pred)

# 2. Length-only regression
X_len = np.array([[r["features"][0]] for r in records])
len_mae, len_r, len_preds = loocv_regression(X_len)
len_acc, _ = loocv_binary(X_len)

# 3. Hand features
X_hand = np.array([r["features"] for r in records])
hand_mae, hand_r, hand_preds = loocv_regression(X_hand)
hand_acc, _ = loocv_binary(X_hand)

# 4. MiniLM (read from predictions.json if router_prototype has been run)
predictions_json = ROOT / "predictions.json"
if predictions_json.exists():
    minilm = json.loads(predictions_json.read_text())
    minilm_by_slug = {p["label"].replace("lc:", ""): p for p in minilm if p["label"].startswith("lc:")}
    minilm_preds_reg = np.array([minilm_by_slug[r["slug"]]["predicted_reduction_pct"] for r in records])
    minilm_preds_cls = [minilm_by_slug[r["slug"]]["predicted_class"] for r in records]
    minilm_mae = float(np.mean(np.abs(y - minilm_preds_reg)))
    minilm_r = float(np.corrcoef(y, minilm_preds_reg)[0, 1])
    minilm_acc = accuracy_score(y_cls, minilm_preds_cls)
else:
    minilm_mae = minilm_r = minilm_acc = float("nan")

# --- Print --------------------------------------------------------------
print("Baseline comparison (LOOCV, n=10)")
print()
print("| Method                      | MAE (pp) | Pearson r | Binary acc |")
print("| --------------------------- | -------: | --------: | ---------: |")
print(f"| Constant (predict mean)     | {constant_mae:>7.2f} | {'n/a':>9} | {constant_acc:>9.2f} |")
print(f"| Length only (1 feature)     | {len_mae:>7.2f} | {len_r:>+9.3f} | {len_acc:>9.2f} |")
print(f"| Hand features (6 features)  | {hand_mae:>7.2f} | {hand_r:>+9.3f} | {hand_acc:>9.2f} |")
print(f"| MiniLM 384-dim (22M params) | {minilm_mae:>7.2f} | {minilm_r:>+9.3f} | {minilm_acc:>9.2f} |")

print()
print(f"Class balance: {int(y_cls.sum())} decompose / {len(y_cls) - int(y_cls.sum())} solo — "
      f"'always decompose' would score {y_cls.mean():.2f}.")

(ROOT / "baseline_compare.json").write_text(json.dumps({
    "n": len(records),
    "threshold_pct": THRESHOLD_PCT,
    "class_balance": {"decompose": int(y_cls.sum()), "solo": len(y_cls) - int(y_cls.sum())},
    "constant":  {"mae_pp": round(constant_mae, 2), "binary_acc": round(constant_acc, 3)},
    "length":    {"mae_pp": round(len_mae, 2),      "pearson_r": round(len_r, 3),  "binary_acc": round(len_acc, 3)},
    "hand":      {"mae_pp": round(hand_mae, 2),     "pearson_r": round(hand_r, 3), "binary_acc": round(hand_acc, 3)},
    "minilm":    {"mae_pp": round(minilm_mae, 2),   "pearson_r": round(minilm_r, 3), "binary_acc": round(minilm_acc, 3)},
}, indent=2))
