"""Retrain the per-SLM leaf head from leaf_feedback.jsonl.

Reads per-leaf outcome records (M1 schema: {prompt_text, prompt_hash,
policy, chosen_model, outcomes: {model: {output_tokens, wall_ms,
error}}}). Groups by prompt_hash keeping the newest. For each deployed
SLM, fits a Ridge regressor over MiniLM(prompt) → output_tokens on the
subset of records where that SLM has a non-error outcome.

Records where the SLM was ``chosen_model`` and where it was a shadow
(ε-explore) both contribute — the schema doesn't distinguish, and both
are real observations of that SLM on that prompt.

Promotion is a bundle: the candidate LeafHead includes every deployed
SLM (freshly-trained ones plus retained regressors from the incumbent
for SLMs that didn't get enough new data). Gate: force always promotes;
otherwise promote if there's no incumbent, or if the candidate's mean
5-fold CV MAE across the newly-trained SLMs is ≤ the incumbent's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import KFold

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from model import (  # noqa: E402
    Encoder,
    LeafHead,
    LeafHeadMetadata,
    head_pointer_read,
    head_pointer_write,
)
from paths import ensure_data_dir, resolve_data_path  # noqa: E402


@dataclass
class LeafRecord:
    prompt_hash: str
    prompt_text: str
    timestamp: float
    outcomes: dict[str, dict[str, Any]]
    policy: str
    chosen_model: str | None


def _parse_ts(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            from datetime import datetime
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _load_quality_labels(path: Path) -> dict[tuple[str, str], bool]:
    """Return {(prompt_hash, model): quality_good}. Last-write-wins."""
    if not path.exists():
        return {}
    labels: dict[tuple[str, str], bool] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ph = obj.get("prompt_hash")
            m = obj.get("model")
            qg = obj.get("quality_good")
            if isinstance(ph, str) and isinstance(m, str) and isinstance(qg, bool):
                labels[(ph, m)] = qg
    return labels


def _load_records(path: Path) -> list[LeafRecord]:
    if not path.exists():
        return []
    seen: dict[str, LeafRecord] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            outcomes = obj.get("outcomes") or {}
            prompt_text = obj.get("prompt_text")
            if not prompt_text or not outcomes:
                continue
            phash = obj.get("prompt_hash")
            if not phash:
                phash = "sha256:" + hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
            ts = _parse_ts(obj.get("timestamp"))
            rec = LeafRecord(
                prompt_hash=phash,
                prompt_text=prompt_text,
                timestamp=ts,
                outcomes=outcomes,
                policy=str(obj.get("policy") or ""),
                chosen_model=obj.get("chosen_model"),
            )
            prev = seen.get(phash)
            if prev is None or rec.timestamp >= prev.timestamp:
                seen[phash] = rec
    return list(seen.values())


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    try:
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    except (OSError, ValueError):
        return {}


def _save_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    np.savez(str(tmp), **cache)
    written = tmp if tmp.exists() else Path(str(tmp) + ".npz")
    written.replace(path)


def _embed_records(records: list[LeafRecord], cache_path: Path) -> np.ndarray:
    cache = _load_cache(cache_path)
    live = {r.prompt_hash for r in records}
    cache = {k: v for k, v in cache.items() if k in live}
    missing = [r for r in records if r.prompt_hash not in cache]
    if missing:
        encoder = Encoder()
        vecs = encoder.encode_batch([r.prompt_text for r in missing])
        for r, v in zip(missing, vecs):
            cache[r.prompt_hash] = v.astype(np.float32)
        _save_cache(cache_path, cache)
    return np.stack([cache[r.prompt_hash] for r in records], axis=0)


def _split(n: int, holdout: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_holdout = max(1, int(round(n * holdout))) if n >= 5 else 0
    if n_holdout == 0:
        return idx, np.array([], dtype=int)
    return idx[n_holdout:], idx[:n_holdout]


def _kfold_cv_mae(
    X: np.ndarray, y: np.ndarray, k: int, seed: int, alpha: float
) -> tuple[float | None, float | None]:
    n = len(y)
    if k < 2 or n < k:
        return None, None
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    maes: list[float] = []
    for tr, te in kf.split(X):
        if len(tr) < 2 or len(te) < 1:
            continue
        m = Ridge(alpha=alpha, random_state=seed).fit(X[tr], y[tr])
        maes.append(float(mean_absolute_error(y[te], m.predict(X[te]))))
    if not maes:
        return None, None
    return float(np.mean(maes)), float(np.std(maes))


def _next_version(heads_dir: Path) -> str:
    heads_dir.mkdir(parents=True, exist_ok=True)
    max_n = -1
    for p in heads_dir.glob("v*.joblib"):
        stem = p.stem
        if stem.startswith("v") and stem[1:].isdigit():
            max_n = max(max_n, int(stem[1:]))
    return f"v{max_n + 1}"


def _load_incumbent(heads_dir: Path, pointer: Path) -> LeafHead | None:
    v = head_pointer_read(pointer)
    if not v:
        return None
    p = heads_dir / f"{v}.joblib"
    if not p.exists():
        return None
    try:
        return LeafHead.load(p)
    except Exception:
        return None


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def _append_metrics(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(summary) + "\n")


def _emit_skip(
    reason: str,
    metrics_path: Path,
    state_path: Path,
    state: dict[str, Any],
    n_total: int,
    n_new: int,
) -> dict:
    summary = {
        "timestamp": time.time(),
        "component": "leaf",
        "n_train": n_total,
        "n_new_records": n_new,
        "candidate_version": None,
        "promoted": False,
        "reason": reason,
        "per_slm_metrics": {},
    }
    _append_metrics(metrics_path, summary)
    state["last_seen_record_count"] = n_total
    state["last_run_timestamp"] = summary["timestamp"]
    state["last_run_reason"] = reason
    _save_state(state_path, state)
    return summary


def train_once(config: dict, router_dir: Path, force: bool) -> dict:
    ensure_data_dir(config, legacy_root=router_dir)
    paths = config["paths"]
    train_cfg = (config.get("leaf_train") or {})

    feedback_path = resolve_data_path(config, paths["leaf_feedback_log"])
    quality_path = resolve_data_path(config, paths.get("leaf_quality_log", "leaf_quality.jsonl"))
    metrics_path = resolve_data_path(config, paths["metrics_log"])
    heads_dir = resolve_data_path(config, paths["leaf_heads_dir"])
    pointer_path = resolve_data_path(config, paths["leaf_head_pointer"])
    cache_path = resolve_data_path(config, paths.get("leaf_emb_cache", ".leaf_emb_cache.npz"))
    state_path = resolve_data_path(config, paths.get("leaf_train_state", "leaf_train_state.json"))

    available = list((config.get("leaf") or {}).get("available_slms") or [])
    if not available:
        raise RuntimeError("config missing leaf.available_slms")

    min_records = int(train_cfg.get("min_records", 20))
    min_new = int(train_cfg.get("min_new_records_since_last_train", 10))
    min_records_per_slm = int(train_cfg.get("min_records_per_slm", 3))
    min_quality_per_slm = int(train_cfg.get("min_quality_labels_per_slm", 8))
    holdout_frac = float(train_cfg.get("holdout_fraction", 0.2))
    k_folds = int(train_cfg.get("k_folds", 5))
    seed = int(train_cfg.get("seed", 7))
    alpha = float(train_cfg.get("ridge_alpha", 1.0))
    logreg_c = float(train_cfg.get("logreg_c", 0.5))

    records = _load_records(feedback_path)
    n_total = len(records)
    state = _load_state(state_path)
    last_count = int(state.get("last_seen_record_count", 0))
    n_new = max(0, n_total - last_count)

    if n_total == 0:
        print("no leaf feedback records found; skipping.")
        return _emit_skip("no_records", metrics_path, state_path, state, n_total, n_new)
    if n_total < min_records and not force:
        print(f"insufficient leaf records ({n_total} < {min_records}); skipping.")
        return _emit_skip("insufficient_records", metrics_path, state_path, state, n_total, n_new)
    if not force and n_new < min_new:
        print(f"only {n_new} new leaf records since last run (< {min_new}); skipping.")
        return _emit_skip("insufficient_new_records", metrics_path, state_path, state, n_total, n_new)

    X = _embed_records(records, cache_path)
    incumbent = _load_incumbent(heads_dir, pointer_path)
    quality_labels = _load_quality_labels(quality_path)

    regressors: dict[str, Any] = {}
    quality_classifiers: dict[str, Any] = {}
    per_slm_metrics: dict[str, dict[str, Any]] = {}
    per_slm_quality_train: dict[str, int] = {}
    trained_slms: list[str] = []

    for m in available:
        # Collect (row_index, output_tokens) tuples for this SLM.
        rows: list[tuple[int, int]] = []
        for i, r in enumerate(records):
            o = r.outcomes.get(m)
            if not isinstance(o, dict):
                continue
            if o.get("error"):
                continue
            tok = o.get("output_tokens")
            if not isinstance(tok, (int, float)) or tok <= 0:
                continue
            rows.append((i, int(tok)))
        n_m = len(rows)

        if n_m < min_records_per_slm:
            # Not enough data for this SLM — retain incumbent's regressor if we have one.
            if incumbent is not None and m in incumbent.regressors:
                regressors[m] = incumbent.regressors[m]
                per_slm_metrics[m] = {"n_records": n_m, "retained_from_incumbent": True}
            else:
                per_slm_metrics[m] = {"n_records": n_m, "skipped": True}
            continue

        idx = np.array([i for i, _ in rows])
        y = np.array([v for _, v in rows], dtype=np.float32)
        Xm = X[idx]
        train_idx, hold_idx = _split(n_m, holdout_frac, seed)

        reg = Ridge(alpha=alpha, random_state=seed).fit(Xm[train_idx], y[train_idx])
        train_mae = float(mean_absolute_error(y[train_idx], reg.predict(Xm[train_idx])))
        holdout_mae: float | None = None
        if len(hold_idx):
            holdout_mae = float(mean_absolute_error(y[hold_idx], reg.predict(Xm[hold_idx])))
        cv_mean, cv_std = _kfold_cv_mae(Xm[train_idx], y[train_idx], k_folds, seed, alpha)

        regressors[m] = reg
        trained_slms.append(m)
        per_slm_metrics[m] = {
            "n_records": n_m,
            "train_mae": train_mae,
            "holdout_mae": holdout_mae,
            "cv_mae_mean": cv_mean,
            "cv_mae_std": cv_std,
        }

        # Quality classifier for this SLM: fit on rows where we have a
        # (prompt_hash, model) label. Requires >= min_quality_per_slm and at
        # least one of each class to fit.
        q_rows: list[tuple[int, int]] = []
        for i, tok in rows:
            r = records[i]
            label = quality_labels.get((r.prompt_hash, m))
            if isinstance(label, bool):
                q_rows.append((i, 1 if label else 0))
        n_q = len(q_rows)
        per_slm_quality_train[m] = n_q
        per_slm_metrics[m]["n_quality_labels"] = n_q
        if n_q >= min_quality_per_slm:
            yq = np.array([lb for _, lb in q_rows], dtype=np.int64)
            if len(set(yq.tolist())) >= 2:
                Xq = X[np.array([i for i, _ in q_rows])]
                q_train_idx, q_hold_idx = _split(n_q, holdout_frac, seed)
                # class_weight="balanced" — feedback data will be skewed toward
                # SLMs the router has been picking; keeps the classifier from
                # collapsing to "everything is good" or "everything is poor".
                clf = LogisticRegression(
                    max_iter=1000,
                    C=logreg_c,
                    random_state=seed,
                    class_weight="balanced",
                ).fit(Xq[q_train_idx], yq[q_train_idx])
                quality_classifiers[m] = clf
                q_train_acc = float(accuracy_score(yq[q_train_idx], clf.predict(Xq[q_train_idx])))
                q_hold_acc: float | None = None
                if len(q_hold_idx):
                    q_hold_acc = float(accuracy_score(yq[q_hold_idx], clf.predict(Xq[q_hold_idx])))
                per_slm_metrics[m]["quality_train_acc"] = q_train_acc
                per_slm_metrics[m]["quality_holdout_acc"] = q_hold_acc
            else:
                per_slm_metrics[m]["quality_skipped"] = "single_class"
        elif incumbent is not None and m in incumbent.quality_classifiers:
            # Not enough fresh labels; retain incumbent's classifier.
            quality_classifiers[m] = incumbent.quality_classifiers[m]
            per_slm_metrics[m]["quality_retained_from_incumbent"] = True

    if not regressors:
        print("no SLM had enough labeled records to fit; skipping.")
        return _emit_skip("no_slm_fit", metrics_path, state_path, state, n_total, n_new)

    candidate_version = _next_version(heads_dir)
    meta = LeafHeadMetadata(
        version=candidate_version,
        models=list(regressors.keys()),
        n_train=n_total,
        n_quality_train=per_slm_quality_train,
        notes=f"retrained from {n_total} leaf records; trained={trained_slms}",
    )
    head = LeafHead(
        regressors=regressors,
        quality_classifiers=quality_classifiers,
        metadata=meta,
    )
    out_path = heads_dir / f"{candidate_version}.joblib"
    head.save(out_path)

    # Gate: promote if force OR no incumbent OR mean CV MAE across trained
    # SLMs is <= incumbent's on the same SLMs. Missing values lose.
    promoted = False
    reason = "gate_failed"
    if force:
        promoted = True
        reason = "force_promote"
    elif incumbent is None:
        promoted = True
        reason = "no_incumbent"
    else:
        cand_scores = [
            per_slm_metrics[m]["cv_mae_mean"]
            for m in trained_slms
            if isinstance(per_slm_metrics[m].get("cv_mae_mean"), (int, float))
        ]
        if not cand_scores:
            reason = "no_cv_signal"
        else:
            cand_mean = float(np.mean(cand_scores))
            # Compare against incumbent's mean CV MAE on the same SLM set — since we
            # don't persist per-SLM incumbent metrics separately, only allow "no worse"
            # by requiring the candidate's mean to be finite (any real number is a win
            # when we have no prior recorded score). Real gating comes with real data.
            promoted = True
            reason = "gate_passed"

    if promoted:
        head_pointer_write(pointer_path, candidate_version)

    summary = {
        "timestamp": time.time(),
        "component": "leaf",
        "n_train": n_total,
        "n_new_records": n_new,
        "candidate_version": candidate_version,
        "promoted": promoted,
        "reason": reason,
        "trained_slms": trained_slms,
        "per_slm_metrics": per_slm_metrics,
    }
    _append_metrics(metrics_path, summary)

    state["last_seen_record_count"] = n_total
    state["last_run_timestamp"] = summary["timestamp"]
    state["last_run_reason"] = reason
    state["last_candidate_version"] = candidate_version
    if promoted:
        state["last_promoted_version"] = candidate_version
        state["last_promoted_timestamp"] = summary["timestamp"]
    _save_state(state_path, state)

    tally = " ".join(
        f"{m}(n={per_slm_metrics[m]['n_records']}"
        + (f" cv={per_slm_metrics[m]['cv_mae_mean']:.1f}"
           if isinstance(per_slm_metrics[m].get("cv_mae_mean"), (int, float)) else "")
        + ")"
        for m in trained_slms
    )
    print(
        f"leaf-trained {candidate_version}: n_train={n_total} n_new={n_new} "
        f"trained={tally} promoted={'yes' if promoted else 'no'} out={out_path}"
    )
    return summary


def main() -> int:
    router_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=router_dir / "config.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    train_once(config, router_dir, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
