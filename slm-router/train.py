"""Scheduled retraining loop for the router head.

Reads feedback.jsonl, embeds prompts (with an on-disk cache keyed by
prompt_hash), fits a fresh (cost, quality) head pair, computes train /
holdout / K-fold-CV metrics, runs the auto-promote gate, updates the
train-state file, and refreshes the static dashboard.

The N-new-records gate lets this be invoked on a short launchd interval
without retraining every time — the run no-ops unless enough new
feedback has landed since the last training attempt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import KFold

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from model import (  # noqa: E402
    Encoder,
    Head,
    HeadMetadata,
    head_pointer_read,
    head_pointer_write,
)
from paths import ensure_data_dir, resolve_data_path  # noqa: E402


@dataclass
class Record:
    prompt_hash: str
    prompt_text: str
    reduction_pct: float
    quality_label: int | None
    timestamp: float


def _load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text())


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


def _quality_label(verdict: Any) -> int | None:
    # DESIGN §4.2: A|B|tie|null. A = solo wins. Target is "mixture not worse".
    if verdict is None:
        return None
    v = str(verdict).strip().lower()
    if v in {"a", "solo"}:
        return 0
    if v in {"b", "mixture", "tie"}:
        return 1
    return None


def _load_feedback(path: Path) -> list[Record]:
    if not path.exists():
        return []
    seen: dict[str, Record] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            outcome = obj.get("outcome") or {}
            red = outcome.get("observed_reduction_pct")
            phash = obj.get("prompt_hash")
            ptext = obj.get("prompt_text") or obj.get("prompt")
            if red is None or not phash or not ptext:
                continue
            ts = _parse_ts(obj.get("timestamp"))
            rec = Record(
                prompt_hash=phash,
                prompt_text=ptext,
                reduction_pct=float(red),
                quality_label=_quality_label(outcome.get("quality_verdict")),
                timestamp=ts,
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
    # np.savez appends .npz if the target doesn't already end in .npz.
    written = tmp if tmp.exists() else Path(str(tmp) + ".npz")
    written.replace(path)


def _embed_records(records: list[Record], cache_path: Path) -> np.ndarray:
    cache = _load_cache(cache_path)
    live_hashes = {r.prompt_hash for r in records}
    cache = {k: v for k, v in cache.items() if k in live_hashes}

    missing = [r for r in records if r.prompt_hash not in cache]
    if missing:
        encoder = Encoder()
        vecs = encoder.encode_batch([r.prompt_text for r in missing])
        for r, v in zip(missing, vecs):
            cache[r.prompt_hash] = v.astype(np.float32)
        _save_cache(cache_path, cache)

    return np.stack([cache[r.prompt_hash] for r in records], axis=0)


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 2:
        return None
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return None
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _split(n: int, holdout: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_holdout = max(1, int(round(n * holdout))) if n >= 5 else 0
    if n_holdout == 0:
        return idx, np.array([], dtype=int)
    return idx[n_holdout:], idx[:n_holdout]


def _kfold_cv_mae(
    X: np.ndarray, y: np.ndarray, k: int, seed: int
) -> tuple[float | None, float | None]:
    """Return (mean, std) of MAE across K folds, or (None, None) if not runnable."""
    n = len(y)
    if k < 2 or n < k:
        return None, None
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    maes: list[float] = []
    for tr, te in kf.split(X):
        if len(tr) < 2 or len(te) < 1:
            continue
        m = LinearRegression().fit(X[tr], y[tr])
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


def _current_metrics(heads_dir: Path, pointer_path: Path) -> tuple[str | None, dict]:
    version = head_pointer_read(pointer_path)
    if not version:
        return None, {}
    head_path = heads_dir / f"{version}.joblib"
    if not head_path.exists():
        return version, {}
    try:
        h = Head.load(head_path)
    except Exception:
        return version, {}
    return version, {
        "holdout_mae_pp": h.metadata.holdout_mae_pp,
        "holdout_pearson_r": h.metadata.holdout_pearson_r,
        "holdout_quality_acc": h.metadata.holdout_quality_acc,
        "train_mae_pp": h.metadata.train_mae_pp,
        "cv_mae_pp_mean": h.metadata.cv_mae_pp_mean,
        "cv_mae_pp_std": h.metadata.cv_mae_pp_std,
        "n_train": h.metadata.n_train,
    }


def _better(cand: float | None, cur: float | None, direction: str) -> bool:
    if cand is None:
        return False
    if cur is None:
        return True
    if direction == "lower":
        return cand < cur
    return cand > cur


def _load_train_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_train_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def _regenerate_dashboard(config: dict, router_dir: Path) -> None:
    try:
        from dashboard import render_dashboard  # local import; optional
        render_dashboard(config)
    except Exception as e:
        # Dashboard is a nice-to-have; failure here must not fail training.
        print(f"dashboard regeneration failed: {type(e).__name__}: {e}")


def train_once(config: dict, router_dir: Path, force: bool) -> dict:
    ensure_data_dir(config, legacy_root=router_dir)

    paths = config["paths"]
    train_cfg = config["train"]

    feedback_path = resolve_data_path(config, paths["feedback_log"])
    metrics_path = resolve_data_path(config, paths["metrics_log"])
    heads_dir = resolve_data_path(config, paths["heads_dir"])
    pointer_path = resolve_data_path(config, paths["head_pointer"])
    cache_path = resolve_data_path(config, paths.get("emb_cache", ".emb_cache.npz"))
    state_path = resolve_data_path(config, paths.get("train_state", "train_state.json"))

    min_records = int(train_cfg["min_records"])
    min_quality = int(train_cfg["min_quality_labels"])
    min_new = int(train_cfg.get("min_new_records_since_last_train", 10))
    holdout_frac = float(train_cfg["holdout_fraction"])
    k_folds = int(train_cfg.get("k_folds", 5))
    seed = int(train_cfg["seed"])

    records = _load_feedback(feedback_path)
    n_total = len(records)
    state = _load_train_state(state_path)
    last_count = int(state.get("last_seen_record_count", 0))
    n_new = max(0, n_total - last_count)

    def _emit_skip(reason: str) -> dict:
        summary = {
            "timestamp": time.time(),
            "n_train": n_total,
            "n_new_records": n_new,
            "n_quality_train": sum(1 for r in records if r.quality_label is not None),
            "candidate_version": None,
            "promoted": False,
            "reason": reason,
            "candidate_metrics": None,
            "current_metrics": _current_metrics(heads_dir, pointer_path)[1],
        }
        _append_metrics(metrics_path, summary)
        # Record that we looked, so the gate advances even on no-op runs.
        state["last_seen_record_count"] = n_total
        state["last_run_timestamp"] = summary["timestamp"]
        state["last_run_reason"] = reason
        _save_train_state(state_path, state)
        _regenerate_dashboard(config, router_dir)
        return summary

    if n_total == 0:
        print("no feedback records found; skipping.")
        return _emit_skip("no_records")

    if n_total < min_records and not force:
        print(f"insufficient records ({n_total} < {min_records}); skipping.")
        return _emit_skip("insufficient_records")

    if not force and n_new < min_new:
        print(
            f"only {n_new} new records since last run "
            f"(< {min_new}); skipping."
        )
        return _emit_skip("insufficient_new_records")

    X = _embed_records(records, cache_path)
    y = np.array([r.reduction_pct for r in records], dtype=np.float32)
    train_idx, hold_idx = _split(n_total, holdout_frac, seed)

    cost = LinearRegression().fit(X[train_idx], y[train_idx])

    train_mae = float(mean_absolute_error(y[train_idx], cost.predict(X[train_idx])))
    cv_mean, cv_std = _kfold_cv_mae(X[train_idx], y[train_idx], k_folds, seed)

    mae = None
    pearson = None
    if len(hold_idx):
        preds = cost.predict(X[hold_idx])
        mae = float(mean_absolute_error(y[hold_idx], preds))
        pearson = _pearson(y[hold_idx], preds)

    quality = None
    quality_acc = None
    q_idx_all = [i for i, r in enumerate(records) if r.quality_label is not None]
    n_quality = len(q_idx_all)
    if n_quality >= min_quality:
        y_q = np.array([records[i].quality_label for i in q_idx_all], dtype=np.int64)
        q_train_mask = np.isin(q_idx_all, train_idx)
        q_hold_mask = np.isin(q_idx_all, hold_idx)
        q_train_idx = np.array(q_idx_all)[q_train_mask]
        q_hold_idx = np.array(q_idx_all)[q_hold_mask]
        y_q_train = y_q[q_train_mask]
        if len(q_train_idx) >= 2 and len(set(y_q_train.tolist())) >= 2:
            quality = LogisticRegression(max_iter=1000, C=0.5).fit(X[q_train_idx], y_q_train)
            if len(q_hold_idx) >= 1:
                y_q_hold = y_q[q_hold_mask]
                preds_q = quality.predict(X[q_hold_idx])
                quality_acc = float(accuracy_score(y_q_hold, preds_q))

    candidate_version = _next_version(heads_dir)
    cand_meta = HeadMetadata(
        version=candidate_version,
        n_train=n_total,
        n_quality_train=n_quality if quality is not None else 0,
        holdout_mae_pp=mae,
        holdout_pearson_r=pearson,
        holdout_quality_acc=quality_acc,
        train_mae_pp=train_mae,
        cv_mae_pp_mean=cv_mean,
        cv_mae_pp_std=cv_std,
        notes=f"retrained from {n_total} records",
    )
    head = Head(cost_regressor=cost, quality_classifier=quality, metadata=cand_meta)
    out_path = heads_dir / f"{candidate_version}.joblib"
    head.save(out_path)

    _, cur_metrics = _current_metrics(heads_dir, pointer_path)

    promoted = False
    reason = "gate_failed"
    if force:
        promoted = True
        reason = "force_promote"
    else:
        mae_ok = _better(mae, cur_metrics.get("holdout_mae_pp"), "lower")
        r_ok = _better(pearson, cur_metrics.get("holdout_pearson_r"), "higher")
        if quality is not None and cur_metrics.get("holdout_quality_acc") is not None:
            q_ok = _better(quality_acc, cur_metrics.get("holdout_quality_acc"), "higher")
        else:
            q_ok = True
        if mae_ok and r_ok and q_ok:
            promoted = True
            reason = "gate_passed"

    if promoted:
        head_pointer_write(pointer_path, candidate_version)

    summary = {
        "timestamp": time.time(),
        "n_train": n_total,
        "n_new_records": n_new,
        "n_quality_train": cand_meta.n_quality_train,
        "candidate_version": candidate_version,
        "promoted": promoted,
        "reason": reason,
        "candidate_metrics": {
            "holdout_mae_pp": mae,
            "holdout_pearson_r": pearson,
            "holdout_quality_acc": quality_acc,
            "train_mae_pp": train_mae,
            "cv_mae_pp_mean": cv_mean,
            "cv_mae_pp_std": cv_std,
        },
        "current_metrics": cur_metrics,
    }
    _append_metrics(metrics_path, summary)

    state["last_seen_record_count"] = n_total
    state["last_run_timestamp"] = summary["timestamp"]
    state["last_run_reason"] = reason
    state["last_candidate_version"] = candidate_version
    if promoted:
        state["last_promoted_version"] = candidate_version
        state["last_promoted_timestamp"] = summary["timestamp"]
    _save_train_state(state_path, state)

    print(
        f"trained {candidate_version}: n_train={n_total} n_new={n_new} "
        f"train_mae={_fmt(train_mae)} holdout_mae={_fmt(mae)} "
        f"cv_mae={_fmt(cv_mean)}±{_fmt(cv_std)} r={_fmt(pearson)} "
        f"qacc={_fmt(quality_acc)} promoted={'yes' if promoted else 'no'} "
        f"out={out_path}"
    )

    _regenerate_dashboard(config, router_dir)
    return summary


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _append_metrics(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(summary) + "\n")


def main() -> int:
    router_dir = Path(__file__).resolve().parent
    default_config = router_dir / "config.yaml"

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=default_config)
    ap.add_argument(
        "--force",
        action="store_true",
        help="bypass min_records and min_new_records gates; force-promote the candidate.",
    )
    ap.add_argument(
        "--force-promote",
        action="store_true",
        help="deprecated alias for --force",
    )
    args = ap.parse_args()

    config = _load_config(args.config)
    train_once(config, router_dir, args.force or args.force_promote)

    # M1: chain the leaf-router retrain on the same schedule. Independent
    # gate + state; failure here must not fail the arm-router run.
    try:
        import leaf_train  # local import; avoids cost when only train.py is imported
        leaf_train.train_once(config, router_dir, args.force or args.force_promote)
    except Exception as e:
        print(f"leaf retrain skipped: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
