"""Score leaf outputs with a frontier judge to produce quality labels.

Reads leaf_feedback.jsonl, finds outcomes with output_text but no quality
label in leaf_quality.jsonl yet, calls the Anthropic API to score each
one against a rubric (accuracy 1-5), and appends the labels to
leaf_quality.jsonl keyed by (prompt_hash, model).

Labels are consumed by leaf_train.py to fit per-SLM LogisticRegression
classifiers that predict P(quality good | prompt, model). "Good" =
rubric score >= score_threshold in config.

Skips cleanly (prints why, exits 0) when:
  - ANTHROPIC_API_KEY is not set
  - anthropic SDK is not installed
  - no unlabeled outcomes exist

Designed to be chained from train.py before leaf_train, so the leaf-train
gate sees fresh labels on the same schedule as the retrain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from paths import ensure_data_dir, resolve_data_path  # noqa: E402


JUDGE_SYSTEM = """You are a strict but fair grader of AI assistant output.
Score how well the model's answer addresses the user's prompt on a 1-5 rubric:

  5 — fully correct, complete, and clearly expressed
  4 — mostly correct with minor issues (small omissions, awkward phrasing)
  3 — partially correct; noticeable gaps or errors
  2 — mostly wrong, misleading, or off-topic
  1 — completely wrong, empty, or refuses to attempt the task

Return exactly this JSON, no prose:

  {"score": <1-5 integer>, "reason": "<one short sentence>"}"""


JUDGE_USER_TEMPLATE = """PROMPT:
{prompt}

MODEL'S ANSWER:
{answer}

Score the answer on the 1-5 rubric and return JSON."""


@dataclass
class Unlabeled:
    prompt_hash: str
    prompt_text: str
    model: str
    output_text: str


def _iter_feedback(path: Path):
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _iter_quality(path: Path):
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _labeled_keys(quality_path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in _iter_quality(quality_path):
        ph = row.get("prompt_hash")
        m = row.get("model")
        if isinstance(ph, str) and isinstance(m, str):
            keys.add((ph, m))
    return keys


def _collect_unlabeled(
    feedback_path: Path, quality_path: Path, limit: int
) -> list[Unlabeled]:
    already = _labeled_keys(quality_path)
    seen: dict[tuple[str, str], Unlabeled] = {}
    for row in _iter_feedback(feedback_path):
        outcomes = row.get("outcomes") or {}
        prompt_text = row.get("prompt_text")
        if not isinstance(prompt_text, str) or not prompt_text:
            continue
        phash = row.get("prompt_hash") or (
            "sha256:" + hashlib.sha256(prompt_text.encode()).hexdigest()
        )
        for model, o in outcomes.items():
            if not isinstance(o, dict):
                continue
            if o.get("error"):
                continue
            text = o.get("output_text")
            if not isinstance(text, str) or not text.strip():
                continue
            key = (phash, model)
            if key in already or key in seen:
                continue
            seen[key] = Unlabeled(
                prompt_hash=phash,
                prompt_text=prompt_text,
                model=model,
                output_text=text,
            )
            if len(seen) >= limit:
                return list(seen.values())
    return list(seen.values())


_SCORE_RE = re.compile(r'"score"\s*:\s*(\d)')


def _parse_score(text: str) -> tuple[int | None, str | None]:
    try:
        obj = json.loads(text)
        s = obj.get("score")
        if isinstance(s, int) and 1 <= s <= 5:
            return s, obj.get("reason")
    except (json.JSONDecodeError, TypeError):
        pass
    # Salvage a bare score from a malformed reply.
    m = _SCORE_RE.search(text or "")
    if m:
        try:
            v = int(m.group(1))
            if 1 <= v <= 5:
                return v, None
        except ValueError:
            pass
    return None, None


def _call_anthropic_judge(
    client: Any, model: str, prompt: str, answer: str, max_tokens: int
) -> tuple[int | None, str | None, str]:
    """Return (score, reason, raw_response_text). Score is None on parse failure."""
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        system=JUDGE_SYSTEM,
        messages=[{
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(prompt=prompt, answer=answer),
        }],
    )
    text_parts = []
    for block in resp.content:
        t = getattr(block, "text", None)
        if isinstance(t, str):
            text_parts.append(t)
    text = "".join(text_parts).strip()
    score, reason = _parse_score(text)
    return score, reason, text


def _try_load_anthropic():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None, "ANTHROPIC_API_KEY not set"
    try:
        import anthropic  # type: ignore
    except ImportError:
        return None, "anthropic SDK not installed (pip install anthropic)"
    try:
        return anthropic.Anthropic(api_key=key), None
    except Exception as e:  # noqa: BLE001
        return None, f"anthropic client init failed: {type(e).__name__}: {e}"


def _append_quality(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def run_once(config: dict) -> dict:
    ensure_data_dir(config)
    paths_cfg = config.get("paths") or {}
    feedback_path = resolve_data_path(config, paths_cfg["leaf_feedback_log"])
    quality_path = resolve_data_path(
        config, paths_cfg.get("leaf_quality_log", "leaf_quality.jsonl")
    )
    reviewer_cfg = config.get("leaf_reviewer") or {}
    judge_model = str(reviewer_cfg.get("judge_model", "claude-sonnet-4-6"))
    threshold = int(reviewer_cfg.get("score_threshold", 4))
    batch_max = int(reviewer_cfg.get("batch_max", 200))
    max_tokens = int(reviewer_cfg.get("request_max_tokens", 300))

    if not feedback_path.exists():
        return {"skipped": True, "reason": "no_feedback_log", "n_scored": 0}

    unlabeled = _collect_unlabeled(feedback_path, quality_path, batch_max)
    if not unlabeled:
        return {"skipped": True, "reason": "no_unlabeled_rows", "n_scored": 0}

    client, err = _try_load_anthropic()
    if client is None:
        return {"skipped": True, "reason": err, "n_scored": 0, "n_pending": len(unlabeled)}

    n_scored = 0
    n_failed = 0
    for row in unlabeled:
        try:
            score, reason, _raw = _call_anthropic_judge(
                client,
                judge_model,
                row.prompt_text,
                row.output_text,
                max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            print(f"judge error for ({row.prompt_hash[:16]},{row.model}): {type(e).__name__}: {e}")
            n_failed += 1
            continue
        if score is None:
            n_failed += 1
            continue
        _append_quality(quality_path, {
            "timestamp": time.time(),
            "prompt_hash": row.prompt_hash,
            "model": row.model,
            "score": score,
            "quality_good": bool(score >= threshold),
            "reason": reason,
            "judge_model": judge_model,
            "threshold": threshold,
        })
        n_scored += 1

    return {
        "skipped": False,
        "reason": "ok",
        "n_scored": n_scored,
        "n_failed": n_failed,
        "n_total_batch": len(unlabeled),
        "judge_model": judge_model,
        "threshold": threshold,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_HERE / "config.yaml")
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    summary = run_once(config)
    if summary.get("skipped"):
        print(f"leaf reviewer skipped: {summary['reason']} "
              f"(n_pending={summary.get('n_pending', 0)})")
    else:
        print(
            f"leaf reviewer: scored {summary['n_scored']}/{summary['n_total_batch']} "
            f"(failed={summary.get('n_failed', 0)}) via {summary['judge_model']} "
            f"threshold>={summary['threshold']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
