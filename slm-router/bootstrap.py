"""Seed the router with heads/v0.joblib from existing bench + experiment artifacts.

Sources (design §7.7):
  1. LC frontier bench     - 10 (prompt, reduction%) rows; cost only.
  2. sxs-real-2 harness    - up to 10 (prompt, reduction%, quality) rows;
                             reduction% is computed from frontier input tokens,
                             prompt text is read from the case YAMLs.
  3. Synthetic anchors     - 7 short prompts labelled ~0-negative reduction;
                             copied verbatim from the router prototype.
  4. slowandsweet SQLite   - only cost signal, and only if `estimated_solo_tokens`
                             is populated AND the row carries prompt text.
                             Today the schema does not persist the prompt, so
                             we skip these rows and note the count.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LinearRegression, LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Encoder, Head, HeadMetadata, head_pointer_write  # noqa: E402


SYNTHETIC_ANCHORS: list[tuple[str, float]] = [
    ("What is 2 + 2?", -10.0),
    ("Return True if n is even.", -5.0),
    ("Reverse a string in Python.", -8.0),
    ("What year did WW2 end?", -12.0),
    ("Print hello world.", -15.0),
    ("Is 7 a prime number?", -5.0),
    ("Sort this list: [3, 1, 2].", -8.0),
]


@dataclass
class Row:
    prompt: str
    reduction_pct: float
    quality_label: int | None   # 1 = mixture not worse than solo; 0 = solo strictly better; None = no verdict
    source: str


def _verdict_to_label(winner: str | None) -> int | None:
    # sxs harness reports lowercase solo/mixture/tie in review.winner.
    # DESIGN §4.2 uses A|B|tie|null with A = solo wins. Classifier target is
    # "mixture is not worse", i.e. verdict in {B, tie} -> 1, A -> 0.
    if winner is None:
        return None
    w = winner.strip().lower()
    if w == "solo":
        return 0
    if w in {"mixture", "tie"}:
        return 1
    return None


def _load_lc_bench(repo_root: Path) -> list[Row]:
    bench = repo_root / "slm-experiments" / "results" / "lc_frontier_bench"
    problems = json.loads((bench / "problems.json").read_text())
    comparison = json.loads((bench / "comparison.json").read_text())
    red_by_slug = {r["slug"]: float(r["reduction_pct"]) for r in comparison["rows"]}
    rows: list[Row] = []
    for p in problems:
        slug = p["slug"]
        if slug not in red_by_slug:
            continue
        rows.append(Row(
            prompt=p["statement"],
            reduction_pct=red_by_slug[slug],
            quality_label=None,
            source="lc_bench",
        ))
    return rows


def _load_sxs(repo_root: Path) -> list[Row]:
    sxs_path = repo_root / "slm-experiments" / "results" / "sxs-real-2.json"
    cases_dir = repo_root / "slm-experiments" / "cases"
    if not sxs_path.exists() or not cases_dir.is_dir():
        return []
    sxs = json.loads(sxs_path.read_text())
    # Index yaml case files by id.
    prompt_by_id: dict[str, str] = {}
    for yml in sorted(cases_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(yml.read_text())
        except yaml.YAMLError:
            continue
        if isinstance(data, dict) and "id" in data and "solo_prompt" in data:
            prompt_by_id[str(data["id"])] = str(data["solo_prompt"]).strip()

    rows: list[Row] = []
    for case in sxs.get("cases", []):
        cid = case.get("case_id")
        prompt = prompt_by_id.get(cid)
        if not prompt:
            continue
        solo = case.get("solo", {}).get("frontier", {}).get("input_tokens")
        mix = case.get("mixture", {}).get("frontier", {}).get("input_tokens")
        if not solo or mix is None:
            continue
        reduction = (solo - mix) / solo * 100.0
        winner = (case.get("review") or {}).get("winner")
        rows.append(Row(
            prompt=prompt,
            reduction_pct=float(reduction),
            quality_label=_verdict_to_label(winner),
            source="sxs",
        ))
    return rows


def _load_synth() -> list[Row]:
    return [
        Row(prompt=t, reduction_pct=r, quality_label=None, source="synthetic")
        for t, r in SYNTHETIC_ANCHORS
    ]


def _load_sqlite(repo_root: Path) -> tuple[list[Row], int]:
    # The DB persists cost outcomes but no prompt text, so rows here can't be
    # embedded. We still count how many usable-if-we-had-prompts rows exist
    # so the summary line surfaces the gap.
    db_path = Path.home() / ".slowandsweet" / "state.db"
    if not db_path.exists():
        return [], 0
    try:
        conn = sqlite3.connect(str(db_path))
        # The schema has no prompt column, so any row with a cost signal is a
        # skipped-for-lack-of-prompt row. estimated_solo_tokens is the
        # counterfactual field the router would consume.
        cur = conn.execute(
            "SELECT COUNT(*) FROM calls WHERE estimated_solo_tokens IS NOT NULL"
        )
        (n,) = cur.fetchone()
        conn.close()
        return [], int(n)
    except sqlite3.Error:
        return [], 0


def bootstrap(repo_root: Path, out_path: Path, pointer_path: Path) -> dict:
    lc = _load_lc_bench(repo_root)
    sxs = _load_sxs(repo_root)
    synth = _load_synth()
    sqlite_rows, sqlite_skipped = _load_sqlite(repo_root)
    rows = lc + sxs + synth + sqlite_rows
    if not rows:
        raise RuntimeError("no bootstrap rows found; nothing to fit")

    encoder = Encoder()
    X = encoder.encode_batch([r.prompt for r in rows])
    y_red = np.array([r.reduction_pct for r in rows], dtype=np.float32)

    cost = LinearRegression().fit(X, y_red)

    q_rows = [(i, r) for i, r in enumerate(rows) if r.quality_label is not None]
    quality = None
    n_quality = len(q_rows)
    if n_quality >= 5 and len({r.quality_label for _, r in q_rows}) >= 2:
        Xq = X[[i for i, _ in q_rows]]
        yq = np.array([r.quality_label for _, r in q_rows], dtype=np.int64)
        quality = LogisticRegression(max_iter=1000, C=0.5).fit(Xq, yq)

    meta = HeadMetadata(
        version="v0",
        n_train=len(rows),
        n_quality_train=n_quality if quality is not None else 0,
        notes="cold-start bootstrap",
    )
    head = Head(cost_regressor=cost, quality_classifier=quality, metadata=meta)
    head.save(out_path)
    head_pointer_write(pointer_path, "v0")

    return {
        "n_lc": len(lc),
        "n_sxs": len(sxs),
        "n_synth": len(synth),
        "n_sqlite_used": len(sqlite_rows),
        "n_sqlite_skipped_no_prompt": sqlite_skipped,
        "n_train": len(rows),
        "n_quality_train": meta.n_quality_train,
        "out_path": str(out_path),
    }


def main() -> int:
    router_dir = Path(__file__).resolve().parent
    default_repo = router_dir.parent
    default_out = router_dir / "heads" / "v0.joblib"
    default_pointer = router_dir / "heads" / "HEAD"

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=default_repo)
    ap.add_argument("--out", type=Path, default=default_out)
    ap.add_argument("--head-pointer", type=Path, default=default_pointer)
    args = ap.parse_args()

    summary = bootstrap(args.repo_root, args.out, args.head_pointer)
    print(
        f"bootstrapped v0: lc={summary['n_lc']} sxs={summary['n_sxs']} "
        f"synth={summary['n_synth']} sqlite_used={summary['n_sqlite_used']} "
        f"sqlite_skipped_no_prompt={summary['n_sqlite_skipped_no_prompt']} | "
        f"n_train={summary['n_train']} n_quality_train={summary['n_quality_train']} | "
        f"out={summary['out_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
