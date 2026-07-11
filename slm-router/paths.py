"""Resolve router state paths against a per-user data directory.

All persistent router state (feedback.jsonl, metrics.jsonl, heads/,
embedding cache, train_state.json, dashboard.html) lives outside the
repo so retraining and inspection tools can share it. Resolution order:

  1. SLOWANDSWEET_HOME env var
  2. `data_dir` key in config.yaml
  3. ~/.slowandsweet/

Paths listed under `paths:` in config.yaml resolve relative to that
directory. Absolute paths pass through untouched.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = "~/.slowandsweet"

_LEGACY_ITEMS = (
    "metrics.jsonl",
    "feedback.jsonl",
    ".emb_cache.npz",
    "heads",
)


def resolve_data_dir(config: dict[str, Any]) -> Path:
    env = os.environ.get("SLOWANDSWEET_HOME")
    raw = env or config.get("data_dir") or DEFAULT_DATA_DIR
    return Path(raw).expanduser().resolve()


def resolve_data_path(config: dict[str, Any], rel: str) -> Path:
    p = Path(rel).expanduser()
    if p.is_absolute():
        return p.resolve()
    return resolve_data_dir(config) / p


def ensure_data_dir(config: dict[str, Any], legacy_root: Path | None = None) -> Path:
    """Create the data dir if missing and move any legacy in-repo state into it.

    Migration runs at most once per file: if the target already exists in
    the data dir, the legacy copy is left alone so nothing is clobbered.
    """
    data_dir = resolve_data_dir(config)
    data_dir.mkdir(parents=True, exist_ok=True)

    if legacy_root is None:
        return data_dir

    for name in _LEGACY_ITEMS:
        src = legacy_root / name
        dst = data_dir / name
        if not src.exists() or dst.exists():
            continue
        try:
            shutil.move(str(src), str(dst))
        except OSError:
            pass
    return data_dir
