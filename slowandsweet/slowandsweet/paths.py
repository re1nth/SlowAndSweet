"""Filesystem layout for ~/.slowandsweet/ and repo discovery."""
from __future__ import annotations

from pathlib import Path

STATE_DIR = Path.home() / ".slowandsweet"
TOKEN_PATH = STATE_DIR / "token"
DB_PATH = STATE_DIR / "state.db"
# Hooks in the Claude Code plugin append one JSON object per line here.
CALLS_JSONL = STATE_DIR / "calls.jsonl"


def ensure_state_dir() -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR


def find_repo_root(start: Path | None = None) -> Path | None:
    """Walk up looking for a SlowAndSweet checkout.

    The marker is README.md + slm-deploy/ in the same directory — both must
    exist so we don't false-positive on an unrelated repo. We search from
    this file's install location (covers `pipx install -e .` from the repo)
    and then from CWD (covers running from the repo).
    """
    candidates: list[Path] = []
    here = Path(__file__).resolve()
    candidates.extend(here.parents)
    if start is not None:
        candidates.append(start.resolve())
        candidates.extend(start.resolve().parents)
    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.extend(cwd.parents)

    seen: set[Path] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if (c / "README.md").is_file() and (c / "slm-deploy").is_dir():
            return c
    return None


def slms_yaml_path() -> Path | None:
    root = find_repo_root()
    if root is None:
        return None
    p = root / "slm-deploy" / "slms.yaml"
    return p if p.is_file() else None
