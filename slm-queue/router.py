"""Prompt -> model name routing.

Picks one of the deployed model names for a given prompt using simple
rule-based heuristics. The available list comes from slm-deploy/slms.yaml
at server startup, so the router only ever returns a model the dispatcher
has workers for.

Also exports `choose_arm` for the arm-level (solo vs mixture) decision,
which optionally consults the `slm-router` sidecar service — see
`slm-router/DESIGN.md` §15.1.
"""
from __future__ import annotations

import hashlib
import os
import sys
import threading
from pathlib import Path


_stash_lock = threading.Lock()
_decision_stash: dict[str, dict] = {}


def _first_match(available: list[str], prefix: str) -> str | None:
    for m in available:
        if m.startswith(prefix):
            return m
    return None


def choose_model(prompt: str, available: list[str]) -> str:
    """Pick a model from `available` for `prompt`.

    Heuristics, in order:
      - code-ish prompts -> qwen2.5 (strong on code among these SLMs)
      - math/reasoning   -> llama3.2
      - long prompts     -> gemma2  (handles wider context comfortably)
      - default          -> smollm2 (fastest TTFT, cheapest)
    """
    p = prompt.lower()

    code_markers = ("```", "def ", "function", "python", "javascript",
                    "typescript", "sql", "code", "regex", "bash ", "shell")
    if any(k in p for k in code_markers):
        m = _first_match(available, "qwen2.5")
        if m:
            return m

    math_markers = ("calculate", "compute", "solve", "equation", "prove",
                    "derivative", "integral", " sum ", "math ")
    if any(k in p for k in math_markers):
        m = _first_match(available, "llama3.2")
        if m:
            return m

    if len(prompt) > 500:
        m = _first_match(available, "gemma2")
        if m:
            return m

    m = _first_match(available, "smollm2")
    if m:
        return m
    return available[0]


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _load_explore_module():
    try:
        from slm_router import explore  # type: ignore
        return explore
    except ImportError:
        pass
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "slm-router"
    if (candidate / "explore.py").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        try:
            import explore  # type: ignore
            return explore
        except Exception:  # noqa: BLE001
            return None
    return None


def _stash(prompt_hash: str, decision: dict) -> None:
    with _stash_lock:
        _decision_stash[prompt_hash] = decision


def choose_arm(prompt: str) -> str:
    """Return "mixture" or "solo" for this prompt.

    Consults the `slm-router` service when `SLM_ROUTER_URL` is set; falls
    back to the conservative default ("mixture", preserving pre-router
    behaviour) when the env var is unset or the service call fails.
    """
    url = os.environ.get("SLM_ROUTER_URL")
    if not url:
        return "mixture"

    explore = _load_explore_module()
    if explore is None:
        return "mixture"

    try:
        decision = explore.route_prompt(prompt)
    except Exception:  # noqa: BLE001
        return "mixture"
    if not decision:
        return "mixture"

    _stash(_prompt_hash(prompt), decision)

    arm = decision.get("decision")
    if arm in ("mixture", "explore"):
        return "mixture"
    return "solo"


def get_stashed_decision(prompt: str) -> dict | None:
    """Pop and return the stashed router Decision for `prompt`, if any."""
    key = _prompt_hash(prompt)
    with _stash_lock:
        return _decision_stash.pop(key, None)
