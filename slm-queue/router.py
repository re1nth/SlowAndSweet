"""Prompt -> model name routing.

Picks one of the deployed model names for a given prompt. When
`SLM_ROUTER_URL` is set, consults the slm-router service's
`/route_leaf` endpoint (M0 learned leaf head) first; otherwise, and
on any failure, falls back to the rule-based heuristics inline. The
available list comes from slm-deploy/slms.yaml at server startup, so
the router only ever returns a model the dispatcher has workers for.

Also exports `choose_arm` for the arm-level (solo vs mixture) decision,
which consults the same slm-router service — see
`slm-router/DESIGN.md` §15.1.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path


_LEAF_ROUTE_TIMEOUT_S = 1.5
_LEAF_FEEDBACK_TIMEOUT_S = 2.0


_stash_lock = threading.Lock()
_decision_stash: dict[str, dict] = {}


def _first_match(available: list[str], prefix: str) -> str | None:
    for m in available:
        if m.startswith(prefix):
            return m
    return None


def _route_leaf_via_router_service(prompt: str, available: list[str]) -> str | None:
    """Call slm-router POST /route_leaf; return the chosen model name or None.

    Returns None (silent fall-through) when SLM_ROUTER_URL is unset, when
    the service can't be reached, when it returns a non-200, or when the
    chosen model isn't in the available list.
    """
    url = os.environ.get("SLM_ROUTER_URL")
    if not url:
        return None
    endpoint = url.rstrip("/") + "/route_leaf"
    payload = json.dumps({"prompt": prompt, "available": available}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("SLM_ROUTER_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=_LEAF_ROUTE_TIMEOUT_S) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    model = body.get("model") if isinstance(body, dict) else None
    if isinstance(model, str) and model in available:
        return model
    return None


def choose_model(prompt: str, available: list[str]) -> str:
    """Pick a model from `available` for `prompt`.

    Preference order:
      1. slm-router /route_leaf (learned head) when SLM_ROUTER_URL is set
         and the call succeeds.
      2. Keyword heuristics (below), used as fallback and cold-start:
         - code-ish prompts -> qwen2.5 (strong on code among these SLMs)
         - math/reasoning   -> llama3.2
         - long prompts     -> gemma2  (handles wider context comfortably)
         - default          -> smollm2 (fastest TTFT, cheapest)
    """
    routed = _route_leaf_via_router_service(prompt, available)
    if routed:
        return routed

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


def post_leaf_feedback(record: dict) -> None:
    """Fire-and-forget POST to slm-router /leaf_feedback.

    Called from a background thread; failures are swallowed. Skips silently
    when `SLM_ROUTER_URL` isn't set.
    """
    url = os.environ.get("SLM_ROUTER_URL")
    if not url:
        return
    endpoint = url.rstrip("/") + "/leaf_feedback"
    payload = json.dumps(record).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("SLM_ROUTER_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=_LEAF_FEEDBACK_TIMEOUT_S) as resp:
            resp.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
