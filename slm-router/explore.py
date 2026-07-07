"""Client-side helpers for talking to the slm-router service.

Imported by `slm-queue` (and other callers) to consult `/route` and post
`/feedback`. See DESIGN.md sections 7.6 and 8. Every function degrades
gracefully when the router service is unreachable — callers must be able
to fall back to a default policy.
"""
from __future__ import annotations

import json
import os
import random
import sys
import urllib.error
import urllib.request


_DEFAULT_URL = "http://127.0.0.1:8092"


def _base_url(override: str | None = None) -> str | None:
    if override:
        return override.rstrip("/")
    env = os.environ.get("SLM_ROUTER_URL")
    if env:
        return env.rstrip("/")
    return None


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    tok = os.environ.get("SLM_ROUTER_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _warn(msg: str) -> None:
    try:
        sys.stderr.write(f"[slm-router.explore] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _post(url: str, body: dict, timeout_s: float) -> dict | None:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200)
            if status != 200:
                _warn(f"POST {url} returned status {status}")
                return None
            raw = resp.read()
    except urllib.error.HTTPError as e:
        _warn(f"POST {url} HTTPError {e.code}")
        return None
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        _warn(f"POST {url} failed: {type(e).__name__}: {e}")
        return None
    except Exception as e:  # noqa: BLE001
        _warn(f"POST {url} unexpected error: {type(e).__name__}: {e}")
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        _warn(f"POST {url} bad response body: {e}")
        return None


def route_prompt(
    prompt: str,
    timeout_s: float = 0.5,
    url: str | None = None,
) -> dict | None:
    base = _base_url(url) or (_DEFAULT_URL if url == "" else None)
    if base is None:
        # SLM_ROUTER_URL unset and caller passed no URL: treat as down.
        return None
    resp = _post(f"{base}/route", {"prompt": prompt}, timeout_s)
    if resp is None:
        return None
    if not isinstance(resp, dict) or "decision" not in resp:
        _warn("route response missing 'decision' field")
        return None
    return resp


def report_feedback(
    feedback: dict,
    timeout_s: float = 1.0,
    url: str | None = None,
) -> bool:
    base = _base_url(url)
    if base is None:
        return False
    resp = _post(f"{base}/feedback", feedback, timeout_s)
    if resp is None:
        return False
    return bool(resp.get("accepted"))


def should_dual_run(decision: dict, dual_run_prob: float = 0.0) -> bool:
    if not isinstance(decision, dict):
        return False
    if decision.get("policy") == "explore":
        return True
    if dual_run_prob <= 0.0:
        return False
    if dual_run_prob >= 1.0:
        return True
    return random.random() < dual_run_prob
