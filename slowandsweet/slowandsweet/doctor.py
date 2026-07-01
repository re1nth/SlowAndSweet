"""`slowandsweet doctor` — health checks against ~/.slowandsweet/ and the local servers."""
from __future__ import annotations

import json
import os
import socket
import sys
import urllib.error
import urllib.request
from typing import Any

from slowandsweet import state
from slowandsweet.paths import (
    DB_PATH,
    STATE_DIR,
    TOKEN_PATH,
    slms_yaml_path,
)

OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
QUEUE_STATUS_URL = "http://127.0.0.1:8080/status"
MCP_URL = "http://127.0.0.1:8090/mcp"

OK = "OK"
FAIL = "FAIL"
WARN = "WARN"

# Soft checks: the queue and MCP server commonly aren't running yet; a missing
# server is informational, not a hard failure.
SOFT_CHECKS = {"queue server reachable", "MCP server reachable"}


def _http_status(url: str, timeout: float = 1.0) -> int:
    """Return the response status code; raises urllib errors on failure."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def _check_state_dir() -> tuple[str, str]:
    if STATE_DIR.is_dir():
        return OK, str(STATE_DIR)
    return FAIL, f"{STATE_DIR} does not exist (run `slowandsweet init`)"


def _check_token() -> tuple[str, str]:
    if not TOKEN_PATH.exists():
        return FAIL, f"{TOKEN_PATH} missing (run `slowandsweet init`)"
    try:
        mode = os.stat(TOKEN_PATH).st_mode & 0o777
    except OSError as e:
        return FAIL, f"stat failed: {type(e).__name__}: {e}"
    if mode != 0o600:
        return FAIL, f"{TOKEN_PATH} mode is {oct(mode)}, expected 0o600"
    return OK, f"{TOKEN_PATH} mode 0o600"


def _check_db() -> tuple[str, str]:
    if not DB_PATH.exists():
        return FAIL, f"{DB_PATH} missing (run `slowandsweet init`)"
    v = state.schema_version(DB_PATH)
    if v != state.CURRENT_SCHEMA_VERSION:
        return FAIL, f"schema_version is {v}, expected {state.CURRENT_SCHEMA_VERSION}"
    return OK, f"schema v{v}"


def _check_ollama() -> tuple[str, str, list[str]]:
    try:
        req = urllib.request.Request(OLLAMA_TAGS_URL, method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as r:
            if r.status != 200:
                return FAIL, f"GET /api/tags returned {r.status}", []
            payload = json.loads(r.read())
    except urllib.error.URLError as e:
        return FAIL, f"unreachable: {e.reason}", []
    except (socket.timeout, TimeoutError):
        return FAIL, "timed out", []
    except Exception as e:  # noqa: BLE001
        return FAIL, f"{type(e).__name__}: {e}", []
    models = [m.get("name") for m in payload.get("models", []) if isinstance(m, dict)]
    return OK, f"{len(models)} model(s) loaded", [m for m in models if m]


def _required_models() -> list[str]:
    path = slms_yaml_path()
    if path is None:
        return []
    try:
        import yaml  # type: ignore
    except ImportError:
        return []
    try:
        docs = list(yaml.safe_load_all(path.read_text()))
    except Exception:  # noqa: BLE001
        return []
    out: list[str] = []
    for d in docs:
        if not isinstance(d, dict) or d.get("kind") != "SLMDeployment":
            continue
        m = (d.get("spec") or {}).get("model")
        if isinstance(m, str) and m not in out:
            out.append(m)
    return out


def _check_queue() -> tuple[str, str]:
    try:
        code = _http_status(QUEUE_STATUS_URL, timeout=1.0)
    except urllib.error.URLError as e:
        return WARN, (
            f"not running ({e.reason}); start with "
            "`python3 slm-queue/server.py --port 8080`"
        )
    except Exception as e:  # noqa: BLE001
        return WARN, f"{type(e).__name__}: {e}"
    if code == 200:
        return OK, f"GET /status -> {code}"
    return WARN, f"unexpected status {code}"


def _check_mcp() -> tuple[str, str]:
    try:
        code = _http_status(MCP_URL, timeout=1.0)
    except urllib.error.URLError as e:
        return WARN, (
            f"not running ({e.reason}); start with "
            "`python3 slm-queue/mcp_server.py --port 8090`"
        )
    except Exception as e:  # noqa: BLE001
        return WARN, f"{type(e).__name__}: {e}"
    # Any HTTP response means it's listening; FastMCP often returns 405/406 on GET.
    if 200 <= code < 500:
        return OK, f"GET /mcp -> {code} (listening)"
    return WARN, f"unexpected status {code}"


def _safe(fn, *args) -> tuple[str, str]:
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001
        return FAIL, f"{type(e).__name__}: {e}"


def collect() -> list[dict[str, str]]:
    results: list[dict[str, str]] = []

    s, d = _safe(_check_state_dir)
    results.append({"name": "state directory", "status": s, "detail": d})

    s, d = _safe(_check_token)
    results.append({"name": "token file", "status": s, "detail": d})

    s, d = _safe(_check_db)
    results.append({"name": "sqlite db", "status": s, "detail": d})

    try:
        s, d, ollama_models = _check_ollama()
    except Exception as e:  # noqa: BLE001
        s, d, ollama_models = FAIL, f"{type(e).__name__}: {e}", []
    results.append({"name": "ollama reachable", "status": s, "detail": d})

    required = []
    try:
        required = _required_models()
    except Exception as e:  # noqa: BLE001
        results.append({"name": "models required", "status": FAIL,
                        "detail": f"{type(e).__name__}: {e}"})
    if required:
        installed = set(ollama_models)
        for m in required:
            present = m in installed
            results.append({
                "name": f"model {m}",
                "status": OK if present else FAIL,
                "detail": "present" if present else "missing (run `slowandsweet init`)",
            })
    else:
        results.append({
            "name": "models required",
            "status": WARN,
            "detail": "could not read slm-deploy/slms.yaml",
        })

    s, d = _safe(_check_queue)
    results.append({"name": "queue server reachable", "status": s, "detail": d})

    s, d = _safe(_check_mcp)
    results.append({"name": "MCP server reachable", "status": s, "detail": d})

    return results


def _print_human(results: list[dict[str, str]]) -> None:
    width = max(len(r["name"]) for r in results)
    for r in results:
        print(f"  [{r['status']:<4}] {r['name']:<{width}}  {r['detail']}")


def run(as_json: bool = False) -> int:
    results = collect()
    if as_json:
        json.dump(results, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_human(results)

    for r in results:
        if r["status"] == FAIL and r["name"] not in SOFT_CHECKS:
            return 1
    return 0
