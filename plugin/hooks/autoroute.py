#!/usr/bin/env python3
"""UserPromptSubmit hook — auto-route via the stashed-plan flow.

Two modes, chosen by SLOWANDSWEET_HOOK_MODE:

  context (default)
    Compact `[autoroute]` context block is injected into the session.
    Frontier calls slm_run_stashed(stash_id) then slm_wait_plan(run_id),
    then acknowledges in one line. Depends on frontier compliance with
    the "do not reproduce tool output" directive to save tokens.
    Small tasks lose tokens (see slm-experiments/autoroute_validate.py):
    for tasks whose solo answer is under ~150 tokens, the injected
    directive alone exceeds solo cost.

  deny (experimental)
    Hook waits for SLMs synchronously (up to SLOWANDSWEET_DENY_TIMEOUT_S),
    then DENIES the user prompt entirely and returns the SLM outputs as
    permissionDecisionReason. Frontier never runs — zero frontier tokens
    for the delegated turn. Only path that saves tokens on tiny tasks.
    Trade-offs:
      - Adds SLM latency to the user's perceived response time (blocking hook).
      - Claude Code renders the denial reason as a system notice, which
        may look different from a normal assistant message. UX must be
        validated on the user's actual session before rollout.
      - The turn does not appear in Claude's conversation memory as an
        assistant reply, so follow-up prompts can't reference it via
        "your last message" style references.

Common to both modes:

  1. POST prompt to slm-router `/decompose`. Rule-based, ~1ms.
  2. POST plan to slm-queue `/plans/stash`. Queue holds it in memory.

Any error at any step exits silent and the frontier answers normally.
This hook fires on every user prompt; noise or latency is felt directly.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROUTER_URL = os.environ.get("SLM_ROUTER_URL", "http://127.0.0.1:8092").rstrip("/")
ROUTER_TOKEN = os.environ.get("SLM_ROUTER_TOKEN")
QUEUE_URL = os.environ.get("SLM_QUEUE_URL", "http://127.0.0.1:8080").rstrip("/")
TIMEOUT_S = float(os.environ.get("SLOWANDSWEET_AUTOROUTE_TIMEOUT", "2.0"))

# Mode: "context" (default, safe) or "deny" (experimental, saves more tokens
# on tiny tasks but changes the UX — the hook waits for SLMs and returns the
# result as a denial reason instead of letting the frontier run).
HOOK_MODE = os.environ.get("SLOWANDSWEET_HOOK_MODE", "context").strip().lower()
DENY_TIMEOUT_S = float(os.environ.get("SLOWANDSWEET_DENY_TIMEOUT_S", "20.0"))
DENY_POLL_INTERVAL_S = 0.4

STATE_DIR = Path(os.environ.get("SLOWANDSWEET_HOME", str(Path.home() / ".slowandsweet")))
GLOBAL_DISABLED_FLAG = STATE_DIR / "disabled"
AUTOROUTE_DISABLED_FLAG = STATE_DIR / "autoroute_disabled"
# The queue server itself is unauthenticated on localhost; the bearer token
# from ~/.slowandsweet/token gates the MCP surface, not /plans/stash.

OPT_OUT_PREFIXES = ("/no-delegate", "/solo", "no-delegate:", "solo:")


def _silent_exit() -> None:
    sys.exit(0)


def _read_stdin_prompt() -> str | None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except (json.JSONDecodeError, ValueError):
        return None
    prompt = payload.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    return None


def _short_circuit(prompt: str) -> bool:
    if GLOBAL_DISABLED_FLAG.exists() or AUTOROUTE_DISABLED_FLAG.exists():
        return True
    lowered = prompt.lstrip().lower()
    return any(lowered.startswith(p) for p in OPT_OUT_PREFIXES)


def _post_json(url: str, body: dict, token: str | None = None, timeout: float | None = None) -> dict | None:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout if timeout is not None else TIMEOUT_S) as r:
            if r.status not in (200, 202):
                return None
            body = json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    return body if isinstance(body, dict) else None


def _get_json(url: str, timeout: float | None = None) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout if timeout is not None else TIMEOUT_S) as r:
            if r.status not in (200, 202):
                return None
            body = json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    return body if isinstance(body, dict) else None


def _run_stash_and_wait(stash_id: str) -> dict | None:
    """DENY mode: trigger the stashed plan and poll until it finishes.

    Returns the plan snapshot on success, None on any failure (in which
    case the caller falls back to letting the frontier answer).
    """
    import time
    started = _post_json(f"{QUEUE_URL}/plans/from_stash", {"stash_id": stash_id})
    if started is None:
        return None
    run_id = started.get("run_id")
    if not isinstance(run_id, str):
        return None
    deadline = time.time() + DENY_TIMEOUT_S
    while time.time() < deadline:
        snap = _get_json(f"{QUEUE_URL}/plans/{run_id}")
        if snap is None:
            time.sleep(DENY_POLL_INTERVAL_S)
            continue
        status = snap.get("status")
        if status and status != "running":
            return snap
        time.sleep(DENY_POLL_INTERVAL_S)
    return None


def _format_deny_reason(snap: dict, n: int) -> str:
    """Render the plan snapshot as a user-facing block for permissionDecisionReason."""
    header = f"(delegated to {n} local SLM leaves)"
    nodes = snap.get("nodes") or {}
    order = snap.get("node_order") or list(nodes.keys())
    parts: list[str] = [header, ""]
    for i, nid in enumerate(order, start=1):
        node = nodes.get(nid) or {}
        result = node.get("result")
        error = node.get("error")
        if error:
            parts.append(f"### leaf {i} ({nid}) — ERROR")
            parts.append(str(error))
        else:
            parts.append(f"### leaf {i} ({nid})")
            parts.append(str(result or "").strip())
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _render_context(stash_id: str, n: int, rule: str, description: str) -> str:
    # Compact directive. Load-bearing rule: after the tool call, the SLM
    # results are ALREADY rendered inline in the transcript by Claude Code
    # — the user sees them without the frontier reproducing them. So the
    # frontier's final response is a single acknowledgment line. This is
    # the whole point of auto-route: no re-emitting big content.
    return (
        f"[autoroute] slm-router matched rule={rule}; "
        f"{n} SLM leaves already stashed as `{stash_id}` (description: {description}).\n"
        f"DIRECTIVE:\n"
        f"  1. Call `slm_run_stashed(stash_id=\"{stash_id}\")` — returns run_id.\n"
        f"  2. Call `slm_wait_plan(run_id=<id>)` — returns node results.\n"
        f"  3. The tool output above is ALREADY VISIBLE to the user in the "
        f"transcript. Do NOT reproduce, summarize, quote, reformat, or "
        f"paraphrase any part of it — that would double the tokens.\n"
        f"  4. Your ENTIRE response after the tool calls MUST be exactly "
        f"one line:\n"
        f"     `(delegated to {n} local SLM leaves — results above)`\n"
        f"     No preamble, no closing, no commentary.\n"
        f"If the decomposition is materially wrong for the user's intent, "
        f"skip the tool calls and answer normally without mentioning this hint.\n"
    )


def main() -> None:
    prompt = _read_stdin_prompt()
    if prompt is None:
        _silent_exit()
    if _short_circuit(prompt):  # type: ignore[arg-type]
        _silent_exit()

    decomp = _post_json(f"{ROUTER_URL}/decompose", {"prompt": prompt}, ROUTER_TOKEN)  # type: ignore[arg-type]
    if decomp is None or decomp.get("decision") != "mixture":
        _silent_exit()
    plan = decomp.get("plan")
    if not isinstance(plan, dict):
        _silent_exit()

    stashed = _post_json(f"{QUEUE_URL}/plans/stash", {"plan": plan})
    if stashed is None:
        _silent_exit()
    stash_id = stashed.get("stash_id")
    if not isinstance(stash_id, str) or not stash_id:
        _silent_exit()

    n = len(plan.get("nodes", []))
    rule = decomp.get("rule", "?")
    description = plan.get("description", "")

    if HOOK_MODE == "deny":
        # Wait for SLMs to finish; return the results as the denial reason.
        # Frontier never runs -> zero frontier tokens for this turn.
        snap = _run_stash_and_wait(stash_id)
        if snap is None or snap.get("status") != "done":
            # Anything went wrong (timeout, queue error, plan error) —
            # silently fall back to the frontier answering solo.
            _silent_exit()
        reason = _format_deny_reason(snap, n)
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            },
        }))
        return

    # Default: inject context, let the frontier drive the tool calls.
    context = _render_context(stash_id, n, rule, description)
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        },
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _silent_exit()
