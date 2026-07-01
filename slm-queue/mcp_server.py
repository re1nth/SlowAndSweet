"""MCP server exposing the SLM queue as tools to Claude Code.

This server is a thin wrapper over the queue's HTTP endpoints. It lets a
Claude Code session decompose a user prompt into a small DAG, hand the
whole plan to the local SLM workers via this server's `slm_submit_plan`
tool, then `slm_wait_plan` until the workers return results — without
Claude having to do the leaf work itself.

Transport: streamable HTTP at http://127.0.0.1:8090/mcp (configurable).

Setup:
  1. `python3 -m venv .venv && .venv/bin/pip install 'mcp[cli]'`
  2. Start the queue server:    `python3 slm-queue/server.py --port 8080`
  3. Start this MCP server:     `.venv/bin/python slm-queue/mcp_server.py`
  4. Register in `.mcp.json`:
       {
         "mcpServers": {
           "slm-queue": {"type": "http", "url": "http://127.0.0.1:8090/mcp"}
         }
       }
  5. Restart Claude Code so it picks up the new server.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

QUEUE_BASE = os.environ.get("SLM_QUEUE_BASE", "http://127.0.0.1:8080")
TOKEN_PATH = Path(os.path.expanduser("~/.slowandsweet/token"))
DISABLED_FLAG = Path(os.path.expanduser("~/.slowandsweet/disabled"))


class DelegationDisabled(RuntimeError):
    """Raised when the operator has turned delegation off via `slowandsweet disable`."""


def _load_token() -> str | None:
    """Read the shared bearer token. Returns None if the file is missing.

    Missing-file path is intentional: dev workflows that haven't run
    `slowandsweet init` should still work, just unauthenticated.
    """
    try:
        return TOKEN_PATH.read_text().strip() or None
    except FileNotFoundError:
        print(
            f"warning: {TOKEN_PATH} not found; MCP server running without auth.",
            file=sys.stderr,
        )
        return None
    except OSError as e:
        print(f"warning: could not read {TOKEN_PATH}: {e}; running without auth.",
              file=sys.stderr)
        return None


def _http_post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _http_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def _make_server(host: str, port: int) -> FastMCP:
    mcp = FastMCP(
        "slm-queue",
        host=host,
        port=port,
        instructions=(
            "Local pool of small language models behind a queue. Use this "
            "to delegate fan-out subtasks (parallel research, drafting, "
            "summarization) that don't need a frontier model. Submit a "
            "DAG plan with slm_submit_plan, then slm_wait_plan until the "
            "workers return. Compose the final answer yourself."
        ),
    )

    @mcp.tool()
    def slm_submit_plan(plan: dict[str, Any]) -> dict:
        """Submit a DAG plan of small prompts to the local SLM queue.

        The plan schema is:

            {
              "plan_id": "<short identifier>",
              "description": "<one-line description>",
              "nodes": [
                {
                  "id": "<node id, unique within plan>",
                  "prompt": "<the prompt text>",
                  "depends_on": ["<other_id>", ...]
                },
                ...
              ]
            }

        Within a node's prompt you may reference the output of a *direct*
        upstream dependency using the placeholder `{{node_id.result}}`.
        Every such placeholder must correspond to an entry in that node's
        depends_on list, or the queue rejects the plan.

        The DAG must be acyclic. Independent nodes (no shared deps) run
        in parallel on the worker pool.

        Returns: {"run_id": "<id>", "plan_id": "<id>"}.

        After submitting, call slm_wait_plan(run_id) to block until all
        nodes complete and retrieve their results.
        """
        if DISABLED_FLAG.exists():
            raise DelegationDisabled(
                "delegation disabled via `slowandsweet disable`; "
                "re-enable with `slowandsweet enable`"
            )
        try:
            return _http_post(f"{QUEUE_BASE}/plans", {"plan": plan})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"queue rejected plan: {e.code} {detail}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"queue not reachable at {QUEUE_BASE} ({e.reason}). "
                "Start it with: python3 slm-queue/server.py --port 8080"
            ) from None

    @mcp.tool()
    def slm_wait_plan(run_id: str, timeout_s: float = 180.0) -> dict:
        """Block until a plan run reaches a terminal state and return the
        full snapshot.

        Polls the queue every 0.5s. Raises TimeoutError if the run is still
        `running` after `timeout_s` seconds (default 180).

        The returned snapshot has:
          - status: "done" | "error"
          - node_order: list of node ids in plan order
          - nodes: {node_id: {status, model, worker, result|error,
                              eval_count, prompt (resolved), elapsed}}

        On success you get every node's generated text in `nodes[id].result`.
        """
        deadline = time.time() + timeout_s
        last_status = None
        while time.time() < deadline:
            try:
                snap = _http_get(f"{QUEUE_BASE}/plans/{run_id}")
            except urllib.error.URLError as e:
                raise RuntimeError(f"queue not reachable: {e.reason}") from None
            last_status = snap["status"]
            if last_status != "running":
                return snap
            time.sleep(0.5)
        raise TimeoutError(
            f"plan {run_id} still {last_status!r} after {timeout_s}s"
        )

    return mcp


class _BearerTokenMiddleware:
    """Minimal ASGI middleware that gates every request on a shared bearer token.

    FastMCP's built-in auth (`settings.auth` + a `TokenVerifier`) is designed
    for OAuth-style resource servers and forces protected-resource metadata
    endpoints. For our single-user shared-token case that's overkill, so we
    wrap the Starlette ASGI app FastMCP builds in `streamable_http_app()`
    with this tiny middleware instead. The token is read once at startup.

    TODO: if FastMCP grows a first-class shared-secret auth seam, switch to it.
    """

    def __init__(self, app, token: str):
        self.app = app
        self.expected = f"Bearer {token}"

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        got = headers.get(b"authorization", b"").decode("latin-1")
        if got != self.expected:
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [(b"content-type", b"application/json")],
            })
            await send({
                "type": "http.response.body",
                "body": b'{"error":"unauthorized"}',
            })
            return
        await self.app(scope, receive, send)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()
    server = _make_server(args.host, args.port)
    print(f"MCP slm-queue server on http://{args.host}:{args.port}/mcp")
    print(f"  backing queue: {QUEUE_BASE}")

    token = _load_token()
    if token is None:
        server.run(transport="streamable-http")
        return

    # Token present: run the streamable-http ASGI app under uvicorn ourselves
    # so we can wrap it with bearer-token auth. Mirrors
    # FastMCP.run_streamable_http_async() but adds our middleware.
    import uvicorn  # FastMCP already depends on uvicorn.

    app = _BearerTokenMiddleware(server.streamable_http_app(), token)
    print(f"  auth: bearer token from {TOKEN_PATH}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
