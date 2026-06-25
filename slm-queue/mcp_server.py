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
import time
import urllib.error
import urllib.request
from typing import Any

from mcp.server.fastmcp import FastMCP

QUEUE_BASE = os.environ.get("SLM_QUEUE_BASE", "http://127.0.0.1:8080")


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()
    server = _make_server(args.host, args.port)
    print(f"MCP slm-queue server on http://{args.host}:{args.port}/mcp")
    print(f"  backing queue: {QUEUE_BASE}")
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
