"""SLM queue adapter.

Talks to the local `slm-queue/server.py` HTTP plan surface. Mirrors the
slm_submit_plan / slm_wait_plan MCP tools but cuts the MCP layer out so
the harness can run from plain Python without an MCP client.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from metrics import SLMUsage


@dataclass
class PlanRunSnapshot:
    run_id: str
    status: str  # "done" | "error" | "running"
    nodes: dict  # node_id -> node state
    elapsed_seconds: float
    leaf_results: dict  # node_id -> result text (only `done` nodes)
    usage: SLMUsage


class SLMQueueAdapter:
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url.rstrip("/")

    def submit(self, plan: dict) -> str:
        body = json.dumps({"plan": plan}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/plans",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read())
        return payload["run_id"]

    def wait(self, run_id: str, *, timeout_s: float = 300, poll_s: float = 0.5) -> PlanRunSnapshot:
        deadline = time.time() + timeout_s
        last: dict = {}
        while time.time() < deadline:
            req = urllib.request.Request(f"{self.base_url}/plans/{run_id}")
            with urllib.request.urlopen(req, timeout=30) as resp:
                last = json.loads(resp.read())
            if last.get("status") in ("done", "error"):
                break
            time.sleep(poll_s)
        else:
            raise TimeoutError(f"plan {run_id} not done after {timeout_s}s")

        nodes = last.get("nodes", {})
        leaf_results = {nid: n["result"] for nid, n in nodes.items() if n.get("status") == "done"}
        usage = SLMUsage(
            output_tokens=sum(int(n.get("eval_count", 0) or 0) for n in nodes.values()),
            nodes=len(nodes),
            wall_seconds=float(last.get("finished_at", 0) or 0) - float(last.get("created_at", 0) or 0),
            models_used=sorted({n.get("model", "") for n in nodes.values() if n.get("model")}),
        )
        return PlanRunSnapshot(
            run_id=run_id,
            status=last.get("status", "unknown"),
            nodes=nodes,
            elapsed_seconds=usage.wall_seconds,
            leaf_results=leaf_results,
            usage=usage,
        )

    def submit_and_wait(self, plan: dict, *, timeout_s: float = 300) -> PlanRunSnapshot:
        run_id = self.submit(plan)
        return self.wait(run_id, timeout_s=timeout_s)
