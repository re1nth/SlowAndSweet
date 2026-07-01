r"""DAG plan loading, validation, and execution.

A Plan is a directed acyclic graph of prompt nodes. Each node may depend on
other nodes; its prompt may reference upstream node results via
`{{node_id.result}}` placeholders, substituted right before the node is
submitted to the SLM queue.

Status lifecycle per node:
    pending  -> queued  -> running  -> done   (success path)
                                    \-> error (worker failed)

A PlanRunner walks the DAG in a background thread, submitting ready nodes
to a Dispatcher, polling for results, and unlocking dependents.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server import Dispatcher


_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\.result\s*\}\}")


def _record_run_metric(run: "PlanRun", terminal_status: str) -> None:
    """Best-effort write to the SlowAndSweet SQLite metrics.

    Lazy-imported so the queue keeps working when the `slowandsweet` wheel
    isn't installed. Swallows all errors — metrics must never take down a
    plan run.
    """
    try:
        from slowandsweet import state  # type: ignore
    except Exception:  # noqa: BLE001
        return
    try:
        slm_tokens_out = sum(
            (n.get("eval_count") or 0) for n in run.nodes.values()
        )
        wall_ms = None
        if run.finished_at is not None:
            wall_ms = int((run.finished_at - run.created_at) * 1000)
        status = "delegated" if terminal_status == "done" else "failed"
        state.record_call(
            call_id=run.run_id,
            status=status,
            slm_tokens_out=slm_tokens_out or None,
            wall_ms=wall_ms,
        )
    except Exception:  # noqa: BLE001
        return


class PlanError(ValueError):
    pass


class Plan:
    def __init__(self, plan_id: str, description: str, nodes: list[dict]):
        self.plan_id = plan_id
        self.description = description
        self.nodes = nodes
        self._validate()

    @classmethod
    def from_file(cls, path: Path) -> "Plan":
        data = json.loads(path.read_text())
        return cls(
            plan_id=data.get("plan_id") or path.stem,
            description=data.get("description", ""),
            nodes=data["nodes"],
        )

    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        return cls(
            plan_id=data.get("plan_id", "inline"),
            description=data.get("description", ""),
            nodes=data["nodes"],
        )

    def _validate(self) -> None:
        if not self.nodes:
            raise PlanError("plan has no nodes")
        ids = [n["id"] for n in self.nodes]
        if len(set(ids)) != len(ids):
            raise PlanError("duplicate node ids")
        id_set = set(ids)
        for n in self.nodes:
            if "prompt" not in n or not n["prompt"]:
                raise PlanError(f"node {n['id']} missing prompt")
            for dep in n.get("depends_on", []):
                if dep not in id_set:
                    raise PlanError(f"node {n['id']} depends on unknown node {dep}")
            refs = set(_PLACEHOLDER.findall(n["prompt"]))
            declared = set(n.get("depends_on", []))
            stray = refs - declared
            if stray:
                raise PlanError(
                    f"node {n['id']} prompt references {sorted(stray)} "
                    f"but they are not in depends_on"
                )
        # Cycle detection via topological sort (Kahn's algorithm).
        indeg = {n["id"]: len(n.get("depends_on", [])) for n in self.nodes}
        children: dict[str, list[str]] = {n["id"]: [] for n in self.nodes}
        for n in self.nodes:
            for dep in n.get("depends_on", []):
                children[dep].append(n["id"])
        queue = [nid for nid, d in indeg.items() if d == 0]
        seen = 0
        while queue:
            nid = queue.pop()
            seen += 1
            for c in children[nid]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        if seen != len(self.nodes):
            raise PlanError("plan contains a cycle")


class PlanRun:
    """Mutable state for one execution of a Plan."""

    def __init__(self, plan: Plan):
        self.run_id = uuid.uuid4().hex[:12]
        self.plan = plan
        self.created_at = time.time()
        self.nodes: dict[str, dict] = {
            n["id"]: {
                "id": n["id"],
                "prompt_template": n["prompt"],
                "depends_on": list(n.get("depends_on", [])),
                "status": "pending",
            }
            for n in plan.nodes
        }
        self.lock = threading.Lock()
        self.status = "running"  # running | done | error
        self.finished_at: float | None = None

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "run_id": self.run_id,
                "plan_id": self.plan.plan_id,
                "description": self.plan.description,
                "created_at": self.created_at,
                "finished_at": self.finished_at,
                "status": self.status,
                "node_order": [n["id"] for n in self.plan.nodes],
                "nodes": {nid: dict(self.nodes[nid]) for nid in self.nodes},
            }


class PlanRunner:
    """Drives a PlanRun against a Dispatcher in a background thread."""

    def __init__(self, run: PlanRun, dispatcher: "Dispatcher"):
        self.run = run
        self.dispatcher = dispatcher
        self.thread = threading.Thread(
            target=self._loop, daemon=True, name=f"planrun-{run.run_id}"
        )

    def start(self) -> None:
        self.thread.start()

    def _ready_node_ids(self) -> list[str]:
        out = []
        for nid, n in self.run.nodes.items():
            if n["status"] != "pending":
                continue
            if all(self.run.nodes[d]["status"] == "done" for d in n["depends_on"]):
                out.append(nid)
        return out

    def _resolve(self, template: str) -> str:
        def sub(m):
            return self.run.nodes[m.group(1)].get("result", "")
        return _PLACEHOLDER.sub(sub, template)

    def _loop(self) -> None:
        in_flight: dict[str, str] = {}  # node_id -> dispatcher task_id

        while True:
            # 1. Enqueue any newly-ready nodes.
            with self.run.lock:
                for nid in self._ready_node_ids():
                    node = self.run.nodes[nid]
                    prompt = self._resolve(node["prompt_template"])
                    task_id, model = self.dispatcher.submit(prompt)
                    node.update({
                        "status": "queued",
                        "task_id": task_id,
                        "model": model,
                        "prompt": prompt,
                        "submitted_at": time.time(),
                    })
                    in_flight[nid] = task_id

            # 2. Mirror task state into node state.
            for nid, tid in list(in_flight.items()):
                r = self.dispatcher.get(tid)
                if r is None:
                    continue
                with self.run.lock:
                    node = self.run.nodes[nid]
                    if r["status"] == "running" and node["status"] == "queued":
                        node["status"] = "running"
                        node["started_at"] = r.get("started_at")
                        node["worker"] = r.get("worker")
                    if r["status"] == "done":
                        node["status"] = "done"
                        node["started_at"] = node.get("started_at") or r.get("started_at")
                        node["worker"] = node.get("worker") or r.get("worker")
                        node["result"] = r.get("result", "")
                        node["eval_count"] = r.get("eval_count")
                        node["finished_at"] = r.get("finished_at")
                        del in_flight[nid]
                    elif r["status"] == "error":
                        node["status"] = "error"
                        node["error"] = r.get("error")
                        node["finished_at"] = r.get("finished_at")
                        del in_flight[nid]

            # 3. Termination check.
            terminal: str | None = None
            with self.run.lock:
                statuses = [n["status"] for n in self.run.nodes.values()]
                if all(s == "done" for s in statuses):
                    self.run.status = "done"
                    self.run.finished_at = time.time()
                    terminal = "done"
                # No more progress possible if nothing in flight and nothing newly ready.
                elif not in_flight and not self._ready_node_ids():
                    self.run.status = "error"
                    self.run.finished_at = time.time()
                    terminal = "error"
            if terminal is not None:
                _record_run_metric(self.run, terminal)
                return

            time.sleep(0.3)


class PlanRegistry:
    """Holds all PlanRuns in memory."""

    def __init__(self, plans_dir: Path):
        self.plans_dir = plans_dir
        self.runs: dict[str, PlanRun] = {}
        self.lock = threading.Lock()

    def list_plan_files(self) -> list[str]:
        if not self.plans_dir.exists():
            return []
        return sorted(p.name for p in self.plans_dir.glob("*.json"))

    def load_from_file(self, name: str) -> Plan:
        path = self.plans_dir / name
        if not path.exists():
            raise PlanError(f"plan file not found: {name}")
        return Plan.from_file(path)

    def register(self, run: PlanRun) -> None:
        with self.lock:
            self.runs[run.run_id] = run

    def get(self, run_id: str) -> PlanRun | None:
        with self.lock:
            return self.runs.get(run_id)

    def list_runs(self) -> list[dict]:
        with self.lock:
            return [
                {
                    "run_id": r.run_id,
                    "plan_id": r.plan.plan_id,
                    "status": r.status,
                    "created_at": r.created_at,
                    "finished_at": r.finished_at,
                    "node_count": len(r.nodes),
                }
                for r in sorted(self.runs.values(), key=lambda x: -x.created_at)
            ]
