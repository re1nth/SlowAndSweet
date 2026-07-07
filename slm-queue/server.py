"""HTTP queue server for SLM task dispatch.

Reads slm-deploy/slms.yaml, spawns `replicas` worker threads per deployment,
exposes:

  POST /tasks       {"prompt": "..."}  -> 202 {"task_id": "...", "model": "..."}
  GET  /tasks/<id>                     -> 200 task state
  GET  /status                         -> 200 queue depths + counters

All workers share one local `ollama serve`. To get real concurrent generation
when a model has replicas > 1, start ollama with OLLAMA_NUM_PARALLEL >= the
max replicas of any single model.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "slm-deploy"))
from validate import collect_deployments, load_yaml_docs  # noqa: E402

from router import choose_arm, choose_model  # noqa: E402
from planner import Plan, PlanError, PlanRegistry, PlanRun, PlanRunner  # noqa: E402
import ui  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"
PLANS_DIR = HERE / "plans"


class Dispatcher:
    """Per-model queues, worker pool, and the in-memory results table."""

    def __init__(self, deployments: list[dict]):
        self.deployments = deployments
        self.models: list[str] = [d["spec"]["model"] for d in deployments]
        self.queues: dict[str, queue.Queue] = {m: queue.Queue() for m in self.models}
        self.results: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.workers: list[threading.Thread] = []

    def start_workers(self) -> None:
        for d in self.deployments:
            model = d["spec"]["model"]
            name = d["metadata"]["name"]
            n = int(d["spec"].get("replicas", 1))
            for i in range(n):
                t = threading.Thread(
                    target=self._worker_loop,
                    args=(model, i),
                    name=f"worker-{name}-{i}",
                    daemon=True,
                )
                t.start()
                self.workers.append(t)
        print(f"started {len(self.workers)} workers across {len(self.models)} models:")
        for d in self.deployments:
            print(f"  {d['metadata']['name']:<10} model={d['spec']['model']:<14} "
                  f"replicas={d['spec'].get('replicas', 1)}")

    def submit(self, prompt: str) -> tuple[str, str]:
        model = choose_model(prompt, self.models)
        task_id = uuid.uuid4().hex[:12]
        with self.lock:
            self.results[task_id] = {
                "task_id": task_id,
                "status": "pending",
                "model": model,
                "prompt": prompt,
                "submitted_at": time.time(),
            }
        self.queues[model].put((task_id, prompt))
        return task_id, model

    def get(self, task_id: str) -> dict | None:
        with self.lock:
            r = self.results.get(task_id)
            return dict(r) if r else None

    def snapshot(self) -> dict:
        with self.lock:
            depths = {m: self.queues[m].qsize() for m in self.models}
            counts = {s: 0 for s in ("pending", "running", "done", "error")}
            for r in self.results.values():
                counts[r["status"]] = counts.get(r["status"], 0) + 1
            return {
                "models": self.models,
                "queue_depths": depths,
                "tasks_total": len(self.results),
                "task_counts": counts,
                "workers": len(self.workers),
            }

    def _worker_loop(self, model: str, replica_idx: int) -> None:
        q = self.queues[model]
        worker_label = f"{model}#{replica_idx}"
        while True:
            task_id, prompt = q.get()
            t0 = time.time()
            with self.lock:
                self.results[task_id].update({
                    "status": "running",
                    "started_at": t0,
                    "worker": worker_label,
                })
            try:
                resp_text, eval_count = self._call_ollama(model, prompt)
                with self.lock:
                    self.results[task_id].update({
                        "status": "done",
                        "result": resp_text,
                        "eval_count": eval_count,
                        "finished_at": time.time(),
                    })
            except Exception as e:  # noqa: BLE001
                with self.lock:
                    self.results[task_id].update({
                        "status": "error",
                        "error": f"{type(e).__name__}: {e}",
                        "finished_at": time.time(),
                    })
            finally:
                q.task_done()

    def _call_ollama(self, model: str, prompt: str) -> tuple[str, int]:
        body = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"seed": 42, "temperature": 0.7, "num_predict": 256},
        }).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read())
        return payload.get("response", ""), int(payload.get("eval_count", 0))


def _start_plan_run(plan: Plan, dispatcher: Dispatcher, registry: PlanRegistry) -> PlanRun:
    run = PlanRun(plan)
    registry.register(run)
    PlanRunner(run, dispatcher).start()
    return run


def make_handler(d: Dispatcher, registry: PlanRegistry):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, indent=2).encode("utf-8")
            self._respond(code, body, "application/json")

        def _html(self, code: int, body: str) -> None:
            self._respond(code, body.encode("utf-8"), "text/html; charset=utf-8")

        def _respond(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):  # quiet stdout
            return

        def _read_json(self) -> dict | None:
            n = int(self.headers.get("Content-Length", 0))
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"error": "invalid json"})
                return None

        def do_POST(self):
            if self.path == "/tasks":
                payload = self._read_json()
                if payload is None:
                    return
                prompt = payload.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    self._json(400, {"error": "prompt (non-empty string) required"})
                    return
                task_id, model = d.submit(prompt)
                self._json(202, {"task_id": task_id, "model": model})
                return

            if self.path == "/plans":
                payload = self._read_json()
                if payload is None:
                    return
                try:
                    if "plan_file" in payload:
                        plan = registry.load_from_file(payload["plan_file"])
                    elif "plan" in payload:
                        plan = Plan.from_dict(payload["plan"])
                    else:
                        self._json(400, {"error": "provide either plan_file or plan"})
                        return
                except (PlanError, KeyError, FileNotFoundError) as e:
                    self._json(400, {"error": f"{type(e).__name__}: {e}"})
                    return
                # Consult the router (no-op unless SLM_ROUTER_URL is set); the
                # planner reads the stash at feedback time.
                try:
                    choose_arm(plan.description or plan.plan_id)
                except Exception:
                    pass
                run = _start_plan_run(plan, d, registry)
                self._json(202, {"run_id": run.run_id, "plan_id": plan.plan_id})
                return

            self._json(404, {"error": "not found"})

        def do_GET(self):
            path = self.path.split("?", 1)[0]

            if path == "/status":
                self._json(200, d.snapshot())
                return

            if path.startswith("/tasks/"):
                tid = path.split("/", 2)[2]
                r = d.get(tid)
                if r is None:
                    self._json(404, {"error": "unknown task_id"})
                    return
                self._json(200, r)
                return

            if path == "/plans":
                self._json(200, {
                    "runs": registry.list_runs(),
                    "plan_files": registry.list_plan_files(),
                })
                return

            if path.startswith("/plans/from-file/"):
                name = path[len("/plans/from-file/"):]
                try:
                    plan = registry.load_from_file(name)
                except (PlanError, FileNotFoundError) as e:
                    self._html(400, f"<pre>{type(e).__name__}: {e}</pre>")
                    return
                run = _start_plan_run(plan, d, registry)
                # Convenience: redirect to the UI page for the new run.
                self.send_response(303)
                self.send_header("Location", f"/plans/{run.run_id}/ui")
                self.end_headers()
                return

            if path.startswith("/plans/") and path.endswith("/ui"):
                run_id = path[len("/plans/"):-len("/ui")]
                run = registry.get(run_id)
                if run is None:
                    self._html(404, "<p>unknown run_id</p>")
                    return
                self._html(200, ui.render_plan_run(run.snapshot()))
                return

            if path.startswith("/plans/"):
                run_id = path.split("/", 2)[2]
                run = registry.get(run_id)
                if run is None:
                    self._json(404, {"error": "unknown run_id"})
                    return
                self._json(200, run.snapshot())
                return

            if path in ("/ui", "/"):
                self._html(200, ui.render_index(registry.list_runs(), registry.list_plan_files()))
                return

            self._json(404, {"error": "not found"})

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slms", default=str(ROOT / "slm-deploy" / "slms.yaml"),
                    help="path to SLMDeployment YAML")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    deployments = collect_deployments(load_yaml_docs(Path(args.slms)))
    if not deployments:
        print("no SLMDeployment docs found", file=sys.stderr)
        return 1

    d = Dispatcher(deployments)
    d.start_workers()

    registry = PlanRegistry(PLANS_DIR)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(d, registry))
    print(f"\nlistening on http://127.0.0.1:{args.port}")
    print("  POST /tasks            GET /tasks/<id>     GET /status")
    print("  POST /plans            GET /plans          GET /plans/<run_id>[/ui]")
    print(f"  GET  /ui               (plan files in {PLANS_DIR.relative_to(ROOT)}/)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
