"""Mixed-DAG executor.

Walks an annotated DAG and dispatches each node to its backend:
  - SLM nodes -> slm-queue /tasks (single task, not a whole plan)
  - LLM nodes -> frontier.complete()

Routing precedence per node:
  1. explicit `backend` override on the NodeSpec
  2. policy lookup using `type` (from the DAG or the classifier)
  3. classifier fallback (calls a small Ollama model)

Parent results are substituted into a child's prompt via the same
`{{node_id.result}}` placeholder convention the slm-queue planner uses,
so nodes can flow results into one another regardless of backend.

The executor parallelizes nodes whose dependencies have all completed,
so a wave of SLM and LLM nodes at the same level all run concurrently.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import re
import threading
import time
import urllib.error
import urllib.request

from classifier import Classifier
from dag import NodeResult, NodeSpec, PrismRun
from policy import Policy


_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\.result\s*\}\}")


class ExecutorError(RuntimeError):
    pass


class PrismExecutor:
    def __init__(
        self,
        *,
        frontier,                          # adapters.frontier.FrontierAdapter
        policy: Policy,
        classifier: Classifier | None = None,
        slm_task_url: str = "http://127.0.0.1:8080",
        max_parallel: int = 4,
    ):
        self.frontier = frontier
        self.policy = policy
        self.classifier = classifier
        self.slm_task_url = slm_task_url.rstrip("/")
        self.max_parallel = max_parallel

    # ------------------------------------------------------------------ #
    # Routing                                                            #
    # ------------------------------------------------------------------ #
    def _route(self, spec: NodeSpec, run: PrismRun) -> tuple[str, str | None]:
        """Return (backend, resolved_type)."""
        if spec.backend in ("slm", "llm"):
            return spec.backend, spec.type

        if spec.type:
            return self.policy.backend_for(spec.type), spec.type

        if self.classifier is None:
            return self.policy.default_unknown, "unknown"

        cls = self.classifier.classify(spec.prompt)
        run.classifier_calls += 1
        run.classifier_seconds += cls.wall_seconds
        backend = self.policy.backend_for(cls.type)
        return backend, cls.type

    # ------------------------------------------------------------------ #
    # Dispatch                                                           #
    # ------------------------------------------------------------------ #
    def _run_llm(self, spec: NodeSpec, prompt: str, ttype: str | None) -> NodeResult:
        t0 = time.time()
        try:
            call = self.frontier.complete(
                system=None,
                user=prompt,
                max_tokens=800,
                temperature=0.2,
            )
            return NodeResult(
                id=spec.id,
                backend="llm",
                type=ttype,
                model=getattr(self.frontier, "name", None),
                prompt=prompt,
                result=call.text,
                wall_seconds=time.time() - t0,
                tokens_in=call.usage.input_tokens,
                tokens_out=call.usage.output_tokens,
            )
        except Exception as e:  # noqa: BLE001
            return NodeResult(
                id=spec.id, backend="llm", type=ttype, model=None,
                prompt=prompt, result="", wall_seconds=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
            )

    def _run_slm(self, spec: NodeSpec, prompt: str, ttype: str | None) -> NodeResult:
        t0 = time.time()
        try:
            body = json.dumps({"prompt": prompt}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.slm_task_url}/tasks",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read())
            task_id = payload["task_id"]

            deadline = time.time() + 300
            state: dict = {}
            while time.time() < deadline:
                req = urllib.request.Request(f"{self.slm_task_url}/tasks/{task_id}")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    state = json.loads(resp.read())
                if state.get("status") in ("done", "error"):
                    break
                time.sleep(0.25)
            else:
                raise TimeoutError(f"slm task {task_id} not done after 300s")

            if state.get("status") != "done":
                raise ExecutorError(state.get("error", "slm task failed"))

            return NodeResult(
                id=spec.id, backend="slm", type=ttype,
                model=state.get("model"),
                prompt=prompt, result=state.get("result", ""),
                wall_seconds=time.time() - t0,
                tokens_out=int(state.get("eval_count", 0) or 0),
            )
        except Exception as e:  # noqa: BLE001
            return NodeResult(
                id=spec.id, backend="slm", type=ttype, model=None,
                prompt=prompt, result="", wall_seconds=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
            )

    # ------------------------------------------------------------------ #
    # DAG walk                                                           #
    # ------------------------------------------------------------------ #
    def run(self, specs: list[NodeSpec], *, plan_id: str, description: str = "") -> PrismRun:
        results: dict[str, NodeResult] = {}
        by_id = {s.id: s for s in specs}
        children: dict[str, set[str]] = {s.id: set() for s in specs}
        for s in specs:
            for dep in s.depends_on:
                if dep not in by_id:
                    raise ExecutorError(f"node {s.id} depends on unknown {dep}")
                children[dep].add(s.id)
        indeg = {s.id: len(s.depends_on) for s in specs}
        ready = [s.id for s, _ in zip(specs, specs) if indeg[s.id] == 0]
        ordered: list[str] = []

        run = PrismRun(
            plan_id=plan_id,
            description=description,
            started_at=time.time(),
            finished_at=0.0,
            final_output="",
            nodes=[],
        )

        lock = threading.Lock()
        with cf.ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            while ready:
                wave = ready
                ready = []
                futs = {}
                for nid in wave:
                    spec = by_id[nid]
                    resolved_prompt = self._substitute(spec.prompt, results)
                    backend, ttype = self._route(spec, run)
                    if backend == "llm":
                        fut = pool.submit(self._run_llm, spec, resolved_prompt, ttype)
                    else:
                        fut = pool.submit(self._run_slm, spec, resolved_prompt, ttype)
                    futs[fut] = nid

                for fut in cf.as_completed(futs):
                    nid = futs[fut]
                    res = fut.result()
                    with lock:
                        results[nid] = res
                        ordered.append(nid)
                        if res.error:
                            run.finished_at = time.time()
                            run.nodes = [results[i] for i in ordered]
                            return run
                        for child in children[nid]:
                            indeg[child] -= 1
                            if indeg[child] == 0:
                                ready.append(child)

        run.nodes = [results[i] for i in ordered]
        run.finished_at = time.time()
        run.final_output = self._final_output(specs, results)
        return run

    @staticmethod
    def _substitute(prompt: str, results: dict[str, NodeResult]) -> str:
        def repl(m: re.Match) -> str:
            nid = m.group(1)
            if nid not in results:
                raise ExecutorError(f"placeholder references missing node {nid}")
            return results[nid].result
        return _PLACEHOLDER.sub(repl, prompt)

    @staticmethod
    def _final_output(specs: list[NodeSpec], results: dict[str, NodeResult]) -> str:
        """Pick the result of the terminal node (no children).

        If multiple terminals exist (unusual), concatenate them in DAG order.
        """
        producers = {s.id for s in specs}
        consumed = {dep for s in specs for dep in s.depends_on}
        terminals = [s.id for s in specs if s.id in (producers - consumed)]
        if not terminals:
            return ""
        return "\n\n".join(results[t].result for t in terminals if t in results)
