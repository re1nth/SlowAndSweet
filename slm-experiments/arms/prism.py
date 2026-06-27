"""Arm C: prism — frontier-orchestrated DAG with per-node routing.

Each case ships a `prism_plan` (annotated DAG). prism's executor walks
that DAG, dispatching nodes to LLM or SLM per the routing precedence
(override > policy[type] > classifier). The terminal node's result is
the final answer.

Frontier + SLM usage is rolled up from per-node accounting and reported
the same way as the solo and mixture arms.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from adapters.frontier import FrontierAdapter
from metrics import ArmResult, FrontierUsage, SLMUsage

# Make the prism package importable as flat modules, matching its own
# CLI's conventions (no package install required).
_PRISM_DIR = Path(__file__).resolve().parent.parent.parent / "prism"
if str(_PRISM_DIR) not in sys.path:
    sys.path.insert(0, str(_PRISM_DIR))

from classifier import Classifier  # noqa: E402
from dag import NodeSpec  # noqa: E402
from executor import PrismExecutor  # noqa: E402
from policy import Policy  # noqa: E402


def _specs_from_plan(plan: dict) -> list[NodeSpec]:
    return [
        NodeSpec(
            id=n["id"],
            prompt=n["prompt"],
            depends_on=list(n.get("depends_on", [])),
            type=n.get("type"),
            backend=n.get("backend"),
            model=n.get("model"),
        )
        for n in plan["nodes"]
    ]


def run_prism(
    case: dict,
    frontier: FrontierAdapter,
    *,
    classifier_model: str = "smollm2:1.7b",
    slm_task_url: str = "http://127.0.0.1:8080",
) -> ArmResult:
    t0 = time.time()
    plan = case.get("prism_plan")
    if not plan:
        return ArmResult(
            arm="prism",
            output="",
            wall_seconds=0.0,
            error="case missing prism_plan",
        )

    policy = Policy.load()
    classifier = Classifier(policy, model=classifier_model)
    executor = PrismExecutor(
        frontier=frontier,
        policy=policy,
        classifier=classifier,
        slm_task_url=slm_task_url,
        max_parallel=int(case.get("max_parallel", 4)),
    )

    try:
        run = executor.run(
            _specs_from_plan(plan),
            plan_id=plan.get("plan_id", case["id"]),
            description=plan.get("description", ""),
        )
    except Exception as e:  # noqa: BLE001
        return ArmResult(
            arm="prism",
            output="",
            wall_seconds=time.time() - t0,
            error=f"{type(e).__name__}: {e}",
        )

    frontier_usage = FrontierUsage(calls=0)
    slm_models: list[str] = []
    slm_tokens = 0
    slm_nodes = 0
    slm_wall = 0.0
    any_error: str | None = None

    for n in run.nodes:
        if n.error:
            any_error = f"node {n.id} ({n.backend}): {n.error}"
            break
        if n.backend == "llm":
            frontier_usage.calls += 1
            frontier_usage.input_tokens += int(n.tokens_in or 0)
            frontier_usage.output_tokens += int(n.tokens_out or 0)
            frontier_usage.wall_seconds += float(n.wall_seconds or 0.0)
        else:  # slm
            slm_tokens += int(n.tokens_out or 0)
            slm_nodes += 1
            slm_wall += float(n.wall_seconds or 0.0)
            if n.model and n.model not in slm_models:
                slm_models.append(n.model)

    if any_error:
        return ArmResult(
            arm="prism", output="",
            frontier=frontier_usage,
            slm=SLMUsage(
                output_tokens=slm_tokens, nodes=slm_nodes,
                wall_seconds=slm_wall, models_used=sorted(slm_models),
            ),
            wall_seconds=time.time() - t0,
            error=any_error,
        )

    return ArmResult(
        arm="prism",
        output=run.final_output,
        frontier=frontier_usage,
        slm=SLMUsage(
            output_tokens=slm_tokens,
            nodes=slm_nodes,
            wall_seconds=slm_wall,
            models_used=sorted(slm_models),
        ),
        wall_seconds=time.time() - t0,
    )
