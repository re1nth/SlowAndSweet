"""Arm B: frontier orchestrates, SLM pool executes leaves.

For a controlled SxS the case YAML ships a pre-authored DAG (Arm B's
"plan") plus a final composition prompt that the frontier runs over the
leaf results. This isolates the variable we care about — what the
frontier offloads to SLMs vs. what it does itself — and removes
DAG-authoring variance from the experiment.

The composer prompt may reference any leaf via `{leaf:<node_id>}`
placeholders, which are substituted with that node's result text
right before the frontier call.
"""
from __future__ import annotations

import re
import time

from adapters.frontier import FrontierAdapter
from adapters.slm import SLMQueueAdapter
from metrics import ArmResult, FrontierUsage, SLMUsage


_LEAF_PLACEHOLDER = re.compile(r"\{leaf:(\w+)\}")


def _fill_composer(prompt: str, leaf_results: dict[str, str]) -> str:
    def repl(m: re.Match) -> str:
        nid = m.group(1)
        if nid not in leaf_results:
            raise KeyError(f"composer references missing leaf {nid!r}")
        return leaf_results[nid]
    return _LEAF_PLACEHOLDER.sub(repl, prompt)


def run_mixture(
    case: dict,
    frontier: FrontierAdapter,
    slm: SLMQueueAdapter,
) -> ArmResult:
    t0 = time.time()
    plan = case["mixture_plan"]
    composer_template: str = case["mixture_composer_prompt"]
    composer_system = case.get("mixture_composer_system")

    try:
        snapshot = slm.submit_and_wait(plan)
        if snapshot.status != "done":
            failed = [
                f"{nid}: {n.get('error', '?')}"
                for nid, n in snapshot.nodes.items()
                if n.get("status") == "error"
            ]
            return ArmResult(
                arm="mixture",
                output="",
                slm=snapshot.usage,
                wall_seconds=time.time() - t0,
                error="plan failed: " + "; ".join(failed),
            )

        composer_prompt = _fill_composer(composer_template, snapshot.leaf_results)
        call = frontier.complete(
            system=composer_system,
            user=composer_prompt,
            max_tokens=int(case.get("max_tokens", 1024)),
            temperature=float(case.get("temperature", 0.2)),
        )
    except Exception as e:  # noqa: BLE001
        return ArmResult(
            arm="mixture",
            output="",
            wall_seconds=time.time() - t0,
            error=f"{type(e).__name__}: {e}",
        )

    return ArmResult(
        arm="mixture",
        output=call.text,
        frontier=call.usage,
        slm=snapshot.usage,
        wall_seconds=time.time() - t0,
    )
