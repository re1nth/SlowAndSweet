"""Frontier-driven decomposition of a free-text prompt into an annotated DAG.

Given a user prompt, asks the frontier to produce a small DAG (1-6
nodes) where each node carries:
  - id
  - prompt
  - depends_on
  - type (one of the policy vocabulary)

The frontier is asked to pre-classify so the SLM classifier is the
fallback path, not the primary one. The strict JSON schema is sent in
the system prompt and marked for prompt caching so re-decomposing
multiple prompts in one session is cheap.
"""
from __future__ import annotations

import json
import re

import sys
from pathlib import Path

# adapters live in slm-experiments; import path injected by the caller (CLI / arm)
HERE = Path(__file__).resolve().parent

from policy import Policy
from dag import NodeSpec


_JSON_BLOB = re.compile(r"\{.*\}", re.DOTALL)


def _system_prompt(policy: Policy) -> list[dict]:
    text = (
        "You decompose a user prompt into a small DAG of subtasks for a "
        "hybrid LLM+SLM executor. Aim for 1-6 nodes. Most prompts decompose "
        "into 2-4 leaf subtasks plus a single terminal `compose` node that "
        "produces the final answer.\n\n"
        "Each subtask gets a `type` from the following taxonomy:\n\n"
        f"{policy.vocab_block()}\n\n"
        "If a subtask doesn't fit any of these, use the type `unknown`.\n\n"
        "Output STRICT JSON only, no prose, in this exact shape:\n"
        "{\n"
        '  "plan_id": "<short-slug>",\n'
        '  "description": "<one line>",\n'
        '  "nodes": [\n'
        '    {"id": "<id>", "depends_on": [], "type": "<type>", '
        '"prompt": "<the prompt for this subtask>"},\n'
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "  - Node ids are lowercase short slugs, unique within the plan.\n"
        "  - depends_on lists OTHER node ids only.\n"
        "  - A node's prompt may reference an upstream node's result using "
        "the placeholder `{{<id>.result}}`. Every such reference must also "
        "appear in depends_on for that node.\n"
        "  - The terminal node (no children) should have type `compose` "
        "and is responsible for the final answer.\n"
    )
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def decompose(prompt: str, *, frontier, policy: Policy) -> dict:
    """Return a dict with `plan_id`, `description`, `nodes`.

    `frontier` is a slm_experiments.adapters.frontier.FrontierAdapter.
    """
    call = frontier.complete(
        system=_system_prompt(policy),
        user=f"USER PROMPT:\n{prompt}",
        max_tokens=1200,
        temperature=0.0,
    )
    m = _JSON_BLOB.search(call.text)
    if not m:
        raise ValueError(f"decomposer returned no JSON: {call.text!r}")
    plan = json.loads(m.group(0))
    if "nodes" not in plan or not isinstance(plan["nodes"], list):
        raise ValueError(f"decomposer plan missing nodes: {plan!r}")
    # Attach usage so the executor can roll it up.
    plan["_decomposer_usage"] = {
        "input_tokens": call.usage.input_tokens,
        "output_tokens": call.usage.output_tokens,
        "wall_seconds": call.usage.wall_seconds,
    }
    return plan


def plan_to_specs(plan: dict) -> list[NodeSpec]:
    specs: list[NodeSpec] = []
    for n in plan["nodes"]:
        specs.append(NodeSpec(
            id=n["id"],
            prompt=n["prompt"],
            depends_on=list(n.get("depends_on", [])),
            type=n.get("type"),
            backend=n.get("backend"),
            model=n.get("model"),
        ))
    return specs
