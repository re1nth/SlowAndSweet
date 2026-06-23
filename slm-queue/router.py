"""Prompt -> model name routing.

Picks one of the deployed model names for a given prompt using simple
rule-based heuristics. The available list comes from slm-deploy/slms.yaml
at server startup, so the router only ever returns a model the dispatcher
has workers for.
"""
from __future__ import annotations


def _first_match(available: list[str], prefix: str) -> str | None:
    for m in available:
        if m.startswith(prefix):
            return m
    return None


def choose_model(prompt: str, available: list[str]) -> str:
    """Pick a model from `available` for `prompt`.

    Heuristics, in order:
      - code-ish prompts -> qwen2.5 (strong on code among these SLMs)
      - math/reasoning   -> llama3.2
      - long prompts     -> gemma2  (handles wider context comfortably)
      - default          -> smollm2 (fastest TTFT, cheapest)
    """
    p = prompt.lower()

    code_markers = ("```", "def ", "function", "python", "javascript",
                    "typescript", "sql", "code", "regex", "bash ", "shell")
    if any(k in p for k in code_markers):
        m = _first_match(available, "qwen2.5")
        if m:
            return m

    math_markers = ("calculate", "compute", "solve", "equation", "prove",
                    "derivative", "integral", " sum ", "math ")
    if any(k in p for k in math_markers):
        m = _first_match(available, "llama3.2")
        if m:
            return m

    if len(prompt) > 500:
        m = _first_match(available, "gemma2")
        if m:
            return m

    m = _first_match(available, "smollm2")
    if m:
        return m
    return available[0]
