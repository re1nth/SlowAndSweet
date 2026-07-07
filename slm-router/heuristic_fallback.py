"""Rule-based fallback used at cold start, when the learned head is missing,
or when the router service is down and the queue must decide locally.

Descendant of the LC frontier bench finding: length + structural markers
give Pearson r ~ 0.62 against actual reduction%. Zero external deps.
"""
from __future__ import annotations

import tiktoken

from model import Decision

_ENC = tiktoken.get_encoding("cl100k_base")

_HEAD_VERSION = "heuristic"


def heuristic_decide(prompt: str) -> Decision:
    tok = len(_ENC.encode(prompt))
    has_examples = ("Example" in prompt) or ("```" in prompt) or ("example:" in prompt.lower())
    has_constraints = "Constraints" in prompt
    has_multiline = prompt.count("\n") >= 3

    # Very short prompts — SLM sketch would be at least as long, skip decomp.
    if tok < 120:
        return Decision(
            decision="solo",
            predicted_reduction_pct=0.0,
            predicted_quality_ok_prob=0.9,
            confidence=0.9,
            head_version=_HEAD_VERSION,
            policy="heuristic",
        )

    # Wordy, structured prompts benefit most.
    if tok > 250 and (has_examples or has_constraints or has_multiline):
        return Decision(
            decision="mixture",
            predicted_reduction_pct=30.0,
            predicted_quality_ok_prob=0.8,
            confidence=0.6,
            head_version=_HEAD_VERSION,
            policy="heuristic",
        )

    # Mid-length prompts — lean solo but with low confidence to invite exploration.
    return Decision(
        decision="solo",
        predicted_reduction_pct=8.0,
        predicted_quality_ok_prob=0.75,
        confidence=0.4,
        head_version=_HEAD_VERSION,
        policy="heuristic",
    )
