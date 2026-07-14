"""Pure decision policy: given head output + config, return a Decision.

No I/O, no side effects. Unit-testable. All thresholds come from config;
callers own the RNG for exploration.
"""
from __future__ import annotations

import random
from typing import Any

from model import Decision, LeafDecision


def decide(
    predicted_reduction_pct: float,
    predicted_quality_ok_prob: float,
    confidence: float,
    *,
    head_version: str,
    policy_config: dict[str, Any],
    rng: random.Random | None = None,
    force_explore: bool = False,
) -> Decision:
    quality_floor = float(policy_config.get("quality_floor", 0.7))
    cost_floor_pct = float(policy_config.get("cost_floor_pct", 20.0))
    confidence_floor = float(policy_config.get("confidence_floor", 0.5))
    epsilon = float(policy_config.get("epsilon", 0.10))

    rng = rng or random.Random()

    if force_explore or (confidence < confidence_floor and rng.random() < epsilon):
        return Decision(
            decision="mixture",  # explore takes the mixture path so we get the counterfactual signal
            predicted_reduction_pct=predicted_reduction_pct,
            predicted_quality_ok_prob=predicted_quality_ok_prob,
            confidence=confidence,
            head_version=head_version,
            policy="explore",
        )

    if predicted_quality_ok_prob < quality_floor:
        return Decision(
            decision="solo",
            predicted_reduction_pct=predicted_reduction_pct,
            predicted_quality_ok_prob=predicted_quality_ok_prob,
            confidence=confidence,
            head_version=head_version,
            policy="learned",
        )

    if predicted_reduction_pct < cost_floor_pct:
        return Decision(
            decision="solo",
            predicted_reduction_pct=predicted_reduction_pct,
            predicted_quality_ok_prob=predicted_quality_ok_prob,
            confidence=confidence,
            head_version=head_version,
            policy="learned",
        )

    if confidence < confidence_floor:
        return Decision(
            decision="unsure",
            predicted_reduction_pct=predicted_reduction_pct,
            predicted_quality_ok_prob=predicted_quality_ok_prob,
            confidence=confidence,
            head_version=head_version,
            policy="learned",
        )

    return Decision(
        decision="mixture",
        predicted_reduction_pct=predicted_reduction_pct,
        predicted_quality_ok_prob=predicted_quality_ok_prob,
        confidence=confidence,
        head_version=head_version,
        policy="learned",
    )


def decide_leaf(
    predictions: dict[str, float],
    *,
    head_version: str,
    available: list[str] | None = None,
    quality_predictions: dict[str, float] | None = None,
    quality_floor: float = 0.0,
) -> LeafDecision:
    """Pick a leaf SLM given per-model predicted output tokens and (optionally)
    per-model P(quality good).

    Policy: filter to the available intersection AND the SLMs whose predicted
    quality clears the floor; return argmin(output_tokens). If the quality
    filter would leave nothing, ignore it — a cheap answer beats no answer —
    but flag that in the policy field.
    """
    if not predictions:
        raise ValueError("empty predictions")
    if available:
        allowed = set(available)
        candidates = {m: v for m, v in predictions.items() if m in allowed}
    else:
        candidates = dict(predictions)
    if not candidates:
        raise ValueError(
            "no overlap between requested `available` list and trained SLMs"
        )

    quality_predictions = quality_predictions or {}
    policy = "learned"

    if quality_floor > 0 and quality_predictions:
        gated = {
            m: v for m, v in candidates.items()
            if quality_predictions.get(m, 1.0) >= quality_floor
        }
        if gated:
            candidates = gated
        else:
            # No SLM cleared the floor; fall through to raw argmin so the
            # request still gets a decision — but tag the policy so callers
            # can see we bypassed the gate.
            policy = "learned_no_quality_pass"

    chosen = min(candidates, key=candidates.__getitem__)
    chosen_qual = quality_predictions.get(chosen)
    return LeafDecision(
        model=chosen,
        predicted_output_tokens=candidates[chosen],
        predicted_quality_prob=chosen_qual,
        alternatives={m: v for m, v in candidates.items() if m != chosen},
        alternatives_quality={
            m: q for m, q in quality_predictions.items() if m != chosen and m in candidates
        },
        head_version=head_version,
        policy=policy,
    )
