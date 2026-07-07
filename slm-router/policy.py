"""Pure decision policy: given head output + config, return a Decision.

No I/O, no side effects. Unit-testable. All thresholds come from config;
callers own the RNG for exploration.
"""
from __future__ import annotations

import random
from typing import Any

from model import Decision


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
