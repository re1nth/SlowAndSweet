"""Arm A: frontier model handles the whole task in one call."""
from __future__ import annotations

import time

from adapters.frontier import FrontierAdapter
from metrics import ArmResult, FrontierUsage, SLMUsage


def run_solo(case: dict, frontier: FrontierAdapter) -> ArmResult:
    t0 = time.time()
    try:
        call = frontier.complete(
            system=case.get("solo_system"),
            user=case["solo_prompt"],
            max_tokens=int(case.get("max_tokens", 1024)),
            temperature=float(case.get("temperature", 0.2)),
        )
    except Exception as e:  # noqa: BLE001
        return ArmResult(
            arm="solo",
            output="",
            frontier=FrontierUsage(),
            slm=SLMUsage(),
            wall_seconds=time.time() - t0,
            error=f"{type(e).__name__}: {e}",
        )
    return ArmResult(
        arm="solo",
        output=call.text,
        frontier=call.usage,
        slm=SLMUsage(),
        wall_seconds=time.time() - t0,
    )
