"""End-to-end orchestration for one experiment run.

For each case:
  1. Run Arm A (solo)  and Arm B (mixture) concurrently.
  2. Once both finish, kick off the blind pairwise reviewer (also runs
     concurrently across cases so the wall clock is dominated by the
     slowest arm, not by a serial review pass).
  3. Persist a JSON snapshot of the run for later reporting.

The runner is intentionally arm-agnostic so adding a third arm (e.g.
dynamic-decomposition mixture) is a matter of adding another callable
to `_ARM_REGISTRY` and updating the reviewer protocol.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import yaml

from adapters.frontier import FrontierAdapter
from adapters.slm import SLMQueueAdapter
from arms.mixture import run_mixture
from arms.solo import run_solo
from metrics import ArmResult, FrontierUsage
from reviewer import review


CASES_DIR = Path(__file__).resolve().parent / "cases"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_cases(case_ids: Iterable[str] | None = None) -> list[dict]:
    files = sorted(CASES_DIR.glob("*.yaml"))
    cases = []
    for f in files:
        with open(f) as fh:
            data = yaml.safe_load(fh)
        data["_file"] = f.name
        cases.append(data)
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c["id"] in wanted]
        if not cases:
            raise SystemExit(f"no cases matched ids: {sorted(wanted)}")
    return cases


def _run_one_case(
    case: dict,
    frontier: FrontierAdapter,
    slm: SLMQueueAdapter,
    rng: random.Random,
) -> dict:
    """Run both arms concurrently, then review."""
    print(f"  [case {case['id']}] starting arms")
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        solo_fut = pool.submit(run_solo, case, frontier)
        mix_fut = pool.submit(run_mixture, case, frontier, slm)
        solo_res: ArmResult = solo_fut.result()
        mix_res: ArmResult = mix_fut.result()
    print(
        f"  [case {case['id']}] arms done — "
        f"solo {solo_res.wall_seconds:.1f}s, mixture {mix_res.wall_seconds:.1f}s"
    )

    if solo_res.error or mix_res.error:
        verdict = None
        verdict_dict = {
            "winner": "skipped",
            "reason": f"arm error — solo: {solo_res.error}, mixture: {mix_res.error}",
        }
    else:
        verdict = review(
            case=case,
            solo_output=solo_res.output,
            mixture_output=mix_res.output,
            frontier=frontier,
            rng=rng,
        )
        verdict_dict = verdict.to_dict()
        print(f"  [case {case['id']}] reviewer → winner={verdict.winner}")

    return {
        "case_id": case["id"],
        "name": case.get("name", case["id"]),
        "category": case.get("category"),
        "solo": solo_res.to_dict(),
        "mixture": mix_res.to_dict(),
        "review": verdict_dict,
    }


def run_experiment(
    *,
    frontier: FrontierAdapter,
    slm: SLMQueueAdapter,
    case_ids: Iterable[str] | None = None,
    max_parallel_cases: int = 1,
    seed: int = 7,
    run_label: str | None = None,
) -> dict:
    cases = load_cases(case_ids)
    rng = random.Random(seed)
    run_id = run_label or f"run-{int(time.time())}"
    print(f"==> run {run_id}: {len(cases)} cases, frontier={frontier.name}")

    t0 = time.time()
    if max_parallel_cases == 1:
        case_results = [_run_one_case(c, frontier, slm, rng) for c in cases]
    else:
        with cf.ThreadPoolExecutor(max_workers=max_parallel_cases) as pool:
            futs = [pool.submit(_run_one_case, c, frontier, slm, rng) for c in cases]
            case_results = [f.result() for f in futs]

    totals = _aggregate(case_results)
    snapshot = {
        "run_id": run_id,
        "frontier": frontier.name,
        "seed": seed,
        "started_at": t0,
        "finished_at": time.time(),
        "wall_seconds": time.time() - t0,
        "case_count": len(case_results),
        "totals": totals,
        "cases": case_results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{run_id}.json"
    with open(out_path, "w") as fh:
        json.dump(snapshot, fh, indent=2, default=str)
    print(f"==> wrote {out_path}")
    return snapshot


def _aggregate(case_results: list[dict]) -> dict:
    def sum_f(arm: str, key: str) -> int:
        return sum(int(c[arm]["frontier"].get(key, 0) or 0) for c in case_results)

    def sum_s(arm: str, key: str) -> int:
        return sum(int(c[arm]["slm"].get(key, 0) or 0) for c in case_results)

    def sum_t(arm: str) -> float:
        return sum(float(c[arm].get("wall_seconds", 0.0) or 0.0) for c in case_results)

    winners = [c["review"].get("winner", "skipped") for c in case_results]
    win_counts = {
        "solo": winners.count("solo"),
        "mixture": winners.count("mixture"),
        "tie": winners.count("tie"),
        "skipped": winners.count("skipped"),
    }

    reviewer_input = sum(
        int((c["review"].get("reviewer_usage", {}) or {}).get("input_tokens", 0) or 0)
        for c in case_results
    )
    reviewer_output = sum(
        int((c["review"].get("reviewer_usage", {}) or {}).get("output_tokens", 0) or 0)
        for c in case_results
    )

    return {
        "winners": win_counts,
        "solo": {
            "frontier_input_tokens": sum_f("solo", "input_tokens"),
            "frontier_output_tokens": sum_f("solo", "output_tokens"),
            "wall_seconds": sum_t("solo"),
        },
        "mixture": {
            "frontier_input_tokens": sum_f("mixture", "input_tokens"),
            "frontier_output_tokens": sum_f("mixture", "output_tokens"),
            "slm_output_tokens": sum_s("mixture", "output_tokens"),
            "wall_seconds": sum_t("mixture"),
        },
        "reviewer": {
            "input_tokens": reviewer_input,
            "output_tokens": reviewer_output,
        },
    }
