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
import hashlib
import json
import os
import random
import secrets
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import yaml

from adapters.frontier import FrontierAdapter
from adapters.slm import SLMQueueAdapter
from arms.mixture import run_mixture
from arms.prism import run_prism
from arms.solo import run_solo
from metrics import ArmResult, FrontierUsage
from reviewer import review, review_nway


CASES_DIR = Path(__file__).resolve().parent / "cases"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


_WINNER_TO_VERDICT = {"solo": "A", "mixture": "B", "tie": "tie"}
_TIKTOKEN_ENC = None


def _count_tokens(text: str) -> int:
    global _TIKTOKEN_ENC
    if not text:
        return 0
    if _TIKTOKEN_ENC is None:
        try:
            import tiktoken  # type: ignore
            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001
            _TIKTOKEN_ENC = False
    if _TIKTOKEN_ENC and _TIKTOKEN_ENC is not False:
        return len(_TIKTOKEN_ENC.encode(text))
    return len(text.split())


def _post_router_feedback(case: dict, arms_results: dict, verdict_dict: dict, run_id: str) -> None:
    """POST a quality-labeled feedback record to slm-router. Gated on SLM_ROUTER_URL;
    swallows every error — a bad POST must never take down an experiment run."""
    url = os.environ.get("SLM_ROUTER_URL")
    if not url:
        return

    solo = arms_results.get("solo")
    mix = arms_results.get("mixture")
    if solo is None or mix is None or solo.error or mix.error:
        return

    winner = verdict_dict.get("winner", "skipped")
    verdict = _WINNER_TO_VERDICT.get(winner)  # None for skipped

    solo_in = int(solo.frontier.input_tokens or 0)
    mix_in = int(mix.frontier.input_tokens or 0)
    reduction = ((solo_in - mix_in) / solo_in * 100.0) if solo_in > 0 else None

    prompt = case.get("solo_prompt", "") or ""
    prompt_hash = "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    record = {
        "decision_id": f"exp_{run_id}_{case['id']}_{secrets.token_hex(4)}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_hash": prompt_hash,
        "prompt_text": prompt,
        "prompt_len_tok": _count_tokens(prompt),
        "decision_made": "mixture",
        "policy": "experiment",
        "head_version": "experiment",
        "outcome": {
            "solo_tokens_estimated": solo_in,
            "composer_tokens_actual": mix_in,
            "observed_reduction_pct": reduction,
            "wall_ms_slm_dag": int((mix.wall_seconds or 0.0) * 1000),
            "wall_ms_saved_vs_solo": int(((solo.wall_seconds or 0.0) - (mix.wall_seconds or 0.0)) * 1000),
            "quality_verdict": verdict,
            "quality_source": "reviewer_shadow",
        },
    }
    try:
        req = urllib.request.Request(
            url.rstrip("/") + "/feedback",
            data=json.dumps(record).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        token = os.environ.get("SLM_ROUTER_TOKEN")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        urllib.request.urlopen(req, timeout=2.0).read()
    except Exception as e:  # noqa: BLE001
        print(f"  [case {case['id']}] router feedback POST failed: {e}", file=sys.stderr)


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
    *,
    include_prism: bool = True,
    run_id: str = "",
) -> dict:
    """Run all arms concurrently, then review."""
    print(f"  [case {case['id']}] starting arms")
    arms_results: dict[str, ArmResult] = {}
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        futs = {
            "solo": pool.submit(run_solo, case, frontier),
            "mixture": pool.submit(run_mixture, case, frontier, slm),
        }
        if include_prism and case.get("prism_plan"):
            futs["prism"] = pool.submit(run_prism, case, frontier)
        for arm, fut in futs.items():
            arms_results[arm] = fut.result()

    times = ", ".join(f"{a} {r.wall_seconds:.1f}s" for a, r in arms_results.items())
    print(f"  [case {case['id']}] arms done — {times}")

    outputs_by_arm: dict[str, str] = {}
    arm_errors: dict[str, str | None] = {}
    for arm, res in arms_results.items():
        arm_errors[arm] = res.error
        if not res.error:
            outputs_by_arm[arm] = res.output

    if len(outputs_by_arm) < 2:
        verdict_dict = {
            "winner": "skipped",
            "reason": "fewer than two arms produced output: " + json.dumps(arm_errors),
        }
    else:
        verdict = review_nway(
            case=case,
            outputs_by_arm=outputs_by_arm,
            frontier=frontier,
            rng=rng,
        )
        verdict_dict = verdict.to_dict()
        print(f"  [case {case['id']}] reviewer → winner={verdict.winner}")

    case_record = {
        "case_id": case["id"],
        "name": case.get("name", case["id"]),
        "category": case.get("category"),
        "review": verdict_dict,
    }
    for arm, res in arms_results.items():
        case_record[arm] = res.to_dict()

    _post_router_feedback(case, arms_results, verdict_dict, run_id)

    return case_record


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
        case_results = [_run_one_case(c, frontier, slm, rng, run_id=run_id) for c in cases]
    else:
        with cf.ThreadPoolExecutor(max_workers=max_parallel_cases) as pool:
            futs = [pool.submit(_run_one_case, c, frontier, slm, rng, run_id=run_id) for c in cases]
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
    arms = ("solo", "mixture", "prism")

    def sum_f(arm: str, key: str) -> int:
        return sum(
            int((c.get(arm) or {}).get("frontier", {}).get(key, 0) or 0)
            for c in case_results
        )

    def sum_s(arm: str, key: str) -> int:
        return sum(
            int((c.get(arm) or {}).get("slm", {}).get(key, 0) or 0)
            for c in case_results
        )

    def sum_t(arm: str) -> float:
        return sum(
            float((c.get(arm) or {}).get("wall_seconds", 0.0) or 0.0)
            for c in case_results
        )

    winners = [c["review"].get("winner", "skipped") for c in case_results]
    win_counts = {
        arm: winners.count(arm) for arm in arms
    } | {"tie": winners.count("tie"), "skipped": winners.count("skipped")}

    reviewer_input = sum(
        int((c["review"].get("reviewer_usage", {}) or {}).get("input_tokens", 0) or 0)
        for c in case_results
    )
    reviewer_output = sum(
        int((c["review"].get("reviewer_usage", {}) or {}).get("output_tokens", 0) or 0)
        for c in case_results
    )

    arms_block = {}
    for arm in arms:
        if not any(arm in c for c in case_results):
            continue
        arms_block[arm] = {
            "frontier_input_tokens": sum_f(arm, "input_tokens"),
            "frontier_output_tokens": sum_f(arm, "output_tokens"),
            "slm_output_tokens": sum_s(arm, "output_tokens"),
            "wall_seconds": sum_t(arm),
        }

    return {
        "winners": win_counts,
        **arms_block,
        "reviewer": {
            "input_tokens": reviewer_input,
            "output_tokens": reviewer_output,
        },
    }
