"""Seed leaf_heads/v0.joblib from choose_model's keyword-rule priors.

The queue's existing `choose_model()` (slm-queue/router.py) picks an SLM
by keyword: code→qwen2.5, math→llama3.2, long→gemma2, default→smollm2.
Those rules encode a decent prior over which SLM will handle a given
prompt efficiently. We turn that prior into a small supervised dataset:

  - synthesize ~10 archetypal prompts per category
  - for each prompt, label every SLM with an "expected output tokens"
    number: LOW for the SLM the rules would prefer, HIGH for the rest
  - fit one LinearRegression per SLM

The policy (argmin predicted output tokens) then reproduces the keyword
rules on cold prompts, and once real feedback lands the heads retrain
against reality.

Rerun any time slm-deploy/slms.yaml changes the deployed set.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.linear_model import Ridge

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from model import (  # noqa: E402
    Encoder,
    LeafHead,
    LeafHeadMetadata,
    head_pointer_write,
)
from paths import ensure_data_dir, resolve_data_path  # noqa: E402


LOW_TOKENS = 80.0      # a fluent, on-task response
HIGH_TOKENS = 400.0    # rambling / off-task / mismatched
NOISE_STD = 20.0


CATEGORY_PROMPTS: dict[str, list[str]] = {
    "code": [
        "Write a Python function that reverses a linked list in place.",
        "Explain what this JavaScript closure does: `const f = (x) => (y) => x + y;`",
        "Refactor this SQL query to eliminate the correlated subquery.",
        "Convert this Bash pipeline to a POSIX-compliant shell script.",
        "Write a regex that matches a valid email address.",
        "Debug this TypeScript compilation error: cannot find name 'foo'.",
        "Implement a binary search tree in Rust with insert and lookup.",
        "Explain how this Python decorator caches its return value.",
        "Refactor the function to reduce cyclomatic complexity.",
        "Write a small React component that displays a countdown timer.",
    ],
    "math": [
        "Calculate 12! + 5^3.",
        "Solve the equation: 3x + 5 = 20.",
        "Compute the derivative of x^2 * sin(x).",
        "Prove that the sum of the first n odd numbers is n^2.",
        "Compute the integral of e^(-x^2) from 0 to infinity.",
        "Solve for x: log_2(x) + log_2(x - 6) = 4.",
        "Compute the eigenvalues of the matrix [[2, 1], [1, 2]].",
        "Solve the system: 2x + 3y = 12, x - y = 1.",
        "Compute the probability of drawing 2 aces from a 52-card deck.",
        "Solve the recurrence T(n) = 2 T(n/2) + n with T(1)=1.",
    ],
    "long": [
        # These strings are intentionally >500 chars so the keyword rule
        # (len > 500) fires; content itself is filler-esque on purpose.
        "Summarize the following document. " + ("The report discusses "
         "renewable energy adoption across seventeen jurisdictions, drawing "
         "on regulatory filings, industrial output statistics, and stakeholder "
         "interviews. Each jurisdiction is analyzed along five axes: policy "
         "incentives, grid readiness, capital availability, workforce, and "
         "public acceptance. The synthesis concludes with a comparative "
         "scorecard and recommendations for policymakers considering similar "
         "programs. ") * 2,
        "Read the transcript below and produce action items. " + ("Speaker A "
         "opened the meeting by revisiting the OKRs for the current quarter "
         "and flagged two that were at risk of missing target. Speaker B "
         "responded with concrete mitigation proposals covering staffing, "
         "roadmap adjustments, and stakeholder communication. Speaker C "
         "raised a related concern about downstream dependencies and asked "
         "whether the migration project should be re-scoped. ") * 3,
        "Given the passage that follows, extract every proper noun and its "
         "role. " + ("Dr. Elena Sokolova, chief scientist at Meridian Labs, "
         "collaborated with Professor Aditya Rao of the Bangalore Institute "
         "on a joint paper submitted to the Journal of Applied Physics. The "
         "work built on earlier findings by Chen and Watanabe (2021), and "
         "was supported by grants from the Novak Foundation and the Fritz "
         "Consortium. Reviewers included Dr. Yara Adebayo and Prof. Mikhail "
         "Zabolotin. ") * 3,
        "Compare and contrast the three product proposals below along cost, "
         "feasibility, and market fit. " + ("Proposal Alpha focuses on a "
         "consumer-grade subscription with a low ARPU and broad reach. "
         "Proposal Beta targets mid-market SaaS with an annual contract and "
         "usage-based upsell. Proposal Gamma is an enterprise deal built on "
         "custom integrations, a dedicated CSM, and multi-year commitments. "
         "Each proposal implies a distinct go-to-market motion, capital "
         "profile, and organizational shape. ") * 3,
        "Read the incident postmortem below and rewrite it as a customer-"
         "facing status update. " + ("At 03:14 UTC our primary database "
         "cluster began emitting elevated replication lag, and by 03:22 UTC "
         "reads were serving stale data to a subset of tenants. Root cause "
         "was a mis-tuned autovacuum policy interacting with a schema "
         "migration that landed the previous evening. Impact was limited to "
         "the analytics dashboard; transactional writes were unaffected. "
         "Full mitigation was in place by 04:10 UTC and normal service "
         "resumed by 04:47 UTC. ") * 2,
        "Analyze the following user research notes and identify the top "
         "three usability issues in priority order. " + ("Participant 1 "
         "struggled to locate the export button and eventually gave up, "
         "using a workaround via the URL. Participant 2 confused the "
         "'archive' action with 'delete' and had to be reassured that "
         "her work was recoverable. Participant 3 found the onboarding "
         "flow abrupt and could not identify next steps after finishing "
         "the initial tutorial. Participant 4 was unable to discover the "
         "keyboard shortcut palette. ") * 2,
        "Given the meeting agenda that follows, produce a two-paragraph "
         "briefing note for a stakeholder who cannot attend. " + ("The "
         "agenda covers Q3 hiring plans, an update from Platform on the "
         "auth-service split, a proposal from Growth to sunset the free "
         "trial in favor of a limited-feature perpetual free tier, and a "
         "read-out from Legal on the new EU regulation. Each agenda item "
         "has a designated owner and a target decision. ") * 3,
        "Summarize this legal contract's key obligations, risks, and "
         "termination triggers. " + ("The agreement between the Provider "
         "and the Customer sets out a three-year term with automatic "
         "annual renewal absent written notice ninety days prior. The "
         "Provider is obligated to maintain 99.95% monthly availability, "
         "and the Customer is obligated to prompt payment within thirty "
         "days of invoice. Termination for cause requires a cure period "
         "of thirty days after written notice; termination for convenience "
         "is available to either party subject to a fee schedule. ") * 3,
    ],
    "generic": [
        "What is the capital of France?",
        "Who painted the Mona Lisa?",
        "How many continents are there?",
        "What year did the Berlin Wall fall?",
        "Name three primary colors.",
        "Convert 30 degrees Celsius to Fahrenheit.",
        "What language is spoken in Brazil?",
        "Who wrote 'To Kill a Mockingbird'?",
        "Summarize this sentence: The quick brown fox jumps over the lazy dog.",
        "What is the boiling point of water at sea level in Celsius?",
    ],
}


# Which SLM prefix each category maps to under the current keyword rules.
CATEGORY_PREFERRED_PREFIX: dict[str, str] = {
    "code": "qwen2.5",
    "math": "llama3.2",
    "long": "gemma2",
    "generic": "smollm2",
}


def _match_prefix(available: list[str], prefix: str) -> str | None:
    for m in available:
        if m.startswith(prefix):
            return m
    return None


def _preferred_for(category: str, available: list[str]) -> str:
    prefix = CATEGORY_PREFERRED_PREFIX[category]
    m = _match_prefix(available, prefix)
    if m:
        return m
    # If the exact preferred SLM isn't deployed, fall back to the first in
    # `available` so bootstrap still fits something coherent.
    return available[0]


def _synthesize(available: list[str], seed: int) -> tuple[list[str], dict[str, np.ndarray]]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    prompts: list[str] = []
    preferred: list[str] = []
    for cat, pool in CATEGORY_PROMPTS.items():
        pref = _preferred_for(cat, available)
        for p in pool:
            prompts.append(p)
            preferred.append(pref)

    # Shuffle the pairs together so the fit isn't order-biased.
    pairs = list(zip(prompts, preferred))
    rng.shuffle(pairs)
    prompts, preferred = [list(t) for t in zip(*pairs)]

    per_slm_targets: dict[str, np.ndarray] = {}
    for m in available:
        y = np.array(
            [LOW_TOKENS if pref == m else HIGH_TOKENS for pref in preferred],
            dtype=np.float32,
        )
        y = y + np_rng.normal(0.0, NOISE_STD, size=y.shape).astype(np.float32)
        y = np.clip(y, 10.0, None)  # never predict fewer than 10 tokens
        per_slm_targets[m] = y
    return prompts, per_slm_targets


def bootstrap(config: dict[str, Any], seed: int, alpha: float) -> dict[str, Any]:
    ensure_data_dir(config)
    paths_cfg = config.get("paths") or {}
    leaf_dir = resolve_data_path(config, paths_cfg.get("leaf_heads_dir", "leaf_heads"))
    pointer = resolve_data_path(config, paths_cfg.get("leaf_head_pointer", "leaf_heads/HEAD"))

    available = (config.get("leaf") or {}).get("available_slms") or []
    if not available:
        raise RuntimeError("config missing leaf.available_slms")

    prompts, per_slm_targets = _synthesize(available, seed)

    encoder = Encoder()
    X = encoder.encode_batch(prompts)

    regressors: dict[str, Any] = {}
    for m in available:
        reg = Ridge(alpha=alpha, random_state=seed).fit(X, per_slm_targets[m])
        regressors[m] = reg

    meta = LeafHeadMetadata(
        version="v0",
        models=list(available),
        n_train=len(prompts),
        notes="cold-start bootstrap from keyword-rule priors",
    )
    head = LeafHead(regressors=regressors, metadata=meta)

    out_path = leaf_dir / "v0.joblib"
    head.save(out_path)
    head_pointer_write(pointer, "v0")

    return {
        "n_train": len(prompts),
        "models": list(available),
        "out_path": str(out_path),
        "pointer": str(pointer),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_HERE / "config.yaml")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization")
    args = ap.parse_args()

    config = yaml.safe_load(args.config.read_text())
    summary = bootstrap(config, args.seed, args.alpha)
    print(
        f"bootstrapped leaf {summary['out_path']}: "
        f"n_train={summary['n_train']} models={summary['models']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
