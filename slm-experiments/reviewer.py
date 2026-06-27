"""Blind pairwise SxS reviewer.

A frontier-model judge receives both arm outputs labeled `A` and `B`
(order randomized per case so the reviewer can't tell which label maps
to which arm). It scores each on a fixed 1-5 rubric and picks a winner.

The rubric and the output-format scaffolding are stable across cases,
so we mark them as `cache_control: ephemeral` to get prompt-cache hits
on subsequent reviews in the same run.
"""
from __future__ import annotations

import json
import random
import re

from adapters.frontier import FrontierAdapter
from metrics import ReviewVerdict


RUBRIC_SYSTEM = [
    {
        "type": "text",
        "text": (
            "You are a strict, calibrated evaluator comparing two answers to "
            "the same prompt. Score each answer on a 1-5 integer scale across "
            "three dimensions:\n\n"
            "  - accuracy: factual correctness and faithfulness to the prompt's "
            "constraints. 5 = no errors; 1 = major errors or hallucinations.\n"
            "  - completeness: covers what the prompt asks for. 5 = fully "
            "addresses every part; 1 = misses the main ask.\n"
            "  - clarity: well-structured, concise, easy to read. 5 = excellent; "
            "1 = confusing or rambling.\n\n"
            "Then decide a winner. Prefer the more accurate answer; break ties on "
            "completeness, then clarity. Return `tie` only when the answers are "
            "genuinely indistinguishable on all three dimensions.\n\n"
            "Output STRICT JSON ONLY, no prose, in this exact shape:\n"
            "{\n"
            '  "scores": {\n'
            '    "A": {"accuracy": <int>, "completeness": <int>, "clarity": <int>},\n'
            '    "B": {"accuracy": <int>, "completeness": <int>, "clarity": <int>}\n'
            "  },\n"
            '  "winner": "A" | "B" | "tie",\n'
            '  "confidence": <float 0..1>,\n'
            '  "reasoning": "<<=3 short sentences>"\n'
            "}\n"
        ),
        "cache_control": {"type": "ephemeral"},
    }
]


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> dict:
    m = _JSON_BLOCK.search(text)
    if not m:
        raise ValueError(f"reviewer returned no JSON: {text!r}")
    return json.loads(m.group(0))


RUBRIC_SYSTEM_NWAY = [
    {
        "type": "text",
        "text": (
            "You are a strict, calibrated evaluator comparing N candidate "
            "answers to the same prompt, labeled A, B, C (and so on). "
            "Score each candidate on a 1-5 integer scale across three "
            "dimensions:\n\n"
            "  - accuracy: factual correctness and faithfulness to the prompt's "
            "constraints.\n"
            "  - completeness: covers what the prompt asks for.\n"
            "  - clarity: well-structured, concise, easy to read.\n\n"
            "Then pick the single best candidate. Prefer the most accurate; "
            "break ties on completeness, then clarity. Use `tie` only when "
            "two or more are truly indistinguishable.\n\n"
            "Output STRICT JSON ONLY, no prose, in this exact shape:\n"
            "{\n"
            '  "scores": {\n'
            '    "A": {"accuracy": <int>, "completeness": <int>, "clarity": <int>},\n'
            '    "B": {"accuracy": <int>, "completeness": <int>, "clarity": <int>},\n'
            '    ...\n'
            "  },\n"
            '  "winner": "A" | "B" | ... | "tie",\n'
            '  "confidence": <float 0..1>,\n'
            '  "reasoning": "<<=3 short sentences>"\n'
            "}\n"
        ),
        "cache_control": {"type": "ephemeral"},
    }
]


def review_nway(
    *,
    case: dict,
    outputs_by_arm: dict[str, str],
    frontier: FrontierAdapter,
    rng: random.Random | None = None,
) -> ReviewVerdict:
    """Multi-way blind ranking. Anonymizes arms as A/B/C/... in a randomized order."""
    rng = rng or random.Random()
    arms = list(outputs_by_arm.keys())
    rng.shuffle(arms)
    labels = [chr(ord("A") + i) for i in range(len(arms))]
    label_to_arm = dict(zip(labels, arms))

    sections = [
        f"---\nANSWER {label}:\n{outputs_by_arm[arm]}"
        for label, arm in zip(labels, arms)
    ]
    user = (
        f"PROMPT GIVEN TO ALL ANSWERS:\n{case['solo_prompt']}\n\n"
        + "\n\n".join(sections)
        + f"\n\nScore each of the {len(arms)} answers and pick the single best. "
        "Return the JSON object only."
    )

    call = frontier.complete(
        system=RUBRIC_SYSTEM_NWAY,
        user=user,
        max_tokens=900,
        temperature=0.0,
    )

    try:
        parsed = _parse_json(call.text)
    except (ValueError, json.JSONDecodeError) as e:
        return ReviewVerdict(
            winner="tie",
            confidence=0.0,
            scores={arm: {} for arm in arms} | {"_parse_error": str(e)},
            reasoning=f"reviewer JSON parse failed: {call.text[:300]}",
            reviewer_usage=call.usage,
        )

    raw_winner = parsed.get("winner", "tie")
    winner = label_to_arm.get(raw_winner, "tie") if raw_winner in label_to_arm else "tie"

    scores = {label_to_arm[label]: parsed.get("scores", {}).get(label, {}) for label in labels}

    return ReviewVerdict(
        winner=winner,
        confidence=float(parsed.get("confidence", 0.0) or 0.0),
        scores=scores,
        reasoning=str(parsed.get("reasoning", "")),
        reviewer_usage=call.usage,
    )


def review(
    *,
    case: dict,
    solo_output: str,
    mixture_output: str,
    frontier: FrontierAdapter,
    rng: random.Random | None = None,
) -> ReviewVerdict:
    rng = rng or random.Random()
    flip = rng.random() < 0.5  # 50% of the time A=mixture, B=solo
    if flip:
        a_text, b_text = mixture_output, solo_output
        a_label, b_label = "mixture", "solo"
    else:
        a_text, b_text = solo_output, mixture_output
        a_label, b_label = "solo", "mixture"

    user = (
        f"PROMPT GIVEN TO BOTH ANSWERS:\n{case['solo_prompt']}\n\n"
        f"---\nANSWER A:\n{a_text}\n\n"
        f"---\nANSWER B:\n{b_text}\n\n"
        "Score A and B and pick a winner. Return the JSON object only."
    )

    call = frontier.complete(
        system=RUBRIC_SYSTEM,
        user=user,
        max_tokens=600,
        temperature=0.0,
    )

    try:
        parsed = _parse_json(call.text)
    except (ValueError, json.JSONDecodeError) as e:
        return ReviewVerdict(
            winner="tie",
            confidence=0.0,
            scores={"solo": {}, "mixture": {}, "_parse_error": str(e)},
            reasoning=f"reviewer JSON parse failed: {call.text[:300]}",
            reviewer_usage=call.usage,
        )

    label_for = {"A": a_label, "B": b_label}
    raw_winner = parsed.get("winner", "tie")
    winner = label_for.get(raw_winner, "tie") if raw_winner in ("A", "B") else "tie"

    scores = {
        a_label: parsed.get("scores", {}).get("A", {}),
        b_label: parsed.get("scores", {}).get("B", {}),
    }

    return ReviewVerdict(
        winner=winner,
        confidence=float(parsed.get("confidence", 0.0) or 0.0),
        scores=scores,
        reasoning=str(parsed.get("reasoning", "")),
        reviewer_usage=call.usage,
    )
