"""Live side-by-side token audit using the real Anthropic API.

The tiktoken-based validator (autoroute_validate.py) is a *model* — it
estimates token counts and dollar cost from prompts and expected outputs.
This script is the *ground truth*: it fires actual `messages.create` calls
against `claude-sonnet-4-6` and reads the `usage` field back from Anthropic.

For each hit prompt in the test set we make two calls per model:

  1. SOLO: user prompt only, no tools. Whatever the model writes IS the
     answer. Record `usage.input_tokens` and `usage.output_tokens`.

  2. MIXTURE v3: user prompt + the exact `additionalContext` block the
     auto-route hook would inject, plus `slm_run_stashed` and
     `slm_wait_plan` as tool schemas. The model calls the tools; we mock
     the tool results with deterministic strings sized to match what real
     SLMs would produce. Record total usage across all turns.

Both runs use fresh conversations (each `messages.create` is independent),
so there's no cross-contamination between paths — this is the SxS the
user asked for.

Requires:
  - ANTHROPIC_API_KEY in the environment
  - `pip install --user anthropic`

Cost estimate for a full run of 9 hit prompts x 2 paths = 18 calls at
Sonnet 4.6 rates: ~$0.05-0.15 total depending on outputs. Won't break
the bank but not free.

Usage:
  ANTHROPIC_API_KEY=... python3 slm-experiments/autoroute_sxs_live.py
  ANTHROPIC_API_KEY=... python3 slm-experiments/autoroute_sxs_live.py --n 3   # subset
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "slm-router"))
from decompose import decompose  # noqa: E402

import anthropic  # noqa: E402

# ---- Config ------------------------------------------------------------
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048
PRICE_IN_PER_M = 3.00
PRICE_OUT_PER_M = 15.00


def dollars(in_toks: int, out_toks: int) -> float:
    return in_toks * PRICE_IN_PER_M / 1e6 + out_toks * PRICE_OUT_PER_M / 1e6


# ---- Test prompts (same as autoroute_validate.py hits) ----------------
TEST_PROMPTS: list[tuple[str, str]] = [
    # (label, prompt) — only positive-rule prompts; solo-vs-mixture is meaningless on rejects.
    ("variant.6tweets", "write six variants of a launch tweet for a new SaaS product"),
    ("variant.10puns", "come up with 10 pun-based titles about databases"),
    ("variant.8hooks", "give me 8 different opening hooks for a wedding toast"),
    ("foreach.brewing.4",
     "for each of the following bullets, write a one-paragraph description:\n"
     "- espresso\n- pour-over\n- french press\n- aeropress"),
    ("foreach.desk.4",
     "for each item below, output a one-line sales pitch:\n"
     "1. adjustable standing desk\n2. mechanical keyboard\n3. curved monitor\n4. ergonomic chair"),
    ("foreach.tickers.5",
     "for every ticker in this list, give me a one-line 2024 recap:\n"
     "- AAPL\n- MSFT\n- GOOGL\n- AMZN\n- NVDA"),
    ("summarize.3paras",
     "summarize each of the following paragraphs:\n\n"
     "This first paragraph is a fairly long block of prose that runs across several sentences to cross the eighty character floor and read as substantive text worth summarizing.\n\n"
     "The second paragraph likewise contains multiple sentences with concrete content and enough characters to clear the length filter used by the decomposer.\n\n"
     "A third block of prose again clocks over eighty characters and represents another chunk of text the SLM pool could each summarize independently."),
    ("summarize.4bullets",
     "tl;dr each of these:\n"
     "- Bullet one is verbose enough to actually be summarized and includes a full sentence.\n"
     "- Bullet two also carries enough prose to warrant a summary of its own.\n"
     "- Bullet three has enough content that the SLM will actually condense something.\n"
     "- Bullet four rounds this out as another list entry with real content."),
]


# ---- Injected context (mirrors plugin/hooks/autoroute.py verbatim) ----
def render_autoroute_context(stash_id: str, n: int, rule: str, description: str) -> str:
    return (
        f"[autoroute] slm-router matched rule={rule}; "
        f"{n} SLM leaves already stashed as `{stash_id}` (description: {description}).\n"
        f"DIRECTIVE:\n"
        f"  1. Call `slm_run_stashed(stash_id=\"{stash_id}\")` — returns run_id.\n"
        f"  2. Call `slm_wait_plan(run_id=<id>)` — returns node results.\n"
        f"  3. The tool output above is ALREADY VISIBLE to the user in the "
        f"transcript. Do NOT reproduce, summarize, quote, reformat, or "
        f"paraphrase any part of it — that would double the tokens.\n"
        f"  4. Your ENTIRE response after the tool calls MUST be exactly "
        f"one line:\n"
        f"     `(delegated to {n} local SLM leaves — results above)`\n"
        f"     No preamble, no closing, no commentary.\n"
        f"If the decomposition is materially wrong for the user's intent, "
        f"skip the tool calls and answer normally without mentioning this hint.\n"
    )


TOOL_SCHEMAS = [
    {
        "name": "slm_run_stashed",
        "description": "Trigger a plan the auto-route hook already stashed. Pass the stash_id from the [autoroute] context.",
        "input_schema": {
            "type": "object",
            "properties": {"stash_id": {"type": "string"}},
            "required": ["stash_id"],
        },
    },
    {
        "name": "slm_wait_plan",
        "description": "Block until a plan run completes and return the full snapshot with per-node results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "timeout_s": {"type": "number", "default": 180.0},
            },
            "required": ["run_id"],
        },
    },
]


# ---- SLM output mocking -----------------------------------------------
# Sizes calibrated to what smollm2/gemma2 would actually produce for
# these prompts (verified against slm-bench numbers).
MOCK_ITEM_SIZES = {
    "multi_variant": 12,   # ~one short line each
    "for_each":      70,   # a real paragraph/pitch
    "summarize":     40,   # 1-2 sentence summary
}


def _mock_content(n_tokens: int, seed: str) -> str:
    """Generate deterministic filler content of roughly n_tokens tokens."""
    # ~4 chars per token; use readable filler so it looks plausible.
    filler_word = "wombat "  # 7 chars incl. space -> ~1.75 tokens
    words_needed = max(3, int(n_tokens / 1.75))
    words = [f"{seed}-mock"] + [filler_word.strip()] * (words_needed - 1)
    return " ".join(words) + "."


def mock_slm_wait_response(plan_nodes: list[dict], rule: str) -> dict:
    """What the queue would return from GET /plans/<run_id> once done."""
    per_item = MOCK_ITEM_SIZES.get(rule, 40)
    nodes: dict[str, dict] = {}
    node_order: list[str] = []
    for i, node in enumerate(plan_nodes, start=1):
        nid = node["id"]
        node_order.append(nid)
        nodes[nid] = {
            "status": "done",
            "model": "smollm2:1.7b",
            "worker": "w1",
            "result": _mock_content(per_item, f"leaf{i}"),
            "eval_count": per_item,
            "elapsed_ms": 850,
        }
    return {
        "run_id": "mock-run-abc",
        "plan_id": "mock",
        "status": "done",
        "node_order": node_order,
        "nodes": nodes,
    }


# ---- Single-prompt SxS runner -----------------------------------------
@dataclass
class RunResult:
    label: str
    n_nodes: int
    rule: str
    solo_in: int
    solo_out: int
    mix_in: int
    mix_out: int
    mix_tool_turns: int
    mix_final_text: str


def _run_solo(client: anthropic.Anthropic, prompt: str) -> tuple[int, int, str]:
    r = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = ""
    for block in r.content:
        if getattr(block, "type", None) == "text":
            text += block.text
    return r.usage.input_tokens, r.usage.output_tokens, text


def _run_mixture(
    client: anthropic.Anthropic,
    prompt: str,
    plan: dict,
    rule: str,
) -> tuple[int, int, int, str]:
    """Run a full tool-use loop: hook context -> tool calls -> mocked results -> final."""
    stash_id = "auto-mocklive01"
    ctx = render_autoroute_context(
        stash_id=stash_id,
        n=len(plan["nodes"]),
        rule=rule,
        description=plan.get("description", ""),
    )
    # Structured message: the hook's context is injected as an assistant-visible
    # system-style block. We put it in the user turn to match how Claude Code
    # actually delivers `additionalContext`.
    user_content = f"<autoroute_context>\n{ctx}\n</autoroute_context>\n\n{prompt}"
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]

    total_in = 0
    total_out = 0
    tool_turns = 0
    final_text = ""

    for step in range(6):  # tight cap so we can't loop forever
        r = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        total_in += r.usage.input_tokens
        total_out += r.usage.output_tokens

        if r.stop_reason != "tool_use":
            for block in r.content:
                if getattr(block, "type", None) == "text":
                    final_text += block.text
            break

        # Handle each tool_use block, produce a matching tool_result.
        tool_turns += 1
        assistant_blocks: list[dict[str, Any]] = []
        for block in r.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                assistant_blocks.append({"type": "text", "text": block.text})
            elif btype == "tool_use":
                assistant_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        messages.append({"role": "assistant", "content": assistant_blocks})

        tool_results = []
        for block in r.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            if block.name == "slm_run_stashed":
                result = {"run_id": "mock-run-abc", "plan_id": "mock"}
            elif block.name == "slm_wait_plan":
                result = mock_slm_wait_response(plan["nodes"], rule)
            else:
                result = {"error": f"unknown tool {block.name}"}
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })
        messages.append({"role": "user", "content": tool_results})

    return total_in, total_out, tool_turns, final_text


def run_one(client: anthropic.Anthropic, label: str, prompt: str) -> RunResult | None:
    plan, rule = decompose(prompt)
    if plan is None:
        print(f"  [{label}] decomposer rejected — skipping")
        return None
    n_nodes = len(plan["nodes"])

    t0 = time.perf_counter()
    solo_in, solo_out, solo_text = _run_solo(client, prompt)
    solo_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    mix_in, mix_out, mix_turns, mix_final = _run_mixture(client, prompt, plan, rule)
    mix_ms = (time.perf_counter() - t0) * 1000

    print(
        f"  [{label:<22}] rule={rule:<14} N={n_nodes:>2}  "
        f"solo: in={solo_in:>5} out={solo_out:>5} ({solo_ms:>5.0f}ms)  "
        f"mix: in={mix_in:>6} out={mix_out:>5} turns={mix_turns} ({mix_ms:>5.0f}ms)"
    )
    return RunResult(
        label=label, n_nodes=n_nodes, rule=rule,
        solo_in=solo_in, solo_out=solo_out,
        mix_in=mix_in, mix_out=mix_out,
        mix_tool_turns=mix_turns, mix_final_text=mix_final,
    )


# ---- Reporting ---------------------------------------------------------
def _fmt_pct(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x*100:.0f}%"


def report(results: list[RunResult]) -> None:
    if not results:
        print("no successful runs.")
        return
    print()
    print("=" * 80)
    print(f"Live SxS results (model={MODEL})")
    print("=" * 80)
    print(f"  Pricing used: ${PRICE_IN_PER_M}/M in, ${PRICE_OUT_PER_M}/M out")
    print()
    hdr = (
        f"  {'label':<22}{'N':>3}  "
        f"{'solo_in':>7}{'solo_out':>8}{'mix_in':>7}{'mix_out':>7}  "
        f"{'solo_$':>9}{'mix_$':>9}{'Δ%':>7}{'verdict':>8}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    tot_solo = tot_mix = 0.0
    wins = losses = 0
    complied = compliance_checked = 0
    for r in results:
        solo_usd = dollars(r.solo_in, r.solo_out)
        mix_usd = dollars(r.mix_in, r.mix_out)
        d = (mix_usd - solo_usd) / solo_usd if solo_usd else 0
        marker = "WIN" if mix_usd < solo_usd else "LOSS"
        if mix_usd < solo_usd:
            wins += 1
        else:
            losses += 1
        # Did the model comply with the "one-line ack" directive?
        compliance_checked += 1
        final_stripped = r.mix_final_text.strip()
        if "delegated" in final_stripped.lower() and len(final_stripped) < 120:
            complied += 1
        print(
            f"  {r.label:<22}{r.n_nodes:>3}  "
            f"{r.solo_in:>7}{r.solo_out:>8}{r.mix_in:>7}{r.mix_out:>7}  "
            f"${solo_usd*1000:>7.2f}m${mix_usd*1000:>7.2f}m{_fmt_pct(d):>7}{marker:>8}"
        )
        tot_solo += solo_usd
        tot_mix += mix_usd
    print("  " + "-" * (len(hdr) - 2))
    d_tot = (tot_mix - tot_solo) / tot_solo if tot_solo else 0
    print(
        f"  {'TOTALS':<22}{'':>3}  {'':>7}{'':>8}{'':>7}{'':>7}  "
        f"${tot_solo*1000:>7.2f}m${tot_mix*1000:>7.2f}m{_fmt_pct(d_tot):>7}"
    )
    print()
    print(f"  wins: {wins}   losses: {losses}   overall: {_fmt_pct(d_tot)}")
    print(f"  directive compliance: {complied}/{compliance_checked} runs "
          f"({100*complied/compliance_checked:.0f}%) followed the one-line-ack rule")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(TEST_PROMPTS),
                    help="run only the first N prompts (cheaper smoke test)")
    ap.add_argument("--model", default=MODEL, help="model id")
    args = ap.parse_args()

    global MODEL
    MODEL = args.model

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 2

    client = anthropic.Anthropic()
    print(f"Running live SxS against {MODEL}. This makes real API calls.")
    print(f"Test set: {min(args.n, len(TEST_PROMPTS))} prompts x 2 paths.")
    print()
    results: list[RunResult] = []
    for label, prompt in TEST_PROMPTS[: args.n]:
        try:
            r = run_one(client, label, prompt)
        except anthropic.APIError as e:
            print(f"  [{label}] API error: {type(e).__name__}: {e}")
            continue
        if r is not None:
            results.append(r)

    report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
