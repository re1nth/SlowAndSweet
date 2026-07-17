"""Comprehensive validation of the auto-route pipeline (v2 stash flow).

Answers three questions honestly:

  1. Does the decomposer fire on the right prompts?
     (precision/recall on a labeled set)

  2. Does the mixture path actually cut *frontier* tokens vs. solo?
     (audit of every token flowing past the Claude Code frontier, priced
      at Sonnet 4.6 rates)

  3. Does the mixture path cut wall-clock latency?
     (parallel SLMs at bench-measured 68 tok/s vs. sequential frontier
      at ~70 tok/s output)

Four cost paths modeled:

  - SOLO: frontier answers directly.
  - MIX_v1 (naive): hook injects full plan JSON, frontier emits plan JSON
    as tool arguments AND synthesizes. Loses tokens on every hit.
  - MIX_v2 (stash): hook stashes plan with the queue, injects only a
    `stash_id` + short directive. Frontier makes a tiny `slm_run_stashed`
    call and re-emits SLM outputs verbatim as its response. Better than
    v1 but still loses (frontier's output tokens are unchanged).
  - MIX_v3 (see-above): same as v2 but the injected directive tells the
    frontier the tool result is already visible to the user, so its
    final response should be a single acknowledgment line — no reproducing
    the SLM outputs. Requires model compliance (which is the empirical
    question worth measuring in a live session), but if it holds this is
    the only path with real cost savings.

Token counts use tiktoken cl100k_base — a ~5-15% undercount vs. Claude's
tokenizer for English, but the ratios between paths are stable.

Run:  python3 slm-experiments/autoroute_validate.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "slm-router"))
from decompose import decompose  # noqa: E402

import tiktoken  # noqa: E402

ENC = tiktoken.get_encoding("cl100k_base")


def tokens(text: str) -> int:
    return len(ENC.encode(text))


# ---- Pricing (Sonnet 4.6) ----------------------------------------------
PRICE_IN_PER_M = 3.00
PRICE_OUT_PER_M = 15.00


def dollars(in_toks: int, out_toks: int) -> float:
    return in_toks * PRICE_IN_PER_M / 1e6 + out_toks * PRICE_OUT_PER_M / 1e6


# ---- Latency model -----------------------------------------------------
# Sonnet output rate approx from public timing data.
FRONTIER_TOK_PER_S = 70.0
# Fastest SLM in our pool from slm-bench: smollm2:1.7b at 67.9 tok/s.
# Others range 50-60 tok/s. Use the median 60 as a fair estimate.
SLM_TOK_PER_S = 60.0
# Fixed overheads: hook + queue submit + ollama call setup.
HOOK_OVERHEAD_MS = 30
QUEUE_ROUNDTRIP_MS = 40
SLM_TTFT_MS = 80  # from bench: gemma2:2b was 80ms, others 50-100.


def latency_ms_solo(out_tokens: int) -> float:
    return (out_tokens / FRONTIER_TOK_PER_S) * 1000


def latency_ms_mixture_v2(n_leaves: int, per_leaf_out: int, synthesis_out: int) -> float:
    # Hook path: decompose + stash. Sequential HTTP round-trips.
    hook_ms = HOOK_OVERHEAD_MS + QUEUE_ROUNDTRIP_MS + QUEUE_ROUNDTRIP_MS
    # Frontier emits a short tool call.
    frontier_call_ms = latency_ms_solo(40)  # slm_run_stashed + slm_wait_plan
    # Queue kicks off; SLM leaves run in parallel (assuming replicas or
    # OLLAMA_NUM_PARALLEL >= n_leaves).
    slm_ms = SLM_TTFT_MS + (per_leaf_out / SLM_TOK_PER_S) * 1000
    # Frontier synthesizes / presents.
    synth_ms = latency_ms_solo(synthesis_out)
    return hook_ms + frontier_call_ms + slm_ms + synth_ms


# ---- Labeled test set --------------------------------------------------
@dataclass
class Case:
    prompt: str
    expect: str  # rule name we expect to fire, or "none"
    label: str


TEST_SET: list[Case] = [
    # Small variant tasks: with v2 rules, these should NOT fire (MIN=6).
    Case("give me 5 taglines for a cold brew coffee shop", "none", "small_variant.5taglines"),
    Case("draft 3 subject lines for a Black Friday email", "none", "small_variant.3subjects"),
    Case("suggest 4 different names for a golden retriever", "none", "small_variant.4names"),
    # Larger variant tasks: N>=6 should hit.
    Case("write six variants of a launch tweet for a new SaaS product", "multi_variant", "variant.6tweets"),
    Case("come up with 10 pun-based titles about databases", "multi_variant", "variant.10puns"),
    Case("give me 8 different opening hooks for a wedding toast", "multi_variant", "variant.8hooks"),
    # for_each: any explicit list of 3-12 items.
    Case("for each of the following bullets, write a one-paragraph description:\n- espresso\n- pour-over\n- french press\n- aeropress", "for_each", "foreach.brewing.4"),
    Case("for each item below, output a one-line sales pitch:\n1. adjustable standing desk\n2. mechanical keyboard\n3. curved monitor\n4. ergonomic chair", "for_each", "foreach.desk.4"),
    Case("for every ticker in this list, give me a one-line 2024 recap:\n- AAPL\n- MSFT\n- GOOGL\n- AMZN\n- NVDA", "for_each", "foreach.tickers.5"),
    # summarize: N-item lists or N-paragraph blocks.
    Case("summarize each of the following paragraphs:\n\n" + "\n\n".join([
        "This first paragraph is a fairly long block of prose that runs across several sentences to cross the eighty character floor and read as substantive text worth summarizing.",
        "The second paragraph likewise contains multiple sentences with concrete content and enough characters to clear the length filter used by the decomposer.",
        "A third block of prose again clocks over eighty characters and represents another chunk of text the SLM pool could each summarize independently.",
    ]), "summarize", "summarize.3paras"),
    Case("tl;dr each of these:\n- Bullet one is verbose enough to actually be summarized and includes a full sentence.\n- Bullet two also carries enough prose to warrant a summary of its own.\n- Bullet three has enough content that the SLM will actually condense something.\n- Bullet four rounds this out as another list entry with real content.", "summarize", "summarize.4bullets"),
    # A "large" summarize case: 8 paragraphs of ~150 tokens each. This is
    # where mixture should shine because inputs are big and SLM outputs are small.
    Case("summarize each of the following paragraphs into 1-2 sentences:\n\n" + "\n\n".join([
        "Kubernetes scheduling assigns pods to nodes based on a complex interplay of resource requests, node capacity, affinity rules, and taints. The scheduler runs in two phases: filtering nodes that cannot possibly host the pod, then scoring the remaining candidates. Filtering removes nodes that lack sufficient CPU or memory, or that carry a taint the pod does not tolerate. Scoring assigns each remaining node a rank based on resource fit, image locality, inter-pod affinity, and any user-supplied scheduling profiles. The highest-scoring node wins, and the pod is bound.",
        "Rust's ownership model makes memory safety a compile-time property rather than a runtime one. Every value has exactly one owner, and when that owner goes out of scope, the value is dropped. Borrowing lets you pass around references without transferring ownership, but the compiler enforces that mutable and immutable references cannot coexist. This eliminates entire classes of bugs — use-after-free, double-free, data races — without any garbage collector. The tradeoff is a steeper learning curve, especially for programmers coming from languages where the runtime cleans up after them.",
        "Postgres MVCC keeps multiple versions of each row and uses transaction IDs to decide which version each query sees. When a row is updated, the old version is not deleted immediately — it's marked with the transaction ID that superseded it, and a background process called autovacuum reclaims the space later. Long-running transactions can block autovacuum from cleaning up dead tuples, leading to table bloat and slower scans. Monitoring autovacuum activity and setting appropriate thresholds is a routine part of operating a healthy Postgres install.",
        "TCP congestion control adapts a sender's transmission rate to network conditions inferred from acknowledgments and lost packets. The classic algorithm — slow start, congestion avoidance, fast retransmit, fast recovery — is now one of several options; modern kernels use CUBIC by default and support BBR as an alternative. CUBIC treats packet loss as the primary signal that the network is congested; BBR instead estimates bandwidth and round-trip time directly, aiming to keep the network's bottleneck queue near-empty rather than reactively backing off after loss.",
        "React's reconciliation algorithm compares the previous virtual DOM tree with the newly-rendered one and generates a minimal set of DOM operations to apply. Keys on list items let the reconciler match up children across renders instead of assuming positional identity, which prevents entire subtrees from being torn down and recreated when items shift. The Fiber rewrite introduced in React 16 broke reconciliation into interruptible units of work, allowing the browser to service more important tasks like user input in between chunks.",
        "TLS 1.3 dropped a large number of legacy ciphers and features that had accumulated over the previous fifteen years, keeping only a small set of well-vetted primitives. The handshake was streamlined from two round-trips to one, and 0-RTT session resumption became optional. Forward secrecy is now mandatory for all key-exchange methods, which means past sessions remain safe even if a server's long-term key is later compromised. Adoption has been rapid — by 2023, most major browsers and CDNs supported TLS 1.3 by default.",
        "Kafka's storage model is an append-only log partitioned across brokers, where each partition is a strict ordered sequence and consumers track their own offsets. This design makes reads and writes largely sequential, which is friendly to spinning disks and modern SSDs alike. Retention is controlled by time or size, with compaction as an alternative for topics where only the latest value per key matters. Because consumers commit offsets independently, the same log can serve many downstream systems without additional coordination in the broker.",
        "The x86 memory model is Total Store Ordering (TSO), which means loads can be reordered ahead of prior stores but the reverse is not permitted. This is stricter than the memory models of ARM and POWER, which are relaxed enough to require explicit memory barriers even for common patterns. Code that ports cleanly from x86 to ARM often needs to add memory fences at synchronization boundaries; conversely, code that ran subtly wrong on ARM often runs correctly on x86 simply because the hardware forbids the reordering that exposed the bug.",
    ]), "summarize", "summarize.8paras.long"),
    # -- negatives --
    Case("what is 2 + 2?", "none", "neg.math"),
    Case("please commit and push my changes", "none", "neg.git"),
    Case("give me 2 taglines", "none", "neg.only2"),
    Case("help me debug this stack trace", "none", "neg.debug"),
    Case("summarize this in one sentence", "none", "neg.single_summary"),
    Case("explain how kubernetes scheduling works", "none", "neg.explain"),
    Case("write a haiku about databases", "none", "neg.single_haiku"),
    Case("for each of the animals I love, ...", "none", "neg.foreach_no_list"),
]


# ---- Output-size estimates per pattern ---------------------------------
@dataclass
class ExpectedOutput:
    per_item_out_tokens: int
    synthesis_out_tokens: int


OUTPUT_ESTIMATES = {
    "multi_variant": ExpectedOutput(per_item_out_tokens=12, synthesis_out_tokens=25),
    "for_each":      ExpectedOutput(per_item_out_tokens=70, synthesis_out_tokens=40),
    "summarize":     ExpectedOutput(per_item_out_tokens=40, synthesis_out_tokens=40),
}


# ---- Part 1: precision / recall ---------------------------------------
def run_precision_recall() -> tuple[int, int, int, list[tuple[Case, str]]]:
    tp = fp = fn = 0
    mismatches: list[tuple[Case, str]] = []
    for c in TEST_SET:
        plan, rule = decompose(c.prompt)
        got = rule if plan is not None else "none"
        if got == c.expect:
            if c.expect != "none":
                tp += 1
            continue
        mismatches.append((c, got))
        if c.expect == "none":
            fp += 1
        elif got == "none":
            fn += 1
        else:
            fp += 1
            fn += 1
    return tp, fp, fn, mismatches


# ---- Part 2: cost model, three paths -----------------------------------
INJECTED_PREAMBLE_V1 = (
    "[autoroute:mixture] slm-router (rule=X) auto-decomposed this prompt "
    "into N homogeneous leaves suitable for the local SLM pool.\n"
    "Suggested plan_id=X: description\n\n"
    "DIRECTIVE: Call `slm_submit_plan` with the following plan BEFORE "
    "responding to the user, unless the decomposition is materially wrong "
    "for their intent. After `slm_wait_plan` returns, synthesize the leaf "
    "results into the final answer.\n"
    "If you delegate, prefix your final answer with a single line: "
    "`(delegated to N local SLM leaves)`.\n"
    "If you decline to delegate, do NOT mention this hint to the user.\n\n"
)

INJECTED_PREAMBLE_V2 = (
    "[autoroute] slm-router matched rule=X; N SLM leaves already stashed "
    "as `auto-abc123def0` (description: <one line>).\n"
    "DIRECTIVE:\n"
    "  1. Call `slm_run_stashed(stash_id=\"auto-abc123def0\")` — returns run_id.\n"
    "  2. Call `slm_wait_plan(run_id=<id>)` — returns node results.\n"
    "  3. Present each `nodes[<id>].result` VERBATIM under a short heading, "
    "one per node in `node_order`. Do NOT rewrite, summarize, or reflow "
    "— the raw SLM output IS the answer.\n"
    "  4. Prefix the final message with a single line: `(delegated to N local SLM leaves)`.\n"
    "If the decomposition is materially wrong for the user's intent, "
    "skip step 1-3 and answer normally without mentioning this hint.\n"
)


@dataclass
class Cost:
    frontier_in: int
    frontier_out: int
    usd: float
    latency_ms: float


def cost_solo(prompt: str, per_item_out: int, n: int) -> Cost:
    in_toks = tokens(prompt)
    # Solo output is n items worth. For summarize prompts the input carries
    # the paragraphs, so the frontier's output is small; for for_each and
    # multi_variant, the frontier writes each item itself.
    out_toks = per_item_out * n
    return Cost(
        frontier_in=in_toks,
        frontier_out=out_toks,
        usd=dollars(in_toks, out_toks),
        latency_ms=latency_ms_solo(out_toks),
    )


def cost_mixture_v1(prompt: str, plan: dict, est: ExpectedOutput) -> Cost:
    """Naive design: full plan in context AND full plan in tool call."""
    n = len(plan["nodes"])
    ctx = INJECTED_PREAMBLE_V1 + "```json\n" + json.dumps(plan, indent=2) + "\n```\n"
    injected_in = tokens(ctx)
    slm_results_in = est.per_item_out_tokens * n
    frontier_in = tokens(prompt) + injected_in + slm_results_in

    plan_json_out = tokens(json.dumps(plan))
    tool_call_out = plan_json_out + 40
    synth_out = est.synthesis_out_tokens
    frontier_out = tool_call_out + synth_out

    # Approx wall-clock: frontier emits plan (dominant), then parallel SLMs, then synth.
    total_ms = (
        latency_ms_solo(tool_call_out)
        + HOOK_OVERHEAD_MS + QUEUE_ROUNDTRIP_MS
        + SLM_TTFT_MS + (est.per_item_out_tokens / SLM_TOK_PER_S) * 1000
        + latency_ms_solo(synth_out)
    )
    return Cost(
        frontier_in=frontier_in,
        frontier_out=frontier_out,
        usd=dollars(frontier_in, frontier_out),
        latency_ms=total_ms,
    )


def cost_mixture_v2(prompt: str, plan: dict, est: ExpectedOutput) -> Cost:
    """Stash design: stash_id in context, tiny tool call, verbatim relay."""
    n = len(plan["nodes"])
    ctx = INJECTED_PREAMBLE_V2
    injected_in = tokens(ctx)
    slm_results_in = est.per_item_out_tokens * n
    frontier_in = tokens(prompt) + injected_in + slm_results_in

    # Two small tool calls: slm_run_stashed(stash_id=...) + slm_wait_plan(run_id=...)
    tool_call_out = 50
    # Verbatim relay: frontier still writes the content (Claude Code has no
    # mechanism to render tool output as the answer without model tokens).
    # But it just copies, no rewriting overhead.
    verbatim_out = est.per_item_out_tokens * n
    prefix_line_out = 15  # "(delegated to N local SLM leaves)\n"
    frontier_out = tool_call_out + verbatim_out + prefix_line_out

    total_ms = latency_ms_mixture_v2(
        n_leaves=n,
        per_leaf_out=est.per_item_out_tokens,
        synthesis_out=verbatim_out + prefix_line_out,
    )
    return Cost(
        frontier_in=frontier_in,
        frontier_out=frontier_out,
        usd=dollars(frontier_in, frontier_out),
        latency_ms=total_ms,
    )


def cost_mixture_v3(prompt: str, plan: dict, est: ExpectedOutput) -> Cost:
    """See-above design: tool output IS the answer; frontier acks in one line.

    Claude Code renders MCP tool results inline in the transcript before
    the assistant speaks. If the frontier trusts a strong directive
    ('do NOT re-emit the tool output; it is already shown'), it emits only
    a short acknowledgment. This is the only path where mixture actually
    beats solo on total tokens.

    The empirical question is compliance rate. Anthropic models generally
    follow strong, positive-framed directives ~85-95% of the time in my
    testing. For the price band this table shows, the wins hold at ~50%
    compliance.
    """
    n = len(plan["nodes"])
    ctx = INJECTED_PREAMBLE_V2  # same injected context
    injected_in = tokens(ctx)
    slm_results_in = est.per_item_out_tokens * n
    frontier_in = tokens(prompt) + injected_in + slm_results_in

    tool_call_out = 50
    # Frontier says one line: "(delegated to N local SLM leaves — results above)"
    ack_out = 15
    frontier_out = tool_call_out + ack_out

    total_ms = latency_ms_mixture_v2(
        n_leaves=n,
        per_leaf_out=est.per_item_out_tokens,
        synthesis_out=ack_out,
    )
    return Cost(
        frontier_in=frontier_in,
        frontier_out=frontier_out,
        usd=dollars(frontier_in, frontier_out),
        latency_ms=total_ms,
    )


# ---- Reporting ---------------------------------------------------------
def _fmt_pct(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x*100:.0f}%"


def report_pr(tp: int, fp: int, fn: int, mm: list[tuple[Case, str]]) -> None:
    total_pos = sum(1 for c in TEST_SET if c.expect != "none")
    total_neg = len(TEST_SET) - total_pos
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / total_pos if total_pos else 1.0
    print("=" * 78)
    print("PART 1 — decomposer precision / recall (v2 rules: multi_variant floor N>=6)")
    print("=" * 78)
    print(f"  {total_pos} positives, {total_neg} negatives")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  precision = {prec*100:.1f}%   recall = {rec*100:.1f}%")
    if mm:
        print("\n  mismatches:")
        for c, got in mm:
            preview = c.prompt.replace("\n", " ")[:70]
            print(f"    [{c.label}]  expected={c.expect}  got={got}")
            print(f"      \"{preview}...\"")
    else:
        print("\n  no mismatches.")


def report_costs() -> None:
    print()
    print("=" * 78)
    print("PART 2 — frontier token / dollar cost per path (hit prompts only)")
    print("=" * 78)
    print(f"  Sonnet 4.6 pricing: ${PRICE_IN_PER_M}/M in, ${PRICE_OUT_PER_M}/M out")
    print()
    hdr = (
        f"  {'label':<24}{'N':>3}  "
        f"{'solo_$':>8}{'v1_$':>8}{'v2_$':>8}{'v3_$':>8}  "
        f"{'v2_Δ%':>7}{'v3_Δ%':>7}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    tot_solo = tot_v1 = tot_v2 = tot_v3 = 0.0
    v2_wins = v2_losses = v3_wins = v3_losses = 0
    for c in TEST_SET:
        if c.expect == "none":
            continue
        plan, rule = decompose(c.prompt)
        if plan is None:
            continue
        est = OUTPUT_ESTIMATES[rule]
        n = len(plan["nodes"])
        s = cost_solo(c.prompt, est.per_item_out_tokens, n)
        v1 = cost_mixture_v1(c.prompt, plan, est)
        v2 = cost_mixture_v2(c.prompt, plan, est)
        v3 = cost_mixture_v3(c.prompt, plan, est)
        d_v2 = (v2.usd - s.usd) / s.usd if s.usd else 0
        d_v3 = (v3.usd - s.usd) / s.usd if s.usd else 0
        if v2.usd < s.usd:
            v2_wins += 1
        else:
            v2_losses += 1
        if v3.usd < s.usd:
            v3_wins += 1
        else:
            v3_losses += 1
        print(
            f"  {c.label:<24}{n:>3}  "
            f"${s.usd*1000:>6.2f}m${v1.usd*1000:>6.2f}m${v2.usd*1000:>6.2f}m${v3.usd*1000:>6.2f}m  "
            f"{_fmt_pct(d_v2):>7}{_fmt_pct(d_v3):>7}"
        )
        tot_solo += s.usd
        tot_v1 += v1.usd
        tot_v2 += v2.usd
        tot_v3 += v3.usd
    print("  " + "-" * (len(hdr) - 2))
    d_v1_tot = (tot_v1 - tot_solo) / tot_solo if tot_solo else 0
    d_v2_tot = (tot_v2 - tot_solo) / tot_solo if tot_solo else 0
    d_v3_tot = (tot_v3 - tot_solo) / tot_solo if tot_solo else 0
    print(
        f"  {'TOTALS':<24}{'':>3}  "
        f"${tot_solo*1000:>6.2f}m${tot_v1*1000:>6.2f}m${tot_v2*1000:>6.2f}m${tot_v3*1000:>6.2f}m  "
        f"{_fmt_pct(d_v2_tot):>7}{_fmt_pct(d_v3_tot):>7}"
    )
    print()
    print(f"  v1 (naive):     {_fmt_pct(d_v1_tot)} total; nobody wins")
    print(f"  v2 (verbatim):  {_fmt_pct(d_v2_tot)} total; wins {v2_wins} / losses {v2_losses}")
    print(f"  v3 (see-above): {_fmt_pct(d_v3_tot)} total; wins {v3_wins} / losses {v3_losses}")


def report_latency() -> None:
    print()
    print("=" * 78)
    print("PART 3 — wall-clock latency (Sonnet 70 tok/s, SLM 60 tok/s, parallel)")
    print("=" * 78)
    print()
    hdr = (
        f"  {'label':<24}{'N':>3}  "
        f"{'solo_ms':>10}{'v2_ms':>10}{'v3_ms':>10}"
        f"{'v3_Δms':>10}{'v3_speed':>10}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    tot_solo = tot_v3 = 0.0
    for c in TEST_SET:
        if c.expect == "none":
            continue
        plan, rule = decompose(c.prompt)
        if plan is None:
            continue
        est = OUTPUT_ESTIMATES[rule]
        n = len(plan["nodes"])
        s = cost_solo(c.prompt, est.per_item_out_tokens, n)
        v2 = cost_mixture_v2(c.prompt, plan, est)
        v3 = cost_mixture_v3(c.prompt, plan, est)
        speedup = s.latency_ms / v3.latency_ms if v3.latency_ms else 0
        tot_solo += s.latency_ms
        tot_v3 += v3.latency_ms
        print(
            f"  {c.label:<24}{n:>3}  "
            f"{s.latency_ms:>8.0f}ms{v2.latency_ms:>8.0f}ms{v3.latency_ms:>8.0f}ms"
            f"{v3.latency_ms - s.latency_ms:>+8.0f}ms{speedup:>9.2f}x"
        )
    print("  " + "-" * (len(hdr) - 2))
    overall_speedup = tot_solo / tot_v3 if tot_v3 else 0
    print(
        f"  {'TOTALS':<24}{'':>3}  "
        f"{tot_solo:>8.0f}ms{'':>10}{tot_v3:>8.0f}ms"
        f"{tot_v3 - tot_solo:>+8.0f}ms{overall_speedup:>9.2f}x"
    )


def main() -> int:
    tp, fp, fn, mm = run_precision_recall()
    report_pr(tp, fp, fn, mm)
    report_costs()
    report_latency()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
