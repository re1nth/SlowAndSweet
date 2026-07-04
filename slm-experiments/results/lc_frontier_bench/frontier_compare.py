"""SxS: how many tokens does Claude Code receive in each pattern?

Case 1 (solo)      — Claude Code sees the full problem statement.
Case 2 (mixture)   — SLM DAG distills the problem into a short algorithm sketch;
                     Claude Code sees only that sketch as its composer input.

The SLM's own input/output tokens are NOT counted against Claude — the whole
point is that they're absorbed by the local pool. We only compare what the
frontier ingests.

Inputs are read from the sibling files:
  problems.json         — the 10 scraped LeetCode statements
  slm_results.json      — per-problem SLM DAG run (has the sketch text)

Output: comparison.json + printed markdown table.
"""
import json, pathlib, tiktoken

ROOT = pathlib.Path(__file__).parent
ENC = tiktoken.get_encoding("cl100k_base")


def solo_prompt(problem: dict) -> str:
    return (
        "Solve the following LeetCode problem. Give a short approach paragraph, "
        "then a Python solution in a ```python``` block.\n\n"
        f"PROBLEM: {problem['title']}\n\n{problem['statement']}"
    )


def mixture_composer_prompt(problem: dict, slm_sketch: str) -> str:
    """What Claude Code sees in the mixture pattern: only the SLM-distilled sketch."""
    return (
        "An SLM has already sketched the algorithm for a LeetCode problem. "
        "Using ONLY that sketch, write the final Python solution as a single "
        "function inside a ```python``` block. No prose, no re-explanation.\n\n"
        f"PROBLEM TITLE: {problem['title']}\n\n"
        "ALGORITHM SKETCH (from local SLM):\n"
        f"{slm_sketch}"
    )


def main():
    problems = json.loads((ROOT / "problems.json").read_text())
    slm_rows = json.loads((ROOT / "slm_results.json").read_text())
    slm_by_slug = {r["slug"]: r for r in slm_rows}

    rows = []
    for p in problems:
        slm = slm_by_slug[p["slug"]]
        sketch = next(n for n in slm["nodes"] if n["id"] == "sketch")
        sketch_text = sketch.get("result_text", "")

        solo_in = len(ENC.encode(solo_prompt(p)))
        mix_in = len(ENC.encode(mixture_composer_prompt(p, sketch_text)))
        rows.append({
            "slug": p["slug"],
            "title": p["title"],
            "difficulty": p["difficulty"],
            "case1_solo_in_tok": solo_in,
            "case2_mixture_in_tok": mix_in,
            "delta_tok": solo_in - mix_in,
            "reduction_pct": round(100 * (solo_in - mix_in) / solo_in, 1),
            "sketch_tok": len(ENC.encode(sketch_text)),
            "statement_tok": len(ENC.encode(p["statement"])),
        })

    tot_solo = sum(r["case1_solo_in_tok"] for r in rows)
    tot_mix = sum(r["case2_mixture_in_tok"] for r in rows)

    header = ("| Problem                            | Diff   | Case 1 in | Case 2 in | Δ    | Reduction |\n"
              "| ---------------------------------- | ------ | --------: | --------: | ---: | --------: |")
    print(header)
    for r in rows:
        sign = "+" if r["delta_tok"] > 0 else ""
        print(f"| {r['title']:34} | {r['difficulty']:6} | "
              f"{r['case1_solo_in_tok']:>9} | {r['case2_mixture_in_tok']:>9} | "
              f"{sign}{r['delta_tok']:>4} | {r['reduction_pct']:>7.1f}% |")

    print()
    print(f"TOTAL  case1={tot_solo}  case2={tot_mix}  delta={tot_solo - tot_mix}  "
          f"reduction={100 * (tot_solo - tot_mix) / tot_solo:.1f}%")

    (ROOT / "comparison.json").write_text(json.dumps({
        "rows": rows,
        "totals": {
            "case1_solo_in_tok": tot_solo,
            "case2_mixture_in_tok": tot_mix,
            "delta_tok": tot_solo - tot_mix,
            "reduction_pct": round(100 * (tot_solo - tot_mix) / tot_solo, 1),
        },
        "notes": (
            "Tokens sent TO Claude Code only. Case 1 = full problem statement. "
            "Case 2 = SLM-generated algorithm sketch (the SLM's own token cost is not "
            "counted, since it's offloaded to the local pool). Both tokenized with "
            "tiktoken cl100k_base as a shared proxy for Claude token counts."
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
