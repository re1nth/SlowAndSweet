"""Render a markdown report from a run snapshot JSON file."""
from __future__ import annotations

import json
from pathlib import Path


def render(snapshot: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Experiment run `{snapshot['run_id']}`")
    lines.append("")
    lines.append(f"- frontier: `{snapshot['frontier']}`")
    lines.append(f"- cases: {snapshot['case_count']}")
    lines.append(f"- wall time: {snapshot['wall_seconds']:.1f}s")
    lines.append(f"- seed: {snapshot['seed']}")
    lines.append("")

    t = snapshot["totals"]
    w = t["winners"]
    lines.append("## Verdicts")
    lines.append("")
    lines.append(
        f"- solo wins: **{w['solo']}** · mixture wins: **{w['mixture']}** · "
        f"tie: {w['tie']} · skipped: {w['skipped']}"
    )
    lines.append("")

    lines.append("## Aggregate usage")
    lines.append("")
    lines.append("| Arm     | Frontier in | Frontier out | SLM out | Wall (s) |")
    lines.append("| ------- | ----------: | -----------: | ------: | -------: |")
    lines.append(
        f"| solo    | {t['solo']['frontier_input_tokens']} | "
        f"{t['solo']['frontier_output_tokens']} | — | "
        f"{t['solo']['wall_seconds']:.1f} |"
    )
    lines.append(
        f"| mixture | {t['mixture']['frontier_input_tokens']} | "
        f"{t['mixture']['frontier_output_tokens']} | "
        f"{t['mixture']['slm_output_tokens']} | "
        f"{t['mixture']['wall_seconds']:.1f} |"
    )
    lines.append(
        f"| reviewer | {t['reviewer']['input_tokens']} | "
        f"{t['reviewer']['output_tokens']} | — | — |"
    )
    lines.append("")

    lines.append("## Per-case")
    lines.append("")
    lines.append(
        "| Case | Winner | Solo tokens (in/out) | Solo wall | "
        "Mix tokens (front in/out · SLM out) | Mix wall |"
    )
    lines.append(
        "| ---- | ------ | -------------------: | --------: | "
        "-----------------------------------: | -------: |"
    )
    for c in snapshot["cases"]:
        s = c["solo"]
        m = c["mixture"]
        winner = c["review"].get("winner", "?")
        lines.append(
            f"| {c['case_id']} | {winner} | "
            f"{s['frontier']['input_tokens']}/{s['frontier']['output_tokens']} | "
            f"{s['wall_seconds']:.1f}s | "
            f"{m['frontier']['input_tokens']}/{m['frontier']['output_tokens']} · "
            f"{m['slm']['output_tokens']} | {m['wall_seconds']:.1f}s |"
        )
    lines.append("")

    lines.append("## Reviewer notes")
    lines.append("")
    for c in snapshot["cases"]:
        rv = c["review"]
        if rv.get("winner") == "skipped":
            lines.append(f"- **{c['case_id']}** — skipped: {rv.get('reason')}")
            continue
        scores = rv.get("scores", {})
        lines.append(
            f"- **{c['case_id']}** → winner **{rv.get('winner')}** "
            f"(conf {rv.get('confidence', 0):.2f}). "
            f"scores solo={scores.get('solo')}, mixture={scores.get('mixture')}. "
            f"{rv.get('reasoning', '').strip()}"
        )
    lines.append("")
    return "\n".join(lines)


def render_file(snapshot_path: Path) -> str:
    with open(snapshot_path) as fh:
        return render(json.load(fh))
