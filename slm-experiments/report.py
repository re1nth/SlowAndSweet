"""Render a markdown report from a run snapshot JSON file."""
from __future__ import annotations

import json
from pathlib import Path


_ARMS = ("solo", "mixture", "prism")


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
    present_arms = [a for a in _ARMS if a in t]
    lines.append("## Verdicts")
    lines.append("")
    win_bits = [f"{a}: **{w.get(a, 0)}**" for a in present_arms]
    lines.append("- " + " · ".join(win_bits) + f" · tie: {w.get('tie', 0)} · skipped: {w.get('skipped', 0)}")
    lines.append("")

    lines.append("## Aggregate usage")
    lines.append("")
    lines.append("| Arm     | Frontier in | Frontier out | SLM out | Σ wall (s) |")
    lines.append("| ------- | ----------: | -----------: | ------: | ---------: |")
    for arm in present_arms:
        row = t[arm]
        slm_cell = str(row.get("slm_output_tokens", "—")) if row.get("slm_output_tokens") else "—"
        lines.append(
            f"| {arm:<7} | {row['frontier_input_tokens']} | "
            f"{row['frontier_output_tokens']} | {slm_cell} | "
            f"{row['wall_seconds']:.1f} |"
        )
    lines.append(
        f"| reviewer | {t['reviewer']['input_tokens']} | "
        f"{t['reviewer']['output_tokens']} | — | — |"
    )
    lines.append("")

    lines.append("## Per-case")
    lines.append("")
    header = "| Case | Winner |"
    sep = "| ---- | ------ |"
    for arm in present_arms:
        header += f" {arm} tokens (front in/out · slm out) | {arm} wall |"
        sep += " ----: | ----: |"
    lines.append(header)
    lines.append(sep)
    for c in snapshot["cases"]:
        winner = c["review"].get("winner", "?")
        row = f"| {c['case_id']} | {winner} |"
        for arm in present_arms:
            ad = c.get(arm) or {}
            fr = ad.get("frontier", {})
            sl = ad.get("slm", {})
            wall = ad.get("wall_seconds", 0.0)
            row += (
                f" {fr.get('input_tokens', 0)}/{fr.get('output_tokens', 0)} · "
                f"{sl.get('output_tokens', 0)} | {wall:.1f}s |"
            )
        lines.append(row)
    lines.append("")

    lines.append("## Reviewer notes")
    lines.append("")
    for c in snapshot["cases"]:
        rv = c["review"]
        if rv.get("winner") == "skipped":
            lines.append(f"- **{c['case_id']}** — skipped: {rv.get('reason')}")
            continue
        scores = rv.get("scores", {})
        score_bits = ", ".join(f"{arm}={scores.get(arm)}" for arm in present_arms if arm in scores)
        lines.append(
            f"- **{c['case_id']}** → winner **{rv.get('winner')}** "
            f"(conf {rv.get('confidence', 0):.2f}). scores {score_bits}. "
            f"{rv.get('reasoning', '').strip()}"
        )
    lines.append("")
    return "\n".join(lines)


def render_file(snapshot_path: Path) -> str:
    with open(snapshot_path) as fh:
        return render(json.load(fh))
