"""HTML rendering for plan runs and the landing page.

No template engine: small enough to be readable inline. Mermaid is loaded
from a CDN; the rest is plain HTML + CSS + a meta-refresh tag.
"""
from __future__ import annotations

import html
import time
from typing import Iterable

_STYLES = """
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         max-width: 1100px; margin: 1.5em auto; padding: 0 1em; color: #111; }
  h1 { margin-bottom: 0.1em; }
  h1 .meta { color: #64748b; font-weight: 400; font-size: 0.65em; margin-left: 0.4em; }
  .desc { color: #475569; margin-top: 0.2em; }
  .status-line { margin: 0.5em 0 1em; color: #475569; font-size: 14px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
           font-weight: 600; font-size: 12px; text-transform: uppercase;
           letter-spacing: 0.04em; }
  .b-pending { background: #f1f5f9; color: #475569; }
  .b-queued  { background: #fef3c7; color: #92400e; }
  .b-running { background: #fde047; color: #713f12; }
  .b-done    { background: #bbf7d0; color: #166534; }
  .b-error   { background: #fecaca; color: #991b1b; }
  table { border-collapse: collapse; width: 100%; margin-top: 1em; font-size: 13px; }
  th, td { border: 1px solid #e2e8f0; padding: 6px 9px; text-align: left;
           vertical-align: top; }
  th { background: #f8fafc; }
  td.result { max-width: 460px; }
  td.result pre { background: #f8fafc; padding: 8px; border-radius: 4px;
                  white-space: pre-wrap; word-break: break-word;
                  max-height: 200px; overflow: auto; font-size: 12px;
                  margin: 0; }
  .mermaid { background: #fff; padding: 1em 0; }
  a { color: #2563eb; text-decoration: none; }
  a:hover { text-decoration: underline; }
  code { background: #f1f5f9; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
"""

_MERMAID_CLASSDEFS = """
  classDef pending fill:#f1f5f9,stroke:#64748b,color:#334155
  classDef queued  fill:#fef3c7,stroke:#92400e,color:#78350f
  classDef running fill:#fde047,stroke:#713f12,color:#713f12
  classDef done    fill:#bbf7d0,stroke:#166534,color:#14532d
  classDef error   fill:#fecaca,stroke:#991b1b,color:#7f1d1d
"""


def _fmt_age(ts: float | None) -> str:
    if ts is None:
        return "—"
    delta = max(0.0, time.time() - ts)
    if delta < 60:
        return f"{delta:.1f}s ago"
    if delta < 3600:
        return f"{delta/60:.1f}m ago"
    return f"{delta/3600:.1f}h ago"


def _fmt_elapsed(node: dict) -> str:
    s = node.get("started_at")
    f = node.get("finished_at")
    if s is None:
        return "—"
    end = f if f is not None else time.time()
    return f"{end - s:.2f}s"


def _badge(status: str) -> str:
    return f'<span class="badge b-{status}">{status}</span>'


def _mermaid_for(snapshot: dict) -> str:
    lines = ["graph TD"]
    order = snapshot["node_order"]
    nodes = snapshot["nodes"]
    for nid in order:
        n = nodes[nid]
        label = html.escape(f"{nid}")
        lines.append(f'  {nid}["{label}"]:::{n["status"]}')
    for nid in order:
        for dep in nodes[nid]["depends_on"]:
            lines.append(f"  {dep} --> {nid}")
    lines.append(_MERMAID_CLASSDEFS)
    return "\n".join(lines)


def _result_cell(node: dict) -> str:
    if node["status"] == "done":
        text = (node.get("result") or "").strip()
        return f"<pre>{html.escape(text)}</pre>"
    if node["status"] == "error":
        return f'<pre>{html.escape(node.get("error") or "")}</pre>'
    if node["status"] in ("queued", "running") and node.get("prompt"):
        return (f'<details><summary>resolved prompt</summary>'
                f'<pre>{html.escape(node["prompt"])}</pre></details>')
    return "—"


def render_plan_run(snapshot: dict) -> str:
    auto_refresh = snapshot["status"] == "running"
    refresh_tag = '<meta http-equiv="refresh" content="2">' if auto_refresh else ""
    diagram = _mermaid_for(snapshot)
    rows = []
    for nid in snapshot["node_order"]:
        n = snapshot["nodes"][nid]
        deps = ", ".join(n["depends_on"]) or "—"
        rows.append(f"""
        <tr>
          <td><code>{html.escape(nid)}</code></td>
          <td>{_badge(n['status'])}</td>
          <td>{html.escape(n.get('model') or '—')}</td>
          <td>{html.escape(n.get('worker') or '—')}</td>
          <td>{html.escape(deps)}</td>
          <td>{_fmt_elapsed(n)}</td>
          <td class="result">{_result_cell(n)}</td>
        </tr>""")
    table = "<table>\n<thead><tr><th>Node</th><th>Status</th><th>Model</th><th>Worker</th><th>Depends on</th><th>Elapsed</th><th>Result / prompt</th></tr></thead>\n<tbody>" + "".join(rows) + "</tbody></table>"

    return f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>{html.escape(snapshot['plan_id'])} · {snapshot['run_id']}</title>
{refresh_tag}
<style>{_STYLES}</style>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({{ startOnLoad: true, theme: 'neutral', flowchart: {{ curve: 'basis' }} }});
</script>
</head><body>
<p><a href="/ui">&larr; all runs</a></p>
<h1>{html.escape(snapshot['plan_id'])} <span class="meta">run {snapshot['run_id']}</span></h1>
<p class="desc">{html.escape(snapshot['description'] or '')}</p>
<p class="status-line">
  Status: {_badge(snapshot['status'])}
  &middot; created {_fmt_age(snapshot['created_at'])}
  {"&middot; finished " + _fmt_age(snapshot['finished_at']) if snapshot.get('finished_at') else ""}
  {"&middot; auto-refresh 2s" if auto_refresh else ""}
</p>
<div class="mermaid">
{diagram}
</div>
{table}
</body></html>
"""


def render_index(runs: list[dict], plan_files: Iterable[str]) -> str:
    rows = []
    if not runs:
        rows.append('<tr><td colspan="5"><em>no runs yet</em></td></tr>')
    for r in runs:
        rows.append(f"""
        <tr>
          <td><a href="/plans/{r['run_id']}/ui"><code>{r['run_id']}</code></a></td>
          <td>{html.escape(r['plan_id'])}</td>
          <td>{_badge(r['status'])}</td>
          <td>{r['node_count']}</td>
          <td>{_fmt_age(r['created_at'])}</td>
        </tr>""")
    runs_table = "<table>\n<thead><tr><th>Run</th><th>Plan</th><th>Status</th><th>Nodes</th><th>Created</th></tr></thead>\n<tbody>" + "".join(rows) + "</tbody></table>"

    file_items = "".join(
        f'<li><code>{html.escape(f)}</code> &mdash; '
        f'<a href="/plans/from-file/{html.escape(f)}">submit</a></li>'
        for f in plan_files
    ) or "<li><em>no plan files in plans/ directory</em></li>"

    return f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>SLM Queue · plan runs</title>
<meta http-equiv="refresh" content="3">
<style>{_STYLES}</style>
</head><body>
<h1>SLM Queue <span class="meta">plan runs</span></h1>
<p class="desc">Submit a hand-authored plan from <code>slm-queue/plans/</code>, then watch it execute as a DAG against the worker pool.</p>

<h2>Plan files</h2>
<ul>{file_items}</ul>

<h2>Runs</h2>
{runs_table}

<p class="status-line"><a href="/status">/status</a> &middot; auto-refresh 3s</p>
</body></html>
"""
