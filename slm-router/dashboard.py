"""Static HTML dashboard for router training metrics.

Reads metrics.jsonl from the data dir and writes a single self-contained
HTML file (Chart.js from CDN, all data inlined). Two sections:

- Arm router (solo vs mixture): headline strip, MAE timeline (train / CV
  band / holdout), Pearson r + quality-acc timeline, training set size,
  per-version bars, recent runs table.
- Leaf router (which local SLM for each subtask): headline strip, per-SLM
  cost MAE and quality-acc timelines, latest per-SLM breakdown (train /
  CV / holdout side-by-side), recent runs table.

The train/CV gap is the overfitting signal in both sections: train ≪ CV
means the model memorized its training slice.

Runnable both standalone (`python dashboard.py`) and from `train.py`
(which calls `render_dashboard(config)` after each training attempt).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from html import escape
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from paths import ensure_data_dir, resolve_data_path  # noqa: E402


CHART_JS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_train_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "&mdash;"
    if isinstance(v, (int, float)):
        return f"{v:.{digits}f}"
    return escape(str(v))


def _fmt_ts(ts: Any) -> str:
    if not isinstance(ts, (int, float)):
        return "&mdash;"
    return escape(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)))


def _series(rows: list[dict[str, Any]], path: list[str]) -> list[Any]:
    out: list[Any] = []
    for r in rows:
        v: Any = r
        for key in path:
            if not isinstance(v, dict):
                v = None
                break
            v = v.get(key)
        out.append(v)
    return out


def render_dashboard(config: dict[str, Any]) -> Path:
    ensure_data_dir(config)
    paths_cfg = config.get("paths") or {}
    metrics_path = resolve_data_path(config, paths_cfg.get("metrics_log", "metrics.jsonl"))
    state_path = resolve_data_path(config, paths_cfg.get("train_state", "train_state.json"))
    leaf_state_path = resolve_data_path(
        config, paths_cfg.get("leaf_train_state", "leaf_train_state.json")
    )
    out_path = resolve_data_path(config, paths_cfg.get("dashboard_html", "dashboard.html"))

    all_rows = _load_metrics(metrics_path)
    state = _load_train_state(state_path)
    leaf_state = _load_train_state(leaf_state_path)
    leaf_slms = list((config.get("leaf") or {}).get("available_slms") or [])

    # Arm-router entries pre-date the `component` tag, so absence == arm.
    rows = [r for r in all_rows if r.get("component") != "leaf"]
    leaf_rows_all = [r for r in all_rows if r.get("component") == "leaf"]

    # Only chart runs that actually trained; skip runs = insufficient_records / insufficient_new_records
    trained = [r for r in rows if r.get("candidate_version") is not None]

    labels = [_fmt_ts(r.get("timestamp")) for r in trained]
    train_mae = _series(trained, ["candidate_metrics", "train_mae_pp"])
    cv_mean = _series(trained, ["candidate_metrics", "cv_mae_pp_mean"])
    cv_std = _series(trained, ["candidate_metrics", "cv_mae_pp_std"])
    holdout_mae = _series(trained, ["candidate_metrics", "holdout_mae_pp"])
    pearson = _series(trained, ["candidate_metrics", "holdout_pearson_r"])
    qacc = _series(trained, ["candidate_metrics", "holdout_quality_acc"])
    n_records = [r.get("n_train") for r in trained]
    versions = [r.get("candidate_version") for r in trained]
    promoted = [bool(r.get("promoted")) for r in trained]

    cv_upper = [
        (m + s) if isinstance(m, (int, float)) and isinstance(s, (int, float)) else None
        for m, s in zip(cv_mean, cv_std)
    ]
    cv_lower = [
        max(0.0, m - s) if isinstance(m, (int, float)) and isinstance(s, (int, float)) else None
        for m, s in zip(cv_mean, cv_std)
    ]

    latest = trained[-1] if trained else None
    cm = (latest or {}).get("candidate_metrics") or {}
    train_v = cm.get("train_mae_pp")
    cv_v = cm.get("cv_mae_pp_mean")
    holdout_v = cm.get("holdout_mae_pp")
    gap = None
    if isinstance(train_v, (int, float)) and isinstance(cv_v, (int, float)):
        gap = cv_v - train_v

    current_head = state.get("last_promoted_version") or "&mdash;"
    last_run_ts = state.get("last_run_timestamp")
    last_run_reason = state.get("last_run_reason") or "&mdash;"
    total_records = latest.get("n_train") if latest else 0

    data_blob = {
        "labels": labels,
        "versions": versions,
        "promoted": promoted,
        "train_mae": train_mae,
        "cv_mean": cv_mean,
        "cv_upper": cv_upper,
        "cv_lower": cv_lower,
        "holdout_mae": holdout_mae,
        "pearson": pearson,
        "qacc": qacc,
        "n_records": n_records,
    }
    data_json = json.dumps(data_blob, default=lambda o: None)

    recent = list(reversed(rows[-20:]))
    table_rows_html = "\n".join(_row_html(r) for r in recent) or _empty_row_html()

    # ------ Leaf section ------
    leaf_trained = [r for r in leaf_rows_all if r.get("candidate_version") is not None]
    leaf_labels = [_fmt_ts(r.get("timestamp")) for r in leaf_trained]
    leaf_versions = [r.get("candidate_version") for r in leaf_trained]
    leaf_promoted = [bool(r.get("promoted")) for r in leaf_trained]

    per_slm_series: dict[str, dict[str, list[Any]]] = {
        m: {
            "train_mae": [],
            "cv_mae_mean": [],
            "holdout_mae": [],
            "n_records": [],
            "quality_train_acc": [],
            "quality_holdout_acc": [],
        }
        for m in leaf_slms
    }
    for r in leaf_trained:
        pm = r.get("per_slm_metrics") or {}
        for m in leaf_slms:
            slm = pm.get(m) or {}
            per_slm_series[m]["train_mae"].append(slm.get("train_mae"))
            per_slm_series[m]["cv_mae_mean"].append(slm.get("cv_mae_mean"))
            per_slm_series[m]["holdout_mae"].append(slm.get("holdout_mae"))
            per_slm_series[m]["n_records"].append(slm.get("n_records"))
            per_slm_series[m]["quality_train_acc"].append(slm.get("quality_train_acc"))
            per_slm_series[m]["quality_holdout_acc"].append(slm.get("quality_holdout_acc"))

    # Latest per-SLM breakdown (grouped bars: train / CV / holdout per SLM).
    latest_leaf = leaf_trained[-1] if leaf_trained else None
    latest_pm = (latest_leaf or {}).get("per_slm_metrics") or {}
    latest_breakdown = {
        "models": leaf_slms,
        "train_mae": [(latest_pm.get(m) or {}).get("train_mae") for m in leaf_slms],
        "cv_mae_mean": [(latest_pm.get(m) or {}).get("cv_mae_mean") for m in leaf_slms],
        "holdout_mae": [(latest_pm.get(m) or {}).get("holdout_mae") for m in leaf_slms],
    }

    # Leaf headline numbers.
    def _mean_of(xs: list[Any]) -> float | None:
        vs = [v for v in xs if isinstance(v, (int, float))]
        return sum(vs) / len(vs) if vs else None

    latest_cost_mae = _mean_of([(latest_pm.get(m) or {}).get("holdout_mae") for m in leaf_slms])
    latest_qual_acc = _mean_of([(latest_pm.get(m) or {}).get("quality_holdout_acc") for m in leaf_slms])
    total_leaf_records = latest_leaf.get("n_train") if latest_leaf else 0
    total_leaf_qual = sum(
        (latest_pm.get(m) or {}).get("n_quality_labels") or 0 for m in leaf_slms
    )

    leaf_current_head = leaf_state.get("last_promoted_version") or "&mdash;"
    leaf_last_run_ts = leaf_state.get("last_run_timestamp")
    leaf_last_reason = leaf_state.get("last_run_reason") or "&mdash;"

    leaf_data_blob = {
        "labels": leaf_labels,
        "versions": leaf_versions,
        "promoted": leaf_promoted,
        "models": leaf_slms,
        "per_slm": per_slm_series,
        "latest_breakdown": latest_breakdown,
    }
    leaf_data_json = json.dumps(leaf_data_blob, default=lambda o: None)

    leaf_recent = list(reversed(leaf_rows_all[-20:]))
    leaf_table_rows_html = (
        "\n".join(_leaf_row_html(r, leaf_slms) for r in leaf_recent)
        or _leaf_empty_row_html()
    )

    html = _TEMPLATE.format(
        chart_js=CHART_JS_CDN,
        generated=_fmt_ts(time.time()),
        current_head=escape(str(current_head)),
        total_records=escape(str(total_records if total_records is not None else 0)),
        last_run=_fmt_ts(last_run_ts),
        last_reason=escape(str(last_run_reason)),
        train_mae=_fmt(train_v),
        cv_mae=_fmt(cv_v),
        holdout_mae=_fmt(holdout_v),
        gap=_fmt(gap),
        gap_class=_gap_class(gap),
        data_json=data_json,
        recent_rows=table_rows_html,
        n_runs=len(rows),
        n_trained=len(trained),
        # Leaf section
        leaf_current_head=escape(str(leaf_current_head)),
        leaf_records=escape(str(total_leaf_records if total_leaf_records is not None else 0)),
        leaf_qual_labels=escape(str(total_leaf_qual)),
        leaf_last_run=_fmt_ts(leaf_last_run_ts),
        leaf_last_reason=escape(str(leaf_last_reason)),
        leaf_cost_mae=_fmt(latest_cost_mae),
        leaf_qual_acc=_fmt(latest_qual_acc),
        leaf_data_json=leaf_data_json,
        leaf_recent_rows=leaf_table_rows_html,
        leaf_n_runs=len(leaf_rows_all),
        leaf_n_trained=len(leaf_trained),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path


def _gap_class(gap: float | None) -> str:
    if gap is None:
        return "muted"
    if gap < 3:
        return "good"
    if gap < 8:
        return "warn"
    return "bad"


def _row_html(r: dict[str, Any]) -> str:
    cm = r.get("candidate_metrics") or {}
    return (
        "<tr>"
        f"<td>{_fmt_ts(r.get('timestamp'))}</td>"
        f"<td>{escape(str(r.get('candidate_version') or '—'))}</td>"
        f"<td>{escape(str(r.get('n_train') or 0))}</td>"
        f"<td>{escape(str(r.get('n_new_records') or 0))}</td>"
        f"<td>{_fmt(cm.get('train_mae_pp'))}</td>"
        f"<td>{_fmt(cm.get('cv_mae_pp_mean'))}</td>"
        f"<td>{_fmt(cm.get('holdout_mae_pp'))}</td>"
        f"<td>{_fmt(cm.get('holdout_pearson_r'))}</td>"
        f"<td>{_fmt(cm.get('holdout_quality_acc'))}</td>"
        f"<td>{'yes' if r.get('promoted') else 'no'}</td>"
        f"<td>{escape(str(r.get('reason') or ''))}</td>"
        "</tr>"
    )


def _empty_row_html() -> str:
    return '<tr><td colspan="11" class="muted">no training runs yet</td></tr>'


def _leaf_row_html(r: dict[str, Any], slms: list[str]) -> str:
    pm = r.get("per_slm_metrics") or {}
    trained_here = r.get("trained_slms") or []
    # Compact per-SLM cell: "smollm2 n=14 cv=92.5 qacc=0.75" for each trained SLM.
    slm_cells: list[str] = []
    for m in slms:
        slm = pm.get(m) or {}
        n = slm.get("n_records")
        cv = slm.get("cv_mae_mean")
        qacc = slm.get("quality_holdout_acc")
        fresh = m in trained_here
        cell = f"{escape(m.split(':')[0])}"
        if n:
            cell += f" n={n}"
        if isinstance(cv, (int, float)):
            cell += f" cv={cv:.1f}"
        if isinstance(qacc, (int, float)):
            cell += f" qacc={qacc:.2f}"
        if not fresh and slm.get("retained_from_incumbent"):
            cell += " (kept)"
        slm_cells.append(cell)
    return (
        "<tr>"
        f"<td>{_fmt_ts(r.get('timestamp'))}</td>"
        f"<td>{escape(str(r.get('candidate_version') or '—'))}</td>"
        f"<td>{escape(str(r.get('n_train') or 0))}</td>"
        f"<td>{escape(str(r.get('n_new_records') or 0))}</td>"
        f"<td class='wrap'>{'<br>'.join(slm_cells) if slm_cells else '—'}</td>"
        f"<td>{'yes' if r.get('promoted') else 'no'}</td>"
        f"<td>{escape(str(r.get('reason') or ''))}</td>"
        "</tr>"
    )


def _leaf_empty_row_html() -> str:
    return '<tr><td colspan="7" class="muted">no leaf training runs yet — waiting on feedback</td></tr>'


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>slm-router training dashboard</title>
<style>
  :root {{
    --bg: #0f1216;
    --panel: #171b22;
    --border: #262b34;
    --text: #e6e8ec;
    --muted: #8892a0;
    --accent: #4ea1ff;
    --good: #55c37a;
    --warn: #e4b350;
    --bad: #e57373;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px;
    background: var(--bg); color: var(--text);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  h1 {{ margin: 0 0 4px 0; font-size: 20px; font-weight: 600; }}
  .subtitle {{ color: var(--muted); margin-bottom: 20px; font-size: 12px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
  .card .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .card .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  .card .value.good {{ color: var(--good); }}
  .card .value.warn {{ color: var(--warn); }}
  .card .value.bad {{ color: var(--bad); }}
  .card .value.muted {{ color: var(--muted); }}
  .card .footnote {{ color: var(--muted); font-size: 11px; margin-top: 6px; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 24px; }}
  @media (max-width: 900px) {{ .charts {{ grid-template-columns: 1fr; }} }}
  .chart-panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
  .chart-panel h2 {{ font-size: 13px; margin: 0 0 12px 0; color: var(--muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }}
  .chart-panel .why {{ color: var(--muted); font-size: 11px; margin-top: 8px; }}
  canvas {{ max-height: 260px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 500; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
  td.muted {{ color: var(--muted); text-align: center; }}
  .table-wrap {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px; overflow-x: auto; }}
  .table-wrap h2 {{ font-size: 13px; margin: 0 0 12px 0; color: var(--muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }}
  td.wrap {{ font-family: ui-monospace, monospace; font-size: 11px; color: var(--muted); }}
  .section-title {{ font-size: 16px; font-weight: 600; margin: 32px 0 12px 0; padding-bottom: 6px; border-bottom: 1px solid var(--border); }}
</style>
</head>
<body>

<h1>slm-router training dashboard</h1>
<div class="subtitle">Generated {generated}</div>

<div class="section-title">Arm router — solo vs mixture</div>
<div class="subtitle">{n_runs} scheduler runs &middot; {n_trained} actual trainings</div>

<div class="grid">
  <div class="card">
    <div class="label">Current head</div>
    <div class="value">{current_head}</div>
    <div class="footnote">Last run: {last_run} ({last_reason})</div>
  </div>
  <div class="card">
    <div class="label">Records used</div>
    <div class="value">{total_records}</div>
    <div class="footnote">most recent training run</div>
  </div>
  <div class="card">
    <div class="label">Latest holdout MAE (pp)</div>
    <div class="value">{holdout_mae}</div>
    <div class="footnote">unseen 20% split</div>
  </div>
  <div class="card">
    <div class="label">Latest CV MAE (pp)</div>
    <div class="value">{cv_mae}</div>
    <div class="footnote">mean over 5 folds</div>
  </div>
  <div class="card">
    <div class="label">Latest train MAE (pp)</div>
    <div class="value">{train_mae}</div>
    <div class="footnote">fit on same 80%</div>
  </div>
  <div class="card">
    <div class="label">Overfitting gap (CV &minus; train)</div>
    <div class="value {gap_class}">{gap}</div>
    <div class="footnote">smaller is better; large gap = memorization</div>
  </div>
</div>

<div class="charts">
  <div class="chart-panel">
    <h2>MAE across training runs</h2>
    <canvas id="chart_mae"></canvas>
    <div class="why">Train (green) &lt;&lt; CV (blue) &lt;&lt; Holdout (orange) indicates overfitting. All three tracking together is healthy.</div>
  </div>
  <div class="chart-panel">
    <h2>Regression quality &amp; classifier accuracy</h2>
    <canvas id="chart_quality"></canvas>
    <div class="why">Holdout Pearson r for the cost regressor; holdout accuracy for the quality classifier.</div>
  </div>
  <div class="chart-panel">
    <h2>Training set size</h2>
    <canvas id="chart_n"></canvas>
    <div class="why">Records used per training run. Metric charts above should be read with this in mind — small N makes them noisy.</div>
  </div>
  <div class="chart-panel">
    <h2>Per-version holdout MAE</h2>
    <canvas id="chart_bars"></canvas>
    <div class="why">Green bars were auto-promoted; grey were rejected by the gate.</div>
  </div>
</div>

<div class="table-wrap">
  <h2>Recent runs</h2>
  <table>
    <thead>
      <tr>
        <th>Time</th><th>Version</th><th>N</th><th>New</th>
        <th>Train MAE</th><th>CV MAE</th><th>Holdout MAE</th>
        <th>Pearson r</th><th>Qual acc</th><th>Promoted</th><th>Reason</th>
      </tr>
    </thead>
    <tbody>
      {recent_rows}
    </tbody>
  </table>
</div>

<div class="section-title">Leaf router — which local SLM per subtask</div>
<div class="subtitle">{leaf_n_runs} scheduler runs &middot; {leaf_n_trained} actual trainings</div>

<div class="grid">
  <div class="card">
    <div class="label">Current leaf head</div>
    <div class="value">{leaf_current_head}</div>
    <div class="footnote">Last run: {leaf_last_run} ({leaf_last_reason})</div>
  </div>
  <div class="card">
    <div class="label">Feedback records</div>
    <div class="value">{leaf_records}</div>
    <div class="footnote">unique prompts in most recent leaf run</div>
  </div>
  <div class="card">
    <div class="label">Quality labels</div>
    <div class="value">{leaf_qual_labels}</div>
    <div class="footnote">sum across SLMs, most recent leaf run</div>
  </div>
  <div class="card">
    <div class="label">Mean cost holdout MAE</div>
    <div class="value">{leaf_cost_mae}</div>
    <div class="footnote">avg across trained SLMs, last run</div>
  </div>
  <div class="card">
    <div class="label">Mean quality holdout acc</div>
    <div class="value">{leaf_qual_acc}</div>
    <div class="footnote">avg across trained SLMs, last run</div>
  </div>
</div>

<div class="charts">
  <div class="chart-panel">
    <h2>Per-SLM holdout MAE across runs</h2>
    <canvas id="leaf_cost_timeline"></canvas>
    <div class="why">Cost-regressor holdout error per SLM over training runs. Divergent lines = SLMs learning at different rates from ε-explore data.</div>
  </div>
  <div class="chart-panel">
    <h2>Per-SLM quality holdout accuracy</h2>
    <canvas id="leaf_quality_timeline"></canvas>
    <div class="why">Reviewer-verdict classifier's holdout accuracy per SLM. 0.5 = uninformed prior; watch for drift below.</div>
  </div>
  <div class="chart-panel">
    <h2>Latest per-SLM MAE breakdown</h2>
    <canvas id="leaf_breakdown"></canvas>
    <div class="why">Train vs 5-fold CV vs holdout MAE for the most recent leaf training run. Big gaps between train and CV = per-SLM overfitting.</div>
  </div>
  <div class="chart-panel">
    <h2>Per-version leaf promotion</h2>
    <canvas id="leaf_versions"></canvas>
    <div class="why">Green bars promoted, grey rejected. Height = mean of holdout MAE across SLMs in that candidate.</div>
  </div>
</div>

<div class="table-wrap">
  <h2>Recent leaf runs</h2>
  <table>
    <thead>
      <tr>
        <th>Time</th><th>Version</th><th>N</th><th>New</th>
        <th>Per-SLM</th><th>Promoted</th><th>Reason</th>
      </tr>
    </thead>
    <tbody>
      {leaf_recent_rows}
    </tbody>
  </table>
</div>

<script src="{chart_js}"></script>
<script>
  const DATA = {data_json};
  const COLORS = {{
    train: "#55c37a",
    cv: "#4ea1ff",
    cvBand: "rgba(78, 161, 255, 0.15)",
    holdout: "#e4b350",
    pearson: "#4ea1ff",
    qacc: "#c07dd8",
    n: "#8892a0",
    promoted: "#55c37a",
    rejected: "#4a505a",
  }};
  const AXIS = {{ ticks: {{ color: "#8892a0" }}, grid: {{ color: "#262b34" }} }};
  const LEGEND = {{ labels: {{ color: "#e6e8ec" }} }};

  if (DATA.labels.length === 0) {{
    document.querySelectorAll("canvas").forEach(c => {{
      c.replaceWith(Object.assign(document.createElement("div"), {{
        textContent: "no training runs yet — dashboard will populate after the first successful run",
        style: "color:#8892a0;font-size:12px;padding:20px 0;text-align:center;"
      }}));
    }});
  }} else {{
    new Chart(document.getElementById("chart_mae"), {{
      type: "line",
      data: {{
        labels: DATA.labels,
        datasets: [
          {{ label: "CV upper (mean+std)", data: DATA.cv_upper, borderColor: "transparent", backgroundColor: COLORS.cvBand, fill: "+1", pointRadius: 0, tension: 0.2 }},
          {{ label: "CV lower (mean-std)", data: DATA.cv_lower, borderColor: "transparent", backgroundColor: COLORS.cvBand, fill: false, pointRadius: 0, tension: 0.2 }},
          {{ label: "Train MAE", data: DATA.train_mae, borderColor: COLORS.train, backgroundColor: COLORS.train, tension: 0.2 }},
          {{ label: "5-fold CV MAE (mean)", data: DATA.cv_mean, borderColor: COLORS.cv, backgroundColor: COLORS.cv, tension: 0.2 }},
          {{ label: "Holdout MAE", data: DATA.holdout_mae, borderColor: COLORS.holdout, backgroundColor: COLORS.holdout, tension: 0.2 }},
        ]
      }},
      options: {{
        plugins: {{ legend: LEGEND, tooltip: {{ mode: "index", intersect: false }} }},
        scales: {{ x: AXIS, y: {{ ...AXIS, title: {{ display: true, text: "MAE (pp)", color: "#8892a0" }} }} }},
        interaction: {{ mode: "nearest", axis: "x", intersect: false }},
      }}
    }});

    new Chart(document.getElementById("chart_quality"), {{
      type: "line",
      data: {{
        labels: DATA.labels,
        datasets: [
          {{ label: "Holdout Pearson r", data: DATA.pearson, borderColor: COLORS.pearson, backgroundColor: COLORS.pearson, tension: 0.2, yAxisID: "y" }},
          {{ label: "Holdout quality acc", data: DATA.qacc, borderColor: COLORS.qacc, backgroundColor: COLORS.qacc, tension: 0.2, yAxisID: "y" }},
        ]
      }},
      options: {{
        plugins: {{ legend: LEGEND }},
        scales: {{ x: AXIS, y: {{ ...AXIS, suggestedMin: 0, suggestedMax: 1 }} }},
      }}
    }});

    new Chart(document.getElementById("chart_n"), {{
      type: "bar",
      data: {{
        labels: DATA.labels,
        datasets: [{{ label: "Records", data: DATA.n_records, backgroundColor: COLORS.n }}]
      }},
      options: {{ plugins: {{ legend: LEGEND }}, scales: {{ x: AXIS, y: AXIS }} }}
    }});

    new Chart(document.getElementById("chart_bars"), {{
      type: "bar",
      data: {{
        labels: DATA.versions,
        datasets: [{{
          label: "Holdout MAE",
          data: DATA.holdout_mae,
          backgroundColor: DATA.promoted.map(p => p ? COLORS.promoted : COLORS.rejected),
        }}]
      }},
      options: {{ plugins: {{ legend: {{ display: false }} }}, scales: {{ x: AXIS, y: AXIS }} }}
    }});
  }}

  // ---- Leaf router charts ----------------------------------------------
  const LEAF = {leaf_data_json};
  const SLM_PALETTE = ["#55c37a", "#e4b350", "#4ea1ff", "#c07dd8", "#e57373", "#7fd0d1", "#d18f52", "#a6b1c2"];
  const slmColor = (i) => SLM_PALETTE[i % SLM_PALETTE.length];

  if (LEAF.labels.length === 0) {{
    ["leaf_cost_timeline","leaf_quality_timeline","leaf_breakdown","leaf_versions"].forEach(id => {{
      const el = document.getElementById(id);
      if (el) el.replaceWith(Object.assign(document.createElement("div"), {{
        textContent: "no leaf training runs yet — waiting on feedback",
        style: "color:#8892a0;font-size:12px;padding:20px 0;text-align:center;"
      }}));
    }});
  }} else {{
    // Per-SLM holdout MAE timeline (one line per SLM).
    new Chart(document.getElementById("leaf_cost_timeline"), {{
      type: "line",
      data: {{
        labels: LEAF.labels,
        datasets: LEAF.models.map((m, i) => ({{
          label: m,
          data: (LEAF.per_slm[m] || {{}}).holdout_mae || [],
          borderColor: slmColor(i),
          backgroundColor: slmColor(i),
          tension: 0.2,
          spanGaps: true,
        }})),
      }},
      options: {{
        plugins: {{ legend: LEGEND, tooltip: {{ mode: "index", intersect: false }} }},
        scales: {{ x: AXIS, y: {{ ...AXIS, title: {{ display: true, text: "MAE (tokens)", color: "#8892a0" }} }} }},
        interaction: {{ mode: "nearest", axis: "x", intersect: false }},
      }}
    }});

    // Per-SLM quality holdout accuracy timeline.
    new Chart(document.getElementById("leaf_quality_timeline"), {{
      type: "line",
      data: {{
        labels: LEAF.labels,
        datasets: LEAF.models.map((m, i) => ({{
          label: m,
          data: (LEAF.per_slm[m] || {{}}).quality_holdout_acc || [],
          borderColor: slmColor(i),
          backgroundColor: slmColor(i),
          tension: 0.2,
          spanGaps: true,
        }})),
      }},
      options: {{
        plugins: {{ legend: LEGEND }},
        scales: {{ x: AXIS, y: {{ ...AXIS, suggestedMin: 0, suggestedMax: 1 }} }},
      }}
    }});

    // Latest per-SLM breakdown: train / CV / holdout as grouped bars.
    new Chart(document.getElementById("leaf_breakdown"), {{
      type: "bar",
      data: {{
        labels: LEAF.latest_breakdown.models,
        datasets: [
          {{ label: "Train MAE",   data: LEAF.latest_breakdown.train_mae,   backgroundColor: COLORS.train }},
          {{ label: "CV MAE",      data: LEAF.latest_breakdown.cv_mae_mean, backgroundColor: COLORS.cv }},
          {{ label: "Holdout MAE", data: LEAF.latest_breakdown.holdout_mae, backgroundColor: COLORS.holdout }},
        ]
      }},
      options: {{
        plugins: {{ legend: LEGEND, tooltip: {{ mode: "index", intersect: false }} }},
        scales: {{ x: AXIS, y: AXIS }},
      }}
    }});

    // Per-version summary: mean holdout MAE across SLMs, colored by promotion.
    const perVersionMean = LEAF.labels.map((_, runIdx) => {{
      const vals = LEAF.models
        .map(m => (LEAF.per_slm[m]||{{}}).holdout_mae?.[runIdx])
        .filter(v => typeof v === "number");
      return vals.length ? vals.reduce((a,b) => a+b, 0) / vals.length : null;
    }});
    new Chart(document.getElementById("leaf_versions"), {{
      type: "bar",
      data: {{
        labels: LEAF.versions,
        datasets: [{{
          label: "Mean holdout MAE",
          data: perVersionMean,
          backgroundColor: LEAF.promoted.map(p => p ? COLORS.promoted : COLORS.rejected),
        }}]
      }},
      options: {{ plugins: {{ legend: {{ display: false }} }}, scales: {{ x: AXIS, y: AXIS }} }}
    }});
  }}
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_HERE / "config.yaml")
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    out = render_dashboard(config)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
