"""Static HTML dashboard for router training metrics.

Reads metrics.jsonl from the data dir and writes a single self-contained
HTML file (Chart.js from CDN, all data inlined) covering:

- Headline strip: current head, latest holdout / CV / train MAE, overfitting
  gap, total records, last run outcome.
- MAE timeline: train vs 5-fold CV (with std band) vs holdout across runs.
- Pearson r and quality accuracy timelines.
- Training set size over time.
- Per-version holdout MAE bar chart, highlighting the currently-promoted
  version.
- Recent runs table.

The train/CV gap is the overfitting signal: train_mae << cv_mae means the
model memorized the training slice.

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
    out_path = resolve_data_path(config, paths_cfg.get("dashboard_html", "dashboard.html"))

    rows = _load_metrics(metrics_path)
    state = _load_train_state(state_path)

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
</style>
</head>
<body>

<h1>slm-router training dashboard</h1>
<div class="subtitle">Generated {generated} &middot; {n_runs} scheduler runs &middot; {n_trained} actual trainings</div>

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
