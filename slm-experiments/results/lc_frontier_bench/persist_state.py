"""One-off: pull the full result text for every run from the live slm-queue
and write a self-contained snapshot to slm_results.json alongside problems.json.

Run this before frontier_compare.py so the comparison doesn't depend on the
queue server still holding the runs in memory.
"""
import json, pathlib, urllib.request

ROOT = pathlib.Path(__file__).parent
SRC_DIR = pathlib.Path("/tmp/lc_bench")

problems = json.loads((SRC_DIR / "problems.json").read_text())
slm_rows = json.loads((SRC_DIR / "slm_results.json").read_text())

for row in slm_rows:
    state = json.loads(urllib.request.urlopen(
        f"http://127.0.0.1:8080/plans/{row['run_id']}", timeout=15
    ).read().decode())
    for node in row["nodes"]:
        live = state["nodes"][node["id"]]
        node["prompt_rendered"] = live.get("rendered_prompt") or live.get("prompt", "")
        node["result_text"] = live.get("result", "")
        node["started_at"] = live.get("started_at")
        node["finished_at"] = live.get("finished_at")

(ROOT / "problems.json").write_text(json.dumps(problems, indent=2))
(ROOT / "slm_results.json").write_text(json.dumps(slm_rows, indent=2))
print(f"Persisted {len(problems)} problems and {len(slm_rows)} SLM runs to {ROOT}")
