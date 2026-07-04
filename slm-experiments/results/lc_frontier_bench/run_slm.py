"""Submit each of the 10 LeetCode problems as an SLM DAG plan and record per-node timings/tokens."""
import json, urllib.request, time, pathlib, tiktoken

ENC = tiktoken.get_encoding("cl100k_base")
BASE = "http://127.0.0.1:8080"


def http_json(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    return json.loads(urllib.request.urlopen(req, timeout=30).read().decode())


def build_plan(problem: dict) -> dict:
    """Two-node DAG: sketch algorithm, then write code that depends on the sketch."""
    stmt = problem["statement"]
    return {
        "plan_id": f"lc_{problem['slug'].replace('-', '_')}",
        "description": f"Solve LeetCode {problem['title']}",
        "nodes": [
            {
                "id": "sketch",
                "depends_on": [],
                "prompt": (
                    "Describe the algorithm to solve the following LeetCode problem in "
                    "3-5 concise bullet points. No code, no preamble.\n\n"
                    f"PROBLEM: {problem['title']}\n\n{stmt}"
                ),
            },
            {
                "id": "solve",
                "depends_on": ["sketch"],
                "prompt": (
                    "Write a single Python function that solves the LeetCode problem "
                    "below. Follow the given algorithm sketch. Return ONLY code inside a "
                    "```python``` block, no explanation.\n\n"
                    f"PROBLEM: {problem['title']}\n\n{stmt}\n\n"
                    "ALGORITHM SKETCH:\n{{sketch.result}}"
                ),
            },
        ],
    }


def run_one(problem: dict) -> dict:
    plan = build_plan(problem)
    started = time.time()
    resp = http_json("POST", "/plans", {"plan": plan})
    run_id = resp["run_id"]
    while True:
        state = http_json("GET", f"/plans/{run_id}")
        statuses = {k: v["status"] for k, v in state["nodes"].items()}
        if all(s == "done" for s in statuses.values()):
            break
        if any(s == "error" for s in statuses.values()):
            raise RuntimeError(f"plan {run_id} errored: {statuses}")
        time.sleep(3)
    wall = time.time() - started

    total_prompt_tok = 0
    total_output_tok = 0
    node_details = []
    for nid, node in state["nodes"].items():
        rendered_prompt = node.get("rendered_prompt") or node.get("prompt", "")
        result = node.get("result", "")
        in_tok = len(ENC.encode(rendered_prompt))
        out_tok = len(ENC.encode(result))
        total_prompt_tok += in_tok
        total_output_tok += out_tok
        node_details.append({
            "id": nid,
            "model": node.get("model"),
            "eval_count_ollama": node.get("eval_count"),
            "tiktoken_input": in_tok,
            "tiktoken_output": out_tok,
        })
    return {
        "slug": problem["slug"],
        "title": problem["title"],
        "difficulty": problem["difficulty"],
        "run_id": run_id,
        "wall_s": round(wall, 1),
        "slm_input_tok_tiktoken": total_prompt_tok,
        "slm_output_tok_tiktoken": total_output_tok,
        "nodes": node_details,
        "final_code": state["nodes"]["solve"].get("result", ""),
    }


def main():
    problems = json.loads(pathlib.Path("/tmp/lc_bench/problems.json").read_text())
    results = []
    for i, p in enumerate(problems, 1):
        print(f"[{i}/{len(problems)}] {p['title']:45}", flush=True)
        r = run_one(p)
        print(f"    run={r['run_id']}  wall={r['wall_s']}s  in={r['slm_input_tok_tiktoken']}  out={r['slm_output_tok_tiktoken']}", flush=True)
        results.append(r)
    pathlib.Path("/tmp/lc_bench/slm_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote /tmp/lc_bench/slm_results.json ({len(results)} rows)")


if __name__ == "__main__":
    main()
