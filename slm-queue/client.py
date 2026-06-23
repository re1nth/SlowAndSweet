"""Demo client: submits a batch of prompts, polls until all are done, prints
results plus a final /status snapshot.

Usage:
    python3 slm-queue/client.py                    # uses built-in demo prompts
    python3 slm-queue/client.py --prompt "..." --prompt "..."
    python3 slm-queue/client.py --base http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


DEMO_PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number.",
    "Calculate 17 * 23 and show your reasoning step by step.",
    "Summarize photosynthesis in one sentence.",
    "Explain SQL JOIN to a beginner with a tiny example.",
    "Name three jazz musicians active in the 1950s.",
]


def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--prompt", action="append",
                    help="prompt to submit (repeatable); overrides demo set")
    ap.add_argument("--poll-interval", type=float, default=0.4)
    args = ap.parse_args()

    prompts = args.prompt or DEMO_PROMPTS

    print("submitting:")
    pending: set[str] = set()
    for p in prompts:
        r = _post(f"{args.base}/tasks", {"prompt": p})
        print(f"  {r['task_id']}  -> {r['model']:<14}  {p[:60]}")
        pending.add(r["task_id"])

    print("\nresults (printed as they complete):")
    while pending:
        time.sleep(args.poll_interval)
        for tid in list(pending):
            r = _get(f"{args.base}/tasks/{tid}")
            if r["status"] in ("done", "error"):
                pending.remove(tid)
                if r["status"] == "done":
                    elapsed = r["finished_at"] - r["submitted_at"]
                    body = r["result"].strip().replace("\n", " ")
                    if len(body) > 140:
                        body = body[:137] + "..."
                    print(f"  done  {elapsed:5.2f}s  {tid}  "
                          f"[{r['worker']}]  {body}")
                else:
                    print(f"  ERR             {tid}  {r['error']}")

    print("\nfinal /status:")
    print(json.dumps(_get(f"{args.base}/status"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
