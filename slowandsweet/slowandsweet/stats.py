"""`slowandsweet stats` — today's delegation totals from the SQLite log."""
from __future__ import annotations

import json
import sys

from slowandsweet import state
from slowandsweet.paths import DB_PATH


def run(as_json: bool = False) -> int:
    if not DB_PATH.exists():
        if as_json:
            json.dump({"error": "db not initialized"}, sys.stdout)
            sys.stdout.write("\n")
        else:
            print("no state db yet — run `slowandsweet init` first")
        return 0

    data = state.read_today(DB_PATH)
    totals = data["totals"]
    reasons = data["top_abstain_reasons"]

    has_activity = (
        totals.get("delegated_calls", 0)
        or totals.get("abstained_calls", 0)
        or totals.get("failed_calls", 0)
    )

    if as_json:
        json.dump(data, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if not has_activity:
        print("no calls recorded today")
        return 0

    print(f"date: {totals['date']}")
    print(f"  delegated_calls:                  {totals['delegated_calls']}")
    print(f"  abstained_calls:                  {totals['abstained_calls']}")
    print(f"  failed_calls:                     {totals['failed_calls']}")
    print(f"  estimated_frontier_tokens_saved:  {totals['frontier_tokens_saved']}")
    print(f"  estimated_wall_ms_saved:          {totals['wall_ms_saved']}")
    if reasons:
        print("  top abstain reasons:")
        for r in reasons:
            print(f"    {r['count']:>4}  {r['reason']}")
    return 0
