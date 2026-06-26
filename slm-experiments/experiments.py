"""CLI entry point for the SxS experimentation harness.

Examples:

  # List the cases that will run
  python slm-experiments/experiments.py list

  # Run all cases (frontier = Anthropic via ANTHROPIC_API_KEY)
  python slm-experiments/experiments.py run

  # Run a subset
  python slm-experiments/experiments.py run --cases multi-doc-summary,code-explain

  # Force the local-frontier stand-in (no API key)
  python slm-experiments/experiments.py run --local-frontier

  # Render markdown for an existing run
  python slm-experiments/experiments.py report --run run-1782....
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from adapters.frontier import build_default  # noqa: E402
from adapters.slm import SLMQueueAdapter  # noqa: E402
from report import render_file  # noqa: E402
from runner import RESULTS_DIR, load_cases, run_experiment  # noqa: E402


def cmd_list(args: argparse.Namespace) -> None:
    cases = load_cases()
    print(f"{len(cases)} case(s):")
    for c in cases:
        print(f"  - {c['id']:<24} [{c.get('category', '?'):<14}] {c.get('name', '')}")


def cmd_run(args: argparse.Namespace) -> None:
    frontier = build_default(prefer_local=args.local_frontier)
    slm = SLMQueueAdapter(base_url=args.queue_url)
    case_ids = [s.strip() for s in args.cases.split(",")] if args.cases else None
    snapshot = run_experiment(
        frontier=frontier,
        slm=slm,
        case_ids=case_ids,
        max_parallel_cases=args.parallel,
        seed=args.seed,
        run_label=args.label,
    )
    md_path = RESULTS_DIR / f"{snapshot['run_id']}.md"
    from report import render
    md_path.write_text(render(snapshot))
    print(f"==> wrote {md_path}")
    print()
    print(render(snapshot))


def cmd_report(args: argparse.Namespace) -> None:
    path = Path(args.run) if args.run.endswith(".json") else RESULTS_DIR / f"{args.run}.json"
    print(render_file(path))


def main() -> None:
    ap = argparse.ArgumentParser(description="SxS experimentation harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_list = sub.add_parser("list", help="list available cases")
    sp_list.set_defaults(func=cmd_list)

    sp_run = sub.add_parser("run", help="run experiment")
    sp_run.add_argument("--cases", default=None, help="comma-separated case ids (default: all)")
    sp_run.add_argument("--queue-url", default="http://127.0.0.1:8080")
    sp_run.add_argument("--parallel", type=int, default=1,
                        help="max cases to run concurrently (default 1)")
    sp_run.add_argument("--seed", type=int, default=7)
    sp_run.add_argument("--label", default=None, help="optional run label")
    sp_run.add_argument("--local-frontier", action="store_true",
                        help="use local Ollama as frontier stand-in")
    sp_run.set_defaults(func=cmd_run)

    sp_rep = sub.add_parser("report", help="render markdown report for a run")
    sp_rep.add_argument("--run", required=True, help="run id or path to snapshot json")
    sp_rep.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
