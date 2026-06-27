"""prism CLI.

Examples:

  # Decompose a free-text prompt and run end-to-end.
  python prism/cli.py run "Summarize the three articles below ..."

  # Execute a pre-authored annotated DAG.
  python prism/cli.py run --plan my_plan.yaml

  # Force the local-frontier stand-in (no Anthropic key).
  python prism/cli.py run --plan my_plan.yaml --local-frontier
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Make prism modules importable as flat names.
sys.path.insert(0, str(HERE))
# Reuse the adapters from slm-experiments so we don't duplicate them.
sys.path.insert(0, str(ROOT / "slm-experiments"))

from adapters.frontier import build_default  # noqa: E402
from classifier import Classifier  # noqa: E402
from dag import NodeSpec  # noqa: E402
from decomposer import decompose, plan_to_specs  # noqa: E402
from executor import PrismExecutor  # noqa: E402
from policy import Policy  # noqa: E402


def _load_plan_yaml(path: Path) -> tuple[str, str, list[NodeSpec]]:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    plan_id = data.get("plan_id", path.stem)
    desc = data.get("description", "")
    specs = [
        NodeSpec(
            id=n["id"],
            prompt=n["prompt"],
            depends_on=list(n.get("depends_on", [])),
            type=n.get("type"),
            backend=n.get("backend"),
            model=n.get("model"),
        )
        for n in data["nodes"]
    ]
    return plan_id, desc, specs


def cmd_run(args: argparse.Namespace) -> None:
    policy = Policy.load()
    frontier = build_default(prefer_local=args.local_frontier)
    classifier = None if args.no_classifier else Classifier(policy, model=args.classifier_model)
    executor = PrismExecutor(
        frontier=frontier,
        policy=policy,
        classifier=classifier,
        slm_task_url=args.queue_url,
        max_parallel=args.parallel,
    )

    if args.plan:
        plan_id, desc, specs = _load_plan_yaml(Path(args.plan))
    elif args.prompt:
        plan = decompose(args.prompt, frontier=frontier, policy=policy)
        plan_id = plan.get("plan_id", "inline")
        desc = plan.get("description", "")
        specs = plan_to_specs(plan)
    else:
        raise SystemExit("provide either --plan PATH or a positional prompt")

    run = executor.run(specs, plan_id=plan_id, description=desc)
    summary = run.to_dict()
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
        return

    print(f"=== prism run `{plan_id}` ===")
    print(f"description: {desc}")
    print(f"nodes: {len(run.nodes)} · wall: {run.wall_seconds:.1f}s · "
          f"classifier calls: {run.classifier_calls} ({run.classifier_seconds:.1f}s)")
    print()
    for n in run.nodes:
        marker = "ERR" if n.error else "ok"
        print(
            f"  [{marker}] {n.id:<14} backend={n.backend:<3} "
            f"type={n.type or '-':<14} model={n.model or '-':<22} "
            f"in/out={n.tokens_in}/{n.tokens_out} t={n.wall_seconds:.1f}s"
        )
        if n.error:
            print(f"        error: {n.error}")
    print()
    print("=== final output ===")
    print(run.final_output)


def main() -> None:
    ap = argparse.ArgumentParser(description="prism — LLM+SLM routed DAG executor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("run", help="run a prompt or a plan")
    sp.add_argument("prompt", nargs="?", default=None,
                    help="free-text prompt to decompose and run")
    sp.add_argument("--plan", default=None,
                    help="path to a pre-authored annotated DAG yaml")
    sp.add_argument("--queue-url", default="http://127.0.0.1:8080")
    sp.add_argument("--local-frontier", action="store_true",
                    help="use a local Ollama model in place of Anthropic")
    sp.add_argument("--no-classifier", action="store_true",
                    help="don't classify untagged nodes; route them to default_unknown")
    sp.add_argument("--classifier-model", default="smollm2:1.7b")
    sp.add_argument("--parallel", type=int, default=4)
    sp.add_argument("--json", action="store_true",
                    help="emit the run snapshot as JSON")
    sp.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
