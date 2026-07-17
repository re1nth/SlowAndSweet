"""argparse dispatcher for the `slowandsweet` binary."""
from __future__ import annotations

import argparse
import sys

from slowandsweet import __version__


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="slowandsweet",
        description="Local-first daemon for the SlowAndSweet SLM delegation pool.",
    )
    p.add_argument("--version", action="version", version=f"slowandsweet {__version__}")
    sub = p.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    sub.add_parser(
        "init",
        help="set up ~/.slowandsweet/, generate a token, and pull required Ollama models",
    )

    pd = sub.add_parser("doctor", help="run health checks against the local install")
    pd.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    ps = sub.add_parser("stats", help="show today's delegation totals")
    ps.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    p_disable = sub.add_parser(
        "disable",
        help="stop delegating; MCP server will refuse plans",
    )
    p_disable.add_argument(
        "scope", nargs="?", default="all", choices=("all", "autoroute"),
        help=(
            "'all' (default) disables both auto-route and manual /delegate; "
            "'autoroute' silences only the UserPromptSubmit hook"
        ),
    )
    p_enable = sub.add_parser("enable", help="re-enable delegation after `disable`")
    p_enable.add_argument(
        "scope", nargs="?", default="all", choices=("all", "autoroute"),
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "init":
        from slowandsweet import init as init_mod
        return init_mod.run()
    if args.command == "doctor":
        from slowandsweet import doctor as doctor_mod
        return doctor_mod.run(as_json=args.json)
    if args.command == "stats":
        from slowandsweet import stats as stats_mod
        return stats_mod.run(as_json=args.json)
    if args.command == "disable":
        from slowandsweet import disable as disable_mod
        return disable_mod.run_disable(args.scope)
    if args.command == "enable":
        from slowandsweet import disable as disable_mod
        return disable_mod.run_enable(args.scope)
    # argparse with required=True guarantees we never get here.
    return 2


if __name__ == "__main__":
    sys.exit(main())
