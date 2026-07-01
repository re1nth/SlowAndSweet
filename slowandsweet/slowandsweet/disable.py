"""Enable/disable the delegation pool via a touch file.

Env vars set in the user's shell don't reach the MCP server process, so
the kill switch is a file at ~/.slowandsweet/disabled. The MCP server's
tool handlers check for it on every call.
"""
from __future__ import annotations

from . import paths


def run_disable() -> int:
    paths.ensure_state_dir()
    paths.DISABLED_FLAG.touch(exist_ok=True)
    print(f"delegation disabled ({paths.DISABLED_FLAG} present)")
    print("re-enable with: slowandsweet enable")
    return 0


def run_enable() -> int:
    if paths.DISABLED_FLAG.exists():
        paths.DISABLED_FLAG.unlink()
        print(f"delegation enabled ({paths.DISABLED_FLAG} removed)")
    else:
        print("delegation already enabled")
    return 0


def is_disabled() -> bool:
    return paths.DISABLED_FLAG.exists()
