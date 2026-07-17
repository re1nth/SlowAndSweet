"""Enable/disable the delegation pool via a touch file.

Env vars set in the user's shell don't reach the MCP server or the
UserPromptSubmit hook process, so kill switches are files under
~/.slowandsweet/.

Two scopes:

- "all" (default): touches DISABLED_FLAG. The MCP server refuses plans;
  the UserPromptSubmit hook also honors this flag and stops injecting.
- "autoroute": touches only AUTOROUTE_DISABLED_FLAG. The auto-route hook
  goes quiet, but explicit `/delegate` + `slm_submit_plan` still work.
"""
from __future__ import annotations

from . import paths


VALID_SCOPES = ("all", "autoroute")


def _flag(scope: str):
    if scope == "all":
        return paths.DISABLED_FLAG
    if scope == "autoroute":
        return paths.AUTOROUTE_DISABLED_FLAG
    raise ValueError(f"unknown scope: {scope!r}")


def _re_enable_hint(scope: str) -> str:
    return "slowandsweet enable" if scope == "all" else f"slowandsweet enable {scope}"


def run_disable(scope: str = "all") -> int:
    paths.ensure_state_dir()
    flag = _flag(scope)
    flag.touch(exist_ok=True)
    if scope == "all":
        print(f"delegation disabled ({flag} present)")
    else:
        print(f"auto-route disabled ({flag} present); manual /delegate still works")
    print(f"re-enable with: {_re_enable_hint(scope)}")
    return 0


def run_enable(scope: str = "all") -> int:
    flag = _flag(scope)
    if flag.exists():
        flag.unlink()
        label = "delegation" if scope == "all" else "auto-route"
        print(f"{label} enabled ({flag} removed)")
    else:
        label = "delegation" if scope == "all" else "auto-route"
        print(f"{label} already enabled")
    return 0


def is_disabled() -> bool:
    return paths.DISABLED_FLAG.exists()


def is_autoroute_disabled() -> bool:
    return paths.DISABLED_FLAG.exists() or paths.AUTOROUTE_DISABLED_FLAG.exists()
