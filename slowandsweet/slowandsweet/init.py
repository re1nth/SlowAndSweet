"""`slowandsweet init` — prepare ~/.slowandsweet and pull required Ollama models."""
from __future__ import annotations

import shutil
import subprocess
import sys

from slowandsweet import state, token
from slowandsweet.paths import (
    DB_PATH,
    STATE_DIR,
    TOKEN_PATH,
    ensure_state_dir,
    slms_yaml_path,
)

DEFAULT_MODELS = ["smollm2:1.7b", "gemma2:2b", "llama3.2:3b", "qwen2.5:3b"]


def _ollama_install_instructions() -> str:
    return (
        "Ollama is required but was not found on PATH.\n"
        "  macOS:   brew install ollama   (or download from ollama.com)\n"
        "  Linux:   curl -fsSL https://ollama.com/install.sh | sh\n"
        "Then start the daemon (`ollama serve &`) and re-run `slowandsweet init`."
    )


def _have_ollama() -> bool:
    if shutil.which("ollama") is None:
        return False
    try:
        r = subprocess.run(
            ["ollama", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def _read_models() -> tuple[list[str], str]:
    """Return (models, source_label)."""
    path = slms_yaml_path()
    if path is None:
        return DEFAULT_MODELS, "built-in defaults (slm-deploy/slms.yaml not found)"
    try:
        import yaml  # type: ignore
    except ImportError:
        return DEFAULT_MODELS, "built-in defaults (PyYAML unavailable)"
    try:
        docs = list(yaml.safe_load_all(path.read_text()))
    except Exception as e:  # noqa: BLE001
        print(f"warning: failed to parse {path}: {e}", file=sys.stderr)
        return DEFAULT_MODELS, "built-in defaults (parse error)"
    models: list[str] = []
    for d in docs:
        if not isinstance(d, dict):
            continue
        if d.get("kind") != "SLMDeployment":
            continue
        m = (d.get("spec") or {}).get("model")
        if isinstance(m, str) and m not in models:
            models.append(m)
    if not models:
        return DEFAULT_MODELS, "built-in defaults (no SLMDeployment docs)"
    return models, str(path)


def _installed_models() -> set[str]:
    try:
        r = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return set()
    if r.returncode != 0:
        return set()
    installed: set[str] = set()
    for i, line in enumerate(r.stdout.splitlines()):
        if i == 0 or not line.strip():
            continue  # header
        installed.add(line.split()[0])
    return installed


def _pull_model(model: str) -> int:
    print(f"  pulling {model} ...")
    try:
        # Stream stdout/stderr through to the user.
        r = subprocess.run(["ollama", "pull", model])
    except (subprocess.SubprocessError, OSError) as e:
        print(f"  failed to pull {model}: {e}", file=sys.stderr)
        return 1
    return r.returncode


def run() -> int:
    ensure_state_dir()
    print(f"state directory: {STATE_DIR}")

    tok = token.write_token()
    print(f"token: {TOKEN_PATH} (32-byte hex, mode 0600)")
    # Don't print the token itself; the file is the source of truth.
    _ = tok

    if not _have_ollama():
        print()
        print(_ollama_install_instructions(), file=sys.stderr)
        return 2

    models, source = _read_models()
    print(f"models (from {source}): {', '.join(models)}")

    installed = _installed_models()
    pull_failures = 0
    for m in models:
        if m in installed:
            print(f"  {m}: already present, skipping")
            continue
        if _pull_model(m) != 0:
            pull_failures += 1

    state.init_db(DB_PATH)
    print(f"db: {DB_PATH} (schema v{state.schema_version(DB_PATH)})")

    print()
    print("setup complete.")
    print("next steps:")
    print("  1. start ollama (if not already):     ollama serve &")
    print("  2. start the queue:                   python3 slm-queue/server.py --port 8080 &")
    print("  3. start the MCP bridge:              python3 slm-queue/mcp_server.py --port 8090 &")
    print("  4. install the Claude Code plugin:    bash plugin/install.sh")
    print("  5. verify everything:                 slowandsweet doctor")
    return 1 if pull_failures else 0
