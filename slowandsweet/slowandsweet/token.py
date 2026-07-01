"""Read/write the local auth token at ~/.slowandsweet/token."""
from __future__ import annotations

import os
import secrets

from slowandsweet.paths import TOKEN_PATH, ensure_state_dir


def read_token() -> str | None:
    if not TOKEN_PATH.exists():
        return None
    try:
        return TOKEN_PATH.read_text().strip() or None
    except OSError:
        return None


def write_token() -> str:
    """Generate token if missing; otherwise return the existing one. Idempotent."""
    existing = read_token()
    if existing:
        return existing
    ensure_state_dir()
    token = secrets.token_hex(32)
    # Write then chmod so the bits we want are applied even if the umask
    # would have allowed group/other read on creation.
    TOKEN_PATH.write_text(token + "\n")
    os.chmod(TOKEN_PATH, 0o600)
    return token
