#!/usr/bin/env bash
# OS-detecting wrapper around install-launchd.sh / install-systemd.sh.
#
# On macOS this delegates to install-launchd.sh (LaunchAgent, nightly 03:00).
# On Linux this delegates to install-systemd.sh (systemd --user timer,
# nightly 03:00). All arguments are forwarded to the underlying script.
#
# Usage:
#   ./scripts/install-scheduler.sh [PYTHON]

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

case "$(uname -s)" in
    Darwin)
        exec "$HERE/install-launchd.sh" "$@"
        ;;
    Linux)
        exec "$HERE/install-systemd.sh" "$@"
        ;;
    *)
        echo "unsupported platform: $(uname -s)" >&2
        echo "supported: macOS (launchd), Linux (systemd)" >&2
        exit 1
        ;;
esac
