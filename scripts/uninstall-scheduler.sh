#!/usr/bin/env bash
# OS-detecting wrapper around uninstall-launchd.sh / uninstall-systemd.sh.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

case "$(uname -s)" in
    Darwin)
        exec "$HERE/uninstall-launchd.sh" "$@"
        ;;
    Linux)
        exec "$HERE/uninstall-systemd.sh" "$@"
        ;;
    *)
        echo "unsupported platform: $(uname -s)" >&2
        exit 1
        ;;
esac
