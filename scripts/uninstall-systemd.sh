#!/usr/bin/env bash
# Remove the slm-router retraining systemd user timer.

set -euo pipefail

UNIT_BASE="slowandsweet-router-train"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_TARGET="$UNIT_DIR/${UNIT_BASE}.service"
TIMER_TARGET="$UNIT_DIR/${UNIT_BASE}.timer"

if ! command -v systemctl >/dev/null 2>&1; then
    echo "systemctl not found; nothing to remove." >&2
    exit 0
fi

systemctl --user disable --now "${UNIT_BASE}.timer" 2>/dev/null || true
systemctl --user stop         "${UNIT_BASE}.service" 2>/dev/null || true

removed=0
for f in "$TIMER_TARGET" "$SERVICE_TARGET"; do
    if [[ -f "$f" ]]; then
        rm "$f"
        echo "removed:    $f"
        removed=1
    fi
done

if [[ "$removed" -eq 0 ]]; then
    echo "nothing to remove; no unit files under $UNIT_DIR"
fi

systemctl --user daemon-reload
