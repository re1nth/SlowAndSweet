#!/usr/bin/env bash
# Remove the slm-router retraining LaunchAgent.

set -euo pipefail

LABEL="com.slowandsweet.router-train"
TARGET="$HOME/Library/LaunchAgents/${LABEL}.plist"

launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true

if [[ -f "$TARGET" ]]; then
    rm "$TARGET"
    echo "removed:    $TARGET"
else
    echo "nothing to remove; $TARGET not present"
fi
