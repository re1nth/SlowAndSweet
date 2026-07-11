#!/usr/bin/env bash
# Install the slm-router retraining LaunchAgent.
#
# Renders the plist template with paths to this repo's Python venv and
# train.py, drops it into ~/Library/LaunchAgents, and loads it. The
# agent wakes nightly at 03:00 local time and calls train.py; train.py
# self-gates on new-record count so a wake with no new data is a
# cheap no-op.
#
# Re-run this script anytime the repo moves or the venv is rebuilt.
#
# Usage:
#   ./scripts/install-launchd.sh [PYTHON]
#
# PYTHON defaults to $REPO/slm-router/.venv/bin/python if present,
# otherwise the first `python3` on PATH.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.slowandsweet.router-train"
TEMPLATE="$REPO_DIR/scripts/${LABEL}.plist.template"
TARGET="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG_DIR="$HOME/.slowandsweet/logs"
LOG_OUT="$LOG_DIR/router-train.out.log"
LOG_ERR="$LOG_DIR/router-train.err.log"
TRAIN_PY="$REPO_DIR/slm-router/train.py"

if [[ ! -f "$TEMPLATE" ]]; then
    echo "template not found: $TEMPLATE" >&2
    exit 1
fi
if [[ ! -f "$TRAIN_PY" ]]; then
    echo "train.py not found: $TRAIN_PY" >&2
    exit 1
fi

PYTHON="${1:-}"
if [[ -z "$PYTHON" ]]; then
    if [[ -x "$REPO_DIR/slm-router/.venv/bin/python" ]]; then
        PYTHON="$REPO_DIR/slm-router/.venv/bin/python"
    elif [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
        PYTHON="$REPO_DIR/.venv/bin/python"
    else
        PYTHON="$(command -v python3 || true)"
    fi
fi
if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
    echo "no usable python interpreter; pass one as the first arg" >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$(dirname "$TARGET")"

sed \
    -e "s|__PYTHON__|$PYTHON|g" \
    -e "s|__TRAIN_PY__|$TRAIN_PY|g" \
    -e "s|__REPO_DIR__|$REPO_DIR|g" \
    -e "s|__LOG_OUT__|$LOG_OUT|g" \
    -e "s|__LOG_ERR__|$LOG_ERR|g" \
    "$TEMPLATE" > "$TARGET"

# Unload if already loaded (bootout is quieter than unload; ignore errors).
launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true

launchctl bootstrap "gui/$(id -u)" "$TARGET"

echo "installed:  $TARGET"
echo "python:     $PYTHON"
echo "train.py:   $TRAIN_PY"
echo "logs:       $LOG_OUT"
echo "            $LOG_ERR"
echo
echo "verify with: launchctl print gui/$(id -u)/${LABEL} | head"
echo "force a run: launchctl kickstart -k gui/$(id -u)/${LABEL}"

# macOS TCC: launchd-launched processes cannot read files under ~/Documents/,
# ~/Downloads/, ~/Desktop/, or removable/network volumes without explicit user
# consent. If the repo lives in one of these locations, the job will fail with
# "Operation not permitted" until Full Disk Access is granted.
case "$REPO_DIR" in
    "$HOME/Documents/"*|"$HOME/Downloads/"*|"$HOME/Desktop/"*)
        cat <<TCC

WARNING: repo is under a TCC-protected directory ($REPO_DIR).
Grant access before the job can run:
  System Settings → Privacy & Security → Full Disk Access
  → toggle on "sh" (or add $PYTHON manually with the + button)
Alternatively, move the repo somewhere outside ~/Documents, ~/Downloads,
and ~/Desktop and re-run this script.

Check $LOG_ERR after the next fire; if you see
'Operation not permitted', TCC is still blocking.
TCC
        ;;
esac
