#!/usr/bin/env bash
# Install the slm-router retraining systemd user timer.
#
# Renders slowandsweet-router-train.{service,timer}.template with paths to
# this repo's Python venv and train.py, drops them into
# ~/.config/systemd/user/, and enables the timer. It wakes nightly at
# 03:00 local time; train.py self-gates on new-record count, so a wake
# with no new data is a cheap no-op.
#
# Re-run this script anytime the repo moves or the venv is rebuilt.
#
# Usage:
#   ./scripts/install-systemd.sh [PYTHON]
#
# PYTHON defaults to $REPO/slm-router/.venv/bin/python if present,
# otherwise the first `python3` on PATH.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_BASE="slowandsweet-router-train"
SERVICE_TEMPLATE="$REPO_DIR/scripts/${UNIT_BASE}.service.template"
TIMER_TEMPLATE="$REPO_DIR/scripts/${UNIT_BASE}.timer.template"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_TARGET="$UNIT_DIR/${UNIT_BASE}.service"
TIMER_TARGET="$UNIT_DIR/${UNIT_BASE}.timer"
LOG_DIR="$HOME/.slowandsweet/logs"
LOG_OUT="$LOG_DIR/router-train.out.log"
LOG_ERR="$LOG_DIR/router-train.err.log"
TRAIN_PY="$REPO_DIR/slm-router/train.py"

if ! command -v systemctl >/dev/null 2>&1; then
    echo "systemctl not found; this script requires systemd." >&2
    echo "On macOS use scripts/install-launchd.sh instead." >&2
    exit 1
fi
if [[ ! -f "$SERVICE_TEMPLATE" ]]; then
    echo "template not found: $SERVICE_TEMPLATE" >&2
    exit 1
fi
if [[ ! -f "$TIMER_TEMPLATE" ]]; then
    echo "template not found: $TIMER_TEMPLATE" >&2
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

mkdir -p "$LOG_DIR" "$UNIT_DIR"

render() {
    sed \
        -e "s|__PYTHON__|$PYTHON|g" \
        -e "s|__TRAIN_PY__|$TRAIN_PY|g" \
        -e "s|__REPO_DIR__|$REPO_DIR|g" \
        -e "s|__LOG_OUT__|$LOG_OUT|g" \
        -e "s|__LOG_ERR__|$LOG_ERR|g" \
        "$1"
}

render "$SERVICE_TEMPLATE" > "$SERVICE_TARGET"
render "$TIMER_TEMPLATE"   > "$TIMER_TARGET"

systemctl --user daemon-reload
systemctl --user enable --now "${UNIT_BASE}.timer"

# Without lingering, user units stop when the user logs out. Warn but
# don't try to enable it (needs sudo and may not match user's policy).
if command -v loginctl >/dev/null 2>&1; then
    if ! loginctl show-user "$(id -un)" -p Linger 2>/dev/null | grep -q "Linger=yes"; then
        echo
        echo "note: user lingering is disabled — the timer will only run"
        echo "      while you're logged in. To let it fire after logout:"
        echo "        sudo loginctl enable-linger \"$(id -un)\""
    fi
fi

echo
echo "installed:  $SERVICE_TARGET"
echo "            $TIMER_TARGET"
echo "python:     $PYTHON"
echo "train.py:   $TRAIN_PY"
echo "logs:       $LOG_OUT"
echo "            $LOG_ERR"
echo
echo "verify with: systemctl --user list-timers ${UNIT_BASE}.timer"
echo "force a run: systemctl --user start ${UNIT_BASE}.service"
echo "tail logs:   journalctl --user -u ${UNIT_BASE}.service -f"
