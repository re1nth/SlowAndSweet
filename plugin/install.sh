#!/bin/sh
# Symlink this plugin into ~/.claude/plugins/slowandsweet so Claude Code loads it.
set -eu

SRC="$(cd "$(dirname "$0")" && pwd)"
DEST_DIR="${HOME}/.claude/plugins"
DEST="${DEST_DIR}/slowandsweet"

mkdir -p "${DEST_DIR}"

if [ -e "${DEST}" ] || [ -L "${DEST}" ]; then
    echo "error: ${DEST} already exists."
    echo "       remove it first: rm \"${DEST}\""
    exit 1
fi

ln -s "${SRC}" "${DEST}"
echo "linked ${DEST} -> ${SRC}"

chmod +x "${SRC}/hooks/log-delegation.sh" 2>/dev/null || true

cat <<'EOF'

Next steps:
  1. pipx install slowandsweet      # installs the CLI daemon
  2. slowandsweet init              # writes ~/.slowandsweet/token and config
  3. restart Claude Code            # picks up the new plugin

To disable later: set SLOWANDSWEET_DISABLE=1, or rm the symlink above.
EOF
