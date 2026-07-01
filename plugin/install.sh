#!/bin/sh
# Install this plugin into ~/.claude/plugins/slowandsweet.
#
# We can't just symlink the whole plugin/ dir because the MCP config needs the
# real bearer token substituted into .mcp.json — Claude Code doesn't expand
# ${SLOWANDSWEET_TOKEN} from the user's shell at load time. So we create a real
# directory, symlink each file/subdir except the .mcp.json.template, and render
# the template into a real .mcp.json with the token from ~/.slowandsweet/token.
set -eu

SRC="$(cd "$(dirname "$0")" && pwd)"
DEST_DIR="${HOME}/.claude/plugins"
DEST="${DEST_DIR}/slowandsweet"
TOKEN_FILE="${HOME}/.slowandsweet/token"

if [ ! -f "${TOKEN_FILE}" ]; then
    echo "error: ${TOKEN_FILE} not found."
    echo ""
    echo "Run \`pipx install ${SRC}/../slowandsweet\` and then \`slowandsweet init\`"
    echo "before installing the plugin."
    exit 1
fi

TOKEN=$(tr -d '\r\n' < "${TOKEN_FILE}")
if [ -z "${TOKEN}" ]; then
    echo "error: ${TOKEN_FILE} is empty; re-run \`slowandsweet init\`."
    exit 1
fi

mkdir -p "${DEST_DIR}"

if [ -e "${DEST}" ] || [ -L "${DEST}" ]; then
    echo "error: ${DEST} already exists."
    echo "       remove it first: rm -rf \"${DEST}\""
    exit 1
fi

mkdir -p "${DEST}"

# Symlink every top-level entry except install.sh itself and the template.
for entry in "${SRC}"/* "${SRC}"/.*; do
    name=$(basename "${entry}")
    case "${name}" in
        .|..|install.sh|.mcp.json.template) continue ;;
    esac
    ln -s "${entry}" "${DEST}/${name}"
done

# Render the MCP config with the real token.
# Use awk (portable, no sed-escape issues even though token is hex-only).
awk -v tok="${TOKEN}" '{
    while (match($0, /\$\{SLOWANDSWEET_TOKEN\}/)) {
        $0 = substr($0, 1, RSTART-1) tok substr($0, RSTART+RLENGTH)
    }
    print
}' "${SRC}/.mcp.json.template" > "${DEST}/.mcp.json"
chmod 0600 "${DEST}/.mcp.json"

chmod +x "${SRC}/hooks/log-delegation.sh" 2>/dev/null || true

echo "installed plugin at ${DEST}"
echo "  .mcp.json rendered with token from ${TOKEN_FILE}"
echo ""
echo "Next: restart Claude Code so it picks up the new plugin."
echo ""
echo "To disable delegation later:   slowandsweet disable"
echo "To remove the plugin entirely: rm -rf \"${DEST}\""
