#!/bin/sh
# PostToolUse hook: append a coarse JSONL log line per tool call.
# Reads the Claude Code tool-call envelope on stdin; never blocks the call.
set -u

LOG_DIR="${HOME}/.slowandsweet"
LOG_FILE="${LOG_DIR}/calls.jsonl"

mkdir -p "${LOG_DIR}" 2>/dev/null || exit 0

INPUT=$(cat)

# Always succeed even if jq is missing or input isn't JSON.
if command -v jq >/dev/null 2>&1; then
    printf '%s' "${INPUT}" | jq -c \
        --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{
            ts: $ts,
            tool: (.tool_name // .toolName // "unknown"),
            duration_ms: (.tool_response.duration_ms // .duration_ms // null),
            session_id: (.session_id // null)
        }' >> "${LOG_FILE}" 2>/dev/null || \
        printf '{"ts":"%s","raw":true}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_FILE}" 2>/dev/null
else
    printf '{"ts":"%s","jq":"missing","raw":%s}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$(printf '%s' "${INPUT}" | wc -c | tr -d ' ')" \
        >> "${LOG_FILE}" 2>/dev/null
fi

exit 0
