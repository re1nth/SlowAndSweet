# SlowAndSweet — Claude Code plugin

This plugin wires Claude Code into the local SlowAndSweet SLM pool. With it
loaded, Claude gains:

- A `slm-batch` subagent that decomposes mechanical fan-out work
  (summarize N files, classify N tickets, translate N strings) and
  delegates the leaves to local Ollama-hosted small models.
- Slash commands: `/delegate <task>` (force-invoke the subagent),
  `/slm-doctor` (daemon health check), `/slm-stats` (usage).
- An MCP server registration pointing at `http://127.0.0.1:8090/mcp` with
  a bearer token sourced from `${SLOWANDSWEET_TOKEN}`.
- A `PostToolUse` hook that appends one JSONL line per tool call to
  `~/.slowandsweet/calls.jsonl` for later inspection.

## Install

See the top-level [`README.md`](../README.md) for the full system setup
(Ollama, queue server, MCP server). Then from the repo root:

```sh
sh plugin/install.sh        # symlinks plugin/ into ~/.claude/plugins/
pipx install slowandsweet   # installs the CLI daemon
slowandsweet init           # writes ~/.slowandsweet/token
# restart Claude Code
```

## Disable

- Temporary: `export SLOWANDSWEET_DISABLE=1` and restart Claude Code.
- Permanent: `rm ~/.claude/plugins/slowandsweet`.

## Layout

```
plugin/
  .claude-plugin/plugin.json   manifest
  .mcp.json                    slm-queue MCP registration
  CLAUDE.md                    delegation rules (informational)
  agents/slm-batch.md          the delegation subagent
  commands/                    /delegate, /slm-doctor, /slm-stats
  hooks/hooks.json             registers the PostToolUse hook
  hooks/log-delegation.sh      the hook script
  install.sh                   symlink installer
```
