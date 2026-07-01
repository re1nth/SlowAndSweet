# SlowAndSweet — Claude Code plugin

This plugin wires Claude Code into the local SlowAndSweet SLM pool. With it
loaded, Claude gains:

- A `slm-batch` subagent that decomposes mechanical fan-out work
  (summarize N files, classify N tickets, translate N strings) and
  delegates the leaves to local Ollama-hosted small models.
- Slash commands: `/delegate <task>` (force-invoke the subagent),
  `/slm-doctor` (daemon health check), `/slm-stats` (usage).
- An MCP server registration pointing at `http://127.0.0.1:8090/mcp` with a
  bearer token injected from `~/.slowandsweet/token` at install time.
- A `PostToolUse` hook (matched to the `slm_*` tools) that appends one JSONL
  line per delegation call to `~/.slowandsweet/calls.jsonl`.

The subagent carries the "when to invoke" and "silently fall through on
abstain" guidance in its `description` field, which is what Claude reads —
this plugin intentionally does not ship a `CLAUDE.md`, since a plugin-root
`CLAUDE.md` is not auto-loaded into sessions.

## Install

See the top-level [`README.md`](../README.md) for the full system setup
(Ollama, queue server, MCP server). Then from the repo root:

```sh
pipx install ./slowandsweet   # installs the CLI daemon
slowandsweet init             # writes ~/.slowandsweet/token
sh plugin/install.sh          # renders .mcp.json with the token, symlinks the rest
# restart Claude Code
```

The install order matters: `install.sh` reads the token from
`~/.slowandsweet/token` and templates it into a real `.mcp.json` at
`~/.claude/plugins/slowandsweet/.mcp.json`. Run `slowandsweet init` first.

## Disable

- Temporary (survives Claude Code restart, no reinstall):
  `slowandsweet disable` — creates `~/.slowandsweet/disabled`; the MCP tool
  refuses plans until you run `slowandsweet enable`.
- Permanent: `rm -rf ~/.claude/plugins/slowandsweet`.

## Layout

```
plugin/
  .claude-plugin/plugin.json   manifest
  .mcp.json.template           slm-queue MCP registration (token placeholder)
  agents/slm-batch.md          the delegation subagent
  commands/                    /delegate, /slm-doctor, /slm-stats
  hooks/hooks.json             registers the PostToolUse hook (slm_* only)
  hooks/log-delegation.sh      the hook script
  install.sh                   symlink + render installer
```
