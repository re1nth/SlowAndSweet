# SlowAndSweet — Claude Code plugin

This plugin wires Claude Code into the local SlowAndSweet SLM pool. With it
loaded, Claude gains:

- A `slm-batch` subagent that decomposes mechanical fan-out work
  (summarize N files, classify N tickets, translate N strings) and
  delegates the leaves to local Ollama-hosted small models.
- **Auto-route**: a `UserPromptSubmit` hook that asks `slm-router`'s
  rule-based decomposer whether the prompt maps to a homogeneous
  multi-leaf pattern. On a match, the hook stashes a ready-to-run DAG
  with the queue and either (a) injects a compact `[autoroute]` context
  block instructing the frontier to trigger the stashed plan (default
  `context` mode) or (b) waits for the SLMs itself, denies the prompt,
  and returns the SLM outputs as the denial reason (`deny` mode; set
  `SLOWANDSWEET_HOOK_MODE=deny`). `context` mode preserves normal
  conversational flow but only saves tokens on larger fan-outs; `deny`
  mode saves tokens on every hit including tiny tasks but changes how
  the assistant turn is rendered — validate against your UI before
  rolling it out. On no match (or if the router is down), the hook stays
  silent and the frontier answers normally. Prefix a prompt with
  `/no-delegate` to force solo for that one turn.
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
  - `slowandsweet disable` — creates `~/.slowandsweet/disabled`; the MCP
    tool refuses plans **and** the auto-route hook goes quiet. Re-enable
    with `slowandsweet enable`.
  - `slowandsweet disable autoroute` — silences only the auto-route hook;
    explicit `/delegate` and manual `slm_submit_plan` still work.
    Re-enable with `slowandsweet enable autoroute`.
- Per-prompt override: prefix your prompt with `/no-delegate ` to force
  the frontier to answer solo for that one turn.
- Permanent: `rm -rf ~/.claude/plugins/slowandsweet`.

## Layout

```
plugin/
  .claude-plugin/plugin.json   manifest
  .mcp.json.template           slm-queue MCP registration (token placeholder)
  agents/slm-batch.md          the delegation subagent
  commands/                    /delegate, /slm-doctor, /slm-stats
  hooks/hooks.json             registers UserPromptSubmit + PostToolUse hooks
  hooks/autoroute.py           auto-route hook: calls slm-router /decompose
  hooks/log-delegation.sh      PostToolUse logger (slm_* only)
  install.sh                   symlink + render installer
```
