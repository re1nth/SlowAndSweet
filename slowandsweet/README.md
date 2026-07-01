# slowandsweet

User-facing CLI daemon for the [SlowAndSweet](../README.md) SLM delegation
pool. Replaces the manual install dance (edit two configs, run two Python
processes) with a single `pipx`-installable command.

## Install

```sh
pipx install -e ./slowandsweet
```

This puts a `slowandsweet` binary on your PATH.

## Commands

- `slowandsweet init` — create `~/.slowandsweet/`, generate a 32-byte hex
  auth token at `~/.slowandsweet/token` (mode 0600), initialize the SQLite
  metrics DB, and `ollama pull` every model declared in
  `slm-deploy/slms.yaml`. Will not auto-install Ollama; prints instructions
  and exits non-zero if it is missing.
- `slowandsweet doctor` — checklist of every prerequisite (state dir, token,
  DB schema, Ollama reachable, required models present, queue server, MCP
  server). `--json` for machine-readable. Queue/MCP being down is a WARN.
- `slowandsweet stats` — today's delegation counts and frontier-token savings
  from the SQLite log. `--json` for machine-readable.

## Files it touches

Everything lives under `~/.slowandsweet/`:

- `token` — local auth token
- `state.db` — SQLite call log + daily rollup
- `calls.jsonl` — written by the Claude Code plugin hooks (not by this CLI)

## License

MIT.
