# Ease of use

## Current state (2026-07-09)

Getting to a working system requires four terminals: `ollama serve`, the
queue server, the router server, and whatever's calling into them. The
queue's [README](../slm-queue/README.md) documents this; the router's
does not yet. Setup takes ~10 minutes end-to-end for someone who
already has Ollama installed, and closer to 30 minutes cold.

The Claude Code plugin (see [`plugin/`](../plugin/)) is the intended
end-user surface — customers shouldn't see any of the SLM machinery.
That is aspirational: the plugin exists, but the mixture arm is not
yet wired into the plugin's request flow.

## Metrics we track

- **First-run time-to-first-successful-plan**: manual stopwatch, no
  telemetry yet. Currently ~30 minutes cold, ~10 minutes warm.
- **Number of processes a user must start**: 3 (Ollama, queue, router),
  4 counting the caller. Target: 1 (via a supervisor or single-binary
  compose).
- **Error message quality**: subjective. Track by counting `except:
  pass` and `except: return` blocks in the codebase — those are places
  we're silently swallowing information the user might need.
  Current count in `slm-queue/planner.py`: 4 broad excepts (see
  `_record_run_metric` and `_report_router_feedback`). Necessary for
  robustness, but each is a spot where a user sees "it just didn't
  work" instead of a diagnostic.

## Log

### 2026-07-08 — Port collision on 8090 discovered
The router defaulted to 8090; `slm-queue/mcp_server.py` was already
listening there. Server crashed on startup with `OSError: Address
already in use`. Bumped router default to 8092 (`7feeb28`). Lesson:
we don't have a port allocation convention. Every service picks its
own port and hopes. A ports table in `consciousness/` or the top-level
README would prevent this recurring. See [scalability.md](scalability.md)
where this shows up again.

### 2026-07-08 — Router feedback fields required a schema deviation
`train.py` couldn't consume real feedback records because it needed
`prompt_text` to embed with MiniLM, but the DESIGN spec (§4.2) only
listed `prompt_hash`. Added `prompt_text` in both emitters. This is
easier now, harder later: when we take this to a multi-user setting,
the raw prompt text will be a privacy concern and we'll want the
option to disable it. Currently there's no config knob for that.

### 2026-07-07 — Router integration wasn't actually invoked
Agent C's original implementation added `choose_arm()` to the queue's
router module but didn't wire it into the `/plans` handler. So the
router was consulted zero times per plan and the feedback loop never
closed. Caught only during the manual end-to-end test. Lesson:
"tests pass" ≠ "end-to-end works"; smoke tests must include the
full expected control flow, not just each piece in isolation.

### 2026-07-04 — venv requirement added to README after user hit it
Initial `slm-queue/README.md` didn't call out `python -m venv .venv`
before `pip install`. First-time-user friction — recorded in commit
`f065440`.
