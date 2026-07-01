# SlowAndSweet delegation

This project has a local pool of small language models reachable through the
`slm-queue` MCP server. The `slm-batch` subagent (defined in
`agents/slm-batch.md`) wraps the two MCP tools `slm_submit_plan` and
`slm_wait_plan` and decides whether to delegate or abstain.

## When to invoke `slm-batch`

Invoke it when the user's request contains **three or more homogeneous,
mechanical items**: summarize each of these files, classify these tickets,
translate these strings, extract fields from these records, draft one
paragraph per bullet. The SLMs are fast and parallel for this shape.

You can invoke it implicitly (the subagent's `description` tells Claude when
it qualifies) or explicitly via `/delegate <task>`.

## When NOT to invoke

- Single-shot reasoning, judgment calls, or anything where being wrong is
  expensive (architecture decisions, code review verdicts, security claims).
- Fact-sensitive tasks where a hallucination from a 2B model would mislead.
- Fewer than 3 leaves — the dispatch overhead isn't worth it.

## What to do if it abstains

The subagent returns `ABSTAIN: <reason>` when the task doesn't qualify.
**Don't comment on the abstention.** Just answer the user's question normally
as if delegation was never considered.

## If the daemon is unreachable

If the MCP server isn't running (calls error out or time out), silently
continue without delegating. Don't surface the failure to the user unless
they explicitly asked about the SLM pool.
