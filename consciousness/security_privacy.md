# Security and privacy

## Current state (2026-07-09)

The whole system runs on `127.0.0.1`. Anyone with a shell on the
box has full read on `feedback.jsonl` (which now contains raw
user prompts), `state.db`, and the router's HTTP endpoints.
Token-based auth exists on the router (`SLM_ROUTER_TOKEN` env)
but is off by default. This is fine for a single-user local
prototype and not fine for anything else.

## Sensitive data on disk

| File                                    | Sensitivity | Retention   |
| --------------------------------------- | ----------- | ----------- |
| `slm-router/feedback.jsonl`             | **High** — raw prompts | forever |
| `slm-router/heads/*.joblib`             | Low — learned weights only | last 30 |
| `slm-router/metrics.jsonl`              | Low — aggregate metrics | forever |
| `slm-router/.emb_cache.npz`             | Low — embeddings (recoverable from prompts) | forever |
| `~/.slowandsweet/state.db`              | Medium — usage patterns, no prompts | forever |
| `slm-experiments/results/*.json`        | Medium — case prompts + outputs | forever |
| Ollama model cache                      | Low — public model weights | forever |

## Attack surface

- **`slm-router` HTTP endpoints on 127.0.0.1** — with auth off,
  any local process can `POST /feedback` with arbitrary data to
  poison the training set. Detection: `train.py`'s auto-promote
  gate should refuse a degraded head, but there's no explicit
  poison-detection.
- **`slm-queue` HTTP on 127.0.0.1** — accepts any plan JSON;
  arbitrary prompts flow through to Ollama. No prompt filtering.
- **`slm-queue/mcp_server.py`** — MCP tools exposed to Claude
  Code plugin. Tokens loaded from env. Same 127.0.0.1 posture.
- **The state DB path** — hard-coded to `~/.slowandsweet/state.db`,
  world-readable by default owner UID. Fine for single-user.

## What we don't do (that we should before multi-user)

- **PII stripping on prompt_text before persisting.** Currently
  we write raw prompts. In a shared environment, this is a
  compliance boundary.
- **Auth-by-default.** `SLM_ROUTER_TOKEN` needs to be required,
  not optional. When we ship a compose file, generate the token
  automatically.
- **Rate limiting.** No rate limits on any endpoint. A tight
  loop against `/route` would eat all encoder threads.
- **Input size caps.** `/route` will happily encode a 10 MB
  prompt (technically MiniLM truncates to 512 tokens, but the
  request body isn't capped upstream).
- **Feedback poisoning detection.** A malicious client could
  post thousands of records claiming mixture always wins by
  99 %. The auto-promote gate catches poisoning that *degrades*
  performance but not poisoning that *inflates* claimed
  performance without corresponding real gains.

## Log

### 2026-07-08 — `prompt_text` in feedback records is the first real privacy debt
Prior to this commit, feedback records had only `prompt_hash`.
`train.py` couldn't consume them, so we added `prompt_text`.
That flipped the sensitivity of `feedback.jsonl` from
Low (aggregate) to High (raw prompts). See
[technical_debt.md](technical_debt.md) for the mitigation
plan (encode-at-emit instead of persist-and-encode-later).

### 2026-07-04 — Bearer token support exists but is opt-in
Router server reads `SLM_ROUTER_TOKEN` env var and requires
`Authorization: Bearer` when set. Default behavior is open.
This is the right default for local dev and the wrong default
for any deployment.
