# `slm-router`

A tiny HTTP sidecar that decides, per prompt, whether to send it straight to Claude Code (**solo**) or through the local SLM decomposition pipeline (**mixture**). One MiniLM encoder plus a small trained head; falls back to a rule-based heuristic when no head is loaded. Full spec: [`DESIGN.md`](./DESIGN.md).

## Layout

```
slm-router/
├── DESIGN.md                    # full design
├── README.md                    # this file
├── requirements.txt
├── config.yaml                  # thresholds, paths, server host/port
│
├── server.py                    # stdlib ThreadingHTTPServer
├── model.py                     # Encoder + Head (frozen MiniLM + sklearn heads)
├── policy.py                    # pure decide(head_output, config) -> Decision
├── heuristic_fallback.py        # cold-start rule-based Decision
│
├── heads/                       # (populated by bootstrap.py / train.py)
│   ├── HEAD                     # text file naming the active version
│   └── <version>.joblib
│
└── feedback.jsonl               # (populated by the queue integration)
```

## Quick start

```
python -m venv .venv
.venv/bin/pip install -r slm-router/requirements.txt
.venv/bin/python slm-router/server.py --port 8092 &
curl -s -X POST http://127.0.0.1:8092/route \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "What is 2 + 2?"}'
```

Flags:
- `--port N`      override `server.port` from `config.yaml`
- `--config PATH` use a different config file

Auth: set `SLM_ROUTER_TOKEN` in the environment to require `Authorization: Bearer <token>` on every request. Unset means open on localhost.

## Endpoints

| Method | Path         | Purpose                                                                                     |
| ------ | ------------ | ------------------------------------------------------------------------------------------- |
| POST   | `/route`     | `{prompt, context?}` -> `Decision` JSON (see DESIGN §4.1). Attaches a fresh `decision_id`.  |
| POST   | `/decompose` | `{prompt}` -> `{decision: "solo"\|"mixture", plan?, rule, decomposer_version}`. Rule-based, no encoder/head, sub-ms typical. Used by the plugin's UserPromptSubmit hook to auto-route homogeneous multi-leaf prompts. |
| POST   | `/feedback`  | Append one outcome record (DESIGN §4.2) to `feedback.jsonl`. Last-write-wins by `decision_id`. |
| GET    | `/metrics`   | Rolling JSON: prediction counts by decision & policy, p50/p99 latency, uptime, head version. |
| POST   | `/reload`    | Re-read `heads/HEAD` and swap in the named head. Also happens automatically on mtime change. |

## State on disk

- `heads/HEAD` and `heads/<version>.joblib` are produced by `bootstrap.py` (M2) and `train.py` (M3). Until they exist, `/route` uses `heuristic_fallback.py` and `head_version` is reported as `"heuristic"`.
- `feedback.jsonl` is written by the queue's `/feedback` callback (M3). Until then the file stays empty and `feedback_records_total` is `0`.

The server is safe to run in either state — it will simply degrade to the heuristic policy and report so via `/metrics`.

## Scheduled retraining

`train.py` is intended to be woken nightly. Install the scheduler via the
OS-detecting wrapper:

```sh
./scripts/install-scheduler.sh   # macOS -> launchd, Linux -> systemd --user
```

Both back-ends fire at 03:00 local time and self-gate on new-record count,
so a wake with no fresh feedback is a cheap no-op. Remove with
`./scripts/uninstall-scheduler.sh`.

## See also

- [`DESIGN.md`](./DESIGN.md) — full spec (schemas, versioning, cold start, counterfactual handling).
- [`../slm-queue/`](../slm-queue/) — the caller that will speak to `/route` and `/feedback` behind the `SLM_ROUTER_URL` env var.
