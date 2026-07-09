# Testing

## Current state (2026-07-09)

There are no unit tests. There is no pytest suite. There is no CI.

There *is* an SxS harness (`slm-experiments/`) that runs the
whole system end-to-end and produces JSON snapshots. That's a
benchmark, not a test — it doesn't fail a build, it produces
evidence.

Every "test" so far has been a manual end-to-end smoke run
recorded in commit messages. That's fine for the current phase
(everything is exploratory) and will not scale past one dev.

## What has coverage

| Path                             | Coverage                       |
| -------------------------------- | ------------------------------ |
| Router `/route` + `/feedback`    | Smoke via curl in ad-hoc runs  |
| `choose_arm` degradation         | Smoke without router service   |
| `Head.save/load`                 | Smoke via foundation smoke test|
| `bootstrap.py` cold start        | Smoke in agent's ack           |
| `train.py` insufficient records  | Smoke on empty log             |
| `train.py` full training         | Smoke on 46 synthetic records  |
| End-to-end plan run              | Smoke on merge-intervals plan  |
| LC frontier bench                | Executable + committed results |

## What has zero coverage

- `policy.decide` — pure function, ideal for unit tests, none exist.
- `heuristic_fallback.heuristic_decide` — same story.
- `planner._composer_prompt_approx` — new helper, no tests.
- `planner._count_tokens` fallback path — the whitespace-split
  fallback triggers only when tiktoken is missing; not covered.
- Plan validation (`Plan._validate`, cycle detection) — exists
  in code but no test verifies the error paths.
- `train.py` auto-promote gate boundary conditions — an
  identical-metrics run refused to promote in one manual test;
  the strictly-better cases are also unit-testable.
- `Router state.append_feedback` under concurrent writes — the
  lock is there but nobody has stressed it.

## Tests we wish existed

**Priority 1 (correctness):**
- Assert `policy.decide` returns "solo" when
  `predicted_quality_ok_prob < quality_floor`, regardless of
  cost.
- Assert `policy.decide` returns "unsure" when confidence is
  under the floor and exploration doesn't trigger.
- Assert `_composer_prompt_approx` output is empty-ish when all
  terminals have empty results (guarantees graceful behavior
  when a plan errored mid-run).
- Assert `Head.load` fails loudly on a corrupted joblib.

**Priority 2 (integration):**
- Router service returns valid JSON under invalid input types
  (POST /route with `prompt: null`, `prompt: 12345`, empty body).
- `bootstrap.py` produces the same head byte-for-byte across
  runs at fixed seed.
- `train.py`'s auto-promote gate refuses to promote when
  candidate MAE is exactly equal to current MAE.
- A round-trip: submit plan → run → feedback → train → new head
  loaded via `/reload` → new predictions differ from old.

**Priority 3 (properties):**
- Any prompt that a heuristic classifies "solo" should never
  cause the learned head to send it to "mixture" without
  exceeding a quality floor. Property test with random prompts
  from a corpus.

## Log

### 2026-07-08 — CI would have caught the wire gap
The `choose_arm` wire miss (server.py never called it) was
found by manual smoke test. A trivial integration test —
"submit a plan, assert one feedback record lands" — would
have caught it in <5 seconds. Fastest ROI on any test we
could add.

### 2026-07-04 — First smoke test disciplined the design
The end-to-end LC bench (10 problems, real SLM DAG runs) was
the first time we verified the SLM route beats naive
heuristics on a real corpus. Not a unit test, but the closest
thing we've had to a "does the thing work" gate.
