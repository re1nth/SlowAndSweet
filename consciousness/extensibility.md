# Extensibility

## Current state (2026-07-09)

Adding a new SLM is one YAML edit. Adding a new *arm* to the
experiments harness is a couple of hundred lines. Adding a new
*router* head architecture is one file plus a new `heads/vN.joblib`
serialisation shape. Adding a whole new component (e.g., a
Prism-style structured DAG generator) has been done once and cost
about a day.

The friction is in the seams:
- Different components have different "how do I add X" stories.
- Nothing enforces the plan-run → feedback-record contract at
  type level; the contract is a JSON schema in `DESIGN.md`.
- The router's Head class assumes a specific 2-head shape (cost
  regressor + quality classifier). Adding a third head (e.g.
  latency-predictor) means editing `model.py` in one place, then
  editing `policy.py`, then editing `train.py`. Three files for
  one concept.

## Extension points that work

| Extension                          | How                                          | Cost      |
| ---------------------------------- | -------------------------------------------- | --------- |
| Add a new SLM                      | Add entry to `slm-deploy/slms.yaml`          | trivial   |
| Add a new plan template            | Drop JSON in `slm-queue/plans/`              | trivial   |
| Add a new experiment case          | Drop YAML in `slm-experiments/cases/`        | trivial   |
| Add a new SxS arm                  | Callable in `slm-experiments/arms/`; wire in `runner.py` | ~200 LOC |
| Add a new heuristic to router      | Edit `slm-router/heuristic_fallback.py`      | small     |
| Add a new decision policy          | Edit `slm-router/policy.py` (pure function)  | small     |
| Add a new head model               | Subclass or replace `Head` in `model.py`     | ~50 LOC   |

## Extension points that are painful

- **Add a new feedback field.** Requires touching the emitter
  (queue or runner), the schema comment in DESIGN.md, and
  `train.py`'s `_load_feedback`. Silent dropping of records that
  don't match the loader is the main hazard.
- **Add a new metric to /metrics.** Requires adding a counter,
  a rollup, and (if we want historical) a persistence layer.
- **Change the head serialisation format.** Everything reads
  `head.joblib` via `Head.load`; a schema bump would break
  `bootstrap.py`'s cold-start heads on the next server boot.
  No versioning story yet.
- **Add a new caller (beyond queue).** Right now only the queue
  talks to the router. If the plugin wanted to route directly,
  it would need its own copy of `explore.py`.

## The seams

- `slm-queue` ↔ `slm-router` — HTTP JSON, env-var gated. Weak
  contract, robust to skew.
- `slm-experiments` ↔ `slm-router` — HTTP JSON, env-var gated,
  same shape. Duplicated payload construction; a shared client
  would remove the drift risk.
- `slm-queue` ↔ `slowandsweet` — direct import from a wheel or
  sibling checkout. Tighter coupling; changes to
  `state.record_call` signature will break the queue silently
  because it's inside a broad except.
- `slm-router` internal — foundation modules (`model.py`,
  `policy.py`, `heuristic_fallback.py`) import each other via
  flat imports; server prepends `slm-router/` to `sys.path`.
  Works, but a namespace package would be cleaner.

## Log

### 2026-07-08 — Four agents fanned out cleanly, one interface friction
Server, training, and queue integration ran in parallel because
their file scopes didn't overlap and the foundation interfaces
were locked before the fan-out. The one friction was
`Head.predict`'s return type — DESIGN.md described it as
returning a `Decision`, but the actual implementation returned
a `(reduction, quality_prob, confidence)` tuple that
`policy.decide` synthesised into a `Decision`. The agents each
noticed independently and worked around it. Cleaner: define the
interface types in a shared `types.py` and let the doc reference
the code, not the other way around.

### 2026-07-04 — Prism was slotted in as a third arm without pain
The experiments harness went from two arms (solo, mixture) to
three (adding prism) with only additions to `runner.py` and a
new file under `arms/`. Nothing existing had to change. That's
the extension story we want on the router side too.
