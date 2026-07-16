# Parallel local inference — plan

How to actually run *N* concurrent inferences over the SLM pool on one
host, and what changes when we go beyond the single-`ollama serve`
model the queue currently assumes.

## 1. Where we are today

`slm-queue/server.py` spawns `replicas` worker threads per SLM
(`slm-deploy/slms.yaml`). Every worker calls the same endpoint:

```
http://localhost:11434/api/generate    (one `ollama serve` process)
```

So "parallelism" today is:

- **Client side (queue):** N worker threads submit concurrently.
- **Server side (Ollama):** whatever `OLLAMA_NUM_PARALLEL` allows,
  serialized behind one process, one weight copy per model, one Metal
  device queue.

The README hint (`OLLAMA_NUM_PARALLEL=2 ollama serve`) is the *only*
lever we currently pull. Everything else is threading on the client
that then queues behind a single generator.

### The real ceiling

On a single Apple Silicon Mac the bottleneck is almost always **GPU
kernel serialization**, not client concurrency. Two things determine
useful parallelism:

1. **KV-cache slots per loaded model.** This is what
   `OLLAMA_NUM_PARALLEL` allocates. More slots → more concurrent
   requests to the *same* model without swapping context.
2. **Number of resident models.** Only one model's weights can occupy
   VRAM at a time by default. `OLLAMA_MAX_LOADED_MODELS` lifts that
   cap; Ollama then multiplexes weights across requests.

More client threads without matching server capacity just grows the
Ollama queue. That's the state we are in.

## 2. Dimensions of parallelism we can actually exploit

| Dimension | What it means | Where it pays off |
| --------- | ------------- | ----------------- |
| **Intra-model batched decoding** | Multiple prompts share one forward pass on the same weights (continuous batching). | High: near-linear throughput up to the batch cap for a given model. |
| **Intra-model KV-cache slots** | Multiple in-flight streams for one model, KV-cache split. | Medium: hides prompt-eval latency; decode still round-robins the GPU. |
| **Inter-model concurrency** | Different models running "at the same time". | Low on a single GPU — kernels serialize. Only real gain is memory-bandwidth overlap and prompt-eval / decode interleave. |
| **Inter-request queue depth** | Many workers push requests concurrently. | Only useful up to the server's capacity above. Beyond that it just fills queues. |

The consequence: **duplicating model instances on the same GPU rarely
gives 2× throughput.** It gives 2× resident memory cost and a modest
overlap win. The lever we want is *continuous batching* + *more KV
slots per model*, not more processes.

## 3. Options survey

### A. Stay on Ollama, tune the knobs (Phase 1 — recommended first)

Environment variables on the daemon:

```
OLLAMA_NUM_PARALLEL=4              # concurrent generations per model
OLLAMA_MAX_LOADED_MODELS=4         # keep all four SLMs resident
OLLAMA_MAX_QUEUE=512               # queue depth before 503
OLLAMA_KEEP_ALIVE=24h              # don't evict between requests
OLLAMA_FLASH_ATTENTION=1           # if the models support it
```

- **Pros:** zero code change; matches the README's existing setup;
  covers the *actual* GPU bottleneck.
- **Cons:** still one process; no per-model isolation; scheduling
  policy is Ollama's, not ours.
- **What we ship:** teach `slowandsweet init`/`doctor` to write and
  verify these vars in a shell-agnostic way (e.g., a launchd/systemd
  drop-in), and record the actual `OLLAMA_NUM_PARALLEL` value in
  `slms.yaml` so the autoscaler cap can respect it.

### B. Multiple Ollama daemons, one per model group (Phase 2)

Run e.g. `ollama serve` on 11434 for the "fast" models and another on
11435 for the "slow" models. The queue's `OLLAMA_URL` becomes a per-model
map.

- **Pros:** isolates model families; failure/OOM in one daemon doesn't
  take the others down; can pin one daemon to CPU-only for cheap
  fallbacks.
- **Cons:** duplicates weights only if the same model runs in both;
  more moving parts to health-check.
- **What we ship:** extend `slms.yaml` with an optional
  `endpoint: http://localhost:11435` per deployment; dispatcher already
  keys by model so the change is small (replace the constant
  `OLLAMA_URL` with a `endpoint_by_model` map).

### C. Swap Ollama for `llama-server` (llama.cpp) — Phase 3, evaluation

`llama.cpp`'s `llama-server` is what Ollama runs under the hood, but
directly:

```
llama-server -m models/smollm2-1.7b.gguf \
  --host 127.0.0.1 --port 8081 \
  --parallel 4 --cont-batching --n-gpu-layers -1
```

One process per model, exposing an OpenAI-compatible `/v1/completions`.

- **Pros:** explicit continuous-batching (`--cont-batching`),
  per-model tuning of `--parallel`, `--ctx-size`, quant format; can
  colocate multiple models on different ports and let the dispatcher
  fan out.
- **Cons:** we own model download/quant selection (no `ollama pull`);
  more surface to install/manage.
- **What we ship:** a `slm-deploy/backend: llama-cpp` block that
  spawns a `llama-server` per deployment; keeps the dispatcher HTTP
  contract unchanged, just switches URLs.

### D. MLX / `mlx-lm` server (Apple-Silicon native) — Phase 3 alternative

`mlx-lm.server` is a native MLX server for Apple Silicon. It uses the
unified memory model directly, which can be materially faster than the
Metal path Ollama uses for some models.

```
mlx_lm.server --model mlx-community/gemma-2-2b-it-4bit --port 8082
```

- **Pros:** best per-token latency on M-series for MLX-converted
  models; unified memory means no host↔GPU copy.
- **Cons:** MLX format is separate from GGUF — model catalog is smaller;
  no built-in multi-request batching in early versions (improving);
  one process per model.
- **What we ship:** same shape as (C) — a backend flag in `slms.yaml`;
  we'd probably use MLX for the smallest SLMs (smollm2, gemma2:2b)
  where per-token latency dominates.

### E. LocalAI (OpenAI-compatible, multi-backend) — considered, not primary

LocalAI wraps llama.cpp/whisper/stable-diffusion/etc. behind one
OpenAI-compatible API. Same trade as (C) but with an extra layer.

- **Pros:** OpenAI compat; multiple model backends behind one URL;
  YAML-declared models (nice fit with our `slms.yaml`).
- **Cons:** extra layer of indirection over the thing that actually
  matters (llama.cpp); container-first UX; less transparent for
  perf work.
- **Verdict:** only pick this if we decide we want the OpenAI-compatible
  surface for external tools; otherwise (C) is closer to the metal.

### F. vLLM / SGLang / TGI — not for this host

Continuous-batching servers built for CUDA/ROCm datacenter GPUs. vLLM
has an experimental Apple Silicon path, but throughput is well below
llama.cpp on M-series today. Keep on the radar for a Linux/CUDA
deployment target; skip for the current Mac.

### G. LM Studio — considered, not primary

LM Studio's local server (`lms server`) is basically llama.cpp with a
GUI. Fine for interactive use, awkward for a daemon we programmatically
manage.

## 3B. Deep dive: A, C, D, F vs. "true parallelism"

The word "parallel" is doing a lot of work in the survey above. Before
we compare, we need to be precise about *what kind* of parallelism each
option actually gives us on a single Apple-Silicon host — because on one
GPU, "N processes" is not the same as "N kernels running at once", and
neither of those is the same as "N tokens generated per forward pass".

### Taxonomy of concurrency, from fake to real

We evaluate each option against these seven mechanisms, in rough order
of how much throughput they buy us on **one** GPU:

| # | Mechanism | What it is | Real parallelism? |
| - | --------- | ---------- | ----------------- |
| T1 | **Time-slicing** | Requests queued; server picks one, runs it, then the next. | No. Just concurrency illusion. |
| T2 | **Continuous batching** | On each decode step, all *ready* requests contribute one token to a single fused forward pass. Batch composition changes every step as requests finish. | **Yes — the primary win on single-GPU.** Sub-linear-but-large throughput gain (2–4× at batch 4–8 for 1–3B models). |
| T3 | **Chunked prefill / prefill–decode interleave** | A new request's prompt-eval is broken into chunks that ride along inside the same GPU step as ongoing decodes. Removes the "prefill stalls all decodes" pathology. | Yes. Hides tail-latency spikes and boosts steady-state throughput 10–30 %. |
| T4 | **Prefix-cache reuse (radix/prefix caching)** | Two requests share a common prefix; the KV cache for that prefix is computed once and reused. | Not parallelism strictly, but under concurrent load it can *look* like a 2–5× speedup on the shared prefix. Huge win for our few-shot / boilerplate leaf prompts. |
| T5 | **Speculative decoding** | A tiny draft model proposes k tokens; the big model verifies them in a single forward pass. Parallel evaluation over sequence positions. | Yes — real parallel-execution win at the token level. 1.5–2.5× decode speedup when draft alignment is high. |
| T6 | **Multi-instance data parallelism** (multiple processes on one GPU) | Separate weight copies, separate KV caches, dispatcher fans out. | **Illusory on one GPU** — kernels time-slice. Buys memory-bandwidth overlap only, at 2× RAM cost. |
| T7 | **Tensor / pipeline parallelism** across multiple GPUs | Model sharded across devices; kernels genuinely execute concurrently on different silicon. | Yes — the only *linear* scaling. Requires ≥2 GPUs; not applicable to a single Mac. |

The interesting question for each option is: **which of T2–T5 do we
actually get, and how well-tuned?** T1 is the floor; T6/T7 are either
uninteresting or unavailable on our hardware.

---

### A. Ollama tuning — deep dive

**How it achieves parallelism.** Ollama is a supervisor + HTTP server
that runs `llama.cpp` (its vendored fork) under the hood, one runner
subprocess per loaded model. `OLLAMA_NUM_PARALLEL=N` tells each runner
to allocate N KV-cache "slots" and enables llama.cpp's continuous
batcher across them (T2). Requests that arrive while a batch is running
land in the daemon's HTTP queue and get picked up on the next step
that has an open slot. `OLLAMA_MAX_LOADED_MODELS=M` lets M distinct
runners coexist; when a request arrives for an unloaded model, Ollama
evicts an LRU runner (respecting `OLLAMA_KEEP_ALIVE`).

**What "true parallelism" it delivers.**

- **T2 (continuous batching): yes**, but per-model only. Cross-model
  batching does not happen — each model's runner is its own scheduler.
- **T3 (chunked prefill): partial.** Newer llama.cpp has parallel
  slots for prefill+decode, but Ollama does not expose the knob
  (`ubatch`) that lets you tune it. You get whatever the vendored
  build defaults to.
- **T4 (prefix caching): weak.** llama.cpp supports simple
  prompt-cache-reuse for identical prefixes, but Ollama's HTTP path
  does not surface a stable session/prefix ID the way vLLM/SGLang do.
- **T5 (speculative decoding): not exposed** — llama.cpp supports it
  (`-md`), but Ollama has no config surface for a draft model.
- T6, T7: N/A.

**Pros.**

- Zero-cost enablement — it's already installed and running.
- Handles multi-model resident memory automatically (LRU + keep-alive).
- Model catalog is a first-class UX (`ollama pull`).
- The `/api/ps` endpoint tells us which runners are hot, which is
  enough to plumb into the queue's autoscaler.
- Version stable — we're not chasing llama.cpp master.

**Cons for true parallelism.**

- **Opacity.** No way to set `--ubatch-size`, `--n-parallel` per model,
  `--flash-attn` toggles per model, KV-cache quantization
  (`-ctk q8_0`), or draft-model paths. All the levers that matter
  for T3/T4/T5 are hidden.
- **Cross-model scheduling is naive.** When two different models both
  have work, they time-slice the GPU without coordination — you get
  the worst of both (T6) rather than intelligent priority.
- **Version drift.** Ollama's vendored llama.cpp lags upstream by
  weeks; features like speculative decoding show up here months later.
- **Prompt caching is weak.** Our workload has boilerplate prefixes
  (system prompt for the SLM, few-shot template). T4 would help
  materially; Ollama doesn't give it to us.

**Practical expectation on our hardware.** On an M-series Mac serving
2–3 concurrent requests to `smollm2:1.7b` or `gemma2:2b`, expect
~1.7–2.3× aggregate throughput vs. sequential. Beyond 4 concurrent
you're memory-bandwidth-bound, and P95 latency starts climbing.

---

### C. `llama-server` (llama.cpp direct) — deep dive

**How it achieves parallelism.** One `llama-server` process per model.
`--parallel N` allocates N slots (T2). `--cont-batching` is on by
default in recent builds. `--batch-size` / `--ubatch-size` tune the
prefill-vs-decode interleave (T3). `--cache-reuse` and stable
session IDs give you per-request prefix caching (T4). `-md` +
`--draft-max` enable speculative decoding with a smaller draft model
(T5). `-fa` turns on flash-attention. `-ctk q8_0 -ctv q8_0` quantizes
the KV cache — often lets you double `--parallel` at the same RAM.

**What "true parallelism" it delivers.**

- **T2: yes**, fully tunable. Same underlying engine as Ollama, but
  every knob is on the command line.
- **T3: yes**, and *you* pick the ubatch size. For a 3B model on M-series,
  `--ubatch-size 256` often wins.
- **T4: yes**, via `--cache-reuse` + session IDs. This is where
  llama-server pulls ahead of Ollama for our boilerplate-prefix
  workload.
- **T5: yes**, and this is potentially the biggest single win — a 500M
  draft in front of a 3B target on Mac Silicon has been reported at
  1.7–2.2× decode speedup on well-aligned tasks (summarization,
  reformatting — exactly our leaves).
- T6: still illusory across processes on one GPU. If we run one
  `llama-server` per model as suggested, we get T6 across models,
  which is the same time-slice tax as Ollama.
- T7: N/A.

**Pros.**

- Full control over the batch/KV/attention knobs — every T2–T5 lever
  is directly settable.
- Speculative decoding is a first-class flag, not a research toy.
- Native OpenAI-compatible endpoints (`/v1/completions`,
  `/v1/chat/completions`) — the dispatcher's HTTP shape stays
  unchanged; just the URL and payload schema shift.
- Prompt cache surface is stable enough to build reliable prefix
  reuse against.
- No supervisor layer between us and the runtime, so bugs are easier
  to reason about.

**Cons for true parallelism.**

- **One process per model** — we manage the lifecycle (start, watchdog,
  restart on OOM, health). Ollama gave us that for free.
- **Same GPU-serialization ceiling** — even with tighter batching,
  cross-model concurrency is T6-only. Not a win over Ollama here.
- **Model catalog on us.** We pick the GGUF quantization
  (`Q4_K_M` vs `Q5_K_M` vs `Q6_K`), and we own the download.
- **Draft-model selection is a research task** — a badly aligned
  draft can *slow* the target model (rejection rate too high).
- **Prefix-cache eviction policy** in llama-server is simpler than
  vLLM's paged store; large concurrent sessions can thrash.

**Practical expectation.** Same base throughput as Ollama for the
same model at the same batch size (they share an engine), but +30–50 %
on top of that once T3/T4/T5 are actually enabled. The delta comes
from the tunables Ollama hides.

---

### D. MLX (`mlx_lm.server`) — deep dive

**How it achieves parallelism.** MLX is Apple's array framework,
purpose-built for Apple Silicon's unified-memory / Metal architecture.
`mlx_lm.server` runs one model per process; the mlx-lm library
implements a batched-generate loop that fuses multiple in-flight
requests into a single Metal command graph (T2). Because unified
memory eliminates host↔device copies, prefill throughput on M-series
is materially higher than the ggml-Metal path llama.cpp takes.

**What "true parallelism" it delivers.**

- **T2: yes**, but the batching implementation is *newer* than
  llama.cpp's and has less scheduling sophistication. Static batch
  windows in older versions; continuous-batching support has been
  landing incrementally through 2025.
- **T3: partial.** Prefill / decode interleave in mlx-lm is less
  granular than llama-server's ubatch; you tend to see prefill spikes
  in P99.
- **T4: limited.** mlx-lm has prompt-cache reuse for exact prefix
  matches, but not a paged/radix store. Works for boilerplate;
  doesn't scale to long-tail branching prefixes.
- **T5: yes.** `mlx_lm.generate` has speculative decoding; solid on
  Llama-family. Same 1.5–2× decode uplift when draft alignment is high.
- T6, T7: N/A / not applicable.

**Pros.**

- **Best per-token latency on Apple Silicon** for models with a good
  MLX port — often 10–25 % faster than the same quant through
  llama.cpp/Ollama, because MLX schedules Metal directly rather than
  through ggml's abstraction.
- **Unified memory is a real architectural win** — no copies, so
  KV-cache growth doesn't compete with model weights the way it does
  when llama.cpp is treating VRAM as a discrete pool.
- Extremely fast prefill on long prompts vs. llama.cpp (Metal graph
  fusion is stronger).
- Easy per-model quantization (`mlx_lm.convert -q --q-bits 4`).

**Cons for true parallelism.**

- **Batching maturity gap.** The batcher is younger than llama.cpp's;
  under real concurrent load (say, 6+ in-flight requests) it's more
  likely to leave the GPU idle between requests. This is exactly the
  regime where "true parallelism" is what we're buying.
- **Smaller model catalog.** `mlx-community` covers the big names
  but the long tail is thin. For obscure fine-tunes you're on your
  own for the MLX conversion.
- **Server surface is thinner.** Less mature OpenAI-compat, no
  grammar / JSON-schema constrained decoding today, no first-class
  session/prefix IDs.
- **Same T6 story** across processes — one GPU means MLX and llama.cpp
  processes competing for it if we mix backends.
- **Version churn.** MLX API and mlx-lm server semantics move fast;
  we'd be pinning versions per release.

**Practical expectation.** For our two smallest SLMs
(`smollm2:1.7b`, `gemma2:2b`) MLX likely wins at single-request
latency and matches Ollama at 2 concurrent. Beyond ~4 concurrent
requests, current MLX server is probably *slower* than a
`--parallel 4` llama-server on the same model. The right use for MLX
is "SLM where P50 latency dominates," not "SLM under sustained
concurrent load".

---

### F. vLLM / SGLang / TGI — deep dive (why they're not for this host)

**How they achieve parallelism.** These are datacenter-grade
continuous-batching servers built around three ideas that
llama.cpp/MLX only partially replicate:

- **PagedAttention (vLLM).** KV cache is broken into fixed-size pages
  managed like OS virtual memory. Fragmentation goes to ~0 %, so you
  can pack many more concurrent sessions into the same VRAM. Direct
  T2 amplifier.
- **Chunked prefill (vLLM 0.5+, SGLang).** Prefill of a new request is
  broken into chunks that share every iteration with ongoing decodes.
  This is T3 done properly — the prefill-decode tradeoff is basically
  eliminated.
- **RadixAttention (SGLang).** A radix-tree of KV-cache prefixes
  shared across *all* requests, with LRU eviction. T4 at industrial
  scale — a system prompt processed once serves N sessions for free.
- **Tensor / pipeline parallelism.** T7 across multi-GPU boxes. This
  is what turns "throughput per server" into "throughput per rack."
- **Speculative decoding** (both vLLM and SGLang). T5 with
  Medusa/EAGLE-style multi-head heads, not just draft-verify.

**What "true parallelism" they deliver.** All of T2, T3, T4, T5, T7 —
essentially the full menu — at industrial-strength implementations.
This is why a single A100 running vLLM outperforms a rack of naive
llama.cpp processes on the same model.

**Pros (in the abstract).**

- **Highest concurrent throughput per GPU** by a wide margin — often
  4–8× llama.cpp on the same hardware once concurrency is > 8.
- **Prefix-caching is not opt-in** — SGLang gets it right by default,
  which for our boilerplate-heavy leaves would be a large win.
- **Multi-GPU scaling** with `tensor-parallel-size` = number of GPUs.
  If we ever run on a Linux GPU box this is what we'd want.
- **Production-grade** — Kubernetes-ready, autoscaler-friendly, well
  understood by anyone who's operated LLM serving.

**Cons for our host.**

- **CUDA-first.** vLLM's Apple-Silicon (`mps`/CPU) path exists but is
  experimental, throughput is well below llama.cpp on M-series, and
  many kernels fall back to Python. In practice, running vLLM on a
  Mac is worse than running Ollama on a Mac. TGI is similar. SGLang
  only supports CUDA in production.
- **Overkill for 1–3B SLMs.** The datacenter machinery pays off at
  high concurrency and large models. Our leaves are short-lived, our
  SLMs are small, we run ≤8 in flight.
- **Operational complexity.** Docker/Linux orientation, per-model
  process, ROCm/CUDA driver management — heavy for a laptop.
- **Model format.** HF safetensors, not GGUF/MLX — different catalog
  again.

**When they become the right answer.** The day we point SlowAndSweet
at a Linux box with even a single L4 / L40S / consumer 4090, vLLM
becomes the correct backend and gives us T2+T3+T4+T5 in one package.
For the current Mac-only shape of the project, **F is on the shelf,
not on the roadmap.**

---

### Summary — which parallelism each option actually gets us

| | T2 batch | T3 chunked prefill | T4 prefix cache | T5 spec-decode | T6 multi-inst | T7 multi-GPU |
| - | --- | --- | --- | --- | --- | --- |
| **A. Ollama** | ✅ per-model, opaque | ⚠ default only | ⚠ weak / no session ID | ❌ hidden | ⚠ across models | ❌ |
| **C. llama-server** | ✅ per-model, tunable | ✅ tunable ubatch | ✅ via cache-reuse | ✅ first-class | ⚠ across models | ❌ |
| **D. mlx-lm** | ⚠ newer batcher | ⚠ coarse | ⚠ exact-match only | ✅ | ⚠ across models | ❌ |
| **F. vLLM/SGLang** | ✅ industry-best | ✅ industry-best | ✅ industry-best (Radix) | ✅ Medusa/EAGLE | n/a — one process suffices | ✅ (needs multi-GPU host) |

The read: **on our current single-Mac host, C is the ceiling for true
parallelism** (real T2/T3/T4/T5 all reachable). A is what we have. D
wins on single-request latency for the smallest models. F is what we'd
want the day we outgrow the laptop.

## 4. Recommended phased plan

**Phase 1 — Extract full throughput from the current stack (days)**

1. Add `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS`,
   `OLLAMA_KEEP_ALIVE` to `slowandsweet init` (write a launchd
   drop-in on macOS; document `systemd` env for Linux).
2. Have `slowandsweet doctor` read the running daemon's `/api/ps` and
   report the effective parallel/loaded caps.
3. Extend the autoscaler in `slm-queue/autoscaler.py` so
   `max_total_replicas` is bounded by `OLLAMA_NUM_PARALLEL * len(models)`
   — no point scaling workers past what the server will actually
   run concurrently.
4. Add a "concurrency" section to `slm-bench` that sweeps
   `OLLAMA_NUM_PARALLEL ∈ {1, 2, 4, 8}` on a fixed prompt set and
   records tokens/sec-per-request and total tokens/sec. This is the
   evidence we need before Phase 2.

**Phase 2 — Multi-endpoint dispatcher (week)**

5. Replace the module-level `OLLAMA_URL` in `slm-queue/server.py` with
   an `endpoint_by_model: dict[str, str]` populated from `slms.yaml`.
   Default remains `http://localhost:11434` if unset.
6. Add `endpoint:` (optional) to each `SLMDeployment` in
   `slms.yaml`; validate.py rejects duplicates on the same endpoint
   that exceed the endpoint's declared `parallel` cap.
7. Provide a `slm-deploy/launch.sh` (or a `slowandsweet up`
   subcommand) that starts one `ollama serve` per declared endpoint
   with the right env, waits for `/api/tags`, and pulls the model into
   the correct daemon.
8. Re-run the concurrency sweep with two daemons; compare against the
   single-daemon baseline.

**Phase 3 — Alternative backends behind the same contract (2 weeks)**

9. Introduce a `backend: ollama | llama-cpp | mlx` field on
   `SLMDeployment`. Dispatcher only sees an HTTP URL; the launcher
   knows how to start the right process.
10. Ship `llama-server` support for the two heaviest models
    (`llama3.2:3b`, `qwen2.5:3b`) with `--cont-batching --parallel 4`.
11. Ship `mlx-lm.server` support for the two smallest
    (`smollm2:1.7b`, `gemma2:2b`) using pre-quantized MLX weights.
12. Bench again. Keep whichever backend wins per model. Update
    `resources.md`.

**Phase 4 — Cross-host (aspirational, off-Mac)**

13. Extend `endpoint:` to accept remote URLs. Ship a small
    `slm-worker` container that packages `llama-server` + a health
    endpoint; the queue treats it as a replica. This is what unlocks
    "run a second Mac / a Linux box next to my laptop" without
    touching the plan/router surface.

## 5. Concrete first commits

Small, isolated, so we can land Phase 1 without a big rewrite:

- `slowandsweet init` writes a `~/Library/LaunchAgents/ai.ollama.plist`
  drop-in (macOS) with `OLLAMA_NUM_PARALLEL=4`,
  `OLLAMA_MAX_LOADED_MODELS=4`, `OLLAMA_KEEP_ALIVE=24h`. `doctor`
  reports the resolved values.
- `slm-bench/concurrency_bench.py`: fires K concurrent requests to one
  model, sweeps K, reports throughput and P50/P95 latency. Writes
  `slm-bench/concurrency_results.json`.
- `slm-queue/server.py`: replace `OLLAMA_URL` constant with
  `dispatcher.endpoint_for(model)`; wire from `slms.yaml`
  `spec.endpoint`. Default unchanged.
- `slm-deploy/validate.py`: reject a deployment set where
  `sum(replicas of models on endpoint E) > E.parallel`.

## 6. Open questions

- Does OLLAMA's `OLLAMA_NUM_PARALLEL` split cleanly across models when
  `MAX_LOADED_MODELS > 1`, or is it a global cap the scheduler shares?
  Determines whether Phase 1 alone suffices or Phase 2 is mandatory.
- What's the real cost of MLX conversion for the four models we ship?
  If we have to maintain custom quantizations, the operational
  overhead may not be worth the per-token gain.
- Does the router benefit from *knowing* the endpoint concurrency
  budget (i.e., should quality-vs-cost scoring include queue depth on
  each endpoint), or is the autoscaler enough?

Answering these is the point of the Phase 1 bench sweep before we
commit to Phase 2 or 3.
