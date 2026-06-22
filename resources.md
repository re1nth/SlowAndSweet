# Compute & Memory Resources

Resource consumption measured while running **10 prompts per model** through Ollama on
this Mac. Each model is warmed once, then the 10 prompts in
[`slm-bench/resources_bench.py`](slm-bench/resources_bench.py) are sent sequentially
with `seed=42`, `temperature=0.7`, `num_predict=256`.

During every request, a background thread polls `ps -axo pid,rss,%cpu,comm` every
100 ms, summing RSS and `%CPU` across every process whose `comm` contains `ollama`
(the API server plus the per-model runner). Peak RSS and time-weighted mean CPU are
reported per model.

## Aggregate over 10 prompts

| Model        | Tokens | Eval (s) | Wall (s) | Tokens/sec | Peak RSS (GB) | Mean CPU (%) | CPU·sec |
| ------------ | -----: | -------: | -------: | ---------: | ------------: | -----------: | ------: |
| smollm2:1.7b |  1 386 |    19.13 |    19.70 |       72.5 |          3.53 |          6.4 |    1.25 |
| gemma2:2b    |  1 872 |    29.65 |    30.50 |       63.1 |          5.43 |         12.2 |    3.71 |
| llama3.2:3b  |  1 776 |    31.36 |    32.30 |       56.6 |          5.83 |          7.9 |    2.54 |
| qwen2.5:3b   |  1 790 |    32.93 |    33.91 |       54.4 |          6.17 |          8.9 |    3.02 |
| phi3:mini    |  2 038 |    37.36 |    38.25 |       54.5 |          5.96 |          5.6 |    2.14 |

- **Tokens** — sum of `eval_count` across the 10 prompts (prompts that hit the
  256-token cap finish there).
- **Eval (s)** — sum of `eval_duration` reported by Ollama (generation time only).
- **Wall (s)** — sum of client-observed request durations.
- **Tokens/sec** — `Tokens / Eval (s)`.
- **Peak RSS** — maximum resident set size summed across all `ollama*` processes
  during the run. This is the relevant memory pressure on the system.
- **Mean CPU** — sample-weighted average of `%CPU` summed across `ollama*` processes
  (Apple `ps` reports per-core %, so values >100 % indicate multiple cores in use).
- **CPU·sec** — `Mean CPU / 100 × Wall`, an approximation of CPU-core-seconds spent.

## Caveats

- **GPU work is not counted.** On Apple Silicon, Ollama offloads transformer math
  to the Metal GPU. `ps` only sees the CPU-side driver thread, so the reported CPU
  numbers undercount the true compute and the GPU's energy/memory bandwidth use is
  invisible here. For GPU utilisation, use `sudo powermetrics --samplers gpu_power`
  alongside this script.
- **Unified memory.** Reported RSS includes weights mapped into unified memory; the
  GPU shares the same pool, so peak RSS is a reasonable proxy for total VRAM-ish
  footprint while a model is resident.
- **Sampling granularity.** A 100 ms sampler can miss sub-100 ms CPU spikes; short
  prompts (e.g. when the model emitted only ~20 tokens) only collect a handful of
  samples, so their per-prompt CPU averages are noisier than the aggregate.
- **Warm cache.** The first call after `ollama serve` starts pays a load cost; the
  numbers above are post-warmup steady-state per model.

Raw per-prompt measurements are in
[`slm-bench/resources_bench_results.json`](slm-bench/resources_bench_results.json).

## Reproducing

```sh
ollama serve  # in another terminal
python3 slm-bench/resources_bench.py
```
