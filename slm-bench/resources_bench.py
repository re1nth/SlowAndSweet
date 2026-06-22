#!/usr/bin/env python3
"""Benchmark compute/memory cost of 10 prompts per SLM via Ollama.

For each model:
  - warm the model with one short call (load weights into memory)
  - run 10 fixed prompts, capturing tokens, eval time, and sampled CPU/RSS
    of all `ollama` processes during the call
  - write per-prompt and aggregate stats to JSON
"""
import json
import subprocess
import threading
import time
import urllib.request

MODELS = ["smollm2:1.7b", "gemma2:2b", "llama3.2:3b", "qwen2.5:3b", "phi3:mini"]
URL = "http://localhost:11434/api/generate"
SAMPLE_INTERVAL_S = 0.1

PROMPTS = [
    "Explain why the sky appears blue in 3 sentences, then give one counterexample.",
    "Summarize the plot of Hamlet in five bullet points.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "List four differences between TCP and UDP.",
    "Translate 'The quick brown fox jumps over the lazy dog' into French and Spanish.",
    "Give three pros and three cons of remote work.",
    "Explain gradient descent to a high-school student in under 120 words.",
    "Write a haiku about an empty coffee cup at dawn.",
    "What is the time complexity of mergesort? Justify briefly.",
    "Name five common causes of memory leaks in C++ programs.",
]


def sample_ollama() -> tuple[float, float]:
    """Return (sum RSS in bytes, sum %CPU) across all ollama processes."""
    out = subprocess.run(
        ["ps", "-axo", "pid,rss,%cpu,comm"],
        capture_output=True, text=True, check=True,
    ).stdout
    rss_kb = 0
    cpu = 0.0
    for line in out.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        comm = parts[3]
        if "ollama" not in comm.lower():
            continue
        try:
            rss_kb += int(parts[1])
            cpu += float(parts[2])
        except ValueError:
            continue
    return rss_kb * 1024, cpu


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()
        self.peak_rss = 0
        self.cpu_samples: list[float] = []

    def run(self):
        while not self.stop_flag.is_set():
            try:
                rss, cpu = sample_ollama()
            except Exception:
                rss, cpu = 0, 0.0
            if rss > self.peak_rss:
                self.peak_rss = rss
            self.cpu_samples.append(cpu)
            time.sleep(SAMPLE_INTERVAL_S)

    def stats(self) -> tuple[int, float, int]:
        n = len(self.cpu_samples)
        avg_cpu = sum(self.cpu_samples) / n if n else 0.0
        return self.peak_rss, avg_cpu, n


def call(model: str, prompt: str, num_predict: int = 256) -> dict:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"seed": 42, "temperature": 0.7, "num_predict": num_predict},
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    return {"wall_s": time.perf_counter() - t0, **data}


def measure(model: str, prompt: str) -> dict:
    sampler = Sampler()
    sampler.start()
    try:
        r = call(model, prompt)
    finally:
        sampler.stop_flag.set()
        sampler.join()
    peak_rss, avg_cpu, n = sampler.stats()
    eval_ns = r.get("eval_duration", 1) or 1
    return {
        "wall_s": r["wall_s"],
        "eval_s": eval_ns / 1e9,
        "eval_tokens": r.get("eval_count", 0),
        "tokens_per_sec": r.get("eval_count", 0) / (eval_ns / 1e9),
        "ttft_s": (r.get("load_duration", 0) + r.get("prompt_eval_duration", 0)) / 1e9,
        "peak_rss_bytes": peak_rss,
        "avg_cpu_pct": avg_cpu,
        "samples": n,
    }


def main():
    summary = []
    for m in MODELS:
        print(f"\n=== {m} ===", flush=True)
        print("  warmup...", flush=True)
        call(m, "Say 'ready'.", num_predict=4)
        per_prompt = []
        for i, p in enumerate(PROMPTS, 1):
            print(f"  prompt {i}/{len(PROMPTS)}...", flush=True, end=" ")
            stats = measure(m, p)
            per_prompt.append(stats)
            print(
                f"{stats['eval_tokens']} tok in {stats['eval_s']:.2f}s "
                f"({stats['tokens_per_sec']:.1f} tok/s) "
                f"peak={stats['peak_rss_bytes']/1e9:.2f} GB "
                f"avg_cpu={stats['avg_cpu_pct']:.0f}%",
                flush=True,
            )
        total_tokens = sum(s["eval_tokens"] for s in per_prompt)
        total_eval_s = sum(s["eval_s"] for s in per_prompt)
        total_wall_s = sum(s["wall_s"] for s in per_prompt)
        peak_rss = max(s["peak_rss_bytes"] for s in per_prompt)
        cpu_weighted = (
            sum(s["avg_cpu_pct"] * s["samples"] for s in per_prompt)
            / max(1, sum(s["samples"] for s in per_prompt))
        )
        summary.append({
            "model": m,
            "prompts": len(PROMPTS),
            "total_tokens": total_tokens,
            "total_eval_s": total_eval_s,
            "total_wall_s": total_wall_s,
            "tokens_per_sec": total_tokens / total_eval_s if total_eval_s else 0,
            "peak_rss_bytes": peak_rss,
            "avg_cpu_pct": cpu_weighted,
            "cpu_seconds": cpu_weighted / 100.0 * total_wall_s,
            "per_prompt": per_prompt,
        })

    out_path = "/Users/reva/Documents/SlowAndSweet/slm-bench/resources_bench_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
