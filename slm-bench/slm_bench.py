#!/usr/bin/env python3
"""Benchmark SLMs on Ollama. Warmup + measured run per model."""
import json
import time
import urllib.request

MODELS = ["smollm2:1.7b", "gemma2:2b", "llama3.2:3b", "qwen2.5:3b", "phi3:mini"]
PROMPT = "Explain why the sky appears blue in 3 sentences, then give one counterexample."
URL = "http://localhost:11434/api/generate"


def call(model: str, prompt: str) -> dict:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"seed": 42, "temperature": 0.7, "num_predict": 256},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    wall = time.perf_counter() - t0
    return {"wall": wall, **data}


results = []
for m in MODELS:
    print(f"\n=== {m} ===", flush=True)
    print("  warmup...", flush=True)
    call(m, "Say 'ready'.")
    print("  measuring...", flush=True)
    r = call(m, PROMPT)
    eval_ns = r.get("eval_duration", 1)
    pe_ns = r.get("prompt_eval_duration", 0)
    load_ns = r.get("load_duration", 0)
    total_ns = r.get("total_duration", 0)
    eval_count = r.get("eval_count", 0)
    tps = eval_count / (eval_ns / 1e9) if eval_ns else 0
    ttft_s = (load_ns + pe_ns) / 1e9
    results.append({
        "model": m,
        "wall_s": r["wall"],
        "total_s": total_ns / 1e9,
        "load_s": load_ns / 1e9,
        "prompt_eval_s": pe_ns / 1e9,
        "eval_s": eval_ns / 1e9,
        "eval_tokens": eval_count,
        "tokens_per_sec": tps,
        "ttft_s": ttft_s,
        "response": r.get("response", "").strip(),
    })
    print(f"  {eval_count} tok in {eval_ns/1e9:.2f}s -> {tps:.1f} tok/s", flush=True)

with open("/tmp/slm_bench_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nWrote /tmp/slm_bench_results.json")
