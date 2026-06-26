"""Frontier-model adapters.

The abstract `FrontierAdapter` lets the experimentation harness swap
in different frontier providers (or a local stand-in for offline runs)
without touching the arms or the reviewer.

The default `AnthropicAdapter` calls the Anthropic Messages API and
returns the response text along with a populated `FrontierUsage` so
both arms and the reviewer can be priced consistently. The adapter
uses prompt caching for reusable system prompts (reviewer rubric,
case scaffolding) — cache hits are reflected separately in the usage.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from metrics import FrontierUsage


@dataclass
class FrontierCall:
    text: str
    usage: FrontierUsage


class FrontierAdapter(ABC):
    name: str

    @abstractmethod
    def complete(
        self,
        *,
        system: str | list[dict] | None,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> FrontierCall:
        ...


class AnthropicAdapter(FrontierAdapter):
    """Calls claude-*-* via the Anthropic SDK.

    `system` may be either a plain string or a list of content blocks
    (e.g. `[{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}]`)
    so callers can opt a stable prefix into prompt caching.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed. Run: "
                ".venv/bin/pip install -r slm-experiments/requirements.txt"
            ) from e
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it or pass api_key=..."
            )
        self.client = Anthropic(api_key=key)
        self.model = model
        self.name = f"anthropic:{model}"

    def complete(
        self,
        *,
        system: str | list[dict] | None,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> FrontierCall:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user}],
        }
        if system is not None:
            kwargs["system"] = system

        t0 = time.time()
        resp = self.client.messages.create(**kwargs)
        elapsed = time.time() - t0

        text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        text = "".join(text_parts)

        u = resp.usage
        usage = FrontierUsage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            calls=1,
            wall_seconds=elapsed,
        )
        return FrontierCall(text=text, usage=usage)


class OllamaFrontierAdapter(FrontierAdapter):
    """Local Ollama model standing in for the frontier (no-API-key mode).

    Token counts come from Ollama's `eval_count` (output) and
    `prompt_eval_count` (input), which we map onto FrontierUsage so the
    rest of the harness doesn't need to special-case it. This is for
    smoke-testing the harness, not for headline numbers.
    """

    def __init__(self, model: str = "llama3.2:3b", url: str = "http://127.0.0.1:11434/api/generate"):
        self.model = model
        self.url = url
        self.name = f"ollama:{model}"

    def complete(
        self,
        *,
        system: str | list[dict] | None,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> FrontierCall:
        import json
        import urllib.request

        sys_text = system if isinstance(system, str) else (
            "\n\n".join(b.get("text", "") for b in (system or []))
        )
        prompt = f"{sys_text}\n\n{user}" if sys_text else user
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read())
        elapsed = time.time() - t0
        usage = FrontierUsage(
            input_tokens=int(payload.get("prompt_eval_count", 0) or 0),
            output_tokens=int(payload.get("eval_count", 0) or 0),
            calls=1,
            wall_seconds=elapsed,
        )
        return FrontierCall(text=payload.get("response", ""), usage=usage)


def build_default(prefer_local: bool = False) -> FrontierAdapter:
    """Pick an adapter based on environment.

    With `prefer_local=False` (default), use Anthropic if a key is
    present; otherwise fall back to the local Ollama stand-in so the
    harness still runs.
    """
    if not prefer_local and os.environ.get("ANTHROPIC_API_KEY"):
        model = os.environ.get("EXPERIMENTS_FRONTIER_MODEL", "claude-sonnet-4-6")
        return AnthropicAdapter(model=model)
    model = os.environ.get("EXPERIMENTS_LOCAL_FRONTIER", "llama3.2:3b")
    return OllamaFrontierAdapter(model=model)
