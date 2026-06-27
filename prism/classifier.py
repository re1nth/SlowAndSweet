"""SLM-driven task classifier.

Given the prompt of one node, picks one task type from the policy
vocabulary using a small local model (default `smollm2:1.7b`). The
classifier is intentionally narrow: it returns one of the known type
names or `unknown`. Anything else falls back to `unknown` and the
router treats it as the policy's `default_unknown` backend.

We memoize per `(prompt_text, model)` in-process so a re-run of the
same DAG doesn't re-classify the same nodes.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from threading import Lock

from policy import Policy


_OLLAMA_URL_DEFAULT = "http://127.0.0.1:11434/api/generate"


@dataclass
class Classification:
    type: str           # one of policy.types or "unknown"
    raw: str            # raw model response
    wall_seconds: float


class Classifier:
    """Wrapper around an Ollama model for one-of-N type classification."""

    def __init__(
        self,
        policy: Policy,
        *,
        model: str = "smollm2:1.7b",
        ollama_url: str = _OLLAMA_URL_DEFAULT,
    ):
        self.policy = policy
        self.model = model
        self.ollama_url = ollama_url
        self._cache: dict[tuple[str, str], Classification] = {}
        self._lock = Lock()

    def classify(self, prompt_text: str) -> Classification:
        key = (prompt_text, self.model)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        instruction = self._build_prompt(prompt_text)
        t0 = time.time()
        body = json.dumps({
            "model": self.model,
            "prompt": instruction,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 24},
        }).encode("utf-8")
        req = urllib.request.Request(
            self.ollama_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read())
        elapsed = time.time() - t0
        raw = (payload.get("response") or "").strip()
        chosen = self._extract_type(raw)
        result = Classification(type=chosen, raw=raw, wall_seconds=elapsed)
        with self._lock:
            self._cache[key] = result
        return result

    def _build_prompt(self, node_prompt: str) -> str:
        # Truncate very long node prompts — the classifier only needs the
        # shape of the work, not the full context.
        snippet = node_prompt.strip()
        if len(snippet) > 600:
            snippet = snippet[:600] + " ..."

        return (
            "You are a task-type classifier. Read the TASK and pick exactly "
            "ONE label from the list below. Respond with the label name only, "
            "no punctuation, no explanation.\n\n"
            "LABELS:\n"
            f"{self.policy.vocab_block()}\n"
            "- unknown: none of the above fit.\n\n"
            "TASK:\n"
            f"{snippet}\n\n"
            "LABEL:"
        )

    def _extract_type(self, raw: str) -> str:
        # Strip whitespace, quotes, punctuation; lowercase.
        token = re.split(r"[\s,.;:!?\"']+", raw.strip(), maxsplit=1)[0].lower()
        if token in self.policy.types:
            return token
        # Sometimes the model emits the label with a leading "- " or "* ".
        if token.startswith("-") or token.startswith("*"):
            token = token.lstrip("-* ").strip()
            if token in self.policy.types:
                return token
        return "unknown"
