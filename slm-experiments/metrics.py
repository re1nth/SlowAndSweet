"""Shared metrics types for arms and reviewer."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class FrontierUsage:
    """Token + time accounting for one or more frontier-model calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    calls: int = 0
    wall_seconds: float = 0.0

    def add(self, other: "FrontierUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.calls += other.calls
        self.wall_seconds += other.wall_seconds

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class SLMUsage:
    """Aggregated SLM usage from a plan run.

    Note: the slm-queue currently only exposes Ollama's eval_count (output
    tokens) per node. Prompt-eval tokens are not tracked yet, so input_tokens
    is reported as None to avoid implying we measured something we didn't.
    """

    output_tokens: int = 0
    input_tokens: int | None = None
    nodes: int = 0
    wall_seconds: float = 0.0
    models_used: list[str] = field(default_factory=list)


@dataclass
class ArmResult:
    """Output of running one arm (solo or mixture) on one case."""

    arm: str  # "solo" or "mixture"
    output: str
    frontier: FrontierUsage = field(default_factory=FrontierUsage)
    slm: SLMUsage = field(default_factory=SLMUsage)
    wall_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["frontier"]["total_tokens"] = self.frontier.total_tokens
        return d


@dataclass
class ReviewVerdict:
    """Reviewer's structured verdict for one case."""

    winner: str  # "solo" | "mixture" | "tie"
    confidence: float  # 0..1
    scores: dict  # {"solo": {"accuracy": 1-5, ...}, "mixture": {...}}
    reasoning: str
    reviewer_usage: FrontierUsage = field(default_factory=FrontierUsage)

    def to_dict(self) -> dict:
        return asdict(self)
