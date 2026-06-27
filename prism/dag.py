"""Shared dataclasses for prism."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NodeSpec:
    """A node in a prism DAG before execution."""

    id: str
    prompt: str
    depends_on: list[str] = field(default_factory=list)
    type: str | None = None        # taxonomy hint (skips classifier when set)
    backend: str | None = None     # explicit override: "slm" | "llm"
    model: str | None = None       # optional model pin (SLM model id, or LLM model)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "depends_on": list(self.depends_on),
            "type": self.type,
            "backend": self.backend,
            "model": self.model,
        }


@dataclass
class NodeResult:
    """Per-node execution record."""

    id: str
    backend: str
    type: str | None
    model: str | None
    prompt: str            # resolved (placeholders substituted)
    result: str
    wall_seconds: float
    tokens_in: int = 0     # only populated for llm or local-frontier nodes
    tokens_out: int = 0    # llm output, or SLM eval_count
    error: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class PrismRun:
    """Full execution snapshot for one prism plan."""

    plan_id: str
    description: str
    started_at: float
    finished_at: float
    final_output: str
    nodes: list[NodeResult]
    classifier_calls: int = 0
    classifier_seconds: float = 0.0

    @property
    def wall_seconds(self) -> float:
        return self.finished_at - self.started_at

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "description": self.description,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wall_seconds": self.wall_seconds,
            "final_output": self.final_output,
            "classifier_calls": self.classifier_calls,
            "classifier_seconds": self.classifier_seconds,
            "nodes": [n.to_dict() for n in self.nodes],
        }
