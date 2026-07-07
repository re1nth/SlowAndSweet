"""Encoder + Head for the router.

Encoder is a frozen SentenceTransformer wrapper. Head bundles a
LinearRegression (cost) and a LogisticRegression (quality) fit on the
same 384-dim embedding space, plus a training-set metadata block used
by the auto-promote gate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np


EMBED_DIM = 384


@dataclass
class Decision:
    decision: str                        # "solo" | "mixture" | "unsure"
    predicted_reduction_pct: float
    predicted_quality_ok_prob: float
    confidence: float
    head_version: str
    policy: str                          # "learned" | "heuristic" | "explore"
    decision_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HeadMetadata:
    version: str = "v0"
    n_train: int = 0
    n_quality_train: int = 0
    holdout_mae_pp: float | None = None
    holdout_pearson_r: float | None = None
    holdout_quality_acc: float | None = None
    trained_at: float = field(default_factory=time.time)
    notes: str = ""


class Encoder:
    """Frozen SentenceTransformer. One instance per process."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, prompt: str) -> np.ndarray:
        vec = self._model.encode(
            prompt, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
        return vec.astype(np.float32).reshape(-1)

    def encode_batch(self, prompts: list[str]) -> np.ndarray:
        return self._model.encode(
            prompts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        ).astype(np.float32)


class Head:
    """Bundles cost regressor + quality classifier + metadata."""

    def __init__(
        self,
        cost_regressor: Any = None,
        quality_classifier: Any = None,
        metadata: HeadMetadata | None = None,
    ):
        self.cost_regressor = cost_regressor
        self.quality_classifier = quality_classifier
        self.metadata = metadata or HeadMetadata()

    def predict(self, x: np.ndarray) -> tuple[float, float, float]:
        """Returns (predicted_reduction_pct, predicted_quality_ok_prob, confidence).

        Confidence is a heuristic: 1 - abs(P(quality) - 0.5) * 2, floored at 0.1.
        This is not calibrated; the design doc calls for later replacement with
        a proper uncertainty estimator (e.g. small ensemble).
        """
        x = x.reshape(1, -1)
        if self.cost_regressor is None:
            predicted_reduction = 0.0
        else:
            predicted_reduction = float(self.cost_regressor.predict(x)[0])

        if self.quality_classifier is None:
            predicted_quality_prob = 0.5
        else:
            predicted_quality_prob = float(self.quality_classifier.predict_proba(x)[0, 1])

        # Confidence: distance from an uninformed 50/50 quality prior. Linear head
        # regressor doesn't expose per-point variance, so we treat quality prob as
        # the main uncertainty signal.
        confidence = max(0.1, abs(predicted_quality_prob - 0.5) * 2)
        return predicted_reduction, predicted_quality_prob, confidence

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(
            {
                "cost_regressor": self.cost_regressor,
                "quality_classifier": self.quality_classifier,
                "metadata": asdict(self.metadata),
            },
            tmp,
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Head":
        raw = joblib.load(Path(path))
        meta = HeadMetadata(**raw["metadata"])
        return cls(
            cost_regressor=raw["cost_regressor"],
            quality_classifier=raw["quality_classifier"],
            metadata=meta,
        )


def head_pointer_read(pointer_path: Path) -> str | None:
    p = Path(pointer_path)
    if not p.exists():
        return None
    return p.read_text().strip() or None


def head_pointer_write(pointer_path: Path, version: str) -> None:
    p = Path(pointer_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(version + "\n")
    tmp.replace(p)
