"""Load and query the routing taxonomy from policies.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


_HERE = Path(__file__).resolve().parent


@dataclass
class Policy:
    types: dict[str, dict]   # type_name -> {"backend": "slm"|"llm", "description": str}
    default_unknown: str

    @classmethod
    def load(cls, path: Path | None = None) -> "Policy":
        p = path or (_HERE / "policies.yaml")
        with open(p) as fh:
            data = yaml.safe_load(fh)
        return cls(
            types=data.get("types", {}),
            default_unknown=data.get("default_unknown", "llm"),
        )

    def backend_for(self, task_type: str | None) -> str:
        if task_type and task_type in self.types:
            return self.types[task_type]["backend"]
        return self.default_unknown

    @property
    def type_names(self) -> list[str]:
        return list(self.types.keys())

    def vocab_block(self) -> str:
        """A compact bullet list for the classifier prompt."""
        return "\n".join(
            f"- {name}: {meta.get('description', '').strip()}"
            for name, meta in self.types.items()
        )
