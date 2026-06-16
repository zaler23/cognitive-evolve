"""Divergent operator protocol and deterministic helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class DivergentOperator(Protocol):
    operator_id: str
    def propose(self, parent: Any, archive: Any, tension_map: Any, k: int = 1) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class PromptOperator:
    operator_id: str
    lens: str
    instruction: str

    def propose(self, parent: Any, archive: Any, tension_map: Any, k: int = 1) -> list[dict[str, Any]]:
        parent_id = str(getattr(parent, "id", "parent") or "parent")
        base = str(getattr(parent, "concise_claim", "") or getattr(parent, "core_mechanism", "") or getattr(parent, "artifact", ""))[:240]
        return [
            {
                "operator": self.operator_id,
                "parent_id": parent_id,
                "descriptor": (self.operator_id, self.lens, index),
                "mutation_instruction": f"{self.instruction} Lens={self.lens}. Parent core={base}",
            }
            for index in range(max(1, int(k or 1)))
        ]


__all__ = ["DivergentOperator", "PromptOperator"]
