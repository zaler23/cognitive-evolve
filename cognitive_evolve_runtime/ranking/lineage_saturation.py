from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.lineage import lineage_counts


@dataclass
class LineageSaturationReport:
    saturated: bool
    saturated_families: list[str] = field(default_factory=list)
    lineage_counts: dict[str, int] = field(default_factory=dict)
    recommended_response: str = "continue"

    def to_dict(self) -> dict[str, Any]:
        return {
            "saturated": self.saturated,
            "saturated_families": self.saturated_families,
            "lineage_counts": self.lineage_counts,
            "recommended_response": self.recommended_response,
        }


def detect_lineage_saturation(candidates: list[CandidateGenome], *, threshold: int = 4) -> LineageSaturationReport:
    counts = lineage_counts(candidates)
    saturated = [family for family, count in counts.items() if count >= threshold]
    return LineageSaturationReport(
        saturated=bool(saturated),
        saturated_families=saturated,
        lineage_counts=counts,
        recommended_response="quarantine_lineage_or_increase_rarity_budget" if saturated else "continue",
    )


__all__ = ["LineageSaturationReport", "detect_lineage_saturation"]
