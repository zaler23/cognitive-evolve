"""MAP-Elites illumination wrapper around the existing quality-diversity archive."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cognitive_evolve_runtime.archives.quality_diversity import QualityDiversityArchive, candidate_bin_key, candidate_search_quality


@dataclass
class MapElitesIllumination:
    archive: QualityDiversityArchive = field(default_factory=QualityDiversityArchive)

    def add(self, candidate: Any) -> dict[str, Any]:
        self.archive.update(candidate)
        descriptor = behavior_descriptor(candidate)
        return {"candidate_id": getattr(candidate, "id", ""), "bin_key": candidate_bin_key(candidate), "descriptor": descriptor, "search_quality": candidate_search_quality(candidate)}

    def sparse_cells(self) -> list[str]:
        occupied = set(self.archive.elites_by_niche)
        return sorted(occupied)

    def to_dict(self) -> dict[str, Any]:
        return {"archive": self.archive.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MapElitesIllumination":
        raw = data or {}
        return cls(archive=QualityDiversityArchive.from_dict(raw.get("archive") or raw))


def behavior_descriptor(candidate: Any) -> tuple[Any, ...]:
    scores = getattr(candidate, "multihead_scores", {}) or {}
    metadata = getattr(candidate, "metadata", {}) if isinstance(getattr(candidate, "metadata", {}), dict) else {}
    family = (getattr(candidate, "artifact_type", "") or getattr(candidate, "core_mechanism", "") or "general")
    complexity = "high" if len(str(getattr(candidate, "artifact", ""))) > 2000 else "low"
    score_band = int(float(scores.get("frontier_score", scores.get("objective_score", 0.0)) or 0.0) * 5)
    failure_class = metadata.get("failure_class") or ("terminal" if metadata.get("terminal_reject") else "open")
    verified = tuple(sorted(str(item) for item in metadata.get("resolved_challenge_ids", []) if item))
    sensitivity = metadata.get("parameter_sensitivity", "unknown")
    return (family, complexity, score_band, failure_class, verified, sensitivity)


__all__ = ["MapElitesIllumination", "behavior_descriptor"]
