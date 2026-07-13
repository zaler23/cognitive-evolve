"""MAP-Elites illumination wrapper around the existing quality-diversity archive."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cognitive_evolve_runtime.archives.quality_diversity import QualityDiversityArchive, candidate_bin_key, candidate_search_quality
from cognitive_evolve_runtime.nexus.search_kernel.descriptor_cells import behavior_descriptor


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



__all__ = ["MapElitesIllumination", "behavior_descriptor"]
