"""Descriptor coverage reports for advisory pool preprocessing."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.archives.quality_diversity import descriptor_cell_distribution, descriptor_population_entropy
from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.fabric.config import PreprocessConfig


def pool_coverage_report(candidates: list[CandidateGenome], *, expected_cells: list[str] | None = None, config: PreprocessConfig | None = None) -> dict[str, Any]:
    cfg = config or PreprocessConfig()
    distribution = descriptor_cell_distribution(candidates)
    occupied = set(distribution)
    expected = {str(item) for item in expected_cells or [] if str(item or "").strip()}
    population_size = len(candidates)
    mean_count = population_size / max(1, len(distribution)) if distribution else 0.0
    sparse_cells = sorted(cell for cell, count in distribution.items() if count <= cfg.sparse_cell_max_count)
    overrepresented_cells = sorted(cell for cell, count in distribution.items() if mean_count > 0.0 and count >= mean_count * cfg.overrepresented_cell_multiplier and count > cfg.sparse_cell_max_count)
    return {
        "advisory": True,
        "population_size": population_size,
        "descriptor_cell_distribution": dict(sorted(distribution.items())),
        "descriptor_entropy": descriptor_population_entropy(candidates),
        "occupied_cell_count": len(occupied),
        "sparse_cells": sparse_cells,
        "overrepresented_cells": overrepresented_cells,
        "missing_cells": sorted(expected - occupied),
    }


__all__ = ["pool_coverage_report"]
