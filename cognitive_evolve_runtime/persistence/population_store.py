"""Durable JSON store for Nexus populations."""
from __future__ import annotations

import json
from pathlib import Path

from cognitive_evolve_runtime.candidates.genome import CandidatePopulation
from cognitive_evolve_runtime.durable.file_lock import atomic_write_json


class PopulationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, population: CandidatePopulation) -> None:
        atomic_write_json(self.path, population.to_dict(), sort_keys=True)

    def load(self) -> CandidatePopulation:
        if not self.path.exists():
            return CandidatePopulation()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"population store must be a JSON object: {self.path}")
        return CandidatePopulation.from_dict(data)


__all__ = ["PopulationStore"]
