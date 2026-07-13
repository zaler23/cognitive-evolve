"""Durable JSON store for Nexus archives."""
from __future__ import annotations

import json
from pathlib import Path

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.durable.file_lock import atomic_write_json


class ArchiveStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, archives: ArchiveManager) -> None:
        atomic_write_json(self.path, archives.to_dict(), sort_keys=True)

    def load(self) -> ArchiveManager:
        if not self.path.exists():
            return ArchiveManager()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"archive store must be a JSON object: {self.path}")
        return ArchiveManager.from_dict(data)


__all__ = ["ArchiveStore"]
