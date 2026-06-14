"""Persist per-candidate verification traces."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.durable.file_lock import atomic_write_json


class VerificationTraceStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, candidate_id: str, trace: list[dict[str, Any]]) -> Path:
        path = self.root / f"{candidate_id}.json"
        atomic_write_json(path, {"candidate_id": candidate_id, "verification_trace": trace}, sort_keys=True)
        return path

    def load(self, candidate_id: str) -> list[dict[str, Any]]:
        path = self.root / f"{candidate_id}.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [dict(item) for item in data.get("verification_trace", []) if isinstance(item, dict)] if isinstance(data, dict) else []


__all__ = ["VerificationTraceStore"]
