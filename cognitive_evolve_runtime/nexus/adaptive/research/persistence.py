"""Safe research artifact shaping helpers."""
from __future__ import annotations

import json
from typing import Any

from cognitive_evolve_runtime.nexus.adaptive.state import sanitize_adaptive_event

MAX_RESEARCH_EVENT_BYTES = 8192
MAX_RESEARCH_SNAPSHOT_BYTES = 200_000


def safe_research_payload(value: Any, *, max_bytes: int = MAX_RESEARCH_SNAPSHOT_BYTES) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    safe = sanitize_adaptive_event(value)
    encoded = json.dumps(safe, ensure_ascii=False, sort_keys=True, default=str)
    if len(encoded.encode("utf-8")) <= max_bytes:
        return safe
    return {"truncated": True, "size_bytes": len(encoded.encode("utf-8"))}


__all__ = ["MAX_RESEARCH_EVENT_BYTES", "MAX_RESEARCH_SNAPSHOT_BYTES", "safe_research_payload"]
