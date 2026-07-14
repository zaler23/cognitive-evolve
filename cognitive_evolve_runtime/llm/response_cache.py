"""Run-local durable replay for completed logical LLM calls."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.durable import stable_hash
from cognitive_evolve_runtime.durable.file_lock import atomic_write_json

from .journal import journal_dir
from .session import current_llm_session

def response_signature(payload: dict[str, Any]) -> str:
    return stable_hash(payload)


def load_response(signature: str) -> dict[str, Any] | None:
    path = _response_path(signature)
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("signature") != signature:
        return None
    return data


def store_response(signature: str, record: dict[str, Any]) -> None:
    path = _response_path(signature)
    if path is None:
        return
    atomic_write_json(path, {"schema_version": "llm-response/v1", "signature": signature, **record}, sort_keys=True)


def _response_path(signature: str) -> Path | None:
    response_root = str(current_llm_session().response_dir or "").strip()
    directory = Path(response_root) if response_root else journal_dir()
    if directory is None:
        return None
    return directory / "llm-responses" / "v1" / f"{signature}.json"

__all__ = ["load_response", "response_signature", "store_response"]
