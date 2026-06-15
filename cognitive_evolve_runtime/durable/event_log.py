"""Append-only durable event log."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..core.redaction import redact
from .file_lock import file_lock
from .step_state import utc_now


@dataclass
class DurableEvent:
    event_type: str
    run_id: str = "run"
    round_id: str = "runtime"
    step_id: str = "step"
    timestamp: str = field(default_factory=utc_now)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventLog:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, event_type: str, payload: dict[str, Any] | None = None, *, run_id: str = "run", round_id: str = "runtime", step_id: str = "step") -> dict[str, Any]:
        record = DurableEvent(event_type=event_type, run_id=run_id, round_id=round_id, step_id=step_id, payload=dict(payload or {})).to_dict()
        append_jsonl(self.path, record)
        return record

    def read(self) -> list[dict[str, Any]]:
        return list(read_jsonl(self.path))

    def rebuild(self) -> dict[str, dict[str, Any]]:
        state: dict[str, dict[str, Any]] = {}
        for event in self.read():
            key = f"{event.get('run_id')}::{event.get('round_id')}::{event.get('step_id')}"
            state.setdefault(key, {})
            state[key].update(dict(event.get("payload") or {}))
            state[key]["last_event_type"] = event.get("event_type")
            state[key]["last_event_at"] = event.get("timestamp")
        return state


def append_jsonl(path: Path | str, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = redact(record)
    with file_lock(target.with_name(target.name + ".lock")):
        with target.open("a", encoding="utf-8") as handle:
            # codeql[py/clear-text-storage-sensitive-data] record is recursively redacted before persistence.
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def read_jsonl(path: Path | str) -> Iterable[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                records.append({"event_type": "corrupt_event_line", "raw": stripped[:500]})
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


__all__ = ["DurableEvent", "EventLog", "append_jsonl", "read_jsonl"]
