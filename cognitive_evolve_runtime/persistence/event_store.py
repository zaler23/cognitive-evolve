"""Append-only Nexus event store with replay."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

from cognitive_evolve_runtime.durable.file_lock import file_lock
from cognitive_evolve_runtime.core.redaction import redact
from cognitive_evolve_runtime.core.serialization import stable_json, utc_now


class EventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = redact(event)
        payload.setdefault("at", utc_now())
        lock_path = self.path.with_name(self.path.name + ".lock")
        with file_lock(lock_path):
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return payload

    def append_once(self, event: dict[str, Any], *, identity_keys: tuple[str, ...] | None = None) -> dict[str, Any] | None:
        """Append an event unless an equivalent event is already present.

        Final persistence can run after live checkpointing or after a resumed
        write.  ``append_once`` keeps the JSONL log append-only while preventing
        duplicate final progress/pipeline events for the same logical round.
        """

        keys = identity_keys or ("type", "round", "stage", "stage_index", "stage_count")
        signature = _event_signature(event, keys)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with file_lock(lock_path):
            if self.path.exists():
                for existing in self.replay():
                    if _event_signature(existing, keys) == signature:
                        return None
            payload = redact(event)
            payload.setdefault("at", utc_now())
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return payload

    def append_many_once(self, events: list[dict[str, Any]], *, identity_keys: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """Append multiple logical events with one replay scan and one lock."""

        keys = identity_keys or ("type", "round", "stage", "stage_index", "stage_count")
        appended: list[dict[str, Any]] = []
        lock_path = self.path.with_name(self.path.name + ".lock")
        with file_lock(lock_path):
            signatures = {_event_signature(existing, keys) for existing in self._replay_unlocked()}
            with self.path.open("a", encoding="utf-8") as handle:
                for event in events:
                    signature = _event_signature(event, keys)
                    if signature in signatures:
                        continue
                    signatures.add(signature)
                    payload = redact(event)
                    payload.setdefault("at", utc_now())
                    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
                    appended.append(payload)
                handle.flush()
                os.fsync(handle.fileno())
        return appended

    def read_all(self) -> list[dict[str, Any]]:
        return list(self.replay())

    def replay(self) -> Iterator[dict[str, Any]]:
        yield from self._replay_unlocked()

    def _replay_unlocked(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                yield data


def _event_signature(event: dict[str, Any], keys: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((key, stable_json(redact(event.get(key)))) for key in keys if key in event)


__all__ = ["EventStore"]
