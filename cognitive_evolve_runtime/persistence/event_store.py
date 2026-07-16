"""Append-only Nexus event store with replay."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator

from cognitive_evolve_runtime.durable.file_lock import file_lock
from cognitive_evolve_runtime.core.redaction import redact
from cognitive_evolve_runtime.core.serialization import stable_json, utc_now

logger = logging.getLogger(__name__)


class EventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = redact(event)
        payload.setdefault("at", utc_now())
        lock_path = self.path.with_name(self.path.name + ".lock")
        with file_lock(lock_path):
            self._append_unlocked([payload])
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
            self._append_unlocked([payload])
            return payload

    def append_many_once(self, events: list[dict[str, Any]], *, identity_keys: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """Append multiple logical events with one replay scan and one lock."""

        keys = identity_keys or ("type", "round", "stage", "stage_index", "stage_count")
        appended: list[dict[str, Any]] = []
        lock_path = self.path.with_name(self.path.name + ".lock")
        with file_lock(lock_path):
            signatures = {_event_signature(existing, keys) for existing in self._replay_unlocked()}
            for event in events:
                signature = _event_signature(event, keys)
                if signature in signatures:
                    continue
                signatures.add(signature)
                payload = redact(event)
                payload.setdefault("at", utc_now())
                appended.append(payload)
            self._append_unlocked(appended)
        return appended

    def watermark(self) -> dict[str, str | int]:
        """Return the byte offset through the last complete JSONL record."""

        lock_path = self.path.with_name(self.path.name + ".lock")
        with file_lock(lock_path):
            return {"path": self.path.name, "offset": self._committed_offset_unlocked()}

    def reconcile(self, watermark: dict[str, Any]) -> dict[str, Any]:
        """Classify event records for audit without changing runtime state."""

        expected_path = str(watermark.get("path") or "")
        if expected_path != self.path.name:
            raise ValueError(f"event watermark path mismatch: expected {expected_path!r}, actual {self.path.name!r}")
        offset = int(watermark["offset"])
        if offset < 0:
            raise ValueError("event watermark offset must be non-negative")

        report: dict[str, Any] = {
            "watermark": {"path": expected_path, "offset": offset},
            "committed": [],
            "post-snapshot": [],
            "truncated": [],
        }
        lock_path = self.path.with_name(self.path.name + ".lock")
        with file_lock(lock_path):
            actual_size = self.path.stat().st_size if self.path.exists() else 0
            if actual_size < offset:
                report["truncated"].append(
                    {
                        "reason": "event_stream_shorter_than_watermark",
                        "expected_offset": offset,
                        "actual_offset": actual_size,
                    }
                )
            if not self.path.exists():
                return report
            cursor = 0
            with self.path.open("rb") as handle:
                for raw_line in handle:
                    start_offset = cursor
                    cursor += len(raw_line)
                    reason = ""
                    event: Any = None
                    if not raw_line.endswith(b"\n"):
                        reason = "incomplete_line"
                    else:
                        try:
                            event = json.loads(raw_line)
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            reason = "invalid_json"
                        if not reason and not isinstance(event, dict):
                            reason = "invalid_event"
                    if reason:
                        report["truncated"].append({"reason": reason, "start_offset": start_offset, "end_offset": cursor})
                    elif cursor <= offset:
                        report["committed"].append(event)
                    elif start_offset >= offset:
                        report["post-snapshot"].append(event)
                    else:
                        report["truncated"].append(
                            {"reason": "watermark_splits_line", "start_offset": start_offset, "end_offset": cursor}
                        )
        return report

    def read_all(self) -> list[dict[str, Any]]:
        return list(self.replay())

    def replay(self) -> Iterator[dict[str, Any]]:
        yield from self._replay_unlocked()

    def _append_unlocked(self, payloads: list[dict[str, Any]]) -> None:
        if not payloads:
            return
        with self.path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            tail_end_offset = handle.tell()
            if tail_end_offset:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    marker = {
                        "at": utc_now(),
                        "reason": "unterminated_jsonl_tail",
                        "tail_end_offset": tail_end_offset,
                        "type": "event_store_tail_quarantined",
                    }
                    handle.write(b"\n")
                    handle.write(_event_line(marker))
            for payload in payloads:
                handle.write(_event_line(payload))
            handle.flush()
            os.fsync(handle.fileno())

    def _committed_offset_unlocked(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size == 0:
                return 0
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) == b"\n":
                return size
            position = size
            while position:
                start = max(0, position - 8192)
                handle.seek(start)
                chunk = handle.read(position - start)
                newline = chunk.rfind(b"\n")
                if newline >= 0:
                    return start + newline + 1
                position = start
        return 0

    def _replay_unlocked(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        corrupted_lines = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                corrupted_lines += 1
                continue
            if isinstance(data, dict):
                yield data
        if corrupted_lines:
            logger.warning("skipped %d corrupted lines while replaying event store", corrupted_lines)


def _event_signature(event: dict[str, Any], keys: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((key, stable_json(redact(event.get(key)))) for key in keys if key in event)


def _event_line(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, ensure_ascii=False, sort_keys=True, default=str) + "\n").encode("utf-8")


__all__ = ["EventStore"]
