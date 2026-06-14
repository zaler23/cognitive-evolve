"""Small cross-process file locking and atomic JSON helpers.

The runtime uses local files as its default durable store.  These helpers make
JSON publication safe for concurrent threads/processes by serializing writers on
a sidecar lock file and publishing only fully validated temporary files.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from cognitive_evolve_runtime.nexus._serde import json_ready

try:  # pragma: no cover - Windows fallback is covered by behavior, not platform.
    import fcntl  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

_THREAD_LOCKS: dict[Path, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.RLock:
    resolved = path.resolve() if path.exists() else path.absolute()
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(resolved)
        if lock is None:
            lock = threading.RLock()
            _THREAD_LOCKS[resolved] = lock
        return lock


@contextmanager
def file_lock(path: Path | str) -> Iterator[None]:
    """Serialize writers across processes when the platform supports it."""

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    thread_lock = _thread_lock(lock_path)
    with thread_lock:
        with lock_path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_DIRECTORY)
    except (AttributeError, OSError):  # pragma: no cover - platform-specific
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(path: Path | str, data: dict[str, Any], *, sort_keys: bool = False, allow_cycles: bool = False) -> None:
    """Write a JSON object using lock + fsync + atomic replace.

    A unique temporary file avoids writer collisions before the sidecar lock is
    acquired/released and guarantees readers never observe a partial JSON body.
    """

    target = Path(path)
    if not isinstance(data, dict):
        raise ValueError(f"atomic JSON target must be a dict object: {target}")
    if not allow_cycles and _has_cycle(data, active=set(), seen=set()):
        raise ValueError(f"checkpoint JSON contains a circular reference: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(target.name + ".lock")
    with file_lock(lock_path):
        tmp = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
        safe_data = json_ready(data)
        if not isinstance(safe_data, dict):
            raise ValueError(f"checkpoint JSON must normalize to an object: {target}")
        payload = json.dumps(safe_data, ensure_ascii=False, indent=2, sort_keys=sort_keys, default=str, allow_nan=False) + "\n"
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            with tmp.open("r", encoding="utf-8") as handle:
                decoded = json.load(handle)
            if not isinstance(decoded, dict):
                raise ValueError(f"checkpoint JSON must be an object: {tmp}")
            os.replace(tmp, target)
            _fsync_dir(target.parent)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def _has_cycle(value: Any, *, active: set[int], seen: set[int]) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return False
    if not isinstance(value, (dict, list, tuple, set)):
        return False
    obj_id = id(value)
    if obj_id in active:
        return True
    if obj_id in seen:
        return False
    active.add(obj_id)
    try:
        if isinstance(value, dict):
            return any(_has_cycle(item, active=active, seen=seen) for item in value.values())
        return any(_has_cycle(item, active=active, seen=seen) for item in value)
    finally:
        active.remove(obj_id)
        seen.add(obj_id)


def atomic_write_text(path: Path | str, text: str) -> None:
    """Write text using lock + fsync + atomic replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(target.name + ".lock")
    with file_lock(lock_path):
        tmp = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
            _fsync_dir(target.parent)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


__all__ = ["atomic_write_json", "atomic_write_text", "file_lock"]
