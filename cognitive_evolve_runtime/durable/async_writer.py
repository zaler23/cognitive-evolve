"""Lossless FIFO write-behind observer for immutable persistence batches."""
from __future__ import annotations

import queue
import threading
import time
from typing import Any


_STOP = object()


class WriteBehindObserver:
    """Freeze on the caller thread and perform full durable writes on one worker."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self._queue: queue.Queue[tuple[float, Any] | object] = queue.Queue()
        self._error: BaseException | None = None
        self._closed = False
        self._lock = threading.Lock()
        self._telemetry = {
            "persistence_batches": 0,
            "persistence_bytes": 0,
            "persistence_queue_wait_ms": 0.0,
            "persistence_write_ms": 0.0,
        }
        self._worker = threading.Thread(target=self._run, name="cogev-persistence", daemon=True)
        self._worker.start()

    def __call__(self, update: dict[str, Any]) -> None:
        self._raise_worker_error()
        if self._closed:
            raise RuntimeError("write-behind observer is closed")
        batch = self.store.freeze(update)
        if batch is None:
            return
        self._queue.put((time.perf_counter(), batch))
        phase = str(update.get("phase") or "")
        if phase in {"post_seeding", "error_checkpoint"}:
            self.flush()
        elif phase == "final_synthesis":
            self.close()

    def flush(self) -> None:
        self._queue.join()
        self._raise_worker_error()

    def close(self) -> None:
        if self._closed:
            self._raise_worker_error()
            return
        self._queue.put(_STOP)
        self._closed = True
        self._queue.join()
        self._worker.join()
        self._raise_worker_error()

    def telemetry(self) -> dict[str, int | float]:
        with self._lock:
            return {
                key: round(value, 3) if isinstance(value, float) else value
                for key, value in self._telemetry.items()
            }

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                enqueued_at, batch = item
                queue_wait_ms = (time.perf_counter() - enqueued_at) * 1000.0
                started = time.perf_counter()
                if self._error is None:
                    self.store.write(batch)
                write_ms = (time.perf_counter() - started) * 1000.0
                with self._lock:
                    self._telemetry["persistence_batches"] += 1
                    self._telemetry["persistence_bytes"] += int(getattr(batch, "serialized_bytes", 0))
                    self._telemetry["persistence_queue_wait_ms"] += queue_wait_ms
                    self._telemetry["persistence_write_ms"] += write_ms
            except BaseException as exc:
                if self._error is None:
                    self._error = exc
            finally:
                self._queue.task_done()

    def _raise_worker_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("asynchronous persistence worker failed") from self._error


__all__ = ["WriteBehindObserver"]
