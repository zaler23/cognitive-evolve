"""Bounded API executors for long CognitiveEvolve jobs."""
from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable


class QueueFullError(RuntimeError):
    """Raised when the bounded API executor has no worker/queue capacity."""


class BoundedExecutor:
    """ThreadPoolExecutor wrapper with a real submit-time queue bound.

    ``ThreadPoolExecutor`` has an unbounded internal queue, so replacing raw
    threads with it alone would only move the resource leak.  The semaphore caps
    running plus queued jobs and fails fast with HTTP 503 when capacity is
    exhausted.
    """

    def __init__(self, *, max_workers: int, max_queue: int, thread_name_prefix: str) -> None:
        self.max_workers = max(1, int(max_workers or 1))
        self.max_queue = max(0, int(max_queue or 0))
        self._capacity = threading.BoundedSemaphore(self.max_workers + self.max_queue)
        self._pool = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix=thread_name_prefix)
        self._shutdown = False
        self._lock = threading.Lock()

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        if not self._capacity.acquire(blocking=False):
            raise QueueFullError("CognitiveEvolve API executor queue is full")
        with self._lock:
            if self._shutdown:
                self._capacity.release()
                raise RuntimeError("CognitiveEvolve API executor is shut down")
            try:
                future = self._pool.submit(self._run_and_release, fn, *args, **kwargs)
            except Exception:
                self._capacity.release()
                raise
        return future

    def shutdown(self, *, wait: bool = False, cancel_futures: bool = True) -> None:
        with self._lock:
            self._shutdown = True
            self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)

    def _run_and_release(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        finally:
            self._capacity.release()


_EXECUTOR_LOCK = threading.Lock()
_JOB_EXECUTOR: BoundedExecutor | None = None
_STREAM_EXECUTOR: BoundedExecutor | None = None


def get_job_executor() -> BoundedExecutor:
    global _JOB_EXECUTOR
    with _EXECUTOR_LOCK:
        if _JOB_EXECUTOR is None:
            _JOB_EXECUTOR = BoundedExecutor(
                max_workers=_env_int("COGEV_API_JOB_WORKERS", 2),
                max_queue=_env_int("COGEV_API_JOB_QUEUE", 8),
                thread_name_prefix="cogev-job",
            )
        return _JOB_EXECUTOR


def get_stream_executor() -> BoundedExecutor:
    global _STREAM_EXECUTOR
    with _EXECUTOR_LOCK:
        if _STREAM_EXECUTOR is None:
            _STREAM_EXECUTOR = BoundedExecutor(
                max_workers=_env_int("COGEV_API_STREAM_WORKERS", 2),
                max_queue=_env_int("COGEV_API_STREAM_QUEUE", 8),
                thread_name_prefix="cogev-stream",
            )
        return _STREAM_EXECUTOR


def shutdown_api_executors(*, wait: bool = False, cancel_futures: bool = True) -> None:
    global _JOB_EXECUTOR, _STREAM_EXECUTOR
    with _EXECUTOR_LOCK:
        executors = [_JOB_EXECUTOR, _STREAM_EXECUTOR]
        _JOB_EXECUTOR = None
        _STREAM_EXECUTOR = None
    for executor in executors:
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=cancel_futures)


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(1, value)


__all__ = ["BoundedExecutor", "QueueFullError", "get_job_executor", "get_stream_executor", "shutdown_api_executors"]
