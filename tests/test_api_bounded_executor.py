from __future__ import annotations

import threading
from pathlib import Path

import pytest

from cognitive_evolve_runtime.api.executor import BoundedExecutor, QueueFullError


def test_bounded_executor_rejects_when_worker_and_queue_are_full() -> None:
    executor = BoundedExecutor(max_workers=1, max_queue=0, thread_name_prefix="test-cogev")
    release = threading.Event()
    future = executor.submit(lambda: release.wait(timeout=2.0))
    try:
        with pytest.raises(QueueFullError):
            executor.submit(lambda: None)
    finally:
        release.set()
        assert future.result(timeout=3.0) is True
        executor.shutdown(wait=True)


def test_api_uses_bounded_executor_instead_of_raw_threads() -> None:
    root = Path(__file__).parents[1]
    for rel in ["cognitive_evolve_runtime/api/openai_compat.py", "cognitive_evolve_runtime/api/streaming.py"]:
        text = (root / rel).read_text(encoding="utf-8")
        assert "threading.Thread(" not in text

