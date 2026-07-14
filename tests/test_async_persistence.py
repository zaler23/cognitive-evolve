from __future__ import annotations

import threading

from cognitive_evolve_runtime.durable.async_writer import WriteBehindObserver


class _Batch:
    def __init__(self, value: int) -> None:
        self.value = value
        self.serialized_bytes = value


class _BlockingStore:
    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Event()
        self.written: list[int] = []

    def freeze(self, update):
        return _Batch(update["value"])

    def write(self, batch):
        self.started.set()
        self.release.wait()
        self.written.append(batch.value)


def test_write_behind_freezes_before_return_and_preserves_fifo_without_loss() -> None:
    store = _BlockingStore()
    observer = WriteBehindObserver(store)

    observer({"phase": "post_mutation", "value": 1})
    assert store.started.wait(1)
    observer({"phase": "post_mutation", "value": 2})
    assert store.written == []
    store.release.set()
    observer.flush()

    assert store.written == [1, 2]
    assert observer.telemetry()["persistence_batches"] == 2
    observer.close()


def test_final_phase_flushes_and_joins_worker() -> None:
    store = _BlockingStore()
    store.release.set()
    observer = WriteBehindObserver(store)

    observer({"phase": "final_synthesis", "value": 3})

    assert store.written == [3]
