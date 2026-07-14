from __future__ import annotations

import threading

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.durable.async_writer import WriteBehindObserver
from cognitive_evolve_runtime.nexus.live_store import LiveNexusStore
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy


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


def test_live_store_freeze_detaches_nested_runtime_state(tmp_path) -> None:
    store = LiveNexusStore(tmp_path, mode="text", contract={}, world={}, max_rounds=1)
    runtime_options = {"nested": {"values": []}}
    search_kernel = {"nested": {"values": []}}
    batch = store.freeze(
        {
            "phase": "post_mutation",
            "round": 1,
            "population": CandidatePopulation([CandidateGenome(id="A")]),
            "archives": ArchiveManager(),
            "policy": EvolutionPolicy(),
            "runtime_options": runtime_options,
            "search_kernel": search_kernel,
        }
    )
    assert batch is not None

    runtime_options["nested"]["values"].append("later")
    search_kernel["nested"]["values"].append("later")

    assert batch.round_snapshot["runtime_options"]["nested"]["values"] == []
    assert batch.round_snapshot["search_kernel"]["nested"]["values"] == []
    checkpoint = next(item.payload for item in batch.snapshot_writes if item.relative_path == "checkpoint.json")
    assert checkpoint["runtime_options"]["nested"]["values"] == []
    assert checkpoint["search_kernel"]["nested"]["values"] == []
