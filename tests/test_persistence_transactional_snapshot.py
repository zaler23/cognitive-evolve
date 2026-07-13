from __future__ import annotations

import json
import threading
from contextlib import contextmanager

import pytest

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.live_store import LiveNexusStore
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.persistence import transactional_snapshot as snapshot_module
from cognitive_evolve_runtime.persistence.checkpoint import CheckpointStore
from cognitive_evolve_runtime.persistence.transactional_snapshot import (
    NexusSnapshotTransaction,
    SnapshotWrite,
    resolve_snapshot_path,
    resolve_snapshot_root,
    snapshot_reader,
)


def _commit(root, generation: str):
    return NexusSnapshotTransaction(root).commit(
        [
            SnapshotWrite("population.json", "json", {"generation": generation}),
            SnapshotWrite("final-answer.md", "text", generation + "\n", sort_keys=False),
        ]
    )


def _read_generation(root) -> tuple[str, str]:
    snapshot_root = resolve_snapshot_root(root)
    population = json.loads((snapshot_root / "population.json").read_text(encoding="utf-8"))
    answer = (snapshot_root / "final-answer.md").read_text(encoding="utf-8").strip()
    return str(population["generation"]), answer


def _write_live_checkpoint(root, round_index: int) -> None:
    store = LiveNexusStore(
        root,
        mode="text",
        contract=NexusObjectiveContract(original_user_goal="goal", normalized_goal="goal"),
        world={"kind": "text"},
        max_rounds=3,
    )
    store(
        {
            "population": CandidatePopulation([CandidateGenome(id=f"C{round_index}", concise_claim="candidate")]),
            "archives": ArchiveManager(),
            "policy": EvolutionPolicy(),
            "phase": "round_end",
            "round": round_index,
            "progress_event": {"type": "evolution_progress", "round": round_index},
        }
    )


def test_snapshot_transaction_publishes_manifest_and_files_atomically(tmp_path):
    result = NexusSnapshotTransaction(tmp_path).commit(
        [
            SnapshotWrite("population.json", "json", {"candidates": []}),
            SnapshotWrite("final-answer.md", "text", "answer\n", sort_keys=False),
        ]
    )

    assert (tmp_path / "population.json").read_text(encoding="utf-8")
    assert (tmp_path / "final-answer.md").read_text(encoding="utf-8") == "answer\n"
    snapshot_root = resolve_snapshot_root(tmp_path)
    manifest = json.loads((snapshot_root / "snapshot-transaction.json").read_text(encoding="utf-8"))
    assert manifest["transaction_id"] == result.transaction_id
    assert set(manifest["files"]) == {"population.json", "final-answer.md"}
    assert manifest["files"]["population.json"]["sha256"]
    assert snapshot_root == tmp_path / "generations" / result.transaction_id
    assert result.manifest_path == str(snapshot_root / "snapshot-transaction.json")
    assert not list(snapshot_root.rglob("*.lock"))


def test_snapshot_transaction_does_not_overwrite_existing_files_on_staging_failure(tmp_path):
    (tmp_path / "population.json").write_text('{"old": true}', encoding="utf-8")
    circular: dict[str, object] = {}
    circular["self"] = circular

    with pytest.raises(ValueError):
        NexusSnapshotTransaction(tmp_path).commit([SnapshotWrite("population.json", "json", circular)])

    assert json.loads((tmp_path / "population.json").read_text(encoding="utf-8")) == {"old": True}


def test_snapshot_transaction_rejects_unsafe_paths(tmp_path):
    with pytest.raises(ValueError):
        NexusSnapshotTransaction(tmp_path).commit([SnapshotWrite("../bad.json", "json", {})])


def test_snapshot_reader_falls_back_to_legacy_flat_layout(tmp_path):
    legacy = tmp_path / "checkpoint.json"
    legacy.write_text('{"round": 4}\n', encoding="utf-8")

    assert resolve_snapshot_root(tmp_path) == tmp_path
    assert resolve_snapshot_path(tmp_path, "checkpoint.json") == legacy


def test_snapshot_reader_acquires_lease_before_first_generation(tmp_path, monkeypatch):
    root = tmp_path / "not-created-yet"
    locks = []

    @contextmanager
    def observed_lock(path):
        locks.append(path)
        yield

    monkeypatch.setattr(snapshot_module, "file_lock", observed_lock)
    with snapshot_reader(root) as resolved:
        assert resolved == root

    assert locks == [root / ".snapshot-transaction.lock"]


def test_snapshot_text_pointer_keeps_flat_compatibility_files(tmp_path, monkeypatch):
    def no_symlink(*args, **kwargs):
        raise OSError("no symlink")

    monkeypatch.setattr(snapshot_module.os, "symlink", no_symlink)

    result = _commit(tmp_path, "portable")

    assert (tmp_path / "CURRENT").is_file()
    assert (tmp_path / "CURRENT").read_text(encoding="utf-8").strip() == result.transaction_id
    assert json.loads((tmp_path / "population.json").read_text(encoding="utf-8"))["generation"] == "portable"
    assert (tmp_path / "final-answer.md").read_text(encoding="utf-8").strip() == "portable"


def test_snapshot_transaction_keeps_old_generation_when_generation_publish_fails(tmp_path, monkeypatch):
    _commit(tmp_path, "old")
    real_replace = snapshot_module.os.replace

    def fail_generation_publish(src, dst):
        source = snapshot_module.Path(src)
        target = snapshot_module.Path(dst)
        if source.parent == tmp_path / "generations" and source.name.endswith(".staging") and target.parent == tmp_path / "generations":
            raise OSError("generation publish fault")
        return real_replace(src, dst)

    monkeypatch.setattr(snapshot_module.os, "replace", fail_generation_publish)
    with pytest.raises(OSError, match="generation publish fault"):
        _commit(tmp_path, "new")

    assert _read_generation(tmp_path) == ("old", "old")


def test_snapshot_transaction_keeps_old_generation_when_current_publish_fails(tmp_path, monkeypatch):
    _commit(tmp_path, "old")
    real_replace = snapshot_module.os.replace

    def fail_current_publish(src, dst):
        if snapshot_module.Path(dst) == tmp_path / "CURRENT":
            raise OSError("CURRENT publish fault")
        return real_replace(src, dst)

    monkeypatch.setattr(snapshot_module.os, "replace", fail_current_publish)
    with pytest.raises(OSError, match="CURRENT publish fault"):
        _commit(tmp_path, "new")

    assert _read_generation(tmp_path) == ("old", "old")


def test_snapshot_transaction_exposes_complete_new_generation_after_current_replace(tmp_path, monkeypatch):
    _commit(tmp_path, "old")
    real_fsync_dir = snapshot_module._fsync_dir

    def fail_after_current_replace(path):
        if snapshot_module.Path(path) == tmp_path:
            raise OSError("post-CURRENT fsync fault")
        return real_fsync_dir(path)

    monkeypatch.setattr(snapshot_module, "_fsync_dir", fail_after_current_replace)
    with pytest.raises(OSError, match="post-CURRENT fsync fault"):
        _commit(tmp_path, "new")

    assert _read_generation(tmp_path) == ("new", "new")


def test_snapshot_reader_pins_one_complete_generation(tmp_path):
    _commit(tmp_path, "old")
    pinned_old_root = resolve_snapshot_root(tmp_path)

    _commit(tmp_path, "new")

    assert json.loads((pinned_old_root / "population.json").read_text(encoding="utf-8"))["generation"] == "old"
    assert (pinned_old_root / "final-answer.md").read_text(encoding="utf-8").strip() == "old"
    assert _read_generation(tmp_path) == ("new", "new")


def test_snapshot_retains_only_current_and_previous_complete_generation(tmp_path):
    first = _commit(tmp_path, "first")
    second = _commit(tmp_path, "second")
    third = _commit(tmp_path, "third")

    retained = {path.name for path in (tmp_path / "generations").iterdir() if path.is_dir()}
    assert retained == {second.transaction_id, third.transaction_id}
    assert not (tmp_path / "generations" / first.transaction_id).exists()
    assert _read_generation(tmp_path) == ("third", "third")


def test_snapshot_reader_lease_prevents_pruning_during_read(tmp_path):
    _commit(tmp_path, "first")
    writer_done = threading.Event()

    def publish_twice() -> None:
        _commit(tmp_path, "second")
        _commit(tmp_path, "third")
        writer_done.set()

    with snapshot_reader(tmp_path) as pinned:
        writer = threading.Thread(target=publish_twice)
        writer.start()
        assert not writer_done.wait(0.05)
        assert (pinned / "population.json").exists()
        assert json.loads((pinned / "population.json").read_text(encoding="utf-8"))["generation"] == "first"

    writer.join(timeout=2)
    assert writer_done.is_set()
    assert _read_generation(tmp_path) == ("third", "third")


def test_live_checkpoint_advances_current_and_checkpoint_restore_reads_latest(tmp_path):
    _write_live_checkpoint(tmp_path, 0)
    previous_root = resolve_snapshot_root(tmp_path)

    _write_live_checkpoint(tmp_path, 1)

    current_root = resolve_snapshot_root(tmp_path)
    restored = CheckpointStore(resolve_snapshot_path(tmp_path, "checkpoint.json")).restore_state()
    assert current_root != previous_root
    assert restored is not None
    assert restored["checkpoint"].round == 1
    assert json.loads((previous_root / "checkpoint.json").read_text(encoding="utf-8"))["round"] == 0


def test_failed_live_checkpoint_keeps_previous_generation_readable(tmp_path, monkeypatch):
    _write_live_checkpoint(tmp_path, 0)
    previous_root = resolve_snapshot_root(tmp_path)
    real_replace = snapshot_module.os.replace

    def fail_generation_publish(src, dst):
        source = snapshot_module.Path(src)
        target = snapshot_module.Path(dst)
        if source.parent == tmp_path / "generations" and source.name.endswith(".staging") and target.parent == tmp_path / "generations":
            raise OSError("live generation publish fault")
        return real_replace(src, dst)

    monkeypatch.setattr(snapshot_module.os, "replace", fail_generation_publish)
    with pytest.raises(OSError, match="live generation publish fault"):
        _write_live_checkpoint(tmp_path, 1)

    assert resolve_snapshot_root(tmp_path) == previous_root
    assert json.loads((previous_root / "checkpoint.json").read_text(encoding="utf-8"))["round"] == 0
