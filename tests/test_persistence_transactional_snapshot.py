from __future__ import annotations

import hashlib
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
from cognitive_evolve_runtime.persistence.event_store import EventStore
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


def test_event_store_append_isolates_unterminated_tail(tmp_path):
    path = tmp_path / "events.jsonl"
    damaged = b'{"type": "committed"}\n{"type": "broken"'
    path.write_bytes(damaged)
    store = EventStore(path)

    assert store.watermark() == {"path": "events.jsonl", "offset": len(b'{"type": "committed"}\n')}
    store.append({"type": "after-damage"})

    raw = path.read_bytes()
    lines = raw.splitlines()
    assert raw.startswith(damaged)
    assert lines[1] == b'{"type": "broken"'
    assert json.loads(lines[2])["type"] == "event_store_tail_quarantined"
    assert json.loads(lines[2])["tail_end_offset"] == len(damaged)
    assert json.loads(lines[3])["type"] == "after-damage"


def test_snapshot_event_watermark_reconciles_committed_post_snapshot_and_truncated(tmp_path):
    store = EventStore(tmp_path / "events.jsonl")
    store.append({"type": "committed-1"})
    store.append({"type": "committed-2"})
    NexusSnapshotTransaction(tmp_path).commit([SnapshotWrite("checkpoint.json", "json", {"round": 2})])
    snapshot_root = resolve_snapshot_root(tmp_path)
    manifest = json.loads((snapshot_root / "snapshot-transaction.json").read_text(encoding="utf-8"))

    assert manifest["event_watermark"] == {"path": "events.jsonl", "offset": (tmp_path / "events.jsonl").stat().st_size}

    store.append({"type": "post-snapshot-1"})
    store.append({"type": "post-snapshot-2"})
    path = tmp_path / "events.jsonl"
    path.write_bytes(path.read_bytes()[:-5])
    truncated_bytes = path.read_bytes()

    report = store.reconcile(manifest["event_watermark"])

    assert path.read_bytes() == truncated_bytes
    assert [event["type"] for event in report["committed"]] == ["committed-1", "committed-2"]
    assert [event["type"] for event in report["post-snapshot"]] == ["post-snapshot-1"]
    assert len(report["truncated"]) == 1
    assert report["truncated"][0]["reason"] == "incomplete_line"


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


def test_checkpoint_restore_rejects_tampered_snapshot_file(tmp_path):
    _write_live_checkpoint(tmp_path, 1)
    snapshot_root = resolve_snapshot_root(tmp_path)
    checkpoint_path = snapshot_root / "checkpoint.json"
    manifest = json.loads((snapshot_root / "snapshot-transaction.json").read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["created_at"] = "tampered"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    expected_hash = manifest["files"]["checkpoint.json"]["sha256"]
    actual_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError) as exc_info:
        CheckpointStore(checkpoint_path).restore_state()

    message = str(exc_info.value)
    assert "checkpoint.json" in message
    assert expected_hash in message
    assert actual_hash in message


def test_checkpoint_restore_accepts_matching_snapshot_manifest(tmp_path):
    _write_live_checkpoint(tmp_path, 1)

    with snapshot_reader(tmp_path) as snapshot_root:
        restored = CheckpointStore(snapshot_root / "checkpoint.json").restore_state()

    assert restored is not None
    assert restored["checkpoint"].round == 1


def test_checkpoint_restore_legacy_flat_layout_logs_and_skips_manifest_verification(tmp_path, caplog):
    source = tmp_path / "source"
    _write_live_checkpoint(source, 1)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "checkpoint.json").write_bytes((resolve_snapshot_root(source) / "checkpoint.json").read_bytes())

    with caplog.at_level("INFO", logger="cognitive_evolve_runtime.persistence.transactional_snapshot"):
        restored = CheckpointStore(legacy / "checkpoint.json").restore_state()

    assert restored is not None
    assert restored["checkpoint"].round == 1
    assert "snapshot manifest missing; skipping hash verification for legacy snapshot file" in caplog.text


def test_checkpoint_restore_rejects_manifest_without_target_hash(tmp_path):
    _write_live_checkpoint(tmp_path, 1)
    snapshot_root = resolve_snapshot_root(tmp_path)
    manifest_path = snapshot_root / "snapshot-transaction.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].pop("checkpoint.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=r"checkpoint\.json.*expected sha256=<missing>.*actual sha256="):
        CheckpointStore(snapshot_root / "checkpoint.json").restore_state()


def test_checkpoint_restore_rejects_missing_manifest_in_published_generation(tmp_path):
    _write_live_checkpoint(tmp_path, 1)
    snapshot_root = resolve_snapshot_root(tmp_path)
    (snapshot_root / "snapshot-transaction.json").unlink()

    with pytest.raises(ValueError, match="snapshot manifest missing for published generation"):
        CheckpointStore(snapshot_root / "checkpoint.json").restore_state()


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
