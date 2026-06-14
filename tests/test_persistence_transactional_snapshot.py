from __future__ import annotations

import json

import pytest

from cognitive_evolve_runtime.persistence.transactional_snapshot import NexusSnapshotTransaction, SnapshotWrite


def test_snapshot_transaction_publishes_manifest_and_files_atomically(tmp_path):
    result = NexusSnapshotTransaction(tmp_path).commit(
        [
            SnapshotWrite("population.json", "json", {"candidates": []}),
            SnapshotWrite("final-answer.md", "text", "answer\n", sort_keys=False),
        ]
    )

    assert (tmp_path / "population.json").read_text(encoding="utf-8")
    assert (tmp_path / "final-answer.md").read_text(encoding="utf-8") == "answer\n"
    manifest = json.loads((tmp_path / "snapshot-transaction.json").read_text(encoding="utf-8"))
    assert manifest["transaction_id"] == result.transaction_id
    assert set(manifest["files"]) == {"population.json", "final-answer.md"}
    assert manifest["files"]["population.json"]["sha256"]


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
