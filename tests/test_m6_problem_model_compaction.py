from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from cognitive_evolve_runtime.outcomes import IntentHypothesis, LatentProblemState
from cognitive_evolve_runtime.outcomes.problem_model import (
    ProblemModelLedger,
    ProblemModelSnapshot,
    initial_problem_model_from_latent_state,
    materialize_problem_model_snapshot,
)
from cognitive_evolve_runtime.outcomes.problem_model_compaction import (
    ProblemModelCompactionError,
    ProblemModelCompactionRecord,
    ProblemModelCompactionStore,
    problem_model_snapshot_from_dict,
    sufficient_problem_model_hashes,
    verify_problem_model_compaction_chain,
)


def _state() -> LatentProblemState:
    return LatentProblemState(
        intents=(
            IntentHypothesis(id="clarity", statement="prefer clarity", posterior=0.5, uncertainty=0.5),
            IntentHypothesis(id="impact", statement="prefer impact", posterior=0.5, uncertainty=0.5),
        )
    )


def _snapshots() -> tuple[ProblemModelSnapshot, ProblemModelSnapshot]:
    base = initial_problem_model_from_latent_state(_state())
    child = replace(
        base,
        id="problem-model-with-maintainability",
        statement="prefer clarity, impact, and maintainability",
        parent_model_hashes=(base.model_hash(),),
        proposal_operator="birth",
        evidence_basis_hashes=(*base.evidence_basis_hashes, "raw:maintainability"),
        subproblems=(*base.subproblems, "maintainability"),
        provenance_ref="test:m6-compaction",
    )
    ledger = ProblemModelLedger()
    ledger.add_model(base, idempotency_key="base")
    first = materialize_problem_model_snapshot(ledger)
    ledger.add_model(child, idempotency_key="child")
    second = materialize_problem_model_snapshot(ledger)
    return first, second


def test_compaction_store_appends_content_addressed_records_and_roundtrips_latest_snapshot(tmp_path: Path) -> None:
    first, second = _snapshots()
    store = ProblemModelCompactionStore(tmp_path / "problem-model-compactions.jsonl")

    first_record = store.append_snapshot(first)
    second_record = store.append_snapshot(second)
    duplicate_second = store.append_snapshot(second, parent_record_hash=first_record.record_hash)
    loaded = store.load_latest_snapshot()
    records = store.load_records()

    assert first_record.record_hash.startswith("pmc:")
    assert second_record.parent_record_hash == first_record.record_hash
    assert duplicate_second.record_hash == second_record.record_hash
    assert len(records) == 2
    assert loaded is not None
    assert loaded.snapshot_hash() == second.snapshot_hash()
    assert problem_model_snapshot_from_dict(loaded.to_dict()).snapshot_hash() == second.snapshot_hash()


def test_parent_record_chain_and_model_parent_hashes_are_verified() -> None:
    first, second = _snapshots()
    first_record = ProblemModelCompactionRecord.from_snapshot(first)
    second_record = ProblemModelCompactionRecord.from_snapshot(second, parent_record_hash=first_record.record_hash)

    verification = verify_problem_model_compaction_chain((first_record, second_record), strict=True)

    assert verification.ok is True
    assert verification.record_count == 2
    assert verification.head_record_hash == second_record.record_hash
    assert set(second.active_model_hashes).issubset(set(verification.known_model_hashes))

    broken_parent = ProblemModelCompactionRecord.from_snapshot(second, parent_record_hash="pmc:missing")
    with pytest.raises(ProblemModelCompactionError, match="parent_record_hash_mismatch"):
        verify_problem_model_compaction_chain((first_record, broken_parent), strict=True)

    orphan_model = replace(second.active_models[-1], parent_model_hashes=("pm:missing-parent",))
    orphan_snapshot = ProblemModelSnapshot(active_models=(orphan_model,), ledger_cursor=1)
    orphan_record = ProblemModelCompactionRecord.from_snapshot(orphan_snapshot)
    with pytest.raises(ProblemModelCompactionError, match="missing_model_parent"):
        verify_problem_model_compaction_chain((orphan_record,), strict=True)


def test_tampered_compaction_record_fails_load(tmp_path: Path) -> None:
    first, _ = _snapshots()
    store = ProblemModelCompactionStore(tmp_path)
    store.append_snapshot(first)
    row = json.loads((tmp_path / "problem-model-compactions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    row["snapshot"]["active_models"][0]["statement"] = "tampered problem model"
    (tmp_path / "problem-model-compactions.jsonl").write_text(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ProblemModelCompactionError, match="record_hash_mismatch"):
        store.load_records()


def test_compaction_is_lossy_only_for_raw_history_not_sufficient_hashes() -> None:
    first, second = _snapshots()
    first_record = ProblemModelCompactionRecord.from_snapshot(first)
    second_record = ProblemModelCompactionRecord.from_snapshot(second, parent_record_hash=first_record.record_hash)
    payload = second_record.to_dict()
    retained = payload["retained_hashes"]

    assert payload["loss_policy"]["omitted"] == ["raw_problem_model_ledger_events"]
    assert payload["loss_policy"]["sufficient_hashes_retained"] is True
    assert "events" not in payload
    assert "ledger_events" not in payload
    assert retained == sufficient_problem_model_hashes(second)
    assert retained["snapshot_hash"] == second.snapshot_hash()
    assert retained["model_space_hash"] == second.model_space_hash()
    assert retained["active_model_hashes"] == list(second.active_model_hashes)
    assert retained["ledger_replay_hash"] == second.ledger_replay_hash
    assert retained["model_parent_hashes"][second.active_models[-1].model_hash()] == [second.active_models[0].model_hash()]
