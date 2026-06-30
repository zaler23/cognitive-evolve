from __future__ import annotations

import json
from pathlib import Path

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.live_store import LiveNexusStore
from cognitive_evolve_runtime.persistence.checkpoint import CheckpointStore, build_checkpoint_state, contract_payload_for_persistence
from cognitive_evolve_runtime.outcomes import (
    IntentHypothesis,
    LatentLedger,
    LatentProblemState,
    PreferenceEvidence,
    materialize_posterior_snapshot,
)


def test_live_store_persists_latent_ledger_once_and_keeps_checkpoint_metadata(tmp_path: Path) -> None:
    state = LatentProblemState(
        intents=(
            IntentHypothesis(id="clarity", statement="prefer clarity", posterior=0.5),
            IntentHypothesis(id="impact", statement="prefer impact", posterior=0.5),
        )
    )
    ledger = LatentLedger()
    event = ledger.add_evidence(
        PreferenceEvidence(
            intent_id="clarity",
            support=0.8,
            contradiction=0.0,
            evidence_ref="verifier:clarity",
            source_type="verifier",
            provenance_ref="verifier:clarity",
        ),
        idempotency_key="verifier:clarity",
    )
    snapshot = materialize_posterior_snapshot(state, ledger)
    contract = NexusObjectiveContract(
        original_user_goal="find the best tradeoff",
        normalized_goal="find the best tradeoff",
        metadata={
            "latent_problem_state": state.to_dict(),
            "latent_ledger": ledger.to_dict(),
            "latent_posterior_snapshot": snapshot.to_dict(),
        },
    )
    store = LiveNexusStore(tmp_path, mode="test", contract=contract, world={"kind": "test"}, max_rounds=2)
    update = {
        "population": CandidatePopulation([CandidateGenome(id="C1", concise_claim="candidate")]),
        "archives": ArchiveManager(),
        "phase": "checkpoint",
        "round": 1,
    }

    store(update)
    latent_events_path = tmp_path / "latent-events.jsonl"
    assert latent_events_path.exists()
    first_lines = latent_events_path.read_text(encoding="utf-8").splitlines()

    store(update)
    second_lines = latent_events_path.read_text(encoding="utf-8").splitlines()

    assert second_lines == first_lines
    rows = [json.loads(line) for line in second_lines]
    assert [row["type"] for row in rows].count("latent_ledger_event") == 1
    assert [row["type"] for row in rows].count("latent_posterior_snapshot") == 1
    assert rows[0]["event_id"] == event.event_id
    ledger_cache = json.loads((tmp_path / "latent-ledger.json").read_text(encoding="utf-8"))
    snapshot_cache = json.loads((tmp_path / "latent-posterior-snapshot.json").read_text(encoding="utf-8"))
    assert ledger_cache["events"][0]["event_id"] == event.event_id
    assert snapshot_cache["ledger_cursor"] == 1

    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    metadata = checkpoint["contract"]["metadata"]
    assert "latent_ledger" not in metadata
    assert metadata["latent_ledger_ref"]["path"] == str(tmp_path / "latent-ledger.json")
    assert metadata["latent_ledger_ref"]["sha256"]
    assert metadata["latent_ledger_ref"]["cursor"] == 1
    assert metadata["latent_posterior_snapshot"]["ledger_cursor"] == 1

    restored = CheckpointStore(tmp_path / "checkpoint.json").restore_state()
    restored_metadata = restored["contract"]["metadata"]
    assert restored_metadata["latent_ledger"]["events"][0]["event_id"] == event.event_id
    assert restored_metadata["latent_ledger_ref"]["cursor"] == 1


def test_checkpoint_and_run_contract_payload_strip_hydrated_latent_ledger() -> None:
    ledger = LatentLedger()
    ledger.add_evidence(
        PreferenceEvidence(
            intent_id="clarity",
            support=0.8,
            contradiction=0.0,
            evidence_ref="verifier:clarity",
            source_type="verifier",
            provenance_ref="verifier:clarity",
        )
    )
    contract = NexusObjectiveContract(
        original_user_goal="find the best tradeoff",
        normalized_goal="find the best tradeoff",
        metadata={
            "latent_ledger": ledger.to_dict(),
            "latent_ledger_ref": {"path": "latent-ledger.json", "sha256": "abc", "cursor": 1},
        },
    )

    checkpoint = build_checkpoint_state(
        round=1,
        max_rounds=2,
        population=CandidatePopulation([CandidateGenome(id="C1", concise_claim="candidate")]),
        archives=ArchiveManager(),
        contract=contract,
    )

    checkpoint_metadata = checkpoint.to_dict()["contract"]["metadata"]
    run_metadata = contract_payload_for_persistence(contract)["metadata"]
    assert "latent_ledger" not in checkpoint_metadata
    assert checkpoint_metadata["latent_ledger_ref"]["cursor"] == 1
    assert "latent_ledger" not in run_metadata
    assert run_metadata["latent_ledger_ref"]["cursor"] == 1
