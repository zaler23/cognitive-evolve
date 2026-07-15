from __future__ import annotations

import copy

import pytest

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.candidates.mutation import MutationOperator
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.evaluators.dossier_grounding import frozen_dossier_hash, ground_frozen_dossier_claims
from cognitive_evolve_runtime.evaluators.evidence import evidence_records
from cognitive_evolve_runtime.evaluators.evidence_authority import stable_artifact_hash
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, EvolutionRound
from cognitive_evolve_runtime.nexus.loop.controller import EvolutionLoopController
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.receipts import (
    ReceiptValidationError,
    TransferReceipt,
    append_intervention_receipt,
    append_transfer_receipt,
    intervention_receipts,
    record_transfer_receipts,
    transfer_credit_decision,
    transfer_receipts,
)
from cognitive_evolve_runtime.nexus.search_kernel.branch_allocator import allocate_productive_branches, productive_outcomes


def _transfer_receipt(artifact: object) -> dict[str, object]:
    return {
        "source_relations": ["queue precedes service"],
        "target_relations": ["claim precedes verification"],
        "mapping": [
            {"source": "queue", "target": "claim"},
            {"source": "service", "target": "verification"},
        ],
        "preserved_invariant": "ordering is preserved",
        "predicted_break_condition": "verification can run before claim creation",
        "probe_ref": "probe-ordering",
        "artifact_hash": stable_artifact_hash(artifact),
    }


def test_receipt_schema_rejections_are_audited() -> None:
    plan: dict[str, object] = {
        "productive_branch_allocation": {"slots": [{"slot_id": "slot-1"}]},
    }
    artifact = {"answer": "mapped"}

    missing_mapping = _transfer_receipt(artifact)
    missing_mapping.pop("mapping")
    with pytest.raises(ReceiptValidationError, match="mapping"):
        append_transfer_receipt(plan, missing_mapping)

    missing_hash = _transfer_receipt(artifact)
    missing_hash.pop("artifact_hash")
    with pytest.raises(ReceiptValidationError, match="artifact_hash"):
        append_transfer_receipt(plan, missing_hash)

    with pytest.raises(ReceiptValidationError, match="unknown branch slot"):
        append_intervention_receipt(
            plan,
            {
                "receipt_id": "receipt-opaque",
                "diagnosed_pressure": {
                    "stagnation_type": "DiversityCollapse",
                    "diagnosis_ref": "diagnosis-1",
                },
                "intervention_type": "diagnosis_guided_reproduction",
                "target": {"axis": "", "family": "", "action": "Repair", "slot": "slot-missing"},
                "recipient_branch_slot_ids": ["slot-missing"],
                "produced_candidate_ids": ["candidate-1"],
                "outcome_refs": ["probe-1"],
            },
        )

    audit = plan["receipt_audit"]
    assert isinstance(audit, list)
    assert [item["status"] for item in audit] == ["rejected", "rejected", "rejected"]
    assert [item["receipt_type"] for item in audit] == ["transfer", "transfer", "intervention"]


def test_transfer_credit_requires_probe_survival_and_dedupes_artifact_hash() -> None:
    artifact = {"answer": "mapped"}
    receipt = TransferReceipt.from_dict(_transfer_receipt(artifact))
    credited_hashes: set[str] = set()

    before_probe = transfer_credit_decision(
        receipt,
        probe_survived=False,
        credited_artifact_hashes=credited_hashes,
    )
    after_probe = transfer_credit_decision(
        receipt,
        probe_survived=True,
        credited_artifact_hashes=credited_hashes,
    )
    repeated = transfer_credit_decision(
        receipt,
        probe_survived=True,
        credited_artifact_hashes=credited_hashes,
    )

    assert before_probe.eligible is True
    assert before_probe.productive_credit is False
    assert after_probe.productive_credit is True
    assert repeated.productive_credit is False
    assert repeated.reason == "duplicate_artifact_hash"

    plan: dict[str, object] = {}
    append_transfer_receipt(plan, receipt)
    assert transfer_receipts(plan, artifact_hash=receipt.artifact_hash, probe_ref=receipt.probe_ref) == [receipt]

    seed = CandidateGenome(
        id="transfer-seed",
        artifact=artifact,
        metadata={
            "search_space": {"seed_axis": "cross_domain_transfer"},
            "transfer_receipt": receipt.to_dict(),
        },
    )
    seed_plan: dict[str, object] = {}
    assert record_transfer_receipts(seed_plan, [seed]) == [receipt]
    assert transfer_receipts(seed_plan, artifact_hash=receipt.artifact_hash) == [receipt]

    candidate = CandidateGenome(
        id="transfer-child",
        parent_ids=["parent"],
        lineage=["parent"],
        artifact=artifact,
        mutation_history=[MutationOperator.TRANSFER],
        metadata={"transfer_receipt": receipt.to_dict()},
    )
    [unprobed] = productive_outcomes([candidate])
    assert unprobed.reward == 0.0
    assert unprobed.reason_codes == ("transfer_probe_not_survived",)

    candidate.verification_trace = [{"probe_ref": "probe-ordering", "passed": True}]
    [credited] = productive_outcomes([candidate])
    assert credited.reward > 0.0
    assert "transfer_invariant_probe_survived" in credited.reason_codes

    parent = CandidateGenome(id="parent", artifact={"answer": "baseline"})
    first_allocation = allocate_productive_branches(
        parents=[parent],
        candidates=[parent, candidate],
        total_slots=1,
    )
    assert first_allocation.credited_transfer_artifact_hashes == (receipt.artifact_hash,)

    repeated_candidate = CandidateGenome(
        id="transfer-child-repeated",
        parent_ids=["parent"],
        lineage=["parent"],
        artifact=artifact,
        mutation_history=[MutationOperator.TRANSFER],
        verification_trace=[{"probe_ref": "probe-ordering", "passed": True}],
        metadata={"transfer_receipt": receipt.to_dict()},
    )
    repeated_allocation = allocate_productive_branches(
        parents=[parent],
        candidates=[parent, repeated_candidate],
        budget_history=[{"generation_plan": {"productive_branch_allocation": first_allocation.to_dict()}}],
        total_slots=1,
    )
    assert repeated_allocation.credited_transfer_artifact_hashes == ()
    assert repeated_allocation.credit_summary["transfer_duplicate_artifact_hash"] == 1


def test_frozen_dossier_grounding_is_hash_locked_span_grounded_and_evidence_backed() -> None:
    dossier = "Alpha mode is enabled. Beta mode is disabled."
    dossier_hash = frozen_dossier_hash(dossier)
    alpha_start = dossier.index("Alpha mode is enabled")
    beta_start = dossier.index("Beta mode is disabled")
    candidate = CandidateGenome(id="candidate-dossier", artifact="answer")

    record = ground_frozen_dossier_claims(
        candidate,
        dossier_text=dossier,
        dossier_hash=dossier_hash,
        claims=[
            {
                "claim_id": "supported",
                "claim": "Alpha mode is enabled",
                "status": "supported",
                "evidence_spans": [{"start": alpha_start, "end": alpha_start + len("Alpha mode is enabled")}],
            },
            {
                "claim_id": "contradicted",
                "claim": "Beta mode is enabled",
                "status": "contradicted",
                "evidence_spans": [{"start": beta_start, "end": beta_start + len("Beta mode is disabled")}],
            },
            {
                "claim_id": "unknown",
                "claim": "Gamma mode is enabled",
                "status": "unknown",
                "explanation": "The model nevertheless says this is supported.",
                "explanation_status": "supported",
                "evidence_spans": [],
            },
            {
                "claim_id": "external",
                "claim": "Delta mode is enabled",
                "status": "supported",
                "source": "https://example.invalid/delta",
                "evidence_spans": [{"start": alpha_start, "end": alpha_start + len("Alpha mode is enabled")}],
            },
        ],
    )

    results = {item["claim_id"]: item for item in record.metadata["claim_grounding"]}
    assert {key: value["status"] for key, value in results.items()} == {
        "supported": "supported",
        "contradicted": "contradicted",
        "unknown": "unknown",
        "external": "unknown",
    }
    assert results["supported"]["evidence_spans"] == [
        {
            "start": alpha_start,
            "end": alpha_start + len("Alpha mode is enabled"),
            "text": "Alpha mode is enabled",
        }
    ]
    assert results["contradicted"]["evidence_spans"][0]["text"] == "Beta mode is disabled"
    assert results["unknown"]["evidence_spans"] == []
    assert results["external"]["reason"] == "external_reference"
    assert all(item["dossier_hash"] == dossier_hash for item in results.values())
    assert evidence_records(candidate)[-1] == record

    with pytest.raises(ValueError, match="dossier hash mismatch"):
        ground_frozen_dossier_claims(
            CandidateGenome(id="hash-mismatch"),
            dossier_text=dossier,
            dossier_hash="sha256:wrong",
            claims=[],
        )


def test_intervention_receipt_chain_is_written_and_queryable_from_generation_history() -> None:
    population = CandidatePopulation(
        [
            CandidateGenome(
                id="parent-a",
                artifact="answer-a",
                core_mechanism="mechanism-a",
                multihead_scores={"answer_likelihood": 0.9, "objective_alignment": 0.8},
            ),
            CandidateGenome(
                id="parent-b",
                artifact="answer-b",
                core_mechanism="mechanism-b",
                multihead_scores={"answer_likelihood": 0.8, "objective_alignment": 0.7},
            ),
        ]
    )
    archives = ArchiveManager()
    budget = EvolutionBudget(max_rounds=2, branch_factor=2)
    round_pipeline = EvolutionRound(model=None, budget=budget)
    policy = EvolutionPolicy()
    contract = NexusObjectiveContract(original_user_goal="answer", normalized_goal="answer")
    evaluation = round_pipeline.evaluate(
        current_round=1,
        population=population,
        archives=archives,
        policy=policy,
        contract=contract,
    )
    diagnosis = SearchDiagnosis(
        stagnation_detected=True,
        stagnation_type="DiversityCollapse",
        under_explored_families=["mechanism-b"],
        recommended_actions=["repair"],
        created_at="2026-07-16T00:00:00+00:00",
    )

    def verify(offspring: list[CandidateGenome]) -> list[dict[str, object]]:
        return [
            {
                "candidate_id": candidate.id,
                "passed": True,
                "probe_ref": f"probe-{candidate.id}",
            }
            for candidate in offspring
        ]

    stop_reason, outcomes, _ = round_pipeline.reproduce(
        current_round=1,
        population=population,
        archives=archives,
        policy=policy,
        contract=contract,
        world=object(),
        rankings=evaluation.rankings,
        diagnosis=diagnosis,
        critiques=evaluation.critiques,
        offspring_verifier=verify,
        repair_parent_candidates=evaluation.repair_parent_candidates,
    )
    plan = copy.deepcopy(round_pipeline.last_generation_plan)
    receipts = intervention_receipts([{"generation_plan": plan}])

    assert stop_reason == ""
    assert outcomes
    assert receipts
    for receipt in receipts:
        assert receipt.diagnosed_pressure["stagnation_type"] == "DiversityCollapse"
        assert receipt.recipient_branch_slot_ids
        assert receipt.produced_candidate_ids
        assert receipt.outcome_refs
        slot_id = receipt.recipient_branch_slot_ids[0]
        candidate_id = receipt.produced_candidate_ids[0]
        assert receipt.target["slot"] == slot_id
        assert intervention_receipts(plan, receipt_id=receipt.receipt_id) == [receipt]
        assert intervention_receipts(plan, slot_id=slot_id, candidate_id=candidate_id) == [receipt]

    controller = EvolutionLoopController(
        population=population,
        archives=archives,
        policy=policy,
        contract=contract,
        world=object(),
        budget=budget,
    )
    controller.round_pipeline = round_pipeline
    budget.history.append({"round": 1})
    controller._record_reproduction_result(1, evaluation, stop_reason, outcomes, {}, None)
    assert evaluation.progress_event["metadata"]["generation_plan_id"] == plan["plan_id"]
    assert evaluation.progress_event["metadata"]["intervention_receipt_ids"] == [
        receipt.receipt_id for receipt in receipts
    ]
