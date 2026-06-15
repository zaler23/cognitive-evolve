from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.archives.quality_diversity import QualityDiversityArchive, candidate_final_quality, candidate_search_quality
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationOperator, MutationPlan
from cognitive_evolve_runtime.evaluators import (
    ArtifactPolicy,
    ChallengeMemory,
    EvaluatorSpec,
    EvidenceRecord,
    ExternalEvaluatorRunner,
    SearchPressure,
    apply_evidence_record,
    challenge_from_diagnostic,
    evidence_state,
    latest_evidence_record,
)
from cognitive_evolve_runtime.evaluators.artifact_normalizer import normalize_artifact
from cognitive_evolve_runtime.nexus.adaptive.config import AdaptiveConfig
from cognitive_evolve_runtime.nexus.adaptive.controller import AdaptiveRuntimeController
from cognitive_evolve_runtime.nexus.final_projection import build_final_projection
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, EvolutionRound
from cognitive_evolve_runtime.nexus.synthesis import SynthesizedResult


def test_public_evidence_api_excludes_removed_protocols() -> None:
    evaluators = importlib.import_module("cognitive_evolve_runtime.evaluators")

    for name in ("ArtifactPolicy", "EvidenceRecord", "SearchPressure", "ChallengeMemory"):
        assert hasattr(evaluators, name)
    for name in ("EvidenceResult", "ChallengeCase", "ChallengeBank"):
        assert not hasattr(evaluators, name)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("cognitive_evolve_runtime.evaluators.challenge_bank")


def test_artifact_normalizer_marks_refolded_artifact_probe_only() -> None:
    candidate = CandidateGenome(
        id="C1",
        artifact="Repair mutation: {'answer': 4, 'steps': ['compare', 'repair']}",
        artifact_type="machine",
    )

    view = normalize_artifact(candidate, artifact_type="machine", machine_artifact_required=True)

    assert view["status"] == "refolded"
    assert view["probe_eligible"] is True
    assert view["final_eligible"] is False
    assert isinstance(view["normalized_artifact"], dict)


def test_external_evaluator_writes_evidence_record_and_scores(tmp_path: Path) -> None:
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(
        "import json, sys\n"
        "data=json.load(open(sys.argv[1]))\n"
        "print(json.dumps({'passed': False, 'metrics': {'score': 0.42, 'challenge_pass_rate': 0.5}, 'diagnostics': ['case-23 failed']}))\n",
        encoding="utf-8",
    )
    candidate = CandidateGenome(id="C1", artifact={"answer": 1}, artifact_type="machine")
    spec = EvaluatorSpec.from_mapping(
        {
            "enabled": True,
            "command": f"{sys.executable} evaluator.py {{candidate_path}}",
            "cwd": str(tmp_path),
            "evidence": {"machine_artifact_required": True, "artifact_type": "machine"},
        }
    )

    results = ExternalEvaluatorRunner().evaluate_population_if_configured([candidate], spec=spec, round_index=3)

    assert len(results) == 1
    record = latest_evidence_record(candidate)
    state = evidence_state(candidate)
    assert record is not None
    assert record.stage == "probe"
    assert record.metadata["status"] == "challenge_failed"
    assert record.metadata["artifact_state"]["status"] == "clean"
    assert record.emitted_challenge_ids
    assert state["final_blocked"] is True
    assert candidate.multihead_scores["frontier_score"] > 0
    assert candidate.multihead_scores["challenge_pass_rate"] == 0.5
    assert candidate.metadata["repair_value"] > 0
    assert "evidence_records" in candidate.metadata
    assert "progressive_evidence" not in candidate.metadata
    assert "challenge_failures" not in candidate.metadata


def test_challenge_memory_dedupes_tracks_pressure_and_resolution() -> None:
    case = challenge_from_diagnostic(candidate_id="C1", source="combo", diagnostic=json.dumps({"mask": 23}), round_index=1)
    record = EvidenceRecord(
        candidate_id="C1",
        source="combo",
        stage="probe",
        score=0.8,
        final_blocked=True,
        repair_value=0.7,
        emitted_challenge_ids=[case["id"]],
        metadata={"challenge_items": [case]},
    )
    memory = ChallengeMemory()

    memory.ingest(record, round_index=1)
    memory.ingest(record, round_index=2)
    pressure = memory.compile_search_pressure(parent_id="C1", scope="candidate", artifact_requirements={"machine_readable_required": True})
    assert isinstance(pressure, SearchPressure)
    assert pressure.target_challenge_ids == [case["id"]]
    assert pressure.artifact_requirements["machine_readable_required"] is True

    memory.mark_targeted("C2", [case["id"]])
    memory.mark_resolved("C2", [case["id"]])

    assert len(memory.items) == 1
    stored = next(iter(memory.items.values()))
    assert stored["kill_count"] == 2
    assert stored["targeted_by_candidate_ids"] == ["C2"]
    assert stored["resolved_by_candidate_ids"] == ["C2"]
    assert memory.targeted_resolution_rate() == 1.0


def test_search_pressure_enters_mutation_plan_and_offspring_metadata() -> None:
    case = challenge_from_diagnostic(candidate_id="P1", source="unit", diagnostic="boundary case failed", round_index=1)
    record = EvidenceRecord(
        candidate_id="P1",
        source="unit",
        stage="probe",
        score=0.7,
        final_blocked=True,
        repair_value=0.7,
        emitted_challenge_ids=[case["id"]],
        metadata={"challenge_items": [case]},
    )
    controller = AdaptiveRuntimeController(config=AdaptiveConfig.from_sources(explicit={"enabled": True, "evidence": {"machine_readable_required": True}}))
    controller.challenge_memory.ingest(record, round_index=1)
    parent = CandidateGenome(id="P1", current_fate=CandidateFate.ACTIVE.value, artifact={"answer": 1})
    round_driver = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=2, branch_factor=1), adaptive=controller)
    plan = MutationPlan(operator=MutationOperator.REPAIR, parent_ids=["P1"], instruction="Repair the candidate.")

    pressured = round_driver._apply_search_pressure_to_plans([plan], parents=[parent])
    child = MutationEngine().mutate(parent, pressured[0])

    assert pressured[0].metadata["target_challenge_ids"] == [case["id"]]
    assert pressured[0].metadata["search_pressure_id"]
    assert "directly addresses these unresolved challenges" in pressured[0].instruction
    assert child.metadata["target_challenge_ids"] == [case["id"]]
    assert child.metadata["search_pressure_id"] == pressured[0].metadata["search_pressure_id"]


def test_archive_keeps_repairable_challenge_failure_out_of_terminal_cull() -> None:
    candidate = CandidateGenome(
        id="C1",
        current_fate=CandidateFate.ACTIVE.value,
        artifact={"x": 1},
        multihead_scores={"objective_alignment": 0.0, "frontier_score": 0.7},
    )
    apply_evidence_record(
        candidate,
        EvidenceRecord(
            candidate_id="C1",
            source="unit",
            stage="probe",
            score=0.7,
            final_blocked=True,
            parent_blocked=False,
            terminal_reject=False,
            repair_value=0.7,
            hints=["repair case"],
        ),
    )
    candidate.failure_lessons.append("probe failed")

    assignments = ArchiveManager().assign_by_policy([candidate], current_round=1, round_limit=5, branch_factor=2)

    assert assignments[0].fate in {CandidateFate.INCUBATING.value, CandidateFate.DORMANT.value, CandidateFate.ACTIVE.value}
    assert assignments[0].fate not in {CandidateFate.CULLED.value, CandidateFate.FAILED.value}


def test_quality_diversity_separates_search_and_final_quality() -> None:
    candidate = CandidateGenome(
        id="C1",
        niche_memberships=["niche"],
        multihead_scores={"frontier_score": 0.9, "repair_value": 0.8, "final_verification": 0.0},
    )
    archive = QualityDiversityArchive()

    archive.update(candidate)

    entry = archive.elites_by_niche["niche"]
    assert entry["search_quality"] > entry["final_quality"]
    assert candidate_search_quality(candidate) > candidate_final_quality(candidate)


def test_final_projection_returns_best_current_without_internal_directives() -> None:
    candidate = CandidateGenome(
        id="C1",
        artifact={"answer": 1},
        artifact_type="machine",
        multihead_scores={"frontier_score": 0.8, "challenge_pass_rate": 0.5, "schema_cleanliness": 1.0},
    )
    case = challenge_from_diagnostic(candidate_id="C1", source="machine", diagnostic="boundary case failed", round_index=1)
    apply_evidence_record(
        candidate,
        EvidenceRecord(
            candidate_id="C1",
            source="machine",
            stage="probe",
            score=0.8,
            final_blocked=True,
            repair_value=0.7,
            emitted_challenge_ids=[case["id"]],
            hints=["fix challenge"],
            metadata={"challenge_items": [case], "artifact_state": {"normalized_artifact": {"answer": 1}, "status": "clean"}},
        ),
    )
    synthesis = SynthesizedResult(status="best_current_route", final_answer="internal repair directive should not be reused", best_candidate_id="C1")

    projection = build_final_projection(population=CandidatePopulation([candidate]), synthesis=synthesis, final_certificate={"blocking_reasons": ["external_evaluator_not_passed"]})
    markdown = projection.to_markdown()

    assert projection.status == "best_current"
    assert projection.objective_solved is False
    assert "internal repair directive should not be reused" not in markdown
    assert "Best current artifact" in markdown
    assert "boundary case failed" in markdown


def test_old_snapshot_evidence_is_migrated_once_and_not_rewritten() -> None:
    candidate = CandidateGenome(id="C1", artifact={"answer": 1}, artifact_type="machine")
    candidate.metadata["progressive_evidence"] = {
        "candidate_id": "C1",
        "status": "challenge_failed",
        "score": 0.6,
        "final_eligible": False,
        "repair_hints": ["old hint"],
    }
    candidate.metadata["repair_value"] = 0.6

    apply_evidence_record(candidate, EvidenceRecord(candidate_id="C1", source="unit", stage="probe", score=0.7, final_blocked=True, repair_value=0.7))

    assert "progressive_evidence" not in candidate.metadata
    assert len(candidate.metadata["evidence_records"]) == 2
    assert candidate.metadata["evidence_records"][0]["source"] == "legacy_progressive_evidence_migration"
    assert candidate.metadata["evidence_state"]["search_score"] == 0.7


def test_adaptive_config_exposes_evidence_control_plane_feature() -> None:
    config = AdaptiveConfig.from_sources(explicit={"enabled": True, "evidence": {"machine_artifact_required": True}})

    assert config.evidence["machine_artifact_required"] is True
    assert config.enabled_features["evidence_control_plane"] is True
    assert "progressive_evidence" not in config.enabled_features


def test_artifact_policy_from_mapping_accepts_machine_artifact_alias() -> None:
    policy = ArtifactPolicy.from_mapping({"machine_artifact_required": True, "allow_refold_for_final": False})

    assert policy.machine_readable_required is True
    assert policy.allow_refold_for_final is False
