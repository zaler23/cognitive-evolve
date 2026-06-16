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
    classify_diagnostic,
    evidence_advisory_features,
    evidence_state,
    latest_evidence_record,
    stable_artifact_identity_hash,
)
from cognitive_evolve_runtime.evaluators.artifact_normalizer import normalize_artifact
from cognitive_evolve_runtime.evaluators.progressive import ProgressiveEvaluator
from cognitive_evolve_runtime.evaluators.result import EvaluatorResult
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


def test_artifact_normalizer_aliases_are_probe_only_not_final() -> None:
    candidate = CandidateGenome(
        id="C1",
        artifact={
            "admission": {"logic": "accept"},
            "eviction_scoring": {"logic": "score by frequency"},
            "parameters": {"frequency_multiplier": 128},
            "state_update": {"on_hit": "increment frequency"},
        },
        artifact_type="cache_policy_json",
    )
    policy = ArtifactPolicy.from_mapping(
        {
            "machine_artifact_required": True,
            "artifact_type": "cache_policy",
            "artifact_type_aliases": {"cache_policy_json": "cache_policy"},
            "field_aliases": {"eviction_scoring": "eviction", "state_update": "update_or_state_update"},
            "required_fields": ["admission", "eviction", "parameters", "update_or_state_update"],
            "final_requires_clean_schema": True,
        }
    )

    view = normalize_artifact(candidate, policy=policy, artifact_type=candidate.artifact_type)

    assert view["artifact_type"] == "cache_policy"
    assert view["status"] == "refolded"
    assert view["probe_eligible"] is True
    assert view["final_eligible"] is False
    assert "eviction" in view["normalized_artifact"]
    assert "update_or_state_update" in view["normalized_artifact"]
    assert "eviction_scoring" not in view["normalized_artifact"]
    assert any("artifact_type_alias_normalized" in item for item in view["diagnostics"])
    assert any("field_alias_normalized" in item for item in view["diagnostics"])
    assert "final_requires_clean_schema" in view["diagnostics"]


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


def test_external_evaluator_receives_normalized_probe_candidate_without_mutating_original(tmp_path: Path) -> None:
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(
        "import json, sys\n"
        "data=json.load(open(sys.argv[1]))\n"
        "artifact=data['artifact']\n"
        "ok=data['artifact_type']=='cache_policy' and all(k in artifact for k in ['admission','eviction','parameters','update_or_state_update'])\n"
        "print(json.dumps({'passed': ok, 'metrics': {'score': 0.72, 'schema_cleanliness': 1.0}, 'diagnostics': [] if ok else ['normalized input missing']}))\n",
        encoding="utf-8",
    )
    candidate = CandidateGenome(
        id="C1",
        artifact={
            "admission": {"logic": "accept"},
            "eviction_scoring": {"logic": "score by frequency"},
            "parameters": {"frequency_multiplier": 128},
            "state_update": {"on_hit": "increment frequency"},
        },
        artifact_type="cache_policy_json",
    )
    spec = EvaluatorSpec.from_mapping(
        {
            "enabled": True,
            "command": f"{sys.executable} evaluator.py {{candidate_path}}",
            "cwd": str(tmp_path),
            "evidence": {
                "machine_artifact_required": True,
                "artifact_type": "cache_policy",
                "artifact_type_aliases": {"cache_policy_json": "cache_policy"},
                "field_aliases": {"eviction_scoring": "eviction", "state_update": "update_or_state_update"},
                "required_fields": ["admission", "eviction", "parameters", "update_or_state_update"],
            },
        }
    )

    result = ExternalEvaluatorRunner().evaluate_population_if_configured([candidate], spec=spec, round_index=1)[0]
    record = latest_evidence_record(candidate)

    assert result.passed is True
    assert candidate.artifact_type == "cache_policy_json"
    assert "eviction_scoring" in candidate.artifact
    assert record is not None
    assert record.metadata["artifact_state"]["status"] == "refolded"
    assert record.final_blocked is True
    assert record.emitted_challenge_ids


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


def test_schema_challenge_classification_and_auto_resolution() -> None:
    assert classify_diagnostic("candidate artifact_type must be cache_policy") == "artifact_type_mismatch"
    assert classify_diagnostic("missing required cache policy sections: eviction") == "missing_required_field"
    assert classify_diagnostic("candidate artifact must be a JSON object") == "machine_parse_failure"
    assert classify_diagnostic("field_alias_normalized: eviction_scoring -> eviction") == "field_alias"
    assert classify_diagnostic("semantic_drift_detected: forbidden_term=checkpoint") == "semantic_drift"
    assert classify_diagnostic("behavior_score_failure: hit_rate_below_threshold") == "behavior_score_failure"
    assert classify_diagnostic("semantic score too low") == "generic"

    schema_case = challenge_from_diagnostic(candidate_id="C1", source="cache_policy", diagnostic="missing required cache policy sections: eviction", round_index=1)
    generic_case = challenge_from_diagnostic(candidate_id="C1", source="cache_policy", diagnostic="semantic score too low", round_index=1)
    memory = ChallengeMemory()
    memory.ingest(EvidenceRecord(candidate_id="C1", source="cache_policy", emitted_challenge_ids=[schema_case["id"], generic_case["id"]], metadata={"challenge_items": [schema_case, generic_case]}), round_index=1)
    memory.mark_targeted("C2", [schema_case["id"], generic_case["id"]])
    fixed_record = EvidenceRecord(
        candidate_id="C2",
        source="cache_policy",
        stage="probe",
        score=0.9,
        final_blocked=True,
        target_challenge_ids=[schema_case["id"], generic_case["id"]],
        diagnostics=[],
        metadata={"status": "passed", "metrics": {"schema_cleanliness": 1.0, "correctness": True}, "artifact_state": {"schema_cleanliness": 1.0, "status": "clean"}},
    )

    resolved = memory.mark_schema_resolved_from_record(fixed_record)

    assert resolved == [schema_case["id"]]
    assert memory.items[schema_case["id"]]["resolved_by_candidate_ids"] == ["C2"]
    assert memory.items[generic_case["id"]]["resolved_by_candidate_ids"] == []


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


def test_instruction_only_search_pressure_enters_mutation_plan_without_target_tracking() -> None:
    controller = AdaptiveRuntimeController(
        config=AdaptiveConfig.from_sources(
            explicit={
                "enabled": True,
                "research": {"enabled": True, "mode": "advisory", "extensions": {"context_pruning": {"enabled": True}}},
            }
        )
    )
    parent = CandidateGenome(id="P1", current_fate=CandidateFate.ACTIVE.value, artifact={"answer": 1})
    round_driver = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=2, branch_factor=1), adaptive=controller)
    plan = MutationPlan(operator=MutationOperator.REPAIR, parent_ids=["P1"], instruction="Repair the candidate.")

    pressured = round_driver._apply_search_pressure_to_plans([plan], parents=[parent])

    assert "search_pressure_id" in pressured[0].metadata
    assert "target_challenge_ids" not in pressured[0].metadata
    assert "Prune mutation context" in pressured[0].instruction


def test_schema_search_pressure_generates_strict_repair_instruction() -> None:
    case = challenge_from_diagnostic(candidate_id="P1", source="cache_policy", diagnostic="candidate artifact_type must be cache_policy", round_index=1)
    memory = ChallengeMemory()
    memory.ingest(EvidenceRecord(candidate_id="P1", source="cache_policy", emitted_challenge_ids=[case["id"]], metadata={"challenge_items": [case]}), round_index=1)

    pressure = memory.compile_search_pressure(
        parent_id="P1",
        scope="candidate",
        artifact_requirements={
            "artifact_type": "cache_policy",
            "artifact_type_aliases": {"cache_policy_json": "cache_policy"},
            "field_aliases": {"eviction_scoring": "eviction", "state_update": "update_or_state_update"},
            "required_fields": ["admission", "eviction", "parameters", "update_or_state_update"],
        },
    )

    assert pressure is not None
    assert pressure.metadata["schema_repair_focus"] is True
    assert "Schema repair has priority" in pressure.mutation_instruction
    assert "artifact_type=cache_policy" in pressure.mutation_instruction
    assert "admission, eviction, parameters, update_or_state_update" in pressure.mutation_instruction
    assert "cache_policy_json" in pressure.mutation_instruction
    assert "eviction_scoring" in pressure.mutation_instruction


def test_search_pressure_prioritizes_contract_schema_over_repeated_generic_final_gate() -> None:
    contract_case = challenge_from_diagnostic(
        candidate_id="__contract__",
        source="artifact_contract_policy",
        diagnostic="contract_artifact_policy_conflict: artifact_type_missing expected=cache_policy",
        round_index=1,
        priority=0.9,
    )
    generic_case = challenge_from_diagnostic(
        candidate_id="C1",
        source="cache_policy",
        diagnostic="external_evaluator_not_passed",
        round_index=1,
        priority=0.9,
    )
    memory = ChallengeMemory()
    memory.ingest(
        EvidenceRecord(
            candidate_id="C1",
            source="cache_policy",
            emitted_challenge_ids=[generic_case["id"]],
            metadata={"challenge_items": [generic_case]},
        ),
        round_index=1,
    )
    for round_index in range(2, 9):
        memory.ingest(
            EvidenceRecord(
                candidate_id=f"C{round_index}",
                source="cache_policy",
                emitted_challenge_ids=[generic_case["id"]],
                metadata={"challenge_items": [generic_case]},
            ),
            round_index=round_index,
        )
    memory.ingest(
        EvidenceRecord(
            candidate_id="__contract__",
            source="artifact_contract_policy",
            emitted_challenge_ids=[contract_case["id"]],
            metadata={"challenge_items": [contract_case]},
        ),
        round_index=2,
    )

    pressure = memory.compile_search_pressure(limit=1, artifact_requirements={"artifact_type": "cache_policy"})

    assert pressure is not None
    assert pressure.target_challenge_ids == [contract_case["id"]]
    assert pressure.metadata["selected_categories"] == ["contract_artifact_policy_conflict"]
    assert "Schema repair has priority" in pressure.mutation_instruction


def test_semantic_drift_blocks_final_and_becomes_search_pressure() -> None:
    candidate = CandidateGenome(
        id="C-drift",
        artifact={
            "admission": {"logic": "restore checkpoint before serving request"},
            "eviction": {"logic": "evict old runtime_state windows"},
            "parameters": {"window": 4},
            "update_or_state_update": {"on_hit": "checkpoint recovery_window"},
        },
        artifact_type="cache_policy",
    )
    spec = EvaluatorSpec.from_mapping(
        {
            "enabled": True,
            "command": "unused",
            "stage": "final",
            "evidence": {
                "machine_artifact_required": True,
                "artifact_type": "cache_policy",
                "required_fields": ["admission", "eviction", "parameters", "update_or_state_update"],
                "metadata": {
                    "domain_vocabulary": ["cache", "admission", "eviction", "hit", "miss", "object"],
                    "forbidden_semantic_terms": ["checkpoint", "runtime_state", "recovery_window"],
                },
            },
        }
    )

    record = ProgressiveEvaluator().evaluate_result(
        candidate,
        EvaluatorResult(candidate_id="C-drift", status="passed", passed=True, metrics={"score": 0.92, "certificate_passed": True}, diagnostics=[]),
        spec=spec,
        round_index=4,
    )
    memory = ChallengeMemory()
    memory.ingest(record, round_index=4)
    pressure = memory.compile_search_pressure(
        artifact_requirements={
            "artifact_type": "cache_policy",
            "required_fields": ["admission", "eviction", "parameters", "update_or_state_update"],
            "metadata": {
                "domain_vocabulary": ["cache", "admission", "eviction", "hit", "miss", "object"],
                "forbidden_semantic_terms": ["checkpoint", "runtime_state", "recovery_window"],
            },
        }
    )

    assert record.final_blocked is True
    assert any("semantic_drift_detected" in item for item in record.diagnostics)
    assert record.emitted_challenge_ids
    assert pressure is not None
    assert pressure.metadata["semantic_drift_focus"] is True
    assert "removing out-of-domain/internal runtime concepts" in pressure.mutation_instruction
    assert "checkpoint" in pressure.mutation_instruction


def test_behavior_diagnostics_are_decomposed_into_behavior_challenges() -> None:
    candidate = CandidateGenome(
        id="C-behavior",
        artifact={"admission": {}, "eviction": {}, "parameters": {}, "update_or_state_update": {}},
        artifact_type="cache_policy",
    )
    spec = EvaluatorSpec.from_mapping(
        {
            "enabled": True,
            "command": "unused",
            "evidence": {
                "machine_artifact_required": True,
                "artifact_type": "cache_policy",
                "required_fields": ["admission", "eviction", "parameters", "update_or_state_update"],
            },
        }
    )

    record = ProgressiveEvaluator().evaluate_result(
        candidate,
        EvaluatorResult(
            candidate_id="C-behavior",
            status="failed",
            passed=False,
            metrics={"score": 0.22, "hit_rate": 0.12, "byte_hit_rate": 0.16, "correctness": False},
            diagnostics=["trace score below threshold"],
        ),
        spec=spec,
        round_index=5,
    )
    memory = ChallengeMemory()
    memory.ingest(record, round_index=5)
    pressure = memory.compile_search_pressure(artifact_requirements={"artifact_type": "cache_policy"})

    assert any(item.startswith("behavior_score_failure") for item in record.diagnostics)
    categories = {item["metadata"]["category"] for item in record.metadata["challenge_items"]}
    assert "behavior_score_failure" in categories
    assert pressure is not None
    assert pressure.metadata["behavior_score_focus"] is True
    assert "improves evaluator behavior" in pressure.mutation_instruction


def test_evidence_advisory_rewards_resolved_clean_schema_over_unresolved_semantic_drift() -> None:
    resolved = CandidateGenome(
        id="C-resolved",
        artifact={"admission": {}, "eviction": {}, "parameters": {}, "update_or_state_update": {}},
        artifact_type="cache_policy",
        multihead_scores={"schema_cleanliness": 1.0},
    )
    unresolved = CandidateGenome(
        id="C-unresolved",
        artifact={"admission": {"logic": "checkpoint route"}},
        artifact_type="cache_policy",
    )
    apply_evidence_record(
        resolved,
        EvidenceRecord(
            candidate_id="C-resolved",
            source="cache_policy",
            score=0.7,
            final_blocked=True,
            repair_value=0.4,
            target_challenge_ids=["case-schema"],
            resolved_challenge_ids=["case-schema"],
            metadata={"artifact_state": {"schema_cleanliness": 1.0, "status": "clean"}},
        ),
    )
    apply_evidence_record(
        unresolved,
        EvidenceRecord(
            candidate_id="C-unresolved",
            source="cache_policy",
            score=0.7,
            final_blocked=True,
            repair_value=0.4,
            target_challenge_ids=["case-semantic"],
            diagnostics=["semantic_drift_detected: forbidden_term=checkpoint"],
            metadata={"challenge_items": [challenge_from_diagnostic(candidate_id="C-unresolved", source="cache_policy", diagnostic="semantic_drift_detected: forbidden_term=checkpoint")]},
        ),
    )

    features = evidence_advisory_features([resolved, unresolved])

    assert features["C-resolved"]["plan_value"] > features["C-unresolved"]["plan_value"]
    assert features["C-unresolved"]["risk"] > features["C-resolved"]["risk"]


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
    assert projection.artifact_type == "machine"
    assert "internal repair directive should not be reused" not in markdown
    assert "Best current artifact" in markdown
    assert "boundary case failed" in markdown
    assert isinstance(projection.to_dict()["artifact"], dict)


def test_final_projection_downgrades_refolded_artifact_even_with_solved_certificate() -> None:
    candidate = CandidateGenome(
        id="C1",
        artifact="{'answer': 1}",
        artifact_type="machine",
        multihead_scores={"frontier_score": 0.8, "schema_cleanliness": 0.85},
    )
    apply_evidence_record(
        candidate,
        EvidenceRecord(
            candidate_id="C1",
            source="machine",
            stage="final",
            score=0.9,
            final_blocked=True,
            metadata={"artifact_state": {"normalized_artifact": {"answer": 1}, "status": "refolded", "final_eligible": False}},
        ),
    )
    synthesis = SynthesizedResult(status="solved", final_answer="solved", best_candidate_id="C1")

    projection = build_final_projection(population=CandidatePopulation([candidate]), synthesis=synthesis, final_certificate={"objective_solved": True, "candidate_id": "C1"})

    assert projection.status == "best_current"
    assert projection.objective_solved is False
    assert "candidate_not_clean_final_eligible" in projection.blocking_issues


def test_final_projection_excludes_failed_and_culled_best_current_candidates() -> None:
    failed = CandidateGenome(id="F1", artifact={"bad": 1}, artifact_type="machine", current_fate=CandidateFate.FAILED.value, multihead_scores={"frontier_score": 0.99})
    active = CandidateGenome(id="A1", artifact={"ok": 1}, artifact_type="machine", current_fate=CandidateFate.ACTIVE.value, multihead_scores={"frontier_score": 0.4})
    synthesis = SynthesizedResult(status="best_current_route", final_answer="", best_candidate_id="F1")

    projection = build_final_projection(population=CandidatePopulation([failed, active]), synthesis=synthesis, final_certificate={"blocking_reasons": ["not_final"]})

    assert projection.status == "best_current"
    assert projection.candidate_id == "A1"


def test_artifact_identity_hash_includes_artifact_type_and_policy() -> None:
    state_a = {"artifact_type": "cache_policy", "normalized_artifact": {"admission": {}}, "status": "clean", "final_eligible": True}
    state_b = {"artifact_type": "other_policy", "normalized_artifact": {"admission": {}}, "status": "clean", "final_eligible": True}
    policy_a = {"artifact_type": "cache_policy", "required_fields": ["admission"], "final_requires_clean_schema": True}
    policy_b = {"artifact_type": "cache_policy", "required_fields": ["admission", "eviction"], "final_requires_clean_schema": True}

    assert stable_artifact_identity_hash(state_a, artifact_policy=policy_a) != stable_artifact_identity_hash(state_b, artifact_policy=policy_a)
    assert stable_artifact_identity_hash(state_a, artifact_policy=policy_a) != stable_artifact_identity_hash(state_a, artifact_policy=policy_b)


def test_cache_trace_evaluator_fixture_scores_clean_policy() -> None:
    candidate = CandidateGenome(
        id="C1",
        artifact={
            "admission": {"logic": "if object.size > cache.average_size * size_threshold_ratio deny else accept"},
            "eviction": {"logic": "score = frequency_multiplier * frequency - base_recency_weight * age"},
            "parameters": {"size_threshold_ratio": 2.5, "frequency_multiplier": 128, "base_recency_weight": 4},
            "update_or_state_update": {"on_hit": "frequency += 1; last_tick = current_tick"},
        },
        artifact_type="cache_policy",
    )
    fixture = Path(__file__).parents[1] / "fixtures" / "evaluators" / "cache_trace_evaluator.py"
    spec = EvaluatorSpec.from_mapping(
        {
            "enabled": True,
            "command": f"{sys.executable} {fixture} {{candidate_path}}",
            "evidence": {
                "machine_artifact_required": True,
                "artifact_type": "cache_policy",
                "required_fields": ["admission", "eviction", "parameters", "update_or_state_update"],
            },
        }
    )

    result = ExternalEvaluatorRunner().evaluate_population_if_configured([candidate], spec=spec, round_index=1)[0]

    assert result.passed is True
    assert result.metrics["schema_cleanliness"] == 1.0
    assert result.metrics["hit_rate"] > 0.0
    assert latest_evidence_record(candidate).metadata["status"] == "passed"


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


def test_evaluator_empty_parsed_diagnostics_do_not_fallback_to_raw_stdout(tmp_path: Path) -> None:
    script = tmp_path / "eval.py"
    script.write_text(
        "import json, sys\n"
        "print(json.dumps({'passed': True, 'metrics': {'score': 0.7}, 'diagnostics': []}))\n",
        encoding="utf-8",
    )
    candidate = CandidateGenome(
        id="clean",
        artifact_type="cache_policy",
        artifact={"admission": {}, "eviction": {}, "parameters": {}, "update_or_state_update": {}},
    )
    spec = EvaluatorSpec.from_mapping(
        {
            "enabled": True,
            "command": f"python {script} {{candidate_path}}",
            "evidence": {
                "machine_artifact_required": True,
                "artifact_type": "cache_policy",
                "required_fields": ["admission", "eviction", "parameters", "update_or_state_update"],
            },
        }
    )

    result = ExternalEvaluatorRunner().evaluate_population_if_configured([candidate], spec=spec, round_index=1)[0]

    assert result.passed is True
    assert result.diagnostics == []
    assert candidate.metadata["evaluator"]["diagnostics"] == []
