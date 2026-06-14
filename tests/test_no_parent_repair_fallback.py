from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, EvolutionRound
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult


def test_reproduce_uses_ranked_repair_parent_when_strict_parent_pool_is_empty() -> None:
    parent = CandidateGenome(
        id="dormant-seed",
        current_fate=CandidateFate.DORMANT,
        artifact_type="project_patch",
        concise_claim="documentation-only seed needs concrete code repair",
        core_mechanism="seed needs repair",
        missing_parts=["runtime_or_test_patch"],
        metadata={
            "failure_micro_guidance": [
                {
                    "blocker": "runtime_code_change_required",
                    "next_action": "replace the docs-only patch with a concrete runtime/test/schema patch",
                    "evidence_needed": ["runtime_or_test_patch", "source_binding"],
                }
            ],
            "stage_eligibility": {
                "parent_eligible": False,
                "hard_reject_reason": "hard_reject_diagnostic:runtime_code_change_required,seed_note_only_patch",
            },
        },
        verification_result={"passed": False, "diagnostics": ["runtime_code_change_required"]},
    )
    population = CandidatePopulation([parent])
    diagnosis = SearchDiagnosis(
        stagnation_detected=True,
        stagnation_type="RouteIncomplete",
        recommended_actions=["reactivate_dormant_candidates", "synthesize_concrete_code_patches_instead_of_docs"],
        notes="all candidates are dormant but ranked repair material exists",
    )
    rankings = RelativeRankingResult(mutation_worthy_ids=["dormant-seed"])
    round_stage = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=48, branch_factor=2))

    stop_reason, _offspring_verification, _compaction = round_stage.reproduce(
        current_round=1,
        population=population,
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="repair", normalized_goal="repair"),
        world=object(),
        rankings=rankings,
        diagnosis=diagnosis,
        critiques=[],
        offspring_verifier=None,
    )

    assert stop_reason != "no_parents_available"
    assert len(population.candidates) > 1
    assert parent.metadata["no_parent_repair_fallback"]["final_answer_blocked"] is True
    assert parent.metadata["final_answer_blocked_until_repaired"] is True


def test_reproduce_does_not_use_dormant_fallback_without_reactivation_signal() -> None:
    parent = CandidateGenome(
        id="quiet-dormant",
        current_fate=CandidateFate.DORMANT,
        missing_parts=["runtime_or_test_patch"],
        metadata={"failure_micro_guidance": [{"blocker": "runtime_code_change_required"}]},
    )
    population = CandidatePopulation([parent])
    round_stage = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=48, branch_factor=2))

    stop_reason, _offspring_verification, _compaction = round_stage.reproduce(
        current_round=1,
        population=population,
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="repair", normalized_goal="repair"),
        world=object(),
        rankings=RelativeRankingResult(mutation_worthy_ids=["quiet-dormant"]),
        diagnosis=SearchDiagnosis(stagnation_detected=False, recommended_actions=["continue"]),
        critiques=[],
        offspring_verifier=None,
    )

    assert stop_reason == "no_parents_available"
    assert len(population.candidates) == 1


def test_docs_only_diagnosis_uses_ranked_repair_parent_when_pool_is_empty() -> None:
    parent = CandidateGenome(
        id="docs-loop-seed",
        current_fate=CandidateFate.DORMANT,
        artifact_type="project_patch",
        concise_claim="seed note only project candidate needs code repair",
        core_mechanism="minimal_patch",
        failure_lessons=["project patch did not touch the required implementation or test surface"],
        metadata={
            "failure_micro_guidance": [
                {
                    "blocker": "seed_note_only_patch",
                    "next_action": "stop modifying NEXUS_SEED_NOTE.md and target implementation or tests directly",
                    "evidence_needed": ["runtime_or_test_patch", "source_binding"],
                }
            ],
            "stage_eligibility": {"parent_eligible": False, "hard_reject_reason": "seed_note_only_patch"},
        },
        verification_result={"passed": False, "diagnostics": ["seed_note_only_patch", "runtime_code_change_required"]},
    )
    population = CandidatePopulation([parent])
    diagnosis = SearchDiagnosis(
        stagnation_detected=True,
        stagnation_type="docs_only_patch_loop",
        recommended_actions=[
            "Discontinue generating markdown-only patch files and target runtime code modifications.",
            "Repair patch structures to ensure unified diff format correctness.",
        ],
        notes="Exploration is trapped in NEXUS_SEED_NOTE markdown notes rather than implementation patches.",
    )
    round_stage = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=48, branch_factor=2))

    stop_reason, _offspring_verification, _compaction = round_stage.reproduce(
        current_round=2,
        population=population,
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="repair runtime", normalized_goal="repair runtime"),
        world=object(),
        rankings=RelativeRankingResult(mutation_worthy_ids=["docs-loop-seed"]),
        diagnosis=diagnosis,
        critiques=[],
        offspring_verifier=None,
    )

    assert stop_reason != "no_parents_available"
    assert parent.metadata["no_parent_repair_fallback"]["final_answer_blocked"] is True
    assert parent.metadata["final_answer_blocked_until_repaired"] is True


def test_patch_application_failed_candidate_can_seed_bounded_repair_when_pool_is_empty() -> None:
    parent = CandidateGenome(
        id="failed-diff-seed",
        current_fate=CandidateFate.FAILED,
        artifact_type="code_patch",
        concise_claim="executor path boundary patch needs diff repair",
        core_mechanism="bounded executor path validation",
        failure_lessons=[
            "patch_application_failed",
            "unified_patch_failed:patch: **** unexpected end of file in patch",
        ],
        verification_result={
            "passed": False,
            "diagnostics": [
                "patch_application_failed",
                "unified_patch_failed:patch: **** unexpected end of file in patch",
                "proof_object_structurally_weak",
            ],
        },
        metadata={
            "failure_micro_guidance": [
                {
                    "blocker": "unified_patch_failed:patch: **** unexpected end of file in patch",
                    "next_action": "rewrite a complete unified diff against exact source context",
                    "evidence_needed": ["valid_unified_diff"],
                }
            ]
        },
    )
    parent.patch_application_result = {
        "status": "failed",
        "diagnostics": ["unified_patch_failed:patch: **** unexpected end of file in patch"],
        "failed_files": ["cognitive_evolve_runtime/api/executor.py"],
    }
    population = CandidatePopulation([parent])
    diagnosis = SearchDiagnosis(
        stagnation_detected=True,
        stagnation_type="patch_application_and_dry_run_failures",
        recommended_actions=["repair_patch_generator_eof_handling_in_executor", "reactivate_dormant_repair_niches"],
        notes="All candidates failed from malformed patch or unexpected EOF, but ranked repair material exists.",
    )
    round_stage = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=48, branch_factor=2))

    stop_reason, _offspring_verification, _compaction = round_stage.reproduce(
        current_round=1,
        population=population,
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="repair runtime", normalized_goal="repair runtime"),
        world=object(),
        rankings=RelativeRankingResult(mutation_worthy_ids=["failed-diff-seed"]),
        diagnosis=diagnosis,
        critiques=[],
        offspring_verifier=None,
    )

    assert stop_reason != "no_parents_available"
    assert len(population.candidates) >= 1
    assert population.candidates[0].parent_ids == ["failed-diff-seed"]
    assert parent.metadata["repair_attempts"] == 1
    assert population.candidates[0].metadata["repair_attempts"] == 1
    assert parent.metadata["no_parent_repair_fallback"]["final_answer_blocked"] is True
    assert parent.metadata["final_answer_blocked_until_repaired"] is True


def test_post_compaction_failed_candidate_can_seed_bounded_repair() -> None:
    parent = CandidateGenome(
        id="compacted-failed-diff",
        current_fate=CandidateFate.FAILED,
        artifact_type="code_patch",
        concise_claim="engine runner patch carries useful repair material",
        core_mechanism="engine runner eof-safe patch repair",
        failure_lessons=[
            "patch_application_failed",
            "unified_patch_failed:patch: **** unexpected end of file in patch",
        ],
        verification_result={
            "passed": False,
            "diagnostics": [
                "patch_application_failed",
                "unified_patch_failed:patch: **** unexpected end of file in patch",
            ],
        },
        metadata={
            "failure_micro_guidance": [
                {
                    "blocker": "unified_patch_failed:patch: **** unexpected end of file in patch",
                    "next_action": "rewrite a complete unified diff against exact source context",
                    "evidence_needed": ["valid_unified_diff"],
                }
            ]
        },
    )
    parent.patch_application_result = {
        "status": "failed",
        "diagnostics": ["unified_patch_failed:patch: **** unexpected end of file in patch"],
        "failed_files": ["cognitive_evolve_runtime/api/engine_runner.py"],
    }
    population = CandidatePopulation([])
    diagnosis = SearchDiagnosis(
        stagnation_detected=True,
        stagnation_type="patch_application_and_dry_run_failures",
        recommended_actions=["repair malformed patch candidates instead of ending with no_parents_available"],
        notes="Population compaction removed terminal Failed candidates, but failed patch repair genes remain useful.",
    )
    round_stage = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=48, branch_factor=2))

    stop_reason, _offspring_verification, _compaction = round_stage.reproduce(
        current_round=1,
        population=population,
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="repair runtime", normalized_goal="repair runtime"),
        world=object(),
        rankings=RelativeRankingResult(mutation_worthy_ids=["compacted-failed-diff"]),
        diagnosis=diagnosis,
        critiques=[],
        offspring_verifier=None,
        repair_parent_candidates=[parent],
    )

    assert stop_reason != "no_parents_available"
    assert len(population.candidates) >= 1
    assert population.candidates[0].parent_ids == ["compacted-failed-diff"]
    assert parent.metadata["no_parent_repair_fallback"]["final_answer_blocked"] is True
    assert parent.metadata["final_answer_blocked_until_repaired"] is True


def test_repairable_failed_offspring_stays_incubating_after_sandbox_failure() -> None:
    parent = CandidateGenome(
        id="active-repair-parent",
        current_fate=CandidateFate.ACTIVE,
        artifact_type="code_patch",
        concise_claim="executor retry patch seed",
        core_mechanism="executor retry patch seed",
        multihead_scores={"objective_alignment": 0.6, "answer_likelihood": 0.4, "verifiability": 0.3},
    )
    population = CandidatePopulation([parent])
    diagnosis = SearchDiagnosis(
        stagnation_detected=True,
        stagnation_type="patch_application_and_dry_run_failures",
        recommended_actions=["repair malformed patch candidates"],
        notes="Malformed offspring should be repaired, not terminally removed before the next round.",
    )
    round_stage = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=48, branch_factor=2))

    def verifier(candidates: list[CandidateGenome]) -> list[dict[str, object]]:
        child = candidates[0]
        child.mark_fate(CandidateFate.FAILED.value)
        child.failure_lessons.append("patch_application_failed")
        child.patch_application_result = {
            "status": "failed",
            "diagnostics": ["unified_patch_failed:patch: **** unexpected end of file in patch"],
            "failed_files": ["cognitive_evolve_runtime/api/executor.py"],
        }
        return [
            {
                "candidate_id": child.id,
                "passed": False,
                "patch_result": child.patch_application_result,
                "tool_feedback": [
                    {
                        "status": "failed",
                        "diagnostics": ["unified_patch_failed:patch: **** unexpected end of file in patch"],
                        "failed_fragments": ["cognitive_evolve_runtime/api/executor.py"],
                    }
                ],
            }
        ]

    stop_reason, offspring_verification, _compaction = round_stage.reproduce(
        current_round=1,
        population=population,
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="repair runtime", normalized_goal="repair runtime"),
        world=object(),
        rankings=RelativeRankingResult(mutation_worthy_ids=["active-repair-parent"]),
        diagnosis=diagnosis,
        critiques=[],
        offspring_verifier=verifier,
    )

    children = [candidate for candidate in population.candidates if candidate.parent_ids == ["active-repair-parent"]]
    assert stop_reason == ""
    assert offspring_verification and offspring_verification[0]["passed"] is False
    assert children
    assert children[0].current_fate == CandidateFate.INCUBATING.value
    assert children[0].metadata["final_answer_blocked_until_repaired"] is True
    assert children[0].metadata["repair_context"]["category"] == "repairable_patch_syntax_or_context"
    assert children[0].metadata["repair_context"]["repair_targets"] == ["cognitive_evolve_runtime/api/executor.py"]
    assert children[0].metadata["repair_required"]["source"] == "offspring_verification_repair_lane"
    assert children[0].metadata["failure_micro_guidance"]
