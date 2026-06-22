from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.failure_classifier import classify_recovery_eligibility
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, EvolutionRound
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.prompt_view import candidate_prompt_view
from cognitive_evolve_runtime.nexus.repair_reactivation import recover_repairable_dormant_seeds
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult


def _diagnosis() -> SearchDiagnosis:
    return SearchDiagnosis(
        stagnation_detected=True,
        stagnation_type="route_incomplete_no_parents_patch_repair",
        recommended_actions=["reactivate_dormant_repair_candidates", "repair malformed patch candidates"],
        notes="Active pool is empty but dormant patch repair material remains.",
    )


def _repairable_candidate(candidate_id: str = "archive-repair") -> CandidateGenome:
    candidate = CandidateGenome(
        id=candidate_id,
        current_fate=CandidateFate.DORMANT,
        artifact_type="code_patch",
        artifact={
            "path": "cognitive_evolve_runtime/nexus/loop.py",
            "unified_diff": (
                "diff --git a/cognitive_evolve_runtime/nexus/loop.py b/cognitive_evolve_runtime/nexus/loop.py\n"
                "--- a/cognitive_evolve_runtime/nexus/loop.py\n"
                "+++ b/cognitive_evolve_runtime/nexus/loop.py\n"
                "@@ -1 +1 @@\n"
                "-old\n"
            ),
        },
        concise_claim="dormant repair seed for no-parent collapse",
        core_mechanism="repair dormant patch recovery",
        failure_lessons=["patch_application_failed", "unified_patch_failed:patch: **** unexpected end of file in patch"],
        source_bindings=[{"path": "cognitive_evolve_runtime/nexus/loop.py", "kind": "source_file"}],
        verification_result={
            "passed": False,
            "diagnostics": ["unified_patch_failed:patch: **** unexpected end of file in patch"],
        },
        metadata={
            "failure_micro_guidance": [
                {
                    "blocker": "unified_patch_failed:patch: **** unexpected end of file in patch",
                    "next_action": "rewrite a complete unified diff against exact source context",
                    "evidence_needed": ["complete_unified_diff", "post_pass_local_verification"],
                    "source_bindings": [{"path": "cognitive_evolve_runtime/nexus/loop.py", "kind": "source_file"}],
                    "disallowed_repeat_pattern": "do_not_repeat_the_same_malformed_or_context_stale_diff",
                }
            ]
        },
    )
    candidate.patch_application_result = {
        "status": "failed",
        "diagnostics": ["unified_patch_failed:patch: **** unexpected end of file in patch"],
        "failed_files": ["cognitive_evolve_runtime/nexus/loop.py"],
    }
    return candidate


def test_recoverable_dormant_archive_seed_prevents_no_parents_available() -> None:
    archives = ArchiveManager()
    parent = _repairable_candidate()
    archives.dormant_archive.add(parent)
    population = CandidatePopulation([])
    round_stage = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=48, branch_factor=2))

    stop_reason, _offspring_verification, _compaction = round_stage.reproduce(
        current_round=7,
        population=population,
        archives=archives,
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="repair self-evolution", normalized_goal="repair self-evolution"),
        world=object(),
        rankings=RelativeRankingResult(),
        diagnosis=_diagnosis(),
        critiques=[],
        offspring_verifier=None,
    )

    assert stop_reason != "no_parents_available"
    assert population.candidates
    child = population.candidates[0]
    assert child.parent_ids == ["archive-repair"]
    assert child.metadata["repair_seed"]["target_files"] == ["cognitive_evolve_runtime/nexus/loop.py"]
    assert child.metadata["targeted_repair_lane"] is False
    assert archives.dormant_archive.candidates["archive-repair"]["metadata"]["repair_attempts"] == 1


def test_dormant_recovery_keeps_narrative_only_source_free_candidate_as_advisory() -> None:
    archives = ArchiveManager()
    archives.dormant_archive.add(
        CandidateGenome(
            id="narrative-only",
            current_fate=CandidateFate.DORMANT,
            artifact_type="answer",
            artifact="We should add a NicheAllocator and improve diversity, but no patch is supplied.",
            concise_claim="narrative-only proposal",
            core_mechanism="narrative_only",
            verification_result={"passed": False, "diagnostics": ["narrative_only", "source_free_final_claim"]},
        )
    )

    recovered = recover_repairable_dormant_seeds(
        archives=archives,
        diagnosis=_diagnosis(),
        policy=EvolutionPolicy(),
        limit=2,
        current_round=7,
    )

    assert [candidate.id for candidate in recovered] == ["narrative-only"]
    assert recovered[0].metadata["dormant_recovery_advisory"]["category"] == "terminal_narrative_or_source_free_final_claim"


def test_recovery_requires_existing_project_target() -> None:
    candidate = _repairable_candidate("hallucinated-target")
    candidate.artifact = {"path": "NicheAllocator.py", "unified_diff": "--- a/NicheAllocator.py\n+++ b/NicheAllocator.py\n@@\n-old\n"}
    candidate.source_bindings = [{"path": "NicheAllocator.py", "kind": "source_file"}]
    candidate.patch_application_result = {
        "status": "failed",
        "diagnostics": ["unified_patch_failed:patch: **** unexpected end of file in patch"],
        "failed_files": ["NicheAllocator.py"],
    }

    verdict = classify_recovery_eligibility(candidate)

    assert verdict.repairable is False
    assert verdict.category == "terminal_recovery_missing_existing_project_path"


def test_repair_attempt_cap_is_soft_for_dormant_recovery() -> None:
    archives = ArchiveManager()
    exhausted = _repairable_candidate("exhausted-repair")
    exhausted.metadata["repair_attempts"] = 1
    archives.dormant_archive.add(exhausted)
    policy = EvolutionPolicy(
        metadata={
            "eligibility_policy": {
                "dormant_repair_reactivation": {"enabled": True, "max_repair_attempts": 1, "max_seeds": 2}
            }
        }
    )

    recovered = recover_repairable_dormant_seeds(
        archives=archives,
        diagnosis=_diagnosis(),
        policy=policy,
        limit=2,
        current_round=8,
    )

    assert [candidate.id for candidate in recovered] == ["exhausted-repair"]
    assert recovered[0].metadata["dormant_recovery_advisory"]["category"] == "terminal_repair_attempts_exhausted"


def test_max_per_group_is_soft_group_hint_not_hard_cap() -> None:
    archives = ArchiveManager()
    first = _repairable_candidate("group-a")
    second = _repairable_candidate("group-b")
    first.lineage = ["shared-family", first.id]
    second.lineage = ["shared-family", second.id]
    archives.dormant_archive.add(first)
    archives.dormant_archive.add(second)
    policy = EvolutionPolicy(
        metadata={
            "eligibility_policy": {
                "dormant_repair_reactivation": {
                    "enabled": True,
                    "max_repair_attempts": 3,
                    "max_per_group": 1,
                    "max_seeds": 3,
                }
            }
        }
    )

    recovered = recover_repairable_dormant_seeds(
        archives=archives,
        diagnosis=_diagnosis(),
        policy=policy,
        limit=3,
        current_round=10,
    )

    assert {candidate.id for candidate in recovered} == {"group-a", "group-b"}
    hinted = [
        candidate
        for candidate in recovered
        if candidate.metadata.get("candidate_budget_decision", {}).get("reason") == "soft_group_hint"
    ]
    assert len(hinted) == 1
    assert hinted[0].metadata["candidate_budget_decision"]["hard_gate"] is False


def test_repair_seed_prompt_view_exposes_concise_contract() -> None:
    archives = ArchiveManager()
    parent = _repairable_candidate()
    archives.dormant_archive.add(parent)
    recovered = recover_repairable_dormant_seeds(
        archives=archives,
        diagnosis=_diagnosis(),
        policy=EvolutionPolicy(),
        limit=1,
        current_round=7,
    )

    view = candidate_prompt_view(recovered[0])

    assert view["repair_seed_contract"]["target_files"] == ["cognitive_evolve_runtime/nexus/loop.py"]
    assert "answer-first exploration" in view["repair_seed_contract"]["contract"]
