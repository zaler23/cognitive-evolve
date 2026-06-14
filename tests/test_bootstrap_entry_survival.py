from __future__ import annotations

from pathlib import Path

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.nexus.diagnosis import PolicyUpdater, SearchDiagnosis
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.project_verification import ProjectCandidateVerifier
from cognitive_evolve_runtime.nexus.repair_reactivation import recover_failure_archive_repair_seeds
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector


def test_malformed_bootstrap_project_patch_survives_as_incubating_repair_parent(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    candidate = CandidateGenome(
        id="bootstrap-malformed",
        generation=0,
        artifact_type="code_patch",
        artifact={
            "path": "mod.py",
            "unified_diff": "--- a/mod.py\n+++ b/mod.py\n@@ -1,2 +1,2 @@\n def value():\n-    return 1\n",
        },
        core_mechanism="under_explored_repairable_patch",
        source_bindings=[{"path": "mod.py", "kind": "source_file"}],
        metadata={"created_in_round": 0, "exploration_source": "nexus_model_seed_batch"},
    )

    summary = ProjectCandidateVerifier(source_root=repo, sandbox_root=tmp_path / "sandboxes").verify(candidate)

    assert summary.passed is False
    assert candidate.current_fate == CandidateFate.INCUBATING.value
    assert candidate.metadata["final_answer_blocked_until_repaired"] is True
    assert candidate.metadata["repair_required"]["source"] == "project_verification_repair_lane"
    assert candidate.metadata["bootstrap_entry_survival"]["final_answer_blocked"] is True


def test_missing_target_bootstrap_patch_stays_terminal(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    candidate = CandidateGenome(
        id="missing-target",
        generation=0,
        artifact_type="code_patch",
        artifact={"path": "missing.py", "patch": "--- missing.py\n+++ missing.py\n@@ -1 +1 @@\n-old\n+new\n"},
        source_bindings=[{"path": "missing.py", "kind": "source_file"}],
        metadata={"created_in_round": 0, "exploration_source": "nexus_model_seed_batch"},
    )

    summary = ProjectCandidateVerifier(source_root=repo, sandbox_root=tmp_path / "sandboxes").verify(candidate)

    assert summary.passed is False
    assert candidate.current_fate == CandidateFate.FAILED.value
    assert candidate.metadata["failure_classification"]["category"] == "terminal_recovery_missing_existing_project_path"


def test_selection_pressure_penalizes_over_explored_and_boosts_under_explored() -> None:
    minimal = CandidateGenome(
        id="minimal",
        current_fate=CandidateFate.ACTIVE,
        core_mechanism="minimal_patch",
        niche_memberships=["minimal_patch"],
        multihead_scores={"objective_alignment": 0.6, "answer_likelihood": 0.55},
    )
    rare = CandidateGenome(
        id="rare",
        current_fate=CandidateFate.ACTIVE,
        core_mechanism="rare_recall",
        niche_memberships=["rare_recall"],
        edge_knowledge_seeds=["rare_recall"],
        multihead_scores={"objective_alignment": 0.58, "answer_likelihood": 0.52},
    )
    diagnosis = SearchDiagnosis(
        stagnation_detected=True,
        stagnation_type="repeated_semantic_convergence",
        over_explored_families=["minimal_patch"],
        under_explored_families=["rare_recall"],
        prematurely_culled_genes=["internal_forgotten_pattern"],
        recommended_actions=["Force selection of under-explored niches by temporarily penalizing minimal_patch parents."],
    )
    policy = PolicyUpdater().update(EvolutionPolicy(), diagnosis)
    eligibility = policy.metadata["eligibility_policy"]

    selected = ParentSelector().select([minimal, rare], ArchiveManager(), limit=1, eligibility_policy=eligibility)

    assert [candidate.id for candidate in selected] == ["rare"]
    assert minimal.metadata["selection_pressure"]["over_explored_penalty"] == ["minimal_patch"]
    assert rare.metadata["selection_pressure"]["under_explored_bonus"] == ["rare_recall"]


def test_failure_archive_reseed_preserves_useful_patch_lesson_when_population_empty(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "cognitive_evolve_runtime").mkdir()
    (repo / "cognitive_evolve_runtime" / "engine_runner.py").write_text("class JobQueue:\n    pass\n", encoding="utf-8")
    archives = ArchiveManager()
    archives.failure_archive.records["Cbadpatch"] = {
        "candidate_id": "Cbadpatch",
        "failure_signature": (
            "patch_application_failed; unified_patch_failed:patch: **** malformed patch at line 27; "
            "source_binding_missing_symbol; final_update_artifact_absent; "
            "target cognitive_evolve_runtime/engine_runner.py"
        ),
        "inherited_gene_summary": (
            "Use evidence_obligation_tracking to preserve under-explored repair candidates; "
            "patch_application_failed"
        ),
    }
    archives.terminal_tombstones["Cbadpatch"] = {
        "candidate_id": "Cbadpatch",
        "niche_key": "evidence_obligation_tracking",
        "score_summary": {"objective_alignment": 0.6, "evidence_progress": 0.7, "proof_progress": 0.3},
    }
    diagnosis = SearchDiagnosis(
        stagnation_detected=True,
        stagnation_type="repeated_semantic_convergence",
        over_explored_families=["minimal_patch"],
        under_explored_families=["evidence_obligation_tracking"],
        prematurely_culled_genes=["rare_recall_seed"],
        recommended_actions=["Force selection of under-explored niches by temporarily penalizing minimal_patch parents."],
    )

    seeds = recover_failure_archive_repair_seeds(
        archives=archives,
        diagnosis=diagnosis,
        policy=EvolutionPolicy(),
        limit=2,
        current_round=1,
        project_root=repo,
    )

    assert len(seeds) == 1
    seed = seeds[0]
    assert seed.current_fate == CandidateFate.INCUBATING.value
    assert seed.metadata["final_answer_blocked_until_repaired"] is True
    assert seed.metadata["repair_seed"]["source"] == "failure_archive_reseed"
    assert seed.metadata["repair_required"]["source"] == "failure_archive_reseed"
    assert seed.metadata["repair_required"]["source_bindings"][0]["path"] == "cognitive_evolve_runtime/engine_runner.py"


def test_failure_archive_reseed_rejects_seed_note_only_records() -> None:
    archives = ArchiveManager()
    archives.failure_archive.records["Cnote"] = {
        "candidate_id": "Cnote",
        "failure_signature": "seed_note_only_patch; source_binding_missing_path; search_seed_not_final",
        "inherited_gene_summary": "test_first; initial search seed is not a final answer",
    }
    diagnosis = SearchDiagnosis(
        stagnation_detected=True,
        stagnation_type="repeated_semantic_convergence",
        recommended_actions=["Force selection of under-explored niches by temporarily penalizing minimal_patch parents."],
    )

    seeds = recover_failure_archive_repair_seeds(
        archives=archives,
        diagnosis=diagnosis,
        policy=EvolutionPolicy(),
        limit=2,
        current_round=1,
    )

    assert seeds == []
