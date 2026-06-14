from __future__ import annotations

from pathlib import Path

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.tools.verification_stack import NexusVerifierStack


def _project_contract() -> NexusObjectiveContract:
    return NexusObjectiveContract(
        original_user_goal="Update the CognitiveEvolve runtime with a source code patch and tests.",
        normalized_goal="runtime source patch with local tests",
        expected_output_forms=["code_patch", "tests"],
        verification_preferences=["source_binding", "local_tests"],
    )


def test_hybrid_design_seed_with_hallucinated_symbol_is_not_final(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "cognitive_evolve_runtime" / "api").mkdir(parents=True)
    (repo / "cognitive_evolve_runtime" / "api" / "jobs.py").write_text("_JOBS = {}\n_JOBS_LOCK = object()\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    candidate = CandidateGenome(
        id="hybrid-jobqueue",
        artifact_type="hybrid",
        artifact="Design: add JobQueue.save_state/load_state.",
        concise_claim="Persist incubating repair candidates across bootstrap.",
        core_mechanism="Add JobQueue state serialization.",
        source_bindings=[{"path": "cognitive_evolve_runtime/api/jobs.py", "symbol": "JobQueue"}],
        evidence_refs=[{"ref_type": "test", "test_name": "test_candidate_genome_roundtrip", "status": "verified"}],
        obligation_delta={"targeted": ["obl_job_state"]},
        formal_artifacts=[
            {
                "artifact_type": "assertion_set",
                "target_obligation_id": "obl_job_state",
                "assertions": ["jq = JobQueue(); jq.add_incubating(candidate_stub); assert len(jq.incubating_queue) == 1"],
            }
        ],
        missing_parts=["State serialization mechanism for the incubating lane across bootstrap phases."],
    )

    result = NexusVerifierStack(project_root=repo).verify_candidate(candidate, contract=_project_contract(), current_round=30, round_limit=48)

    assert result.passed is True
    assert result.rank_eligible is True
    assert result.final_eligible is False
    diagnostics = candidate.verification_result["final_gate"]["diagnostics"]
    assert "final_artifact_type_not_publishable" in diagnostics
    assert "final_update_artifact_absent" in diagnostics
    assert "source_binding_missing_symbol" in diagnostics
    assert "final_missing_parts_unresolved" in diagnostics
    assert ArchiveManager().is_final_answer_eligible(candidate) is False


def test_final_answer_blocked_until_reverified_overrides_positive_verification(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "cognitive_evolve_runtime").mkdir()
    (repo / "mod.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (repo / "tests" / "test_mod.py").write_text("from mod import value\n\ndef test_value():\n    assert value() == 2\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    candidate = CandidateGenome(
        id="blocked-valid-patch",
        current_fate=CandidateFate.ELITE,
        artifact_type="code_patch",
        artifact={"path": "mod.py", "patch": "--- mod.py\n+++ mod.py\n@@ -1,2 +1,2 @@\n def value():\n-    return 1\n+    return 2\n"},
        source_bindings=[{"path": "mod.py", "symbol": "value"}],
        evidence_refs=[{"kind": "test", "path": "tests/test_mod.py", "status": "verified"}],
        obligation_delta={"targeted": ["obl_patch"], "discharged": ["obl_patch"]},
        metadata={"final_answer_blocked_until_reverified": True},
    )
    candidate.patch_application_result = {"status": "applied", "applied_files": ["mod.py"], "pre_hash": "a", "post_hash": "b"}

    result = NexusVerifierStack(project_root=repo).verify_candidate(candidate, contract=_project_contract(), current_round=8, round_limit=48)

    assert result.final_eligible is False
    assert "final_answer_blocked_until_reverified" in candidate.verification_result["final_gate"]["diagnostics"]
    assert ArchiveManager().is_final_answer_eligible(candidate) is False


def test_source_bound_applied_patch_with_relevant_test_can_be_final(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "cognitive_evolve_runtime").mkdir()
    (repo / "mod.py").write_text("def value():\n    return 2\n", encoding="utf-8")
    (repo / "tests" / "test_mod.py").write_text("from mod import value\n\ndef test_value():\n    assert value() == 2\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    candidate = CandidateGenome(
        id="valid-patch",
        current_fate=CandidateFate.ELITE,
        artifact_type="code_patch",
        artifact={"path": "mod.py", "patch": "--- mod.py\n+++ mod.py\n@@ -1,2 +1,2 @@\n def value():\n-    return 1\n+    return 2\n"},
        source_bindings=[{"path": "mod.py", "symbol": "value"}],
        evidence_refs=[{"kind": "test", "path": "tests/test_mod.py", "status": "verified"}],
        obligation_delta={"targeted": ["obl_patch"], "discharged": ["obl_patch"]},
    )
    candidate.patch_application_result = {"status": "applied", "applied_files": ["mod.py"], "pre_hash": "a", "post_hash": "b"}

    result = NexusVerifierStack(project_root=repo).verify_candidate(candidate, contract=_project_contract(), current_round=8, round_limit=48)

    assert result.passed is True
    assert result.final_eligible is True
    assert candidate.verification_result["final_gate"]["diagnostics"] == []
    assert ArchiveManager().is_final_answer_eligible(candidate) is True


def test_unrelated_test_evidence_does_not_cover_source_binding(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "cognitive_evolve_runtime").mkdir()
    (repo / "mod.py").write_text("def value():\n    return 2\n", encoding="utf-8")
    (repo / "tests" / "test_candidate_genome_roundtrip.py").write_text("def test_roundtrip():\n    assert True\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    candidate = CandidateGenome(
        id="unrelated-evidence",
        current_fate=CandidateFate.ELITE,
        artifact_type="code_patch",
        artifact={"path": "mod.py", "patch": "--- mod.py\n+++ mod.py\n@@ -1,2 +1,2 @@\n def value():\n-    return 1\n+    return 2\n"},
        source_bindings=[{"path": "mod.py", "symbol": "value"}],
        evidence_refs=[{"ref_type": "test", "test_name": "test_candidate_genome_roundtrip", "status": "verified"}],
        obligation_delta={"targeted": ["obl_patch"], "discharged": ["obl_patch"]},
    )
    candidate.patch_application_result = {"status": "applied", "applied_files": ["mod.py"], "pre_hash": "a", "post_hash": "b"}

    result = NexusVerifierStack(project_root=repo).verify_candidate(candidate, contract=_project_contract(), current_round=8, round_limit=48)

    assert result.final_eligible is False
    assert "evidence_ref_not_source_relevant" in candidate.verification_result["final_gate"]["diagnostics"]
