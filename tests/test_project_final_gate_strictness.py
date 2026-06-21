from __future__ import annotations

from pathlib import Path

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.tools.verification_stack import NexusVerifierStack


def _project_contract() -> NexusObjectiveContract:
    return NexusObjectiveContract(
        original_user_goal="Improve source",
        normalized_goal="improve source",
        verification_preferences=["source_binding", "local_tests"],
    )


def test_hallucinated_symbol_is_advisory_not_final_blocking(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "cognitive_evolve_runtime" / "api").mkdir(parents=True)
    (repo / "cognitive_evolve_runtime" / "api" / "jobs.py").write_text("_JOBS = {}\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    candidate = CandidateGenome(
        id="hybrid-jobqueue",
        artifact_type="hybrid",
        artifact="Design: add JobQueue.save_state/load_state.",
        concise_claim="Persist candidates across bootstrap.",
        core_mechanism="Add JobQueue state serialization.",
        source_bindings=[{"path": "cognitive_evolve_runtime/api/jobs.py", "symbol": "JobQueue"}],
        evidence_refs=[{"ref_type": "test", "test_name": "test_candidate_genome_roundtrip", "status": "verified"}],
        missing_parts=["State serialization mechanism."],
    )

    result = NexusVerifierStack(project_root=repo).verify_candidate(candidate, contract=_project_contract(), current_round=30, round_limit=48)

    assert result.passed is True
    assert result.rank_eligible is True
    assert result.final_eligible is True
    assert "source_binding_missing_symbol" in result.diagnostics
    assert result.final_gate["final_eligible"] is True
    assert "source_binding_missing_symbol" in result.final_gate["diagnostics"]


def test_final_answer_blocked_until_reverified_is_advisory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "cognitive_evolve_runtime").mkdir()
    (repo / "mod.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (repo / "tests" / "test_mod.py").write_text("def test_value():\n    assert True\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    candidate = CandidateGenome(
        id="blocked-valid-patch",
        current_fate=CandidateFate.ELITE,
        artifact_type="code_patch",
        artifact={"path": "mod.py", "patch": "diff"},
        source_bindings=[{"path": "mod.py", "symbol": "value"}],
        evidence_refs=[{"kind": "test", "path": "tests/test_mod.py", "status": "verified"}],
        metadata={"final_answer_blocked_until_reverified": True},
    )
    candidate.patch_application_result = {"status": "applied", "applied_files": ["mod.py"], "pre_hash": "a", "post_hash": "b"}

    result = NexusVerifierStack(project_root=repo).verify_candidate(candidate, contract=_project_contract(), current_round=8, round_limit=48)

    assert result.final_eligible is True
    assert "final_answer_blocked_until_reverified" in result.diagnostics
    assert result.final_gate["final_eligible"] is True


def test_unrelated_test_evidence_is_advisory_not_blocking(tmp_path: Path) -> None:
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
        artifact={"path": "mod.py", "patch": "diff"},
        source_bindings=[{"path": "mod.py", "symbol": "value"}],
        evidence_refs=[{"ref_type": "test", "test_name": "test_candidate_genome_roundtrip", "status": "verified"}],
    )
    candidate.patch_application_result = {"status": "applied", "applied_files": ["mod.py"], "pre_hash": "a", "post_hash": "b"}

    result = NexusVerifierStack(project_root=repo).verify_candidate(candidate, contract=_project_contract(), current_round=8, round_limit=48)

    assert result.final_eligible is True
    assert "evidence_ref_not_source_relevant" in result.diagnostics
