from __future__ import annotations

from pathlib import Path

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.tools.verification_stack import NexusVerifierStack


def _project(root: Path) -> None:
    (root / "cognitive_evolve_runtime").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "cognitive_evolve_runtime" / "core.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")


def _contract() -> NexusObjectiveContract:
    return NexusObjectiveContract(
        original_user_goal="Improve runtime with materialized source change.",
        normalized_goal="runtime patch",
        expected_output_forms=["code_patch", "tests"],
        verification_preferences=["source_binding", "local_tests"],
    )


def _candidate(candidate_id: str, *, binding: dict[str, str], patch: str = "diff") -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        artifact_type="code_patch",
        artifact={"patch": patch},
        concise_claim="materialization route",
        core_mechanism="source lineage mechanism",
        source_bindings=[binding],
    )


def test_existing_file_refinement_records_advisory_lineage(tmp_path: Path) -> None:
    _project(tmp_path)
    candidate = _candidate("existing-refinement", binding={"path": "cognitive_evolve_runtime/core.py", "symbol": "value"})
    candidate.patch_application_result = {"status": "applied", "applied_files": ["cognitive_evolve_runtime/core.py"], "pre_hash": "a", "post_hash": "b"}

    result = NexusVerifierStack(project_root=tmp_path).verify(candidate, contract=_contract(), current_round=1, round_limit=48)

    assert result.final_eligible is True
    assert "source_binding_missing_path" not in result.diagnostics
    assert candidate.metadata["source_lineage_gate"]["advisory_only"] is True
    assert candidate.metadata["source_lineage_gate"]["facts"][0]["lineage_mode"] == "existing_file_refinement"


def test_missing_or_isolated_materialization_is_advisory_not_incubating(tmp_path: Path) -> None:
    _project(tmp_path)
    candidate = _candidate(
        "new-file-unmaterialized",
        binding={"path": "cognitive_evolve_runtime/new_lineage_gate.py", "symbol": "NewLineageGate", "binding_mode": "materialize"},
    )
    candidate.patch_application_result = {"status": "applied", "applied_files": ["cognitive_evolve_runtime/core.py"], "pre_hash": "a", "post_hash": "b"}

    result = NexusVerifierStack(project_root=tmp_path).verify(candidate, contract=_contract(), current_round=1, round_limit=48)
    assignments = ArchiveManager().assign_by_policy([candidate], current_round=1, round_limit=48, branch_factor=4)

    assert "declared_new_file_not_created" in result.diagnostics
    assert result.final_eligible is True
    assert assignments[0].fate in {"Elite", "Active", "Dormant"}
    assert candidate.metadata["source_lineage_gate"]["advisory_only"] is True


def test_complete_new_file_materialization_remains_final_eligible(tmp_path: Path) -> None:
    _project(tmp_path)
    candidate = _candidate(
        "new-file-complete",
        binding={"path": "cognitive_evolve_runtime/new_lineage_gate.py", "symbol": "NewLineageGate", "binding_mode": "materialize"},
        patch="new file with class NewLineageGate",
    )
    candidate.patch_application_result = {"status": "applied", "applied_files": ["cognitive_evolve_runtime/new_lineage_gate.py", "tests/test_new_lineage_gate.py"], "pre_hash": "a", "post_hash": "b"}

    result = NexusVerifierStack(project_root=tmp_path).verify(candidate, contract=_contract(), current_round=1, round_limit=48)

    assert result.final_eligible is True
    assert "new_file_integration_absent" not in result.diagnostics
    assert candidate.metadata["source_lineage_gate"]["advisory_only"] is True


def test_root_level_new_file_out_of_scope_is_advisory(tmp_path: Path) -> None:
    _project(tmp_path)
    candidate = _candidate(
        "root-file-out-of-scope",
        binding={"path": "active_evidence_obligation_ledger.py", "symbol": "Ledger", "binding_mode": "materialize"},
        patch="root file",
    )
    candidate.patch_application_result = {"status": "applied", "applied_files": ["active_evidence_obligation_ledger.py"], "pre_hash": "a", "post_hash": "b"}

    result = NexusVerifierStack(project_root=tmp_path).verify(candidate, contract=_contract(), current_round=1, round_limit=48)

    assert "new_file_path_out_of_scope" in result.diagnostics
    assert result.passed is True
    assert result.final_eligible is True
    assert candidate.metadata["stage_eligibility"]["hard_reject_reason"] == ""
