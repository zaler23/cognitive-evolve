from __future__ import annotations

from pathlib import Path

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.project_candidate import PatchOperation, ProjectCandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.tools.verification_stack import NexusVerifierStack


def _contract() -> NexusObjectiveContract:
    return NexusObjectiveContract(
        original_user_goal="Improve runtime with code patch.",
        normalized_goal="runtime code patch",
        expected_output_forms=["code_patch", "tests"],
        verification_preferences=["source_binding", "local_tests"],
    )


def test_noop_patch_result_is_advisory_not_blocking() -> None:
    candidate = ProjectCandidateGenome(
        id="patch-legacy-noop",
        patch_application_result={"status": "applied", "applied_files": [], "pre_hash": "same", "post_hash": "same"},
        artifact="noop patch idea",
    )

    result = NexusVerifierStack().verify_candidate(candidate)

    assert result.passed is True
    assert result.final_eligible is True
    assert "patch_no_effect:no_files_applied" in result.diagnostics or "patch_no_effect:pre_hash_equals_post_hash" in result.diagnostics


def test_docs_or_seed_note_patch_is_advisory_answer_material() -> None:
    candidate = ProjectCandidateGenome(
        id="patch-docs-only",
        artifact="Documentation-only idea still carries answer context.",
        patch_set=[PatchOperation(path="docs/ROADMAP.md", operation="append", content="idea")],
        patch_application_result={"status": "applied", "applied_files": ["docs/ROADMAP.md"], "pre_hash": "a", "post_hash": "b"},
    )

    result = NexusVerifierStack().verify_candidate(candidate, contract=_contract(), current_round=8, round_limit=48)

    assert result.passed is True
    assert result.final_eligible is True
    assert "runtime_code_change_required" in result.diagnostics


def test_missing_source_and_hallucinated_symbol_are_advisory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "cognitive_evolve_runtime" / "api").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (repo / "cognitive_evolve_runtime" / "api" / "engine_runner.py").write_text("def _run_engine():\n    return 'ok'\n", encoding="utf-8")
    candidate = CandidateGenome(
        id="hallucinated-symbol",
        artifact_type="code_patch",
        artifact={"path": "cognitive_evolve_runtime/api/engine_runner.py", "patch": "diff"},
        concise_claim="Add missing selector",
        core_mechanism="source preflight",
        source_bindings=[{"path": "cognitive_evolve_runtime/api/engine_runner.py", "symbol": "select_parents"}],
    )

    result = NexusVerifierStack(project_root=repo).verify_candidate(candidate, contract=_contract(), current_round=3, round_limit=22)

    assert result.passed is True
    assert result.final_eligible is True
    assert "binding_symbol_missing" in result.diagnostics
    assert candidate.metadata["source_lineage_gate"]["advisory_only"] is True


def test_patch_created_symbol_remains_final_eligible_advisory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "cognitive_evolve_runtime" / "api").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (repo / "cognitive_evolve_runtime" / "api" / "engine_runner.py").write_text("def _run_engine():\n    return 'ok'\n", encoding="utf-8")
    candidate = CandidateGenome(
        id="creates-symbol",
        artifact_type="code_patch",
        artifact={"path": "cognitive_evolve_runtime/api/engine_runner.py", "patch": "def select_parents(candidates): return list(candidates)"},
        concise_claim="create symbol",
        core_mechanism="source extension",
        source_bindings=[{"path": "cognitive_evolve_runtime/api/engine_runner.py", "symbol": "select_parents"}],
    )
    candidate.patch_application_result = {"status": "applied", "applied_files": ["cognitive_evolve_runtime/api/engine_runner.py"], "pre_hash": "a", "post_hash": "b"}

    result = NexusVerifierStack(project_root=repo).verify_candidate(candidate, contract=_contract(), current_round=3, round_limit=22)

    assert result.passed is True
    assert result.final_eligible is True
