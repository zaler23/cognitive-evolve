from __future__ import annotations

from pathlib import Path

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.tools.verification_stack import NexusVerifierStack


def _project(root: Path) -> None:
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (root / "cognitive_evolve_runtime").mkdir()
    (root / "cognitive_evolve_runtime" / "__init__.py").write_text("", encoding="utf-8")
    (root / "cognitive_evolve_runtime" / "core.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_core.py").write_text("from cognitive_evolve_runtime.core import value\n\ndef test_value():\n    assert value() == 1\n", encoding="utf-8")


def _contract() -> NexusObjectiveContract:
    return NexusObjectiveContract(
        original_user_goal="Update CognitiveEvolve self-evolution with a source patch and tests.",
        normalized_goal="runtime source patch with tests",
        expected_output_forms=["code_patch", "tests"],
        verification_preferences=["source_binding", "local_tests"],
    )


def _patch_candidate(**kwargs) -> CandidateGenome:
    defaults = dict(
        id="candidate",
        artifact_type="code_patch",
        concise_claim="Update the runtime with a concrete patch.",
        core_mechanism="source-bound patch with test evidence",
        evidence_refs=[{"kind": "test", "path": "tests/test_core.py", "status": "passed"}],
        obligation_delta={"targeted": ["obl_patch"], "discharged": ["obl_patch"]},
    )
    defaults.update(kwargs)
    return CandidateGenome(**defaults)


def test_existing_file_refinement_is_not_penalized_as_non_materialization(tmp_path: Path) -> None:
    _project(tmp_path)
    candidate = _patch_candidate(
        id="existing-refinement",
        artifact={"patch": "--- cognitive_evolve_runtime/core.py\n+++ cognitive_evolve_runtime/core.py\n@@ -1,2 +1,2 @@\n def value():\n-    return 1\n+    return 2\n"},
        source_bindings=[{"path": "cognitive_evolve_runtime/core.py", "symbol": "value"}],
    )
    candidate.patch_application_result = {"status": "applied", "applied_files": ["cognitive_evolve_runtime/core.py"], "pre_hash": "a", "post_hash": "b"}

    result = NexusVerifierStack(project_root=tmp_path).verify(candidate, contract=_contract(), current_round=1, round_limit=48)

    assert result.final_eligible is True
    assert "source_binding_missing_path" not in result.diagnostics
    assert "declared_new_file_not_created" not in result.diagnostics
    assert candidate.metadata["source_lineage_gate"]["facts"][0]["lineage_mode"] == "existing_file_refinement"


def test_existing_file_extension_allows_patch_created_symbol(tmp_path: Path) -> None:
    _project(tmp_path)
    candidate = _patch_candidate(
        id="existing-extension",
        concise_claim="Add a new helper symbol to the existing runtime file.",
        artifact={"patch": "--- cognitive_evolve_runtime/core.py\n+++ cognitive_evolve_runtime/core.py\n@@ -1,2 +1,5 @@\n def value():\n     return 1\n+\n+def improved_value():\n+    return value() + 1\n--- tests/test_core.py\n+++ tests/test_core.py\n@@ -1,4 +1,7 @@\n-from cognitive_evolve_runtime.core import value\n+from cognitive_evolve_runtime.core import improved_value, value\n \n def test_value():\n     assert value() == 1\n+\n+def test_improved_value():\n+    assert improved_value() == 2\n"},
        source_bindings=[{"path": "cognitive_evolve_runtime/core.py", "symbol": "improved_value", "binding_mode": "extend"}],
        evidence_refs=[{"kind": "test", "path": "tests/test_core.py", "test_name": "test_improved_value", "status": "passed"}],
    )
    candidate.patch_application_result = {"status": "applied", "applied_files": ["cognitive_evolve_runtime/core.py", "tests/test_core.py"], "pre_hash": "a", "post_hash": "b"}

    result = NexusVerifierStack(project_root=tmp_path).verify(candidate, contract=_contract(), current_round=1, round_limit=48)

    assert result.final_eligible is True
    assert "source_binding_missing_symbol" not in result.diagnostics
    assert "declared_new_symbol_not_created" not in result.diagnostics
    assert candidate.metadata["source_lineage_gate"]["facts"][0]["lineage_mode"] == "existing_file_extension"


def test_existing_file_extension_missing_symbol_is_repairable_not_phantom_path(tmp_path: Path) -> None:
    _project(tmp_path)
    candidate = _patch_candidate(
        id="extension-unmaterialized",
        concise_claim="Add improved_value to the existing runtime file.",
        artifact={"patch": "--- cognitive_evolve_runtime/core.py\n+++ cognitive_evolve_runtime/core.py\n@@ -1,2 +1,3 @@\n def value():\n+    # TODO wire improved_value\n     return 1\n"},
        source_bindings=[{"path": "cognitive_evolve_runtime/core.py", "symbol": "improved_value", "binding_mode": "extend"}],
    )
    candidate.patch_application_result = {"status": "applied", "applied_files": ["cognitive_evolve_runtime/core.py"], "pre_hash": "a", "post_hash": "b"}

    result = NexusVerifierStack(project_root=tmp_path).verify(candidate, contract=_contract(), current_round=1, round_limit=48)
    assignments = ArchiveManager().assign_by_policy([candidate], current_round=1, round_limit=48, branch_factor=4)

    assert "declared_new_symbol_not_created" in result.diagnostics
    assert "source_binding_missing_symbol" not in result.diagnostics
    assert assignments[0].fate == CandidateFate.INCUBATING.value


def test_new_file_without_creation_patch_becomes_materialization_obligation(tmp_path: Path) -> None:
    _project(tmp_path)
    candidate = _patch_candidate(
        id="new-file-unmaterialized",
        concise_claim="Create a new lineage gate module.",
        artifact={"patch": "--- cognitive_evolve_runtime/core.py\n+++ cognitive_evolve_runtime/core.py\n@@ -1,2 +1,3 @@\n def value():\n+    # prepare hook\n     return 1\n"},
        source_bindings=[{"path": "cognitive_evolve_runtime/new_lineage_gate.py", "symbol": "NewLineageGate", "binding_mode": "materialize"}],
    )
    candidate.patch_application_result = {"status": "applied", "applied_files": ["cognitive_evolve_runtime/core.py"], "pre_hash": "a", "post_hash": "b"}

    result = NexusVerifierStack(project_root=tmp_path).verify(candidate, contract=_contract(), current_round=1, round_limit=48)
    assignments = ArchiveManager().assign_by_policy([candidate], current_round=1, round_limit=48, branch_factor=4)

    assert "declared_new_file_not_created" in result.diagnostics
    assert "new_file_patch_absent" in result.diagnostics
    assert "source_binding_missing_path" not in result.diagnostics
    assert assignments[0].fate == CandidateFate.INCUBATING.value
    repair = candidate.metadata["repair_required"]
    assert "new_file_unified_diff" in repair["evidence_needed"]


def test_complete_new_file_materialization_with_test_integration_can_pass_final(tmp_path: Path) -> None:
    _project(tmp_path)
    candidate = _patch_candidate(
        id="new-file-complete",
        artifact={"patch": "--- /dev/null\n+++ cognitive_evolve_runtime/new_lineage_gate.py\n@@ -0,0 +1,3 @@\n+class NewLineageGate:\n+    def ready(self):\n+        return True\n--- /dev/null\n+++ tests/test_new_lineage_gate.py\n@@ -0,0 +1,5 @@\n+from cognitive_evolve_runtime.new_lineage_gate import NewLineageGate\n+\n+def test_new_lineage_gate_ready():\n+    assert NewLineageGate().ready() is True\n+\n"},
        source_bindings=[{"path": "cognitive_evolve_runtime/new_lineage_gate.py", "symbol": "NewLineageGate", "binding_mode": "materialize"}],
        evidence_refs=[{"kind": "test", "path": "tests/test_new_lineage_gate.py", "test_name": "test_new_lineage_gate_ready", "status": "passed"}],
    )
    candidate.patch_application_result = {"status": "applied", "applied_files": ["cognitive_evolve_runtime/new_lineage_gate.py", "tests/test_new_lineage_gate.py"], "pre_hash": "a", "post_hash": "b"}

    result = NexusVerifierStack(project_root=tmp_path).verify(candidate, contract=_contract(), current_round=1, round_limit=48)

    assert result.final_eligible is True
    assert "source_binding_missing_path" not in result.diagnostics
    assert "declared_new_file_not_created" not in result.diagnostics
    assert "new_file_integration_absent" not in result.diagnostics
    assert candidate.metadata["source_lineage_gate"]["facts"][0]["lineage_mode"] == "new_file_materialization"


def test_isolated_new_file_materialization_is_incubating_not_final(tmp_path: Path) -> None:
    _project(tmp_path)
    candidate = _patch_candidate(
        id="new-file-isolated",
        artifact={"patch": "--- /dev/null\n+++ cognitive_evolve_runtime/new_lineage_gate.py\n@@ -0,0 +1,2 @@\n+class NewLineageGate:\n+    pass\n"},
        source_bindings=[{"path": "cognitive_evolve_runtime/new_lineage_gate.py", "symbol": "NewLineageGate", "binding_mode": "materialize"}],
        evidence_refs=[{"kind": "artifact", "path": "cognitive_evolve_runtime/new_lineage_gate.py", "status": "created"}],
    )
    candidate.patch_application_result = {"status": "applied", "applied_files": ["cognitive_evolve_runtime/new_lineage_gate.py"], "pre_hash": "a", "post_hash": "b"}

    result = NexusVerifierStack(project_root=tmp_path).verify(candidate, contract=_contract(), current_round=1, round_limit=48)
    assignments = ArchiveManager().assign_by_policy([candidate], current_round=1, round_limit=48, branch_factor=4)

    assert result.final_eligible is False
    assert "new_file_integration_absent" in result.diagnostics
    assert assignments[0].fate == CandidateFate.INCUBATING.value


def test_root_level_new_file_is_hard_rejected_as_out_of_scope(tmp_path: Path) -> None:
    _project(tmp_path)
    candidate = _patch_candidate(
        id="root-file-out-of-scope",
        artifact={"patch": "--- /dev/null\n+++ active_evidence_obligation_ledger.py\n@@ -0,0 +1,2 @@\n+class Ledger:\n+    pass\n"},
        source_bindings=[{"path": "active_evidence_obligation_ledger.py", "symbol": "Ledger", "binding_mode": "materialize"}],
    )
    candidate.patch_application_result = {"status": "applied", "applied_files": ["active_evidence_obligation_ledger.py"], "pre_hash": "a", "post_hash": "b"}

    result = NexusVerifierStack(project_root=tmp_path).verify(candidate, contract=_contract(), current_round=1, round_limit=48)
    decision = candidate.metadata["stage_eligibility"]

    assert "new_file_path_out_of_scope" in result.diagnostics
    assert decision["hard_reject_reason"].startswith("hard_reject_diagnostic:new_file_path_out_of_scope")
    assert decision["incubating"] is False
