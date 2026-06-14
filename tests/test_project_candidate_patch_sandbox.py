from __future__ import annotations

from pathlib import Path
import sys

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.candidates.project_candidate import PatchOperation, ProjectCandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.project_verification import ProjectCandidateVerifier
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector
from cognitive_evolve_runtime.tools.verification_stack import NexusVerifierStack
from cognitive_evolve_runtime.tools.patch_sandbox import PatchSandbox, preflight_unified_patch
from cognitive_evolve_runtime.tools.runner import ToolRunner


def test_project_candidate_patch_sandbox_applies_and_runs_compileall(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    candidate = ProjectCandidateGenome(
        id="patch1",
        patch_set=[PatchOperation(path="mod.py", operation="replace", old_text="return 1", new_text="return 2")],
    )

    result = PatchSandbox(repo, tmp_path / "sandboxes").apply(candidate)
    feedback = ToolRunner(timeout_seconds=10).run([sys.executable, "-m", "compileall", "-q", "."], cwd=result.sandbox_path)

    assert result.status == "applied"
    assert result.pre_hash != result.post_hash
    assert result.applied_files == ["mod.py"]
    assert feedback.status == "passed"
    assert candidate.patch_application_result["post_hash"] == result.post_hash


def test_project_candidate_patch_sandbox_reports_failed_replace(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("x = 1\n", encoding="utf-8")
    candidate = ProjectCandidateGenome(id="patch2", patch_set=[PatchOperation(path="mod.py", operation="replace", old_text="missing", new_text="ok")])

    result = PatchSandbox(repo, tmp_path / "sandboxes").apply(candidate)

    assert result.status == "failed"
    assert result.failed_files == ["mod.py"]
    assert "old_text not found" in result.diagnostics[0]


def test_project_verifier_applies_generic_code_patch_unified_diff(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    candidate = CandidateGenome(
        id="generic-code-patch",
        artifact_type="code_patch",
        artifact={
            "path": "mod.py",
            "patch": "--- mod.py\n+++ mod.py\n@@ -1,2 +1,2 @@\n def value():\n-    return 1\n+    return 2\n",
        },
    )

    summaries = ProjectCandidateVerifier(source_root=repo, sandbox_root=tmp_path / "sandboxes").verify_population([candidate])

    assert len(summaries) == 1
    assert summaries[0].passed is True
    assert summaries[0].patch_result["status"] == "applied"
    assert summaries[0].patch_result["applied_files"] == ["mod.py"]
    assert candidate.verification_result["passed"] is True
    sandbox_file = Path(summaries[0].patch_result["sandbox_path"]) / "mod.py"
    assert "return 2" in sandbox_file.read_text(encoding="utf-8")


def test_patch_preflight_catches_truncated_and_malformed_hunks() -> None:
    truncated = (
        "diff --git a/mod.py b/mod.py\n"
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ bad header\n"
        "+x = 1\n"
        "[truncated]"
    )

    result = preflight_unified_patch(truncated)

    assert result["ok"] is False
    joined = ";".join(result["diagnostics"])
    assert "patch_truncated:truncation_marker_present" in joined
    assert "malformed_hunk_header" in joined


def test_patch_sandbox_runs_preflight_before_patch_tool(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("x = 1\n", encoding="utf-8")
    candidate = CandidateGenome(
        id="bad-preflight",
        artifact_type="code_patch",
        artifact={
            "patch": (
                "diff --git a/mod.py b/mod.py\n"
                "--- a/mod.py\n"
                "+++ b/mod.py\n"
                "@@ bad header\n"
                "-x = 1\n"
                "+x = 2\n"
            )
        },
    )

    result = PatchSandbox(repo, tmp_path / "sandboxes").apply(candidate)

    assert result.status == "failed"
    assert any("patch_preflight_failed:malformed_hunk_header" in item for item in result.diagnostics)


def test_project_verifier_rejects_generic_code_patch_missing_target(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    candidate = CandidateGenome(
        id="generic-code-patch-missing",
        artifact_type="code_patch",
        artifact={
            "path": "missing.py",
            "patch": "--- missing.py\n+++ missing.py\n@@ -1 +1 @@\n-old\n+new\n",
        },
    )

    summaries = ProjectCandidateVerifier(source_root=repo, sandbox_root=tmp_path / "sandboxes").verify_population([candidate])

    assert len(summaries) == 1
    assert summaries[0].passed is False
    assert summaries[0].patch_result["status"] == "failed"
    assert summaries[0].patch_result["failed_files"] == ["missing.py"]
    assert candidate.current_fate == CandidateFate.FAILED.value


def test_project_candidate_patch_sandbox_rejects_empty_noop_patch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("x = 1\n", encoding="utf-8")
    candidate = ProjectCandidateGenome(id="patch-noop-empty", patch_set=[])

    result = PatchSandbox(repo, tmp_path / "sandboxes").apply(candidate)

    assert result.status == "no_op"
    assert result.passed is False
    assert "patch_no_effect:no_files_applied" in result.diagnostics


def test_project_candidate_patch_sandbox_rejects_identical_hash_noop(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("x = 1\n", encoding="utf-8")
    candidate = ProjectCandidateGenome(
        id="patch-noop-hash",
        patch_set=[PatchOperation(path="mod.py", operation="replace", old_text="x = 1", new_text="x = 1")],
    )

    result = PatchSandbox(repo, tmp_path / "sandboxes").apply(candidate)

    assert result.status == "no_op"
    assert result.applied_files == ["mod.py"]
    assert result.pre_hash == result.post_hash
    assert "patch_no_effect:pre_hash_equals_post_hash" in result.diagnostics


def test_project_verifier_marks_noop_patch_not_passed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("x = 1\n", encoding="utf-8")
    candidate = ProjectCandidateGenome(
        id="patch-noop-verifier",
        patch_set=[PatchOperation(path="mod.py", operation="replace", old_text="x = 1", new_text="x = 1")],
    )

    summary = ProjectCandidateVerifier(source_root=repo, sandbox_root=tmp_path / "sandboxes").verify(candidate)

    assert summary.passed is False
    assert candidate.verification_result["passed"] is False
    assert candidate.patch_application_result["status"] == "no_op"


def test_generic_verifier_rejects_legacy_applied_noop_patch_result() -> None:
    candidate = ProjectCandidateGenome(
        id="patch-legacy-noop",
        patch_application_result={
            "status": "applied",
            "applied_files": [],
            "pre_hash": "same",
            "post_hash": "same",
        },
    )

    result = NexusVerifierStack().verify_candidate(candidate)

    assert result.passed is False
    assert "patch_no_effect" in result.diagnostics
    assert candidate.verification_result["final_eligible"] is False


def test_generic_verifier_rejects_seed_note_only_runtime_patch() -> None:
    contract = NexusObjectiveContract(
        original_user_goal="Self-evolve the CognitiveEvolve core runtime implementation.",
        normalized_goal="improve runtime implementation and tests",
        expected_output_forms=["patch", "tests"],
        verification_preferences=["source_binding", "local_tests"],
    )
    candidate = ProjectCandidateGenome(
        id="patch-seed-note-only",
        patch_set=[PatchOperation(path="NEXUS_SEED_NOTE.md", operation="write", content="# note\n")],
        patch_application_result={
            "status": "applied",
            "applied_files": ["NEXUS_SEED_NOTE.md"],
            "pre_hash": "before",
            "post_hash": "after",
        },
        source_bindings=[{"path": "NEXUS_SEED_NOTE.md", "kind": "source_file"}],
        evidence_refs=[{"id": "seed-note", "kind": "source_file", "status": "verified"}],
        obligation_delta={"targeted": ["runtime_patch_surface"]},
    )

    result = NexusVerifierStack().verify_candidate(candidate, contract=contract, current_round=8, round_limit=48)

    assert result.passed is False
    assert "seed_note_only_patch" in result.diagnostics
    assert "runtime_code_change_required" in result.diagnostics
    assert candidate.verification_result["final_eligible"] is False


def test_docs_only_runtime_patch_is_hard_rejected_not_incubated() -> None:
    contract = NexusObjectiveContract(
        original_user_goal="Improve the core project algorithm and runtime.",
        normalized_goal="improve core runtime algorithm",
        expected_output_forms=["patch", "tests"],
        verification_preferences=["source_binding", "local_tests"],
    )
    candidate = ProjectCandidateGenome(
        id="patch-docs-only",
        patch_set=[PatchOperation(path="docs/ROADMAP.md", operation="append", content="\nMore ideas.\n")],
        patch_application_result={
            "status": "applied",
            "applied_files": ["docs/ROADMAP.md"],
            "pre_hash": "before",
            "post_hash": "after",
        },
        source_bindings=[{"path": "docs/ROADMAP.md", "kind": "source_file"}],
        evidence_refs=[{"id": "roadmap-doc", "kind": "source_file", "status": "verified"}],
        obligation_delta={"targeted": ["runtime_patch_surface"]},
    )

    result = NexusVerifierStack().verify_candidate(candidate, contract=contract, current_round=8, round_limit=48)
    archives = ArchiveManager()
    archives.update(archives.assign_by_policy([candidate], current_round=8, round_limit=48), candidates=[candidate])

    assert result.passed is False
    assert "runtime_code_change_absent:documentation_only_patch" in result.diagnostics
    assert candidate.current_fate != CandidateFate.INCUBATING.value
    assert "runtime_code_change_required" in candidate.metadata["stage_eligibility"]["hard_reject_reason"]
    assert ParentSelector().select([candidate], archives, limit=1) == []


def test_generic_code_patch_rejects_missing_source_and_patch_target(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "cognitive_evolve_runtime" / "nexus").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
    (repo / "cognitive_evolve_runtime" / "nexus" / "stage_policy.py").write_text("def parse_metric_value(value):\n    return value\n", encoding="utf-8")
    contract = NexusObjectiveContract(
        original_user_goal="Prove and implement a runtime code patch for the CognitiveEvolve core.",
        normalized_goal="prove and implement runtime code patch",
        expected_output_forms=["code_patch", "tests", "proof"],
        verification_preferences=["source_binding", "formal_artifact"],
    )
    candidate = CandidateGenome(
        id="code-patch-missing-path",
        artifact_type="code_patch",
        artifact={
            "path": "cognitive_evolve_runtime/nexus/verifier.py",
            "patch": "--- cognitive_evolve_runtime/nexus/verifier.py\n+++ cognitive_evolve_runtime/nexus/verifier.py\n@@\n-old\n+new\n",
        },
        concise_claim="Patch the verifier lock path.",
        core_mechanism="Targets an exact verifier source path.",
        formal_artifacts=[
            {
                "type": "assertion_set",
                "target_obligation_id": "obl_path_binding",
                "assertions": ["assert Path('cognitive_evolve_runtime/nexus/verifier.py').exists()"],
            }
        ],
        proof_obligations=[{"id": "obl_path_binding", "status": "targeted"}],
        obligation_delta={"targeted": ["obl_path_binding"], "discharged": ["obl_path_binding"]},
        source_bindings=[{"path": "cognitive_evolve_runtime/nexus/verifier.py", "kind": "source_file"}],
        evidence_refs=[{"id": "local-check", "kind": "verification", "status": "verified"}],
    )

    result = NexusVerifierStack(project_root=repo).verify_candidate(candidate, contract=contract, current_round=8, round_limit=48)

    assert result.passed is False
    assert "source_binding_missing_path" in result.diagnostics
    assert "patch_target_missing" in result.diagnostics
    assert candidate.verification_result["final_eligible"] is False
    assert "source_binding_missing_path" in candidate.metadata["stage_eligibility"]["hard_reject_reason"]


def test_generic_code_patch_accepts_existing_source_and_patch_target(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "cognitive_evolve_runtime" / "nexus").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
    (repo / "cognitive_evolve_runtime" / "nexus" / "stage_policy.py").write_text("def parse_metric_value(value):\n    return value\n", encoding="utf-8")
    contract = NexusObjectiveContract(
        original_user_goal="Prove and implement a runtime code patch for the CognitiveEvolve core.",
        normalized_goal="prove and implement runtime code patch",
        expected_output_forms=["code_patch", "tests", "proof"],
        verification_preferences=["source_binding", "formal_artifact"],
    )
    candidate = CandidateGenome(
        id="code-patch-existing-path",
        artifact_type="code_patch",
        artifact={
            "path": "cognitive_evolve_runtime/nexus/stage_policy.py",
            "patch": "--- cognitive_evolve_runtime/nexus/stage_policy.py\n+++ cognitive_evolve_runtime/nexus/stage_policy.py\n@@\n-return value\n+return value\n",
        },
        concise_claim="Patch the existing metric parser path.",
        core_mechanism="Targets the existing stage_policy source path.",
        formal_artifacts=[
            {
                "type": "assertion_set",
                "target_obligation_id": "obl_path_binding",
                "assertions": ["assert Path('cognitive_evolve_runtime/nexus/stage_policy.py').exists()"],
            }
        ],
        proof_obligations=[{"id": "obl_path_binding", "status": "targeted"}],
        obligation_delta={"targeted": ["obl_path_binding"], "discharged": ["obl_path_binding"]},
        source_bindings=[{"path": "cognitive_evolve_runtime/nexus/stage_policy.py", "kind": "source_file"}],
        evidence_refs=[{"id": "local-check", "kind": "verification", "status": "verified"}],
    )

    result = NexusVerifierStack(project_root=repo).verify_candidate(candidate, contract=contract, current_round=8, round_limit=48)

    assert "source_binding_missing_path" not in result.diagnostics
    assert "patch_target_missing" not in result.diagnostics
    assert result.passed is True


def test_source_patch_preflight_rejects_hallucinated_symbol_before_ranking(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "cognitive_evolve_runtime" / "api").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
    (repo / "cognitive_evolve_runtime" / "api" / "engine_runner.py").write_text("def _run_engine():\n    return 'ok'\n", encoding="utf-8")
    contract = NexusObjectiveContract(
        original_user_goal="Improve the self-evolution runtime with a code patch.",
        normalized_goal="runtime code patch",
        expected_output_forms=["code_patch", "tests"],
        verification_preferences=["source_binding", "local_tests"],
    )
    candidate = CandidateGenome(
        id="hallucinated-symbol",
        artifact_type="code_patch",
        artifact={
            "path": "cognitive_evolve_runtime/api/engine_runner.py",
            "patch": "--- cognitive_evolve_runtime/api/engine_runner.py\n+++ cognitive_evolve_runtime/api/engine_runner.py\n@@ -1,2 +1,3 @@\n def _run_engine():\n+    # keep existing behavior\n     return 'ok'\n",
        },
        source_bindings=[{"path": "cognitive_evolve_runtime/api/engine_runner.py", "symbol": "select_parents"}],
        evidence_refs=[{"kind": "test", "path": "tests/test_project_candidate_patch_sandbox.py", "status": "planned"}],
        obligation_delta={"targeted": ["obl_preflight"]},
    )

    result = NexusVerifierStack(project_root=repo).verify_candidate(candidate, contract=contract, current_round=3, round_limit=22)

    assert result.passed is False
    assert result.rank_eligible is False
    assert "source_binding_missing_symbol" in result.diagnostics
    assert candidate.metadata["source_patch_preflight"]["passed"] is False
    assert candidate.metadata["source_patch_preflight"]["missing_symbols"] == [
        {"path": "cognitive_evolve_runtime/api/engine_runner.py", "symbol": "select_parents"}
    ]


def test_source_patch_preflight_allows_patch_created_symbol_as_repair_material(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "cognitive_evolve_runtime" / "api").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
    (repo / "cognitive_evolve_runtime" / "api" / "engine_runner.py").write_text("def _run_engine():\n    return 'ok'\n", encoding="utf-8")
    contract = NexusObjectiveContract(
        original_user_goal="Improve the self-evolution runtime with a code patch.",
        normalized_goal="runtime code patch",
        expected_output_forms=["code_patch", "tests"],
        verification_preferences=["source_binding", "local_tests"],
    )
    candidate = CandidateGenome(
        id="creates-symbol",
        artifact_type="code_patch",
        artifact={
            "path": "cognitive_evolve_runtime/api/engine_runner.py",
            "patch": "--- cognitive_evolve_runtime/api/engine_runner.py\n+++ cognitive_evolve_runtime/api/engine_runner.py\n@@ -1,2 +1,5 @@\n+def select_parents(candidates):\n+    return list(candidates)\n+\n def _run_engine():\n     return 'ok'\n",
        },
        source_bindings=[{"path": "cognitive_evolve_runtime/api/engine_runner.py", "symbol": "select_parents"}],
        evidence_refs=[{"kind": "test", "path": "tests/test_project_candidate_patch_sandbox.py", "status": "planned"}],
        obligation_delta={"targeted": ["obl_preflight"]},
    )

    result = NexusVerifierStack(project_root=repo).verify_candidate(candidate, contract=contract, current_round=3, round_limit=22)

    assert result.passed is False
    assert "proof_object_absent" in result.diagnostics
    assert "source_patch_preflight" not in candidate.metadata


def test_project_verifier_applies_generic_code_patch_unified_diff_from_content(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    candidate = CandidateGenome(
        id="generic-code-patch-content",
        artifact_type="code_patch",
        artifact={
            "path": "mod.py",
            "content": "--- mod.py\n+++ mod.py\n@@ -1,2 +1,2 @@\n def value():\n-    return 1\n+    return 3\n",
        },
    )

    summaries = ProjectCandidateVerifier(source_root=repo, sandbox_root=tmp_path / "sandboxes").verify_population([candidate])

    assert len(summaries) == 1
    assert summaries[0].passed is True
    assert summaries[0].patch_result["status"] == "applied"
    assert summaries[0].patch_result["applied_files"] == ["mod.py"]
    sandbox_file = Path(summaries[0].patch_result["sandbox_path"]) / "mod.py"
    assert "return 3" in sandbox_file.read_text(encoding="utf-8")


def test_project_verifier_applies_generic_code_patch_unified_diff_alias(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    candidate = CandidateGenome(
        id="generic-code-patch-unified-diff",
        artifact_type="code_patch",
        artifact={
            "path": "mod.py",
            "unified_diff": "diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n@@ -1,2 +1,2 @@\n def value():\n-    return 1\n+    return 5\n",
        },
    )

    summaries = ProjectCandidateVerifier(source_root=repo, sandbox_root=tmp_path / "sandboxes").verify_population([candidate])

    assert summaries[0].passed is True
    assert summaries[0].patch_result["status"] == "applied"
    assert summaries[0].patch_result["applied_files"] == ["mod.py"]
    sandbox_file = Path(summaries[0].patch_result["sandbox_path"]) / "mod.py"
    assert "return 5" in sandbox_file.read_text(encoding="utf-8")


def test_generic_code_patch_content_must_look_like_unified_diff(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    candidate = CandidateGenome(
        id="generic-code-patch-narrative-content",
        artifact_type="code_patch",
        artifact={"path": "mod.py", "content": "Please change mod.py so value returns 4, but this is not a diff."},
    )

    result = PatchSandbox(repo, tmp_path / "sandboxes").apply(candidate)

    assert result.status == "no_op"
    assert result.applied_files == []
    assert "patch_no_effect:no_files_applied" in result.diagnostics
