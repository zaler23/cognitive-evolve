from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.nexus.failure_classifier import classify_candidate_failure, classify_recovery_eligibility
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector


def test_classifier_keeps_malformed_diff_with_real_target_repairable() -> None:
    candidate = CandidateGenome(
        id="bad-diff",
        artifact_type="code_patch",
        artifact={
            "path": "cognitive_evolve_runtime/api/executor.py",
            "unified_diff": (
                "diff --git a/cognitive_evolve_runtime/api/executor.py b/cognitive_evolve_runtime/api/executor.py\n"
                "--- a/cognitive_evolve_runtime/api/executor.py\n"
                "+++ b/cognitive_evolve_runtime/api/executor.py\n"
                "@@ -1 +1 @@\n"
                "-old\n"
            ),
        },
        failure_lessons=["patch_application_failed"],
    )
    payload = {
        "passed": False,
        "patch_result": {
            "status": "failed",
            "diagnostics": ["unified_patch_failed:patch: **** unexpected end of file in patch"],
            "failed_files": ["cognitive_evolve_runtime/api/executor.py"],
        },
    }

    verdict = classify_candidate_failure(candidate, payload)

    assert verdict.repairable is True
    assert verdict.category == "repairable_patch_syntax_or_context"
    assert verdict.repair_targets == ["cognitive_evolve_runtime/api/executor.py"]
    assert verdict.failure_guidance
    assert "complete_unified_diff" in verdict.failure_guidance[0]["evidence_needed"]


def test_recovery_eligibility_keeps_malformed_diff_with_current_target_repairable() -> None:
    candidate = CandidateGenome(
        id="recoverable-diff",
        artifact_type="code_patch",
        artifact={"path": "cognitive_evolve_runtime/nexus/loop.py"},
        failure_lessons=["patch_application_failed", "unified_patch_failed:patch: **** unexpected end of file in patch"],
        source_bindings=[{"path": "cognitive_evolve_runtime/nexus/loop.py", "kind": "source_file"}],
    )
    candidate.patch_application_result = {
        "status": "failed",
        "diagnostics": ["unified_patch_failed:patch: **** unexpected end of file in patch"],
        "failed_files": ["cognitive_evolve_runtime/nexus/loop.py"],
    }

    verdict = classify_recovery_eligibility(candidate)

    assert verdict.repairable is True
    assert verdict.repair_targets == ["cognitive_evolve_runtime/nexus/loop.py"]


def test_classifier_keeps_missing_project_path_terminal() -> None:
    candidate = CandidateGenome(
        id="missing-target",
        artifact_type="code_patch",
        artifact={"path": "nexus_model_adapter_schema_repair.py"},
        source_bindings=[{"path": "nexus_model_adapter_schema_repair.py", "kind": "source_file"}],
    )
    payload = {
        "passed": False,
        "patch_result": {
            "status": "failed",
            "diagnostics": ["source_binding_missing_path", "patch_target_missing"],
            "failed_files": ["nexus_model_adapter_schema_repair.py"],
        },
    }

    verdict = classify_candidate_failure(candidate, payload)

    assert verdict.repairable is False
    assert verdict.category == "terminal_missing_project_path"
    assert verdict.failure_guidance == []


def test_classifier_keeps_docs_only_and_seed_note_terminal() -> None:
    docs_candidate = CandidateGenome(id="docs-only", artifact_type="code_patch", artifact={"path": "docs/ROADMAP.md"})
    seed_candidate = CandidateGenome(id="seed-note", artifact_type="code_patch", artifact={"path": "NEXUS_SEED_NOTE.md"})

    assert classify_candidate_failure(docs_candidate, {"passed": False}).repairable is False
    assert classify_candidate_failure(seed_candidate, {"passed": False}).category == "terminal_seed_note_only_patch"


def test_classifier_maps_transport_failures_to_checkpoint_resume_actions() -> None:
    candidate = CandidateGenome(id="transport", artifact_type="answer", artifact="partial")

    expected = {
        "EMPTY_ASSISTANT_CONTENT from direct_http provider": "checkpoint_resume_empty_assistant",
        "TRUNCATED assistant content; finish_reason=length": "checkpoint_resume_truncated_response",
        "Timeout from direct_http provider": "checkpoint_resume_timeout",
        "HTTP 429 rate limit": "checkpoint_resume_rate_limit_429",
        "HTTP 503 service unavailable": "checkpoint_resume_provider_5xx",
    }

    for diagnostic, category in expected.items():
        verdict = classify_candidate_failure(candidate, {"diagnostics": [diagnostic]})
        assert verdict.category == category
        assert verdict.repairable is True
        assert verdict.lifecycle_action == "checkpoint_and_resume"
        assert verdict.retention == "checkpoint_resume"


def test_classifier_maps_schema_and_static_failures_to_repair_lanes() -> None:
    schema_verdict = classify_candidate_failure(
        CandidateGenome(id="schema", artifact_type="answer", artifact="{}"),
        {"diagnostics": ["ModelResponseSchemaError: response failed schema validation"]},
    )
    assert schema_verdict.category == "repairable_model_schema_or_json_contract"
    assert schema_verdict.lifecycle_action == "repair_output_contract"

    static_verdict = classify_candidate_failure(
        CandidateGenome(id="static", artifact_type="code_patch", artifact={"path": "cognitive_evolve_runtime/llm/transport.py"}),
        {
            "diagnostics": ["compileall_failed: SyntaxError"],
            "failed_files": ["cognitive_evolve_runtime/llm/transport.py"],
        },
    )
    assert static_verdict.category == "repairable_static_or_syntax_failure"
    assert static_verdict.lifecycle_action == "bounded_static_repair_lane"


def test_exhausted_incubating_repair_candidate_is_not_selected_as_parent() -> None:
    candidate = CandidateGenome(
        id="exhausted-repair",
        current_fate=CandidateFate.INCUBATING,
        metadata={
            "repair_required": {"blockers": ["unified_patch_failed"], "evidence_needed": ["complete_unified_diff"]},
            "stage_eligibility": {
                "parent_eligible": True,
                "repair_required": True,
                "repair_exhausted": True,
            },
        },
        multihead_scores={"objective_alignment": 0.8, "answer_likelihood": 0.4},
    )

    assert ParentSelector().select([candidate], ArchiveManager(), limit=1) == []
