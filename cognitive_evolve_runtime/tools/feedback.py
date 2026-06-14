"""Structured local tool feedback."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict, coerce_str_list, utc_now
from cognitive_evolve_runtime.validation.result import VerificationResult, verification_result_from_mapping


@dataclass
class ToolFeedback:
    tool_id: str
    status: str
    diagnostics: list[str] = field(default_factory=list)
    counterexamples: list[Any] = field(default_factory=list)
    verified_fragments: list[str] = field(default_factory=list)
    failed_fragments: list[str] = field(default_factory=list)
    cost: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    raw_output_ref: str = ""
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolFeedback":
        return cls(
            tool_id=str(data.get("tool_id") or "unknown"),
            status=str(data.get("status") or "unknown"),
            diagnostics=coerce_str_list(data.get("diagnostics")),
            counterexamples=list(data.get("counterexamples", [])),
            verified_fragments=coerce_str_list(data.get("verified_fragments")),
            failed_fragments=coerce_str_list(data.get("failed_fragments")),
            cost=coerce_dict(data.get("cost")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            raw_output_ref=str(data.get("raw_output_ref") or ""),
            created_at=str(data.get("created_at") or utc_now()),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, default=str)

    def to_verification_result(self) -> VerificationResult:
        return verification_result_from_mapping(
            {
                "status": self.status,
                "confidence": self.confidence,
                "reason": "; ".join(self.diagnostics[:5]),
                "source": self.tool_id,
                "diagnostics": list(self.diagnostics),
                "verified_fragments": list(self.verified_fragments),
                "failed_fragments": list(self.failed_fragments),
            },
            source=self.tool_id,
        )


@dataclass
class FailureMicroGuidance:
    """Compact repair directive distilled from verifier failures."""

    candidate_id: str
    blocker: str
    next_action: str
    evidence_needed: list[str] = field(default_factory=list)
    source_bindings: list[dict[str, Any]] = field(default_factory=list)
    disallowed_repeat_pattern: str = ""
    severity: str = "warning"
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailureMicroGuidance":
        return cls(
            candidate_id=str(data.get("candidate_id") or ""),
            blocker=str(data.get("blocker") or "verification_failure"),
            next_action=str(data.get("next_action") or "repair_with_concrete_evidence_delta"),
            evidence_needed=coerce_str_list(data.get("evidence_needed")),
            source_bindings=[dict(item) for item in data.get("source_bindings", []) if isinstance(item, dict)],
            disallowed_repeat_pattern=str(data.get("disallowed_repeat_pattern") or ""),
            severity=str(data.get("severity") or "warning"),
            created_at=str(data.get("created_at") or utc_now()),
        )


def failure_micro_guidance_from_diagnostics(
    *,
    candidate_id: str,
    diagnostics: list[str],
    source_bindings: list[dict[str, Any]] | None = None,
    limit: int = 5,
) -> list[FailureMicroGuidance]:
    """Create GAAPO-style micro-guidance without bloating prompts."""

    out: list[FailureMicroGuidance] = []
    seen: set[str] = set()
    for diagnostic in diagnostics:
        blocker = str(diagnostic or "").strip()
        if not blocker or blocker in seen:
            continue
        seen.add(blocker)
        out.append(
            FailureMicroGuidance(
                candidate_id=candidate_id,
                blocker=blocker,
                next_action=_next_action_for_blocker(blocker),
                evidence_needed=_evidence_needed_for_blocker(blocker),
                source_bindings=list(source_bindings or [])[:5],
                disallowed_repeat_pattern=_repeat_pattern_for_blocker(blocker),
                severity="error" if blocker.endswith("_absent") or blocker.endswith("_unverified") or "failed" in blocker else "warning",
            )
        )
        if len(out) >= max(0, int(limit or 0)):
            break
    return out


def _next_action_for_blocker(blocker: str) -> str:
    lowered = str(blocker or "").lower()
    if any(token in lowered for token in ("unified_patch_failed", "malformed patch", "unexpected eof", "unexpected end of file", "hunk", "old_text not found")):
        return "regenerate a complete unified diff against exact current source context; include valid ---/+++ headers and complete hunks for existing project-relative paths"
    if "patch_no_effect" in lowered:
        return "replace the no-op patch with a concrete source/test/schema change that applies in the sandbox and changes the project hash"
    mapping = {
        "artifact_object_absent": "produce the object-level work product required by the model-defined artifact contract; do not only describe how to produce it",
        "object_level_artifact_absent": "produce the artifact itself rather than a plan, critique, or routing note",
        "concrete_delta_absent": "attach a measurable artifact_delta showing what changed relative to the parent or prior version",
        "claim_artifact_unbound": "bind the claim to exact artifact content, artifact_delta, or evidence refs",
        "meta_commentary_only": "replace meta commentary with the concrete artifact body requested by the contract",
        "final_gate_self_certifying": "replace self-certifying final criteria with a structural check, external/referee check, or observable comparison",
        "delta_unmeasurable": "rewrite the minimum_concrete_delta so it can be observed without asking the generator to self-grade",
        "proof_object_absent": "instantiate a concrete formal_artifact; for code/runtime tasks use assertion_set with executable assertions/checks/test_cases and target_obligation_id",
        "proof_object_structurally_weak": "replace placeholder proof text with a structurally checkable object such as assertion_set.assertions containing executable assert statements",
        "ledger_non_progressing": "update obligation_delta by targeting, discharging, decomposing, or refuting a named obligation",
        "duplicate_formal_signature": "change the formal signature materially rather than paraphrasing the same object",
        "blocking_obligation_not_targeted": "target the currently blocked obligation id directly",
        "obligation_delta_absent": "attach a named obligation_delta tied to the claimed progress",
        "evidence_ref_absent": "add runtime-verifiable evidence_refs that support the decisive claim",
        "evidence_ref_unverified": "replace or verify evidence_refs with passed/source-backed evidence",
        "source_binding_absent": "bind exact files, schema fields, tests, checkpoints, or events before proposing the repair",
        "source_binding_missing_path": "replace hallucinated source bindings with exact files that exist in the project snapshot, or declare a materialization binding with a creating patch",
        "patch_target_missing": "retarget the patch to an existing project file or mark it as a deliberate new-file patch",
        "declared_new_file_not_created": "create the declared new file in the patch using a new-file diff or patch_set write operation",
        "new_file_patch_absent": "attach the concrete patch that materializes the declared new file instead of only describing it",
        "declared_new_symbol_not_created": "define the declared new function/class in the target file patch before claiming the source binding",
        "new_file_integration_absent": "add a test or existing runtime integration point that imports, calls, or otherwise exercises the new file/symbol",
        "new_file_path_out_of_scope": "move the new file into an allowed project runtime or tests path such as cognitive_evolve_runtime/... or tests/...",
        "runtime_code_change_required": "modify runtime, test, schema, or executable project files instead of only documentation",
        "runtime_code_change_absent:documentation_only_patch": "replace the docs-only patch with a concrete runtime/test/schema patch",
        "seed_note_only_patch": "stop modifying NEXUS_SEED_NOTE.md and target implementation or tests directly",
        "final_artifact_type_not_publishable": "keep the design/hybrid seed as repair material, but synthesize a concrete code_patch/project_patch with verifier-readable source bindings before final answer selection",
        "final_update_artifact_absent": "attach an applied patch, patch_set, or complete unified diff plus post-pass verification instead of only describing the project update",
        "source_binding_missing_symbol": "retarget the binding to a symbol that exists in the file or include the patch that creates that symbol before claiming final eligibility",
        "final_missing_parts_unresolved": "resolve or explicitly narrow missing_parts with evidence before allowing this candidate to become a final answer",
        "evidence_ref_not_source_relevant": "replace unrelated evidence refs with tests or source artifacts that import, exercise, or inspect the bound file and symbol",
        "final_answer_blocked_until_repaired": "repair the candidate and rerun verification before final answer selection",
        "final_answer_blocked_until_reverified": "rerun local verification after repair and clear the reverify blocker only when evidence is current",
        "final_answer_blocked_until_verified": "run the required verifier stack and attach current passing evidence before final answer selection",
    }
    return mapping.get(blocker, "repair the blocker with a concrete evidence_delta and verifier-readable artifact")


def _evidence_needed_for_blocker(blocker: str) -> list[str]:
    lowered = str(blocker or "").lower()
    if any(token in lowered for token in ("unified_patch_failed", "malformed patch", "unexpected eof", "unexpected end of file", "hunk", "old_text not found")):
        return ["complete_unified_diff", "existing_project_relative_path", "post_pass_local_verification"]
    if "patch_no_effect" in lowered:
        return ["runtime_or_test_patch", "source_binding", "post_pass_local_verification"]
    mapping = {
        "artifact_object_absent": ["object_level_artifact", "required_work_product"],
        "object_level_artifact_absent": ["object_level_artifact", "required_work_product"],
        "concrete_delta_absent": ["artifact_delta", "observable_change"],
        "claim_artifact_unbound": ["claim_to_artifact_binding", "artifact_delta"],
        "meta_commentary_only": ["artifact_body", "non_meta_output"],
        "final_gate_self_certifying": ["independent_final_gate", "structural_or_referee_check"],
        "delta_unmeasurable": ["measurable_delta", "observable_signal"],
        "proof_object_absent": ["formal_artifact", "target_obligation_id"],
        "proof_object_structurally_weak": ["formal_artifact", "structural_check", "executable_assertion"],
        "ledger_non_progressing": ["obligation_delta"],
        "duplicate_formal_signature": ["new_formal_signature"],
        "blocking_obligation_not_targeted": ["targeted_obligation_id"],
        "obligation_delta_absent": ["obligation_delta"],
        "evidence_ref_absent": ["verified_evidence_ref"],
        "evidence_ref_unverified": ["verified_evidence_ref"],
        "source_binding_absent": ["source_binding"],
        "source_binding_missing_path": ["existing_source_file", "corrected_source_binding_or_materialization_patch"],
        "patch_target_missing": ["existing_patch_target", "valid_new_file_marker"],
        "declared_new_file_not_created": ["new_file_unified_diff", "allowed_project_relative_path"],
        "new_file_patch_absent": ["materialization_patch", "changed_file_list"],
        "declared_new_symbol_not_created": ["symbol_definition", "source_binding"],
        "new_file_integration_absent": ["test_or_runtime_integration", "source_relevant_evidence_ref"],
        "new_file_path_out_of_scope": ["allowed_runtime_or_test_path"],
        "runtime_code_change_required": ["runtime_or_test_patch", "source_binding"],
        "runtime_code_change_absent:documentation_only_patch": ["runtime_or_test_patch", "source_binding"],
        "seed_note_only_patch": ["runtime_or_test_patch", "source_binding"],
        "final_artifact_type_not_publishable": ["code_patch_or_project_patch", "source_binding", "post_pass_local_verification"],
        "final_update_artifact_absent": ["applied_patch_or_patch_set", "changed_file_list", "post_pass_local_verification"],
        "source_binding_missing_symbol": ["existing_symbol_or_symbol_creating_patch", "corrected_source_binding"],
        "final_missing_parts_unresolved": ["resolved_missing_parts", "evidence_delta"],
        "evidence_ref_not_source_relevant": ["source_relevant_test_or_trace", "bound_file_or_symbol_evidence"],
        "final_answer_blocked_until_repaired": ["repair_patch", "fresh_verification"],
        "final_answer_blocked_until_reverified": ["fresh_verification", "current_evidence_ref"],
        "final_answer_blocked_until_verified": ["fresh_verification", "current_evidence_ref"],
    }
    return mapping.get(blocker, ["evidence_delta"])


def _repeat_pattern_for_blocker(blocker: str) -> str:
    lowered = str(blocker or "").lower()
    if any(token in lowered for token in ("unified_patch_failed", "malformed patch", "unexpected eof", "unexpected end of file", "hunk", "old_text not found")):
        return "do_not_repeat_the_same_malformed_or_context_stale_diff"
    if "patch_no_effect" in lowered:
        return "do_not_emit_no_op_or_comment_only_patches_for_runtime_objectives"
    if blocker in {"proof_object_absent", "proof_object_structurally_weak"}:
        return "do_not_only_describe_the_route_without_a_checkable_formal_object"
    if blocker in {"ledger_non_progressing", "obligation_delta_absent", "blocking_obligation_not_targeted"}:
        return "do_not_claim_progress_without_named_obligation_delta"
    if blocker in {"declared_new_file_not_created", "new_file_patch_absent", "declared_new_symbol_not_created", "new_file_integration_absent"}:
        return "do_not_claim_materialized_source_progress_until the patch creates the file_or_symbol and a test_or_entrypoint exercises it"
    if blocker in {"evidence_ref_absent", "evidence_ref_unverified", "source_binding_absent", "source_binding_missing_path", "patch_target_missing", "new_file_path_out_of_scope"}:
        return "do_not_rank_source_free_or_unverified_claims_as_progress"
    if blocker == "duplicate_formal_signature":
        return "do_not_emit_same_formal_signature_with_new_wording"
    if blocker in {"runtime_code_change_required", "runtime_code_change_absent:documentation_only_patch", "seed_note_only_patch"}:
        return "do_not_repeat_documentation_only_or_seed_note_changes_for_runtime_objectives"
    if blocker in {"final_artifact_type_not_publishable", "final_update_artifact_absent"}:
        return "do_not_mark_design_or_hybrid_candidates_final_without_a_concrete_patch_delta"
    if blocker in {"source_binding_missing_symbol", "evidence_ref_not_source_relevant"}:
        return "do_not_claim_source_grounded_progress_with_hallucinated_symbols_or_unrelated_tests"
    if blocker in {"final_missing_parts_unresolved", "final_answer_blocked_until_repaired", "final_answer_blocked_until_reverified", "final_answer_blocked_until_verified"}:
        return "do_not_promote_candidates_with_explicit_unresolved_final_blockers"
    return "do_not_repeat_the_failed_proposal_shape_without_new_evidence"


__all__ = ["ToolFeedback", "FailureMicroGuidance", "failure_micro_guidance_from_diagnostics"]
