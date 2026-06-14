"""Stage policy constants and diagnostic classes."""
from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateFate
from cognitive_evolve_runtime.nexus.obligations import HARD_EVIDENCE_FAILURES, HARD_PROOF_FAILURES
from cognitive_evolve_runtime.nexus.source_lineage import MATERIALIZATION_HARD_DIAGNOSTICS, MATERIALIZATION_REPAIR_DIAGNOSTICS

EARLY_STAGE = "early"
MIDDLE_STAGE = "middle"
LATE_STAGE = "late"
FINAL_STAGE = "final"
STAGE_ORDER = {EARLY_STAGE: 0, MIDDLE_STAGE: 1, LATE_STAGE: 2, FINAL_STAGE: 3}

PREFINAL_REPAIR_DIAGNOSTICS = {
    # Dynamic-artifact-contract and final-gate completeness failures are final
    # answer blockers, not reasons to terminate a still-evolving route.  Before
    # final synthesis they should create repair obligations and keep useful
    # material rankable/reproducible when the candidate is otherwise non-empty.
    "required_work_product_absent",
    "allowed_artifact_shapes_absent",
    "minimum_concrete_delta_absent",
    "evaluation_dimensions_absent",
    "invalid_outputs_underconstrained",
    "final_gate_absent",
    "final_gate_self_certifying",
    "artifact_object_absent",
    "object_level_artifact_absent",
    "concrete_delta_absent",
    "claim_artifact_unbound",
    "delta_unmeasurable",
    "meta_commentary_only",
    "design_candidate_incomplete",
}

REPAIRABLE_DIAGNOSTICS = HARD_PROOF_FAILURES | HARD_EVIDENCE_FAILURES | MATERIALIZATION_REPAIR_DIAGNOSTICS | PREFINAL_REPAIR_DIAGNOSTICS | {
    "seed_not_final",
    "missing_parts",
    "auxiliary_guard",
    "declared_new_symbol_not_created",
}

HARD_REJECT_DIAGNOSTICS = {
    "contract_hash",
    "candidate tried to evolve against a stale or modified contract",
    "patch_sandbox_failed",
    "project_offspring_failed_sandbox_verification",
    "patch_no_effect",
    "runtime_code_change_required",
    "runtime_code_change_absent:documentation_only_patch",
    "seed_note_only_patch",
    "source_binding_missing_path",
    "patch_target_missing",
    "contract_objective_absent",
    *MATERIALIZATION_HARD_DIAGNOSTICS,
}

TERMINAL_FATES = {CandidateFate.CULLED.value, CandidateFate.FAILED.value}

__all__ = [
    "EARLY_STAGE", "MIDDLE_STAGE", "LATE_STAGE", "FINAL_STAGE", "STAGE_ORDER",
    "PREFINAL_REPAIR_DIAGNOSTICS", "REPAIRABLE_DIAGNOSTICS", "HARD_REJECT_DIAGNOSTICS", "TERMINAL_FATES",
]
