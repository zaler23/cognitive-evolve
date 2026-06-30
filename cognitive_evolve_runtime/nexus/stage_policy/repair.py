"""Repair requirement and hard-reject helpers for stage policy."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.nexus._serde import coerce_dict

from .constants import (
    EARLY_STAGE, FINAL_STAGE, HARD_REJECT_DIAGNOSTICS, PREFINAL_REPAIR_DIAGNOSTICS, TERMINAL_FATES,
)
from .stages import _final_token_present, _has_evidence_progress
from .types import EligibilityDecision, RepairRequirement

def repair_requirement(candidate: CandidateGenome, *, decision: EligibilityDecision | None = None) -> RepairRequirement:
    if decision is None:
        from .eligibility import stage_eligibility
        decision = stage_eligibility(candidate)
    guidance = _repair_guidance(candidate)
    evidence_needed: list[str] = []
    source_bindings: list[dict[str, Any]] = []
    next_actions: list[str] = []
    for item in guidance:
        evidence_needed.extend(str(value) for value in item.get("evidence_needed", []) if value)
        source_bindings.extend(dict(value) for value in item.get("source_bindings", []) if isinstance(value, dict))
        action = str(item.get("next_action") or "").strip()
        if action:
            next_actions.append(action)
    return RepairRequirement(
        blockers=list(dict.fromkeys(decision.repair_blockers)),
        evidence_needed=list(dict.fromkeys(evidence_needed)),
        source_bindings=_dedupe_source_bindings(source_bindings),
        next_actions=list(dict.fromkeys(next_actions)),
        acceptance_criteria=_acceptance_criteria_for_blockers(decision.repair_blockers),
        stage=decision.stage,
    )

def _hard_reject_reason(candidate: CandidateGenome, diagnostics: set[str], *, stage: str = EARLY_STAGE) -> str:
    fate = CandidateFate.normalize(getattr(candidate, "current_fate", ""))
    if fate in TERMINAL_FATES:
        return f"terminal_fate:{fate}"
    hard_diagnostics = set(diagnostics.intersection(HARD_REJECT_DIAGNOSTICS))
    if stage != FINAL_STAGE:
        hard_diagnostics.difference_update(PREFINAL_REPAIR_DIAGNOSTICS)
    if hard_diagnostics:
        return "hard_reject_diagnostic:" + ",".join(sorted(hard_diagnostics))
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    for key in ("terminal_reject_reason", "terminal_failure"):
        reason = str(metadata.get(key) or "").strip()
        if reason:
            return reason
    if bool(metadata.get("semantic_drift") or metadata.get("unrelated_drift")):
        return "unrelated_semantic_drift"
    text = _candidate_text(candidate)
    if _forbidden_phrase_asserted(text, ("second runtime", "parallel runtime", "new ranking authority")):
        return "second_runtime_or_ranking_authority"
    if _forbidden_phrase_asserted(text, ("hidden fallback", "fallback router")):
        return "hidden_fallback"
    return ""

def _repair_guidance(candidate: CandidateGenome) -> list[dict[str, Any]]:
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    raw = metadata.get("failure_micro_guidance")
    result = getattr(candidate, "verification_result", {}) or {}
    if raw is None and isinstance(result, dict):
        raw = result.get("failure_guidance")
    out: list[dict[str, Any]] = []
    for item in raw or []:
        if isinstance(item, dict):
            out.append(dict(item))
    return out

def _candidate_nonempty(candidate: CandidateGenome) -> bool:
    return bool(
        str(candidate.artifact or "").strip()
        or str(candidate.concise_claim or "").strip()
        or str(candidate.core_mechanism or "").strip()
        or candidate.formal_artifacts
        or candidate.evidence_refs
        or candidate.source_bindings
    )

def _candidate_text(candidate: CandidateGenome) -> str:
    parts = [
        candidate.concise_claim,
        candidate.core_mechanism,
        candidate.artifact if isinstance(candidate.artifact, str) else "",
        " ".join(candidate.missing_parts[:5]),
    ]
    return " ".join(str(part or "") for part in parts).lower()

def _forbidden_phrase_asserted(text: str, phrases: tuple[str, ...]) -> bool:
    negators = ("no ", "not ", "without ", "avoid ", "avoiding ", "forbid", "forbidden", "must not", "don't ", "do not ", "不能", "不要", "禁止")
    positive_verbs = ("add", "create", "introduce", "build", "use", "implement", "new", "restore", "resurrect")
    for phrase in phrases:
        start = text.find(phrase)
        if start < 0:
            continue
        before = text[max(0, start - 48) : start]
        if any(negator in before for negator in negators):
            continue
        if any(verb in before.split()[-8:] for verb in positive_verbs):
            return True
        # A bare self-description like "second runtime/ranking authority" is
        # still a violation; a constraint sentence is filtered above.
        return True
    return False

def _explicit_final_claim(candidate: CandidateGenome, text: str) -> bool:
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    if metadata.get("objective_solved") or metadata.get("final_claim") or metadata.get("claims_final_answer"):
        return True
    final_tokens = (
        "final answer",
        "complete proof",
        "proved the theorem",
        "the theorem is proved",
        "objective solved",
        "closed proof",
        "完整证明",
        "最终答案",
        "已经证明",
    )
    return any(_final_token_present(text, token) for token in final_tokens)

def _acceptance_criteria_for_blockers(blockers: list[str]) -> list[str]:
    mapping = {
        "artifact_object_absent": "add the object-level artifact required by the model-defined artifact contract",
        "required_work_product_absent": "materialize the model-defined work product instead of only describing it",
        "allowed_artifact_shapes_absent": "declare the artifact shape expected for this run",
        "minimum_concrete_delta_absent": "state the minimum observable delta that makes this candidate progress",
        "evaluation_dimensions_absent": "add model-defined evaluation dimensions for ranking this candidate",
        "invalid_outputs_underconstrained": "state which empty/meta/restatement outputs must be rejected",
        "final_gate_absent": "add final-answer criteria, while keeping this candidate non-final until verified",
        "concrete_delta_absent": "state a measurable artifact delta relative to the parent or prior version",
        "claim_artifact_unbound": "bind the candidate claim to exact content in the artifact or delta",
        "meta_commentary_only": "replace meta commentary with the artifact itself",
        "final_gate_self_certifying": "replace self-certifying final criteria with a structural, tool, referee, or comparison check",
        "design_candidate_incomplete": "complete mechanism, evaluation dimensions, design diff, and failure conditions",
        "proof_object_absent": "add a verifier-readable formal_artifact",
        "proof_object_structurally_weak": "replace placeholder proof text with a structurally checkable formal_artifact",
        "ledger_non_progressing": "change obligation_delta by targeting/discharging/decomposing/refuting a named obligation",
        "duplicate_formal_signature": "emit a materially new formal signature",
        "blocking_obligation_not_targeted": "target the currently blocked obligation id",
        "obligation_delta_absent": "attach non-empty obligation_delta tied to the claim",
        "evidence_ref_absent": "add runtime-verifiable evidence_refs",
        "evidence_ref_unverified": "make evidence_refs verified or remove the unsupported claim",
        "source_binding_absent": "bind exact source file/schema/test/checkpoint/event refs",
        "source_binding_missing_path": "replace hallucinated file refs with paths that exist in the snapshot",
        "patch_target_missing": "retarget patch to an existing file or an explicit new-file diff",
        "runtime_code_change_required": "modify runtime, test, schema, or executable project files rather than only documentation",
        "runtime_code_change_absent:documentation_only_patch": "replace the docs-only patch with a concrete runtime/test/schema patch",
        "seed_note_only_patch": "stop modifying NEXUS_SEED_NOTE.md and target implementation or tests directly",
        "seed_not_final": "optional: sharpen the initial seed into a clearer direct answer",
        "missing_parts": "resolve or narrow the candidate missing_parts",
    }
    criteria = [mapping.get(blocker, f"repair blocker {blocker}") for blocker in blockers]
    return list(dict.fromkeys(criteria))

def _dedupe_source_bindings(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (str(item.get("path") or ""), str(item.get("ref") or ""), str(item.get("kind") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out[:8]

__all__ = ["repair_requirement"]
