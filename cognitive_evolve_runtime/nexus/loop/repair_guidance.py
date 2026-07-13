"""Repair-parent and repair-guidance helpers for Nexus rounds."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationOperator
from cognitive_evolve_runtime.nexus.obligations import candidate_source_bindings

def _repair_seed_for_parent(parent: CandidateGenome | None) -> dict[str, Any]:
    if parent is None or not isinstance(parent.metadata, dict):
        return {}
    seed = parent.metadata.get("repair_seed")
    return dict(seed) if isinstance(seed, dict) else {}


def _source_integration_points_for_parent(parent: CandidateGenome | None) -> list[dict[str, Any]]:
    if parent is None:
        return []
    points: list[dict[str, Any]] = []
    for binding in candidate_source_bindings(parent):
        path = binding.get("path")
        if path:
            points.append({"path": str(path), "kind": str(binding.get("kind") or "source_binding"), "evidence_need": "post-pass verification"})
    repair = parent.metadata.get("repair_required") if isinstance(parent.metadata, dict) else None
    if isinstance(repair, dict):
        for binding in repair.get("source_bindings", []) or []:
            if not isinstance(binding, dict):
                continue
            path = binding.get("path")
            if path:
                points.append({"path": str(path), "kind": str(binding.get("kind") or "repair_source_binding"), "evidence_need": "repair-target post-pass verification"})
    repair_seed = _repair_seed_for_parent(parent)
    for path in repair_seed.get("target_files", []) if repair_seed else []:
        if path:
            points.append({"path": str(path), "kind": "repair_seed_target", "evidence_need": "pre-fail/post-pass verification"})
    for attr, kind in (("affected_tests", "test"), ("touched_symbols", "symbol")):
        for value in getattr(parent, attr, []) or []:
            points.append({"ref": str(value), "kind": kind, "evidence_need": "pre-fail/post-pass expectation"})
    for ref in parent.evidence_refs:
        if isinstance(ref, dict):
            points.append({"ref": str(ref.get("id") or ref.get("path") or ref.get("kind") or ""), "kind": str(ref.get("kind") or "evidence_ref"), "evidence_need": "preserve-or-improve"})
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for point in points:
        key = (str(point.get("path") or point.get("ref") or ""), str(point.get("kind") or ""))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        deduped.append(point)
    return deduped


def _failure_micro_guidance_for_parent(parent: CandidateGenome | None) -> list[dict[str, Any]]:
    if parent is None:
        return []
    raw = parent.metadata.get("failure_micro_guidance") if isinstance(parent.metadata, dict) else None
    if raw is None and isinstance(parent.verification_result, dict):
        raw = parent.verification_result.get("failure_guidance")
    directives: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        blocker = str(item.get("blocker") or "").strip()
        next_action = str(item.get("next_action") or "").strip()
        if not blocker or not next_action:
            continue
        directives.append(
            {
                "blocker": blocker,
                "next_action": next_action,
                "evidence_needed": [str(value) for value in item.get("evidence_needed", []) if value],
                "source_bindings": [dict(binding) for binding in item.get("source_bindings", []) if isinstance(binding, dict)],
                "disallowed_repeat_pattern": str(item.get("disallowed_repeat_pattern") or ""),
                "severity": str(item.get("severity") or "warning"),
            }
        )
    return directives


def _repair_requirement_for_parent(parent: CandidateGenome | None) -> dict[str, Any]:
    if parent is None or not isinstance(parent.metadata, dict):
        return {}
    repair = parent.metadata.get("repair_required")
    if isinstance(repair, dict) and repair.get("blockers"):
        return dict(repair)
    decision = parent.metadata.get("stage_eligibility")
    if isinstance(decision, dict) and decision.get("repair_required") and decision.get("repair_blockers"):
        return {
            "blockers": [str(item) for item in decision.get("repair_blockers", []) if item],
            "evidence_needed": [],
            "source_bindings": [],
            "next_actions": [],
            "stage": str(decision.get("stage") or ""),
        }
    guidance = _failure_micro_guidance_for_parent(parent)
    if guidance:
        blockers = [str(item.get("blocker") or "") for item in guidance if item.get("blocker")]
        evidence_needed: list[str] = []
        source_bindings: list[dict[str, Any]] = []
        next_actions: list[str] = []
        for item in guidance:
            evidence_needed.extend(str(value) for value in item.get("evidence_needed", []) if value)
            source_bindings.extend(dict(value) for value in item.get("source_bindings", []) if isinstance(value, dict))
            action = str(item.get("next_action") or "").strip()
            if action:
                next_actions.append(action)
        return {
            "blockers": list(dict.fromkeys(blockers)),
            "evidence_needed": list(dict.fromkeys(evidence_needed)),
            "source_bindings": source_bindings,
            "next_actions": list(dict.fromkeys(next_actions)),
            "stage": "repair_guidance",
        }
    return {}


def _repair_operator_for_requirement(repair: dict[str, Any]) -> str:
    blockers = {str(item) for item in repair.get("blockers", []) if item}
    evidence_needed = {str(item) for item in repair.get("evidence_needed", []) if item}
    tokens = blockers | evidence_needed
    if tokens.intersection({"proof_object_absent", "proof_object_structurally_weak", "formal_artifact", "structural_check"}):
        return MutationOperator.INSTANTIATE_FORMAL_ARTIFACT
    if tokens.intersection({"ledger_non_progressing", "obligation_delta_absent", "blocking_obligation_not_targeted", "obligation_delta", "targeted_obligation_id"}):
        return MutationOperator.DISCHARGE_OBLIGATION
    if tokens.intersection({"evidence_ref_absent", "evidence_ref_unverified", "verified_evidence_ref", "source_binding_absent", "source_binding"}):
        return MutationOperator.TOOL_GROUND
    return MutationOperator.REPAIR


__all__ = ["_failure_micro_guidance_for_parent", "_repair_operator_for_requirement", "_repair_requirement_for_parent", "_repair_seed_for_parent", "_source_integration_points_for_parent"]
