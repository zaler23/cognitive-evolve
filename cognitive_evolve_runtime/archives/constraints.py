"""Archive constraint and final-eligibility helpers."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.core.serialization import coerce_dict, stable_hash

def candidate_verification_blocks_final(candidate: CandidateGenome) -> bool:
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    if metadata.get("terminal_failure") or metadata.get("terminal_reject"):
        return True
    stage = coerce_dict(metadata.get("stage_eligibility"))
    if str(stage.get("hard_reject_reason") or "").strip():
        return True
    fate = str(getattr(candidate, "current_fate", "") or "").lower()
    if fate in {"failed", "culled"}:
        return True
    # Verification/source/proof/final-gate diagnostics are advisory in
    # answer-first mode and never block final answer selection.
    return False

def candidate_is_verified_dormant_frontier(candidate: CandidateGenome) -> bool:
    """Permit Dormant synthesis for answer-bearing edge/frontier material.

    Dormant remains a parking state, but answer-first synthesis may surface a
    parked candidate when it carries concrete answer material and either quality
    or frontier diversity signal. Verification diagnostics remain advisory.
    """

    if candidate_verification_blocks_final(candidate):
        return False
    scores = candidate.multihead_scores
    quality = max(
        float(scores.get("answer_likelihood", 0.0) or 0.0),
        float(scores.get("objective_alignment", 0.0) or 0.0),
        float(scores.get("evidence_progress", 0.0) or 0.0),
        float(scores.get("proof_progress", 0.0) or 0.0),
        float(scores.get("verifiability", 0.0) or 0.0),
    )
    answer_signal = bool(str(candidate.artifact or candidate.concise_claim or candidate.core_mechanism).strip())
    frontier_signal = bool(candidate.edge_knowledge_seeds or candidate.novelty_descriptors or candidate.niche_memberships)
    return answer_signal and (quality > 0.0 or frontier_signal)

def verification_failure_signature(candidate: CandidateGenome) -> str:
    return "|".join(dict.fromkeys(sorted(verification_diagnostics(candidate))))[:500]

def verification_diagnostics(candidate: CandidateGenome) -> set[str]:
    result = getattr(candidate, "verification_result", {}) or {}
    if not isinstance(result, dict):
        return set()
    diagnostics = {str(item) for item in result.get("diagnostics", []) if item}
    for section in ("proof_progress", "evidence_obligation"):
        payload = result.get(section)
        if isinstance(payload, dict):
            diagnostics.update(str(item) for item in payload.get("diagnostics", []) if item)
    return diagnostics

def constraint_target(candidate: CandidateGenome) -> str:
    if candidate.lineage:
        return str(candidate.lineage[0])
    return str(candidate.core_mechanism or candidate.concise_claim or candidate.id)

def constraint_id(kind: str, *parts: Any) -> str:
    return kind + "_" + stable_hash({"kind": kind, "parts": parts})[:16]

__all__ = [
    "candidate_is_verified_dormant_frontier",
    "candidate_verification_blocks_final",
    "constraint_id",
    "constraint_target",
    "verification_diagnostics",
    "verification_failure_signature",
]
