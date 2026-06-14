"""Archive constraint and final-eligibility helpers."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash
from cognitive_evolve_runtime.nexus.final_gate import FINAL_BLOCKING_METADATA_FLAGS, final_gate_summary
from cognitive_evolve_runtime.nexus.obligations import HARD_EVIDENCE_FAILURES, HARD_PROOF_FAILURES

def candidate_verification_blocks_final(candidate: CandidateGenome) -> bool:
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    if any(bool(metadata.get(key)) for key in FINAL_BLOCKING_METADATA_FLAGS):
        return True
    result = getattr(candidate, "verification_result", {}) or {}
    if not isinstance(result, dict) or not result:
        return False
    if result.get("passed") is False:
        return True
    if result.get("final_eligible") is False or result.get("rank_eligible") is False:
        return True
    diagnostics = verification_diagnostics(candidate)
    hard = HARD_PROOF_FAILURES | HARD_EVIDENCE_FAILURES
    if diagnostics.intersection(hard):
        return True
    stored_final_gate = result.get("final_gate") if isinstance(result, dict) else None
    if isinstance(stored_final_gate, dict) and stored_final_gate.get("final_eligible") is True:
        return False
    final_summary = final_gate_summary(candidate)
    return final_summary.required and not final_summary.final_eligible

def candidate_is_verified_dormant_frontier(candidate: CandidateGenome) -> bool:
    """Permit Dormant final synthesis only for verified edge/frontier material.

    Dormant is normally a parking state, not a final-answer state.  The only
    exception is the edge-knowledge case observed in real runs: a rare but
    verified candidate may be dormant because the archive wants diversity, yet
    it is still the best admissible answer.  This predicate makes that exception
    explicit and blocks the old deadlock where any Dormant candidate could be
    synthesized merely because it was preserved.
    """

    if candidate_verification_blocks_final(candidate):
        return False
    result = getattr(candidate, "verification_result", {}) or {}
    if not isinstance(result, dict) or not result:
        return False
    if result.get("passed") is not True:
        return False
    if result.get("final_eligible") is False or result.get("rank_eligible") is False:
        return False
    if candidate.metadata.get("source_grounding_required") and not (
        candidate.source_bindings or candidate.evidence_refs or candidate.evidence_delta
    ):
        return False
    scores = candidate.multihead_scores
    quality = max(
        float(scores.get("answer_likelihood", 0.0) or 0.0),
        float(scores.get("objective_alignment", 0.0) or 0.0),
        float(scores.get("evidence_progress", 0.0) or 0.0),
        float(scores.get("proof_progress", 0.0) or 0.0),
        float(scores.get("verifiability", 0.0) or 0.0),
    )
    frontier_signal = bool(
        candidate.edge_knowledge_seeds
        or candidate.formal_artifacts
        or candidate.obligation_delta
        or candidate.evidence_delta
        or candidate.evidence_refs
        or candidate.source_bindings
    )
    return quality > 0.0 and frontier_signal

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
