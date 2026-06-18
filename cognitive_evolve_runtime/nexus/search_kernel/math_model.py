"""Small mathematical primitives for the Nexus search kernel."""
from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from .descriptor_cells import descriptor_cell_key
from .fingerprints import candidate_fingerprint


def similarity(left: CandidateGenome, right: CandidateGenome) -> float:
    """Return a deterministic similarity in [0, 1]."""

    if left.id == right.id:
        return 1.0
    lf = candidate_fingerprint(left)
    rf = candidate_fingerprint(right)
    score = 0.0
    if lf.semantic_signature == rf.semantic_signature:
        score += 0.35
    if lf.artifact_signature == rf.artifact_signature:
        score += 0.25
    if lf.grounded_signature != "UNDEFINED" and lf.grounded_signature == rf.grounded_signature:
        score += 0.15
    if descriptor_cell_key(left) == descriptor_cell_key(right):
        score += 0.12
    if lf.lineage_root and lf.lineage_root == rf.lineage_root:
        score += 0.08
    if lf.failure_signature and lf.failure_signature == rf.failure_signature:
        score += 0.05
    l_tokens = set(lf.descriptor_tokens)
    r_tokens = set(rf.descriptor_tokens)
    if l_tokens or r_tokens:
        score += 0.10 * (len(l_tokens & r_tokens) / max(1, len(l_tokens | r_tokens)))
    return max(0.0, min(1.0, score))


def mmr_score(*, quality: float, max_similarity: float, relevance: float, lambda_quality: float = 0.62) -> float:
    lam = max(0.0, min(1.0, float(lambda_quality)))
    base = (0.70 * max(0.0, min(1.0, quality))) + (0.30 * max(0.0, min(1.0, relevance)))
    return (lam * base) - ((1.0 - lam) * max(0.0, min(1.0, max_similarity)))


def batch_gain(*, accepted_count: int, novel_count: int, batch_size: int, relevant_count: int) -> float:
    if batch_size <= 0:
        return 0.0
    return (0.55 * (novel_count / max(1, batch_size))) + (0.35 * (relevant_count / max(1, batch_size))) + (0.10 * min(1.0, accepted_count / max(1, batch_size)))


__all__ = ["batch_gain", "mmr_score", "similarity"]
