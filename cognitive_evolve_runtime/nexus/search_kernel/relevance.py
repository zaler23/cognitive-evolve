"""Engine-computed relevance scoring for generated candidates."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus._serde import coerce_dict
from cognitive_evolve_runtime.nexus.obligations import candidate_has_obligation_or_evidence_delta
from .fingerprints import candidate_descriptor_tokens, normalize_token


def relevance_score(candidate: CandidateGenome, *, contract: Any | None = None, policy: Any | None = None, world: Any | None = None, diagnosis: Any | None = None) -> float:
    """Return a bounded relevance score independent of model self-report.

    The score is intentionally small and conservative: it asks whether the
    candidate carries a concrete artifact/mechanism/evidence path that can be
    evolved further, not whether it is already good.
    """

    score = 0.0
    if str(candidate.artifact or "").strip() or candidate.artifact not in (None, ""):
        score += 0.20
    if getattr(candidate, "patch_set", None):
        score += 0.30
    if getattr(candidate, "touched_files", None) or getattr(candidate, "expected_effects", None):
        score += 0.10
    if str(candidate.concise_claim or "").strip():
        score += 0.10
    if str(candidate.core_mechanism or "").strip():
        score += 0.15
    if candidate.source_bindings or candidate.evidence_refs or candidate.evidence_delta:
        score += 0.18
    if candidate.verification_trace or candidate.formal_artifacts or candidate.proof_obligations or candidate_has_obligation_or_evidence_delta(candidate):
        score += 0.14
    if candidate.niche_memberships or candidate.novelty_descriptors or candidate.edge_knowledge_seeds:
        score += 0.10
    score += min(0.08, 0.02 * len(candidate.mutation_history or []))
    score += _contract_overlap_bonus(candidate, contract=contract, policy=policy, world=world, diagnosis=diagnosis)
    if candidate.failure_lessons and not (candidate.evidence_delta or candidate.proof_obligations or candidate.verification_trace):
        score -= min(0.20, 0.04 * len(candidate.failure_lessons))
    metadata = coerce_dict(candidate.metadata)
    if metadata.get("hard_reject_reason") or metadata.get("terminal_reject_reason"):
        score -= 0.35
    if str(metadata.get("descriptor_cell") or "").strip():
        score += 0.03
    return max(0.0, min(1.0, score))


def candidate_is_relevant(candidate: CandidateGenome, *, floor: float = 0.35, **kwargs: Any) -> bool:
    return relevance_score(candidate, **kwargs) >= max(0.0, min(1.0, float(floor)))


def _contract_overlap_bonus(candidate: CandidateGenome, *, contract: Any | None, policy: Any | None, world: Any | None, diagnosis: Any | None) -> float:
    tokens = candidate_descriptor_tokens(candidate)
    if not tokens:
        return 0.0
    target_values: list[Any] = []
    for obj in (contract, policy, world, diagnosis):
        if obj is None:
            continue
        if hasattr(obj, "to_dict"):
            data = coerce_dict(obj.to_dict())
        elif isinstance(obj, dict):
            data = coerce_dict(obj)
        else:
            data = {name: getattr(obj, name) for name in dir(obj) if not name.startswith("_") and name in {"objective", "success_criteria", "candidate_niches", "metadata", "kind"}}
        for key in ("objective", "goal", "success_criteria", "candidate_niches", "metadata", "kind"):
            target_values.append(data.get(key))
    target_tokens: set[str] = set()
    for value in target_values:
        if isinstance(value, dict):
            iterable = list(value.keys()) + list(value.values())
        elif isinstance(value, (list, tuple, set)):
            iterable = list(value)
        else:
            iterable = [value]
        for item in iterable:
            token = normalize_token(item)
            if token:
                target_tokens.add(token)
                target_tokens.update(part for part in token.split("_") if len(part) > 2)
    if not target_tokens:
        return 0.0
    overlap = len(tokens & target_tokens) / max(1, len(target_tokens))
    return min(0.12, 0.12 * overlap)


__all__ = ["candidate_is_relevant", "relevance_score"]
