"""Evidence-aware mutation strategy selection."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.evaluators.evidence import progressive_evidence


def mutation_strategy_for_candidate(candidate: Any) -> dict[str, Any]:
    evidence = progressive_evidence(candidate)
    if evidence is None:
        return {"strategy": "baseline_mutation", "reason": "no_progressive_evidence"}
    if evidence.artifact_view is not None and evidence.artifact_view.status == "refolded":
        return {"strategy": "schema_clean_reemit", "reason": "artifact_refolded", "challenge_ids": [case.id for case in evidence.challenge_cases]}
    if evidence.challenge_cases:
        return {"strategy": "challenge_guided_repair", "reason": evidence.status, "challenge_ids": [case.id for case in evidence.challenge_cases], "repair_hints": list(evidence.repair_hints)}
    if evidence.score >= 0.65 and not evidence.final_eligible:
        return {"strategy": "locus_constrained_rewrite", "reason": "frontier_candidate_not_final_eligible"}
    return {"strategy": "baseline_mutation", "reason": evidence.status}


def annotate_mutation_strategies(candidates: list[Any]) -> None:
    for candidate in candidates:
        metadata = candidate.metadata if isinstance(getattr(candidate, "metadata", None), dict) else {}
        metadata["mutation_strategy"] = mutation_strategy_for_candidate(candidate)
        candidate.metadata = metadata


__all__ = ["annotate_mutation_strategies", "mutation_strategy_for_candidate"]
