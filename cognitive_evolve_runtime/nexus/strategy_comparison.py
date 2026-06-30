"""Open carrier for comparing candidate project directions without enums."""
from __future__ import annotations

from typing import Any, Iterable

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.core.serialization import coerce_dict


def strategy_comparison_context(policy: Any = None, population: Iterable[CandidateGenome] | None = None) -> dict[str, Any]:
    metadata = coerce_dict(getattr(policy, "metadata", None) or coerce_dict(policy).get("metadata"))
    payload = coerce_dict(metadata.get("strategy_comparison"))
    observations = []
    for candidate in population or []:
        item = coerce_dict(coerce_dict(candidate.metadata).get("strategy_observation"))
        if item:
            observations.append({"candidate_id": candidate.id, **item})
    if not payload and not observations:
        return {}
    return {
        "schema": "strategy_comparison.v1",
        "open_hypotheses": list(payload.get("open_hypotheses") or [])[:8],
        "decision_questions": list(payload.get("decision_questions") or [])[:8],
        "observations": observations[:12],
        "policy": "compare_open_strategies_without_runtime_enum",
    }


__all__ = ["strategy_comparison_context"]
