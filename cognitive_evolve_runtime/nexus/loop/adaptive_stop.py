"""Adaptive stagnation stop derived from recorded search interventions."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.ranking.parent_selection import evaluator_selection_key


def adaptive_stagnation_exhausted(
    *,
    adaptive: bool,
    best_answer_id: str,
    history: list[dict[str, Any]],
    diagnosis: SearchDiagnosis,
    policy: EvolutionPolicy | None,
    candidates: list[CandidateGenome] | None = None,
) -> bool:
    if not adaptive or not best_answer_id or policy is None:
        return False
    try:
        patience = int(policy.metadata.get("adaptive_stagnation_patience"))
    except (TypeError, ValueError):
        return False
    if patience <= 0:
        return False

    observations = [
        (
            str((item.get("ranking") or {}).get("best_final_answer_id") or ""),
            _quality_key(item.get("best_quality_key")),
            _diagnosis_actions(item.get("diagnosis")),
        )
        for item in history
        if isinstance(item, dict) and isinstance(item.get("ranking"), dict)
    ]
    current = next((candidate for candidate in candidates or [] if candidate.id == best_answer_id), None)
    observations.append((best_answer_id, _candidate_quality_key(current), set(diagnosis.recommended_actions)))
    tail: list[tuple[str, tuple[float, ...], set[str]]] = []
    latest_quality = observations[-1][1]
    for observation in reversed(observations):
        if observation[0] != best_answer_id:
            break
        quality = observation[1]
        if tail and quality and latest_quality and latest_quality > quality:
            break
        tail.append(observation)
        if quality:
            latest_quality = quality
    if len(tail) < patience:
        return False

    configured = policy.metadata.get("stagnation_interventions")
    required = {
        str(action)
        for action in (configured if isinstance(configured, list) else policy.stagnation_actions)
        if str(action) and str(action) != "continue"
    }
    applied = {action for _best, _quality, actions in tail for action in actions}
    return bool(required) and required.issubset(applied)


def _diagnosis_actions(value: Any) -> set[str]:
    if not isinstance(value, dict):
        return set()
    return {str(action) for action in value.get("recommended_actions", []) if str(action)}


def candidate_quality_key(candidate: CandidateGenome | None) -> list[float]:
    return list(_candidate_quality_key(candidate))


def _candidate_quality_key(candidate: CandidateGenome | None) -> tuple[float, ...]:
    if candidate is None:
        return ()
    tier, evaluator_score, _candidate_id = evaluator_selection_key(candidate)
    return (
        float(tier),
        float(evaluator_score),
        float(candidate.multihead_scores.get("answer_likelihood", 0.0) or 0.0),
        float(candidate.multihead_scores.get("objective_score", 0.0) or 0.0),
    )


def _quality_key(value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return ()


__all__ = ["adaptive_stagnation_exhausted", "candidate_quality_key"]
