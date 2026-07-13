"""Auditable candidate-fate transition semantics.

Existing runtime paths can still call ``CandidateGenome.mark_fate`` for backward
compatibility.  New code should prefer ``transition_candidate_fate`` when it
wants an explicit state-machine check and reason trail.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome


class IllegalFateTransition(ValueError):
    pass


VALID_TRANSITIONS: dict[str, set[str]] = {
    CandidateFate.ACTIVE.value: {
        CandidateFate.ELITE.value,
        CandidateFate.INCUBATING.value,
        CandidateFate.DORMANT.value,
        CandidateFate.AUXILIARY.value,
        CandidateFate.CULLED.value,
        CandidateFate.FAILED.value,
    },
    CandidateFate.ELITE.value: {CandidateFate.ACTIVE.value, CandidateFate.CULLED.value},
    CandidateFate.INCUBATING.value: {CandidateFate.ACTIVE.value, CandidateFate.DORMANT.value, CandidateFate.CULLED.value, CandidateFate.FAILED.value},
    CandidateFate.DORMANT.value: {CandidateFate.ACTIVE.value, CandidateFate.CULLED.value},
    CandidateFate.AUXILIARY.value: {CandidateFate.ACTIVE.value, CandidateFate.CULLED.value},
    CandidateFate.CULLED.value: set(),
    CandidateFate.FAILED.value: {CandidateFate.DORMANT.value},
}


@dataclass(frozen=True)
class FateTransition:
    previous: str
    target: str
    reason: str = ""

    def to_dict(self) -> dict[str, str]:
        data = {"previous": self.previous, "target": self.target}
        if self.reason:
            data["reason"] = self.reason
        return data


def validate_transition(current: Any, target: Any) -> FateTransition:
    previous = CandidateFate.normalize(current)
    next_fate = CandidateFate.normalize(target, default="")
    if next_fate not in CandidateFate.ALL:
        raise IllegalFateTransition(f"unknown candidate fate: {target}")
    if previous == next_fate:
        return FateTransition(previous=previous, target=next_fate)
    allowed = VALID_TRANSITIONS.get(previous, set())
    if next_fate not in allowed:
        raise IllegalFateTransition(
            f"Transition {previous} -> {next_fate} is not allowed; valid targets: {sorted(allowed)}"
        )
    return FateTransition(previous=previous, target=next_fate)


def transition_candidate_fate(candidate: CandidateGenome, target: Any, *, reason: str = "") -> CandidateGenome:
    transition = validate_transition(candidate.current_fate, target)
    candidate.mark_fate(transition.target)
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    history = metadata.get("fate_transition_history")
    if not isinstance(history, list):
        history = []
    record = FateTransition(previous=transition.previous, target=transition.target, reason=reason).to_dict()
    history.append(record)
    metadata["fate_transition_history"] = history[-20:]
    if reason:
        metadata["state_transition_reason"] = reason
    candidate.metadata = metadata
    return candidate


__all__ = ["FateTransition", "IllegalFateTransition", "VALID_TRANSITIONS", "transition_candidate_fate", "validate_transition"]
