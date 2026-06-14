"""BOED / Active Inference advisory producer."""
from __future__ import annotations

from .representations import CandidateRepresentation, PopulationRepresentation
from .signals import TheorySignal


def produce_boed_signals(population: PopulationRepresentation) -> tuple[TheorySignal, ...]:
    if not population.candidates:
        return ()
    signals: list[TheorySignal] = []
    for candidate in population.candidates:
        uncertainty = _bounded((len(candidate.missing_parts) + len(candidate.uncertainty_notes)) / 6.0)
        novelty = _bounded((len(candidate.novelty_descriptors) + len(candidate.niche_memberships)) / 6.0)
        evidence_gap = 1.0 if candidate.evidence_ref_count == 0 and candidate.source_binding_count == 0 else 0.25
        plan_value = _bounded((uncertainty + novelty + evidence_gap) / 3.0)
        signals.append(
            TheorySignal(
                source="boed",
                kind="plan_value",
                target_type="candidate",
                cycle_id=population.cycle_id,
                target_id=candidate.candidate_id,
                value=plan_value,
                confidence=0.5,
                provenance=("boed:expected_information_gain",),
                meta={"uncertainty_mass": uncertainty, "novelty_mass": novelty, "evidence_gap": evidence_gap},
            )
        )
    return tuple(signals)


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = ["produce_boed_signals"]
