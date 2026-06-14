"""MDL-style advisory producer for candidate representations."""
from __future__ import annotations

from .representations import CandidateRepresentation, PopulationRepresentation
from .signals import TheorySignal


def produce_mdl_signals(population: PopulationRepresentation) -> tuple[TheorySignal, ...]:
    lengths = {candidate.candidate_id: _description_length(candidate) for candidate in population.candidates}
    if not lengths:
        return ()
    lo = min(lengths.values())
    hi = max(lengths.values())
    span = max(1, hi - lo)
    signals: list[TheorySignal] = []
    for candidate in population.candidates:
        # Shorter descriptions receive a higher advisory prior.  This is a weak
        # prior only; config weight defaults to zero and cannot gate candidates.
        normalized = 1.0 - ((lengths[candidate.candidate_id] - lo) / span)
        signals.append(
            TheorySignal(
                source="mdl",
                kind="rank_prior",
                target_type="candidate",
                cycle_id=population.cycle_id,
                target_id=candidate.candidate_id,
                value=normalized,
                confidence=0.5,
                provenance=("mdl:description_length",),
                meta={"description_length": lengths[candidate.candidate_id]},
            )
        )
    return tuple(signals)


def _description_length(candidate: CandidateRepresentation) -> int:
    text = "|".join(
        [
            candidate.artifact_type,
            candidate.concise_claim,
            candidate.core_mechanism,
            " ".join(candidate.missing_parts),
            " ".join(candidate.uncertainty_notes),
            " ".join(candidate.novelty_descriptors),
            " ".join(candidate.niche_memberships),
        ]
    )
    structural = len(candidate.scores) * 8 + candidate.source_binding_count * 6 + candidate.evidence_ref_count * 6
    return len(text.encode("utf-8")) + structural


__all__ = ["produce_mdl_signals"]
