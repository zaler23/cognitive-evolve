"""Stability, viability, and type-contract diagnostics for M6.7."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .representations import PopulationRepresentation
from .signals import TheorySignal


@dataclass(frozen=True)
class StabilityDiagnostic:
    cycle_id: str
    viable: bool
    risk_score: float
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"reason_codes": list(self.reason_codes)}


def diagnose_population_stability(population: PopulationRepresentation) -> StabilityDiagnostic:
    if not population.candidates:
        return StabilityDiagnostic(population.cycle_id, viable=False, risk_score=1.0, reason_codes=("empty_population",))
    missing_heavy = sum(1 for candidate in population.candidates if candidate.missing_parts or candidate.uncertainty_notes)
    risk = missing_heavy / max(1, len(population.candidates))
    reasons: list[str] = []
    if risk > 0.5:
        reasons.append("high_unresolved_gap_density")
    if len({candidate.fate for candidate in population.candidates}) <= 1:
        reasons.append("low_state_diversity")
    return StabilityDiagnostic(population.cycle_id, viable=True, risk_score=risk, reason_codes=tuple(reasons))


def stability_advisory_signals(population: PopulationRepresentation) -> tuple[TheorySignal, ...]:
    diagnostic = diagnose_population_stability(population)
    return (
        TheorySignal(
            source="stability",
            kind="diagnostic",
            target_type="population",
            cycle_id=population.cycle_id,
            target_id="population",
            value=diagnostic.risk_score,
            confidence=0.5,
            provenance=("stability:population_snapshot",),
            meta={"viable": diagnostic.viable, "reason_codes": list(diagnostic.reason_codes)},
        ),
    )


__all__ = ["StabilityDiagnostic", "diagnose_population_stability", "stability_advisory_signals"]
