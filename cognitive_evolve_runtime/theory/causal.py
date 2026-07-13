"""Read-only causal-estimation advisories for M6.3.

The first landing is explicitly non-identifying: it can summarize intervention-like
snapshot contrasts, but it must not claim a causal effect or control runtime
behavior.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .representations import CompletedEventSnapshot
from .signals import TheorySignal


@dataclass(frozen=True)
class InterventionAttributionAdvisory:
    factor_id: str
    outcome_id: str
    observations: int
    effect_estimate: float = 0.0
    identified: bool = False
    reason_codes: tuple[str, ...] = field(default_factory=lambda: ("non_identified_observational_snapshot",))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"reason_codes": list(self.reason_codes)}


def estimate_intervention_attribution(events: tuple[CompletedEventSnapshot, ...]) -> tuple[InterventionAttributionAdvisory, ...]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for event in events:
        metric_map = dict(event.metrics)
        for name, value in metric_map.items():
            grouped.setdefault((event.event_type, name), []).append(float(value))
    advisories: list[InterventionAttributionAdvisory] = []
    for (factor_id, outcome_id), values in sorted(grouped.items()):
        mean = sum(values) / max(1, len(values))
        advisories.append(
            InterventionAttributionAdvisory(
                factor_id=factor_id,
                outcome_id=outcome_id,
                observations=len(values),
                effect_estimate=mean,
                identified=False,
                reason_codes=("non_identified_observational_snapshot", "no_randomized_or_interventional_basis"),
            )
        )
    return tuple(advisories)


def causal_advisory_signals(events: tuple[CompletedEventSnapshot, ...]) -> tuple[TheorySignal, ...]:
    signals: list[TheorySignal] = []
    for advisory in estimate_intervention_attribution(events):
        signals.append(
            TheorySignal(
                source="causal",
                kind="diagnostic",
                target_type="outcome",
                cycle_id=events[0].cycle_id if events else "cycle:unknown",
                target_id=f"{advisory.factor_id}:{advisory.outcome_id}",
                value=0.0,
                confidence=0.0,
                provenance=("causal:non_identified_snapshot",),
                meta={
                    "factor_id": advisory.factor_id,
                    "outcome_id": advisory.outcome_id,
                    "observations": advisory.observations,
                    "identified": advisory.identified,
                    "reason_codes": list(advisory.reason_codes),
                },
            )
        )
    return tuple(signals)


__all__ = ["InterventionAttributionAdvisory", "causal_advisory_signals", "estimate_intervention_attribution"]
