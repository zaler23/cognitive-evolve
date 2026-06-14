"""M6 advisory-only theory layer.

Theory proposes; M5/M6 proves.  The package exports inert-by-default advisory
objects and never owns promotion, certificate, proof, or gate decisions.
"""
from __future__ import annotations

from .aggregator import aggregate_advisory_features
from .bandit import BudgetSuggestion, OperatorArmStats, suggest_budget_allocation
from .causal import InterventionAttributionAdvisory, causal_advisory_signals, estimate_intervention_attribution
from .cellular import SearchCell, build_search_cells, cellular_advisory_signals
from .config import TheoryConfig
from .errors import TheoryCancelled, TheoryProducerError, TheoryTimeout
from .geometry import GeometrySummary, descriptor, geometry_advisory_signals, summarize_population_geometry
from .layer import TheoryLayer
from .representations import CandidateRepresentation, CompletedEventSnapshot, PopulationRepresentation, build_population_representation
from .signals import AdvisoryRankingFeatures, TheorySignal, forbidden_key_paths, validate_theory_signal_json_safe
from .stability import StabilityDiagnostic, diagnose_population_stability, stability_advisory_signals
from .telemetry import THEORY_TELEMETRY_NAMESPACE, TheoryTelemetry

__all__ = [
    "AdvisoryRankingFeatures",
    "BudgetSuggestion",
    "CandidateRepresentation",
    "CompletedEventSnapshot",
    "GeometrySummary",
    "InterventionAttributionAdvisory",
    "OperatorArmStats",
    "PopulationRepresentation",
    "SearchCell",
    "StabilityDiagnostic",
    "THEORY_TELEMETRY_NAMESPACE",
    "TheoryCancelled",
    "TheoryConfig",
    "TheoryLayer",
    "TheoryProducerError",
    "TheorySignal",
    "TheoryTelemetry",
    "TheoryTimeout",
    "aggregate_advisory_features",
    "build_population_representation",
    "build_search_cells",
    "causal_advisory_signals",
    "cellular_advisory_signals",
    "descriptor",
    "diagnose_population_stability",
    "estimate_intervention_attribution",
    "forbidden_key_paths",
    "geometry_advisory_signals",
    "stability_advisory_signals",
    "summarize_population_geometry",
    "suggest_budget_allocation",
    "validate_theory_signal_json_safe",
]
