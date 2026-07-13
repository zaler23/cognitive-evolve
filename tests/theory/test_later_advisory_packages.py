from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cognitive_evolve_runtime.theory import (
    AdvisoryRankingFeatures,
    BudgetSuggestion,
    OperatorArmStats,
    TheoryConfig,
    TheoryLayer,
)
from cognitive_evolve_runtime.theory.bandit import suggest_budget_allocation
from cognitive_evolve_runtime.theory.causal import causal_advisory_signals, estimate_intervention_attribution
from cognitive_evolve_runtime.theory.cellular import build_search_cells, cellular_advisory_signals
from cognitive_evolve_runtime.theory.geometry import geometry_advisory_signals, summarize_population_geometry
from cognitive_evolve_runtime.theory.representations import CandidateRepresentation, CompletedEventSnapshot, PopulationRepresentation
from cognitive_evolve_runtime.theory.stability import diagnose_population_stability, stability_advisory_signals

ROOT = Path(__file__).resolve().parents[2]


def _population() -> PopulationRepresentation:
    return PopulationRepresentation(
        cycle_id="round:later",
        candidates=(
            CandidateRepresentation(candidate_id="patch", artifact_type="code_patch", concise_claim="tight patch", novelty_descriptors=("repair",), source_binding_count=1),
            CandidateRepresentation(candidate_id="essay", artifact_type="article", concise_claim="long essay plan", niche_memberships=("narrative",), evidence_ref_count=1),
            CandidateRepresentation(candidate_id="gap", artifact_type="code_patch", concise_claim="gap", missing_parts=("test",), uncertainty_notes=("uncertain",)),
        ),
    )


def test_causal_package_is_non_identifying_and_advisory_only() -> None:
    events = (
        CompletedEventSnapshot(cycle_id="r1", event_type="mutate", target_id="C1", metrics=(("score", 0.2),)),
        CompletedEventSnapshot(cycle_id="r1", event_type="mutate", target_id="C2", metrics=(("score", 0.6),)),
    )

    advisories = estimate_intervention_attribution(events)
    signals = causal_advisory_signals(events)

    assert advisories
    assert all(item.identified is False for item in advisories)
    assert all("non_identified_observational_snapshot" in item.reason_codes for item in advisories)
    assert signals
    assert all(signal.source == "causal" for signal in signals)
    assert all(signal.advisory_only is True and signal.confidence == 0.0 and signal.value == 0.0 for signal in signals)
    assert all("gate" not in signal.to_dict() and "certificate" not in signal.to_dict() for signal in signals)


def test_geometry_package_summarizes_diversity_without_archive_access() -> None:
    population = _population()

    summary = summarize_population_geometry(population)
    signals = geometry_advisory_signals(population)

    assert summary.candidate_count == 3
    assert 0.0 <= summary.coverage <= 1.0
    assert 0.0 <= summary.mean_pairwise_distance <= 1.0
    assert {signal.target_id for signal in signals} == {"patch", "essay", "gap"}
    assert all(signal.source == "geometry" and signal.kind == "diversity" for signal in signals)


def test_cellular_package_consumes_sidecar_features_without_runtime_control() -> None:
    population = _population()
    features = {"patch": AdvisoryRankingFeatures(candidate_id="patch", rank_prior=0.2, plan_value=0.3, diversity=0.4)}

    cells = build_search_cells(population, features)
    signals = cellular_advisory_signals(population, features)

    assert cells
    assert signals
    assert all(signal.source == "cellular" and signal.advisory_only is True for signal in signals)
    source = (ROOT / "cognitive_evolve_runtime" / "theory" / "cellular.py").read_text(encoding="utf-8")
    imports = [node.module or "" for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)]
    assert ".mdl" not in imports and ".boed" not in imports and ".observer" not in imports


def test_bandit_package_emits_suggestions_not_budget_mutations() -> None:
    arms = (
        OperatorArmStats(arm_id="known", pulls=10, reward_sum=4.0, risk_sum=1.0),
        OperatorArmStats(arm_id="new", pulls=0, reward_sum=0.0, risk_sum=0.0),
    )

    suggestions = suggest_budget_allocation(arms)

    assert {item.arm_id for item in suggestions} == {"known", "new"}
    assert all(item.advisory_only is True for item in suggestions)
    assert any("unexplored_arm" in item.reason_codes for item in suggestions if item.arm_id == "new")
    with pytest.raises(ValueError, match="advisory"):
        BudgetSuggestion(arm_id="bad", suggestion_score=0.0, advisory_only=False)


def test_stability_package_is_diagnostic_only() -> None:
    population = PopulationRepresentation(
        cycle_id="r1",
        candidates=(
            CandidateRepresentation(candidate_id="a", fate="Dormant", missing_parts=("x",)),
            CandidateRepresentation(candidate_id="b", fate="Dormant", uncertainty_notes=("y",)),
        ),
    )

    diagnostic = diagnose_population_stability(population)
    signals = stability_advisory_signals(population)

    assert diagnostic.viable is True
    assert diagnostic.risk_score == 1.0
    assert "low_state_diversity" in diagnostic.reason_codes
    assert signals[0].source == "stability"
    assert signals[0].kind == "diagnostic"
    assert signals[0].target_type == "population"


def test_later_packages_are_opt_in_through_layer_methods() -> None:
    population = _population()
    events = (CompletedEventSnapshot(cycle_id="r1", event_type="eval", target_id="C1", metrics=(("score", 1.0),)),)
    layer = TheoryLayer()
    disabled = TheoryConfig()

    assert layer.causal_advisories(events, config=disabled) == ()
    assert layer.cellular_advisories(population, {}, config=disabled) == ()
    assert layer.budget_suggestions((OperatorArmStats("arm"),), config=disabled) == ()
    assert layer.stability_advisories(population, config=disabled) == ()

    enabled = TheoryConfig.from_mapping(
        {
            "enabled": True,
            "producers": {"causal": True, "cellular": True, "bandit": True, "stability": True},
        }
    )
    assert layer.causal_advisories(events, config=enabled)
    assert layer.cellular_advisories(population, {}, config=enabled)
    assert layer.budget_suggestions((OperatorArmStats("arm"),), config=enabled)
    assert layer.stability_advisories(population, config=enabled)
