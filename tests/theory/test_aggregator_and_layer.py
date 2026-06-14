from __future__ import annotations

from cognitive_evolve_runtime.theory import TheoryConfig, TheoryLayer
from cognitive_evolve_runtime.theory.aggregator import aggregate_advisory_features
from cognitive_evolve_runtime.theory.representations import CandidateRepresentation, PopulationRepresentation
from cognitive_evolve_runtime.theory.signals import TheorySignal


def _population() -> PopulationRepresentation:
    return PopulationRepresentation(
        cycle_id="round:1",
        candidates=(
            CandidateRepresentation(candidate_id="C1", concise_claim="short"),
            CandidateRepresentation(candidate_id="C2", concise_claim="long" * 20, missing_parts=("gap",)),
        ),
    )


def test_aggregation_is_order_invariant_and_clamped() -> None:
    cfg = TheoryConfig(enabled=True, mdl_enabled=True, boed_enabled=True, mdl_weight=10.0, boed_weight=10.0, clamp_min=-1.0, clamp_max=1.0)
    signals = [
        TheorySignal(source="mdl", kind="rank_prior", target_type="candidate", cycle_id="r1", target_id="C1", value=0.8),
        TheorySignal(source="boed", kind="plan_value", target_type="candidate", cycle_id="r1", target_id="C1", value=0.8),
    ]

    first = aggregate_advisory_features(signals, cfg)
    second = aggregate_advisory_features(list(reversed(signals)), cfg)

    assert first == second
    assert first["C1"].rank_prior == 1.0
    assert first["C1"].plan_value == 1.0


def test_theory_layer_disabled_returns_empty_and_no_telemetry() -> None:
    layer = TheoryLayer()

    assert layer.advisory_features_for_population(_population()) == {}
    assert layer.telemetry.records == []


def test_theory_layer_weight_zero_preserves_zero_features() -> None:
    cfg = TheoryConfig.from_mapping({"enabled": True, "producers": {"mdl": True, "boed": True}, "weights": {"mdl": 0.0, "boed": 0.0}})
    features = TheoryLayer().advisory_features_for_population(_population(), config=cfg)

    assert set(features) == {"C1", "C2"}
    assert all(item.rank_prior == 0.0 and item.plan_value == 0.0 for item in features.values())


def test_theory_layer_producer_timeout_or_error_returns_empty(monkeypatch) -> None:
    import cognitive_evolve_runtime.theory.layer as layer_module

    def boom(_population):
        raise RuntimeError("producer failed")

    monkeypatch.setattr(layer_module, "produce_mdl_signals", boom)
    cfg = TheoryConfig.from_mapping({"enabled": True, "producers": {"mdl": True}, "weights": {"mdl": 1.0}})

    assert TheoryLayer().advisory_features_for_population(_population(), config=cfg) == {}
