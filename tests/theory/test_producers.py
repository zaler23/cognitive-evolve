from __future__ import annotations

from cognitive_evolve_runtime.theory.boed import produce_boed_signals
from cognitive_evolve_runtime.theory.mdl import produce_mdl_signals
from cognitive_evolve_runtime.theory.representations import CandidateRepresentation, CompletedEventSnapshot, PopulationRepresentation
from cognitive_evolve_runtime.theory.observer import observe_completed_events


def test_mdl_is_deterministic_and_redundancy_does_not_improve_length_prior() -> None:
    population = PopulationRepresentation(
        cycle_id="r1",
        candidates=(
            CandidateRepresentation(candidate_id="short", concise_claim="short"),
            CandidateRepresentation(candidate_id="long", concise_claim="short" * 50),
        ),
    )

    first = produce_mdl_signals(population)
    second = produce_mdl_signals(population)

    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    values = {item.target_id: item.value for item in first}
    assert values["short"] >= values["long"]


def test_boed_is_bounded_and_does_not_prune_candidates() -> None:
    population = PopulationRepresentation(
        cycle_id="r1",
        candidates=(
            CandidateRepresentation(candidate_id="a", missing_parts=("gap",), uncertainty_notes=("uncertain",)),
            CandidateRepresentation(candidate_id="b"),
        ),
    )

    signals = produce_boed_signals(population)

    assert {signal.target_id for signal in signals} == {"a", "b"}
    assert all(0.0 <= signal.value <= 1.0 for signal in signals)
    assert {signal.provenance for signal in signals} == {("boed:heuristic_plan_value",)}


def test_observer_consumes_completed_snapshots_only_and_emits_advisory_namespace_ready_signals() -> None:
    event = CompletedEventSnapshot(cycle_id="r1", event_type="completed", target_id="C1", diagnostics=("needs_more_evidence",))

    signals = observe_completed_events((event,))

    assert len(signals) == 1
    assert signals[0].source == "observer"
    assert signals[0].advisory_only is True
