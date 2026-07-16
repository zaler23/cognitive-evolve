from __future__ import annotations

from copy import deepcopy

from cognitive_evolve_runtime.archives.manager import ArchiveManager, FateAssignment
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.evaluators.evidence import select_preliminary_incumbent
from cognitive_evolve_runtime.nexus.live_store import LiveNexusStore
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, EvolutionRound, evolve_once
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.representation_shadow import (
    DeterministicStubRepresentationProvider,
    RepresentationShadowLayer,
    RepresentationVectorStore,
)
from cognitive_evolve_runtime.persistence.checkpoint import CheckpointStore


def _candidates() -> list[CandidateGenome]:
    return [
        CandidateGenome(
            id="A",
            artifact="alpha mechanism",
            concise_claim="alpha",
            core_mechanism="alpha",
            multihead_scores={"objective_alignment": 0.9, "answer_likelihood": 0.8},
            created_at="2026-01-01T00:00:00+00:00",
        ),
        CandidateGenome(
            id="B",
            artifact="alpha variation",
            concise_claim="alpha variation",
            core_mechanism="alpha",
            multihead_scores={"objective_alignment": 0.8, "answer_likelihood": 0.7},
            created_at="2026-01-01T00:00:00+00:00",
        ),
        CandidateGenome(
            id="C",
            artifact="distant mechanism",
            concise_claim="distant",
            core_mechanism="distant",
            multihead_scores={"objective_alignment": 0.7, "answer_likelihood": 0.6},
            created_at="2026-01-01T00:00:00+00:00",
        ),
    ]


def _provider(*, revision: str = "fixture-v1", vectors: dict[str, list[float]] | None = None) -> DeterministicStubRepresentationProvider:
    return DeterministicStubRepresentationProvider(
        vectors=vectors
        or {
            "A": [1.0, 0.0],
            "B": [0.999, 0.001],
            "C": [-1.0, 0.0],
        },
        provider="deterministic-stub",
        model_revision=revision,
        dimension=2,
        normalization="none",
    )


def _active_assignments(candidates: list[CandidateGenome]) -> list[FateAssignment]:
    return [FateAssignment(candidate.id, CandidateFate.ACTIVE.value) for candidate in candidates]


def test_no_provider_leaves_generation_plan_and_event_without_shadow_fields() -> None:
    population = CandidatePopulation(_candidates())
    round_pipeline = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=1, branch_factor=2))

    evaluation = round_pipeline.evaluate(
        current_round=1,
        population=population,
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="answer", normalized_goal="answer"),
    )

    assert "representation_shadow" not in evaluation.generation_plan
    assert "representation_shadow" not in evaluation.progress_event["metadata"]
    assert round_pipeline.representation_store is None


def test_stub_shadow_features_enter_plan_and_event_with_frozen_version() -> None:
    population = CandidatePopulation(_candidates())
    provider = _provider()
    round_pipeline = EvolutionRound(
        model=None,
        budget=EvolutionBudget(max_rounds=1, branch_factor=2),
        representation_provider=provider,
    )

    evaluation = round_pipeline.evaluate(
        current_round=1,
        population=population,
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="answer", normalized_goal="answer"),
    )
    shadow = evaluation.generation_plan["representation_shadow"]

    assert shadow == evaluation.progress_event["metadata"]["representation_shadow"]
    assert shadow["representation_id"] == "deterministic-stub:fixture-v1:dim=2:normalization=none"
    assert shadow["mode"] == "shadow_only"
    assert shadow["activation_gate"]["activated_for_selection"] is False
    assert shadow["activation_gate"]["selection_consumers"] == []
    assert set(shadow["features"]) == {
        candidate.id
        for candidate in population.candidates
        if candidate.current_fate in {CandidateFate.ACTIVE.value, CandidateFate.ELITE.value, CandidateFate.INCUBATING.value}
    }
    assert all("novelty" in item and "local_density" in item and "neighbors" in item for item in shadow["features"].values())
    assert shadow["measurements"]["lexical_distance_correlation"]["pair_count"] > 0
    assert "leave_one_out_neighbor_rank_stability" in shadow["measurements"]


def test_representation_version_change_keeps_vector_namespaces_isolated() -> None:
    candidates = _candidates()
    store = RepresentationVectorStore()
    first = RepresentationShadowLayer(_provider(), store=store, top_k=1).observe(
        round_index=1,
        candidates=candidates,
        fate_assignments=_active_assignments(candidates),
    )
    second_provider = _provider(
        revision="fixture-v2",
        vectors={
            "A": [1.0, 0.0],
            "B": [-1.0, 0.0],
            "C": [0.999, 0.001],
        },
    )
    second = RepresentationShadowLayer(second_provider, store=store, top_k=1).observe(
        round_index=2,
        candidates=candidates,
        fate_assignments=_active_assignments(candidates),
    )

    assert first["representation_id"] != second["representation_id"]
    assert set(store.representation_ids()) == {first["representation_id"], second["representation_id"]}
    assert first["features"]["A"]["neighbors"][0]["candidate_id"] == "B"
    assert second["features"]["A"]["neighbors"][0]["candidate_id"] == "C"
    assert store.get(first["representation_id"], "B") != store.get(second["representation_id"], "B")


def test_shadow_features_do_not_change_selection_allocation_incumbent_or_stop() -> None:
    base_candidates = _candidates()
    contract = NexusObjectiveContract(original_user_goal="answer", normalized_goal="answer")

    plain_population = CandidatePopulation(deepcopy(base_candidates))
    shadow_population = CandidatePopulation(deepcopy(base_candidates))
    plain_archives = ArchiveManager()
    shadow_archives = ArchiveManager()
    plain_round = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=2, branch_factor=2))
    shadow_round = EvolutionRound(
        model=None,
        budget=EvolutionBudget(max_rounds=2, branch_factor=2),
        representation_provider=_provider(),
    )
    plain = plain_round.evaluate(
        current_round=1,
        population=plain_population,
        archives=plain_archives,
        policy=EvolutionPolicy(),
        contract=contract,
    )
    shadow = shadow_round.evaluate(
        current_round=1,
        population=shadow_population,
        archives=shadow_archives,
        policy=EvolutionPolicy(),
        contract=contract,
    )

    assert shadow.rankings.to_dict() == plain.rankings.to_dict()
    assert [(item.id, item.current_fate) for item in shadow_population.candidates] == [
        (item.id, item.current_fate) for item in plain_population.candidates
    ]
    assert shadow.stop_reason == plain.stop_reason
    assert shadow.generation_plan["plan_id"] == plain.generation_plan["plan_id"]
    assert [item.metadata["generation_plan_id"] for item in shadow_population.candidates] == [
        item.metadata["generation_plan_id"] for item in plain_population.candidates
    ]
    assert getattr(select_preliminary_incumbent(shadow_population.candidates), "id", None) == getattr(
        select_preliminary_incumbent(plain_population.candidates), "id", None
    )

    plain_reproduction = plain_round.reproduce(
        current_round=1,
        population=plain_population,
        archives=plain_archives,
        policy=plain.policy,
        contract=contract,
        world=object(),
        rankings=plain.rankings,
        diagnosis=plain.diagnosis,
        critiques=plain.critiques,
        offspring_verifier=None,
        repair_parent_candidates=plain.repair_parent_candidates,
    )
    shadow_reproduction = shadow_round.reproduce(
        current_round=1,
        population=shadow_population,
        archives=shadow_archives,
        policy=shadow.policy,
        contract=contract,
        world=object(),
        rankings=shadow.rankings,
        diagnosis=shadow.diagnosis,
        critiques=shadow.critiques,
        offspring_verifier=None,
        repair_parent_candidates=shadow.repair_parent_candidates,
    )

    assert shadow_reproduction[0] == plain_reproduction[0]
    assert shadow_round.last_generation_plan["parent_ids"] == plain_round.last_generation_plan["parent_ids"]
    assert shadow_round.last_generation_plan["productive_branch_allocation"] == plain_round.last_generation_plan["productive_branch_allocation"]
    assert shadow_round.last_generation_plan["logical_islands"] == plain_round.last_generation_plan["logical_islands"]


def test_synthetic_low_density_region_is_recovered_by_shadow_density() -> None:
    candidates = _candidates()
    shadow = RepresentationShadowLayer(_provider(), top_k=1).observe(
        round_index=1,
        candidates=candidates,
        fate_assignments=_active_assignments(candidates),
    )

    assert shadow["features"]["C"]["local_density"] < shadow["features"]["A"]["local_density"]
    assert shadow["features"]["C"]["local_density"] < shadow["features"]["B"]["local_density"]
    assert shadow["measurements"]["low_density_region_recovery"]["lowest_density_candidate_ids"] == ["C"]


def test_checkpoint_replay_preserves_vectors_hashes_neighbors_and_features(tmp_path) -> None:
    candidates = _candidates()
    provider = _provider()
    store = RepresentationVectorStore()
    first = RepresentationShadowLayer(provider, store=store, top_k=2).observe(
        round_index=1,
        candidates=candidates,
        fate_assignments=_active_assignments(candidates),
    )
    calls_after_first_observation = provider.call_count
    checkpoint_store = CheckpointStore(tmp_path / "checkpoint.json")
    checkpoint_store.save_state(
        round=1,
        max_rounds=2,
        population=CandidatePopulation(candidates),
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="answer", normalized_goal="answer"),
        search_kernel={"representation_shadow_store": store.to_dict()},
    )

    restored = checkpoint_store.restore_state()
    assert restored is not None
    replay_store = RepresentationVectorStore.from_dict(restored["search_kernel"]["representation_shadow_store"])
    replay = RepresentationShadowLayer(provider, store=replay_store, top_k=2).observe(
        round_index=1,
        candidates=candidates,
        fate_assignments=_active_assignments(candidates),
    )

    assert provider.call_count == calls_after_first_observation
    assert replay_store.to_dict() == store.to_dict()
    assert replay["features"] == first["features"]
    assert replay["measurements"] == first["measurements"]


def test_runtime_live_checkpoint_keeps_vectors_in_independent_shadow_store(tmp_path) -> None:
    candidates = _candidates()
    population = CandidatePopulation(candidates)
    archives = ArchiveManager()
    policy = EvolutionPolicy()
    contract = NexusObjectiveContract(original_user_goal="answer", normalized_goal="answer")
    budget = EvolutionBudget(max_rounds=1, branch_factor=2)
    provider = _provider()
    observer = LiveNexusStore(
        tmp_path,
        mode="text",
        contract=contract,
        world={},
        max_rounds=1,
        budget=budget.to_dict(),
    )

    result = evolve_once(
        population=population,
        archives=archives,
        policy=policy,
        contract=contract,
        world={},
        budget=budget,
        observer=observer,
        representation_provider=provider,
    )
    restored = CheckpointStore(tmp_path / "checkpoint.json").restore_state()

    assert restored is not None
    assert restored["search_kernel"]["representation_shadow_store"] == result.representation_store
    assert "representation_store" not in result.to_dict()
    assert "representation_shadow" in result.budget_history[0]["generation_plan"]
    assert "representation_shadow" in result.progress_events[0]["metadata"]
    assert all("representation_shadow" not in candidate.metadata for candidate in result.population.candidates)
