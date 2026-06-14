from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContractBuilder
from cognitive_evolve_runtime.inputs.text_packet import TextInputPacket
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, EvolutionRound
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult


def test_empty_population_reseeds_contract_defined_incubating_parent() -> None:
    packet = TextInputPacket.from_text("Improve the supplied artifact.")
    contract = NexusObjectiveContractBuilder().build_text_contract(user_goal="Improve the supplied artifact.", packet=packet)
    stage = EvolutionRound(model=None, budget=EvolutionBudget(max_rounds=2, branch_factor=1))
    population = CandidatePopulation([])

    stop_reason, offspring_verification, compaction = stage.reproduce(
        current_round=1,
        population=population,
        archives=ArchiveManager(),
        policy=EvolutionPolicy(candidate_niches=[]),
        contract=contract,
        world={"kind": "text"},
        rankings=RelativeRankingResult(raw_notes="empty"),
        diagnosis=SearchDiagnosis(stagnation_detected=True, stagnation_type="population_empty"),
        critiques=[],
        offspring_verifier=None,
    )

    assert stop_reason == ""
    assert offspring_verification == []
    assert isinstance(compaction, dict)
    # The parent itself is not a final answer; deterministic mutation produces
    # offspring from the Incubating reseed material and keeps the run alive.
    assert population.candidates
    assert all(candidate.metadata.get("search_seed_not_final") for candidate in population.candidates)


def test_emergency_reseed_is_final_blocked_search_material() -> None:
    from cognitive_evolve_runtime.nexus.activation_reseed import emergency_activation_reseed

    contract = NexusObjectiveContractBuilder().build_text_contract(user_goal="Revise this scene.", packet={})
    [seed] = emergency_activation_reseed(contract=contract, world={"kind": "text"}, policy=EvolutionPolicy(candidate_niches=[]), limit=1, current_round=3)

    assert seed.current_fate == CandidateFate.INCUBATING.value
    assert seed.metadata["search_seed_not_final"] is True
    assert seed.metadata["final_answer_blocked_until_repaired"] is True
    assert seed.artifact["model_defined_required_work_product"]
