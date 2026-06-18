from __future__ import annotations

from cognitive_evolve_runtime.candidates.crossover import descriptor_tokens, jaccard_similarity, neighborhood_crossover_partner
from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationOperator, MutationPlan
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.loop.offspring import _generate_offspring
from cognitive_evolve_runtime.nexus.loop.round import _cell_activation_map
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.v23_theory_config import CACrossoverConfig


class _World:
    kind = "text"


def _candidate(cid: str, mechanism: str, *, niche: str, search: float = 0.5, final: float = 0.5) -> CandidateGenome:
    return CandidateGenome(
        id=cid,
        artifact=f"{mechanism} artifact",
        concise_claim=mechanism,
        core_mechanism=mechanism,
        niche_memberships=[niche],
        novelty_descriptors=[mechanism],
        multihead_scores={
            "frontier_score": search,
            "continuation_value": search,
            "objective_alignment": final,
            "answer_likelihood": final,
            "core_mechanism_strength": final,
        },
    )


def test_descriptor_tokens_and_jaccard_similarity_are_deterministic() -> None:
    candidate = _candidate("A", "wave frontier", niche="math")
    tokens = descriptor_tokens(candidate)

    assert "wave" in tokens
    assert jaccard_similarity({"a", "b"}, {"b", "c"}) == 1 / 3


def test_ca_crossover_selects_highest_descriptor_neighborhood_similarity_partner() -> None:
    pivot = _candidate("P", "ca wave", niche="math", search=0.4)
    local = _candidate("L", "ca wave expansion", niche="math", search=0.6)
    distant = _candidate("D", "graph oracle", niche="biology", search=0.99)

    partner = neighborhood_crossover_partner(pivot, [pivot, local, distant], CACrossoverConfig(min_shared_descriptor_tokens=1))

    assert partner is local


def test_ca_crossover_uses_configured_global_donor_when_no_shared_descriptor() -> None:
    pivot = _candidate("P", "alpha", niche="a", search=0.1)
    high_search = _candidate("S", "beta", niche="b", search=0.9, final=0.2)
    high_final = _candidate("F", "gamma", niche="c", search=0.1, final=0.95)

    partner = neighborhood_crossover_partner(
        pivot,
        [pivot, high_search, high_final],
        CACrossoverConfig(min_shared_descriptor_tokens=3, global_donor_policy="highest_final_quality"),
    )

    assert partner is high_final


def test_crossover_plan_deterministic_fallback_generates_two_parent_child() -> None:
    parent = _candidate("P", "ca wave", niche="math")
    partner = _candidate("Q", "ca wave proof", niche="math")
    plan = MutationPlan(operator=MutationOperator.CROSSOVER, parent_ids=["P", "Q"], instruction="combine")

    offspring = _generate_offspring(
        model=None,
        mutation_engine=MutationEngine(),
        parents=[parent],
        plans=[plan],
        world=_World(),
        contract=NexusObjectiveContract(original_user_goal="x", normalized_goal="x"),
        policy=EvolutionPolicy(),
        candidate_pool=[parent, partner],
        ca_config=CACrossoverConfig(),
    )

    assert len(offspring) == 1
    assert offspring[0].parent_ids == ["P", "Q"]
    assert offspring[0].metadata["ca_crossover"]["parent_ids"] == ["P", "Q"]


def test_cell_activation_map_records_parent_and_offspring_cells() -> None:
    parent = _candidate("P", "alpha", niche="math")
    child = _candidate("C", "alpha child", niche="math")
    plan = MutationPlan(operator=MutationOperator.DEEPEN, parent_ids=["P"])

    activation = _cell_activation_map(parents=[parent], plans=[plan], offspring=[child])

    assert activation
    assert any("P" in entry["parent_ids"] for entry in activation.values())
    assert any("C" in entry["offspring_ids"] for entry in activation.values())
