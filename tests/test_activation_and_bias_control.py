from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.prompt_view import build_prompt_view
from cognitive_evolve_runtime.nexus.semantics import classify
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRater


class OrderBiasedRankModel:
    def __init__(self) -> None:
        self.orders: list[list[str]] = []

    def relative_rank(self, *, candidates: list[CandidateGenome], **_: Any) -> dict[str, Any]:
        self.orders.append([candidate.id for candidate in candidates])
        first = candidates[0].id
        return {
            "best_final_answer_id": first,
            "strongest_mechanism_id": first,
            "mutation_worthy_ids": [first],
            "edge_value_ids": [],
            "auxiliary_ids": [],
            "dormant_ids": [],
            "dominated_pairs": [],
            "crossover_pairs": [],
            "preserve_incomplete_ids": [],
            "pairwise_preferences": [],
            "multihead_observations": {},
            "raw_notes": "order-biased",
        }


def test_activation_contract_is_model_driven_and_domain_neutral() -> None:
    view = build_prompt_view(
        "nexus_seed_population",
        {
            "contract": NexusObjectiveContract(original_user_goal="improve a work product", normalized_goal="improve a work product"),
            "policy": EvolutionPolicy(),
        },
    )

    activation = view.payload["activation_contract"]
    assert activation["runtime_does_not_choose_domains"] is True
    assert activation["model_decides_activation_need"] is True
    assert "persona" in activation["activation_controls"]
    assert "cross_domain_analogy" in activation["activation_controls"]
    assert "conceptual_blending" in activation["activation_controls"]
    assert "discussion alone" in activation["anti_empty_talk_rule"]
    assert view.payload["search_space_contract"]["model_driven"] is True
    assert "grounding surfaces" in view.payload["artifact_generation_contract"]["search_space_rule"]


def test_relative_rater_runs_reversed_ab_pass_and_merges_by_candidate_id() -> None:
    weaker = CandidateGenome(
        id="A",
        current_fate=CandidateFate.ACTIVE.value,
        multihead_scores={"answer_likelihood": 0.1, "objective_alignment": 0.1, "core_mechanism_strength": 0.2},
    )
    stronger = CandidateGenome(
        id="B",
        current_fate=CandidateFate.ACTIVE.value,
        multihead_scores={"answer_likelihood": 0.9, "objective_alignment": 0.9, "core_mechanism_strength": 0.8},
    )
    model = OrderBiasedRankModel()

    ranking = RelativeRater(model=model).rank(candidates=[weaker, stronger], contract=None, policy=None, archives=ArchiveManager())

    assert model.orders == [["A", "B"], ["B", "A"]]
    assert ranking.best_final_answer_id == "B"
    assert "ab_order_bias_mitigation" in ranking.raw_notes


def test_relative_rank_prompt_tells_judge_to_ignore_position_and_verbosity() -> None:
    view = build_prompt_view(
        "nexus_relative_rank",
        {"candidates": [CandidateGenome(id="A"), CandidateGenome(id="B")], "contract": {}, "policy": {}, "archives": ArchiveManager()},
    )

    mitigation = view.payload["ranking_bias_mitigation"]
    assert "list position" in mitigation["position_bias"]
    assert "longer artifacts" in mitigation["verbosity_bias"]


def test_missing_model_semantic_router_is_conservative_not_authoritative() -> None:
    route = classify("Write a proof-of-concept product note without a model router.")

    assert route.semantic["fallback_only"] is True
    assert route.semantic["router_source"] == "model_unavailable_conservative"
    assert route.semantic["task_type"] == "model_unavailable_unclassified"
    assert route.level == "L4_evolutionary"
    assert route.search is True
    assert "model_unavailable_conservative" in route.reason


def test_project_policy_does_not_inject_minimal_patch_as_default_search_family() -> None:
    from cognitive_evolve_runtime.nexus.policy import EvolutionPolicyBuilder

    class ProjectWorld:
        kind = "project"

    policy = EvolutionPolicyBuilder().build(
        contract=NexusObjectiveContract(original_user_goal="find a more elegant self-evolution architecture", normalized_goal="find a more elegant self-evolution architecture"),
        world=ProjectWorld(),
    )

    assert "minimal_patch" not in policy.candidate_niches
    assert policy.metadata["search_space_plan_required"]
