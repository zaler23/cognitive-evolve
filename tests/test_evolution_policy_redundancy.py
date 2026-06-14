from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.llm.env import LLMResponseError
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicyBuilder


class RefusingPolicyModel:
    def build_evolution_policy(self, *, contract: Any, world: Any) -> dict[str, Any]:
        raise LLMResponseError("LLM response was not valid JSON: I cannot fulfill this request.")


def test_policy_builder_falls_back_when_model_returns_non_json_refusal() -> None:
    contract = {
        "search_space_plan": {
            "candidate_families": [
                {"id": "incidence_geometry_bounds"},
                {"id": "additive_combinatorics_counterexample"},
            ]
        }
    }

    policy = EvolutionPolicyBuilder().build(contract=contract, world={"kind": "text"}, model=RefusingPolicyModel())

    assert "incidence_geometry_bounds" in policy.candidate_niches
    assert "additive_combinatorics_counterexample" in policy.candidate_niches
    fallback = policy.metadata["model_policy_fallback"]
    assert fallback["error_type"] == "LLMResponseError"
    assert fallback["final_answer_blocked"] is True
    assert policy.metadata["search_space_plan"]["source"] == "objective_contract"


def test_policy_builder_falls_back_when_model_policy_shape_is_invalid() -> None:
    class InvalidPolicyModel:
        def build_evolution_policy(self, *, contract: Any, world: Any) -> dict[str, Any]:
            return {"rarity_budget": "not-a-number"}

    policy = EvolutionPolicyBuilder().build(contract={}, world={"kind": "text"}, model=InvalidPolicyModel())

    assert policy.candidate_niches
    assert policy.metadata["model_policy_fallback"]["error_type"] == "ValueError"
