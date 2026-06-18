from __future__ import annotations

from typing import Any

import pytest

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.llm.env import LLMResponseError
from cognitive_evolve_runtime.llm.model_spec import LLMModelSpec
from cognitive_evolve_runtime.nexus.model_adapter import StructuredModelAdapter
from cognitive_evolve_runtime.nexus.model_routes import NexusModelRoutes, NexusModelRole
from cognitive_evolve_runtime.nexus.runtime import NexusRuntime


class _DefaultModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def seed_population(self, **_: Any) -> list[dict[str, Any]]:
        self.calls.append("seed_population")
        return [{"id": "DSEED", "artifact": "default seed", "concise_claim": "default", "core_mechanism": "default"}]

    def build_objective_contract(self, *, user_goal: str, world: Any) -> dict[str, Any]:
        self.calls.append("build_objective_contract")
        return {
            "original_user_goal": user_goal,
            "normalized_goal": user_goal,
            "input_constraints": [],
            "allowed_evidence_sources": ["Input Evidence", "Tool Evidence", "Model Hypothesis"],
            "disallowed_goal_mutations": [],
            "expected_output_forms": ["answer"],
            "uncertainty_policy": "bounded",
            "verification_preferences": [],
            "success_dimensions": ["objective_alignment"],
            "failure_dimensions": ["semantic_drift"],
        }

    def build_evolution_policy(self, *, contract: Any, world: Any) -> dict[str, Any]:
        self.calls.append("build_evolution_policy")
        return {
            "candidate_niches": ["direct", "edge"],
            "fitness_axes": ["objective_alignment", "answer_likelihood", "rarity", "verifiability"],
            "mutation_operators": ["Deepen", "CrossOver"],
            "archive_schema": {"AnswerArchive": {"enabled": True}, "RarityArchive": {"enabled": True}},
            "parent_selection_preferences": {},
            "culling_principles": [],
            "rarity_budget": 0.2,
            "tool_preferences": [],
            "stagnation_actions": ["continue"],
            "synthesis_policy": {},
        }

    def relative_rank(self, *, candidates: list[CandidateGenome], contract: Any, policy: Any, archives: Any) -> dict[str, Any]:
        self.calls.append("relative_rank")
        best = candidates[0]
        return {
            "best_final_answer_id": best.id,
            "strongest_mechanism_id": best.id,
            "mutation_worthy_ids": [best.id],
            "edge_value_ids": [],
            "auxiliary_ids": [],
            "dormant_ids": [],
            "dominated_pairs": [],
            "crossover_pairs": [],
            "preserve_incomplete_ids": [],
            "pairwise_preferences": [],
            "multihead_observations": {best.id: {"objective_alignment": 0.8, "answer_likelihood": 0.8}},
            "raw_notes": "ranked",
        }

    def critique_candidates(self, *, candidates: list[CandidateGenome], round_index: int, contract: Any, policy: Any, archives: Any) -> list[dict[str, Any]]:
        self.calls.append("critique_candidates")
        return []

    def diagnose_search_state(self, *, population: list[CandidateGenome], archives: Any, history: list[dict[str, Any]], contract: Any, policy: Any) -> dict[str, Any]:
        self.calls.append("diagnose_search_state")
        return {"stagnation_detected": False, "stagnation_type": "None", "recommended_actions": ["continue"], "notes": "ok"}

    def update_policy(self, *, policy: Any, diagnosis: Any) -> dict[str, Any]:
        self.calls.append("update_policy")
        return policy.to_dict()

    def plan_mutations(self, *, parents: list[CandidateGenome], actions: list[str], archives: Any, diagnosis: Any, policy: Any) -> list[dict[str, Any]]:
        self.calls.append("plan_mutations")
        return [{"operator": "Deepen", "parent_ids": [parent.id], "instruction": "deepen"} for parent in parents]

    def generate_offspring(self, *, plans: list[Any], parents: list[CandidateGenome], world: Any, contract: Any, policy: Any) -> list[dict[str, Any]]:
        self.calls.append("generate_offspring")
        return [
            {
                "id": f"O{index}",
                "parent_ids": [parent.id],
                "generation": parent.generation + 1,
                "artifact": f"offspring {parent.id}",
                "concise_claim": "offspring",
                "core_mechanism": "offspring",
                "multihead_scores": {"objective_alignment": 0.85, "answer_likelihood": 0.85},
            }
            for index, parent in enumerate(parents)
        ]

    def synthesize_result(self, *, population: list[CandidateGenome], archives: Any, contract: Any, world: Any) -> dict[str, Any]:
        self.calls.append("synthesize_result")
        return {"status": "done", "final_answer": "default synthesized", "best_candidate_id": population[0].id if population else "", "warnings": []}

    def should_stop(self, *, budget: Any, diagnosis: Any, best_answer_id: str, population: list[CandidateGenome]) -> dict[str, Any]:
        self.calls.append("should_stop")
        return {"should_stop": False, "reason": "test"}


class _SeedModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def seed_population(self, *, contract: Any, world: Any, policy: Any) -> list[dict[str, Any]]:
        self.calls.append("seed_population")
        return [
            {"id": "S0", "artifact": "seed route candidate", "concise_claim": "seed", "core_mechanism": "seed", "multihead_scores": {"rarity": 0.8}},
            {"id": "S1", "artifact": "second seed route candidate", "concise_claim": "seed2", "core_mechanism": "seed2"},
        ]


class _FailingSeedModel:
    def __init__(self) -> None:
        self.calls = 0

    def seed_population(self, *, contract: Any, world: Any, policy: Any) -> list[dict[str, Any]]:
        self.calls += 1
        raise LLMResponseError("seed route unavailable")


def test_runtime_single_model_keeps_legacy_seed_behavior(tmp_path) -> None:
    model = _DefaultModel()
    result = NexusRuntime(model=model, output_dir=tmp_path).run_text("route test", max_rounds=1, min_population_size=2, branch_factor=2)

    assert result.final_answer
    assert "seed_population" in model.calls
    assert result.evolution["runtime_metadata"]["model_routes"]["seed_uses_default"] is True


def test_seed_route_isolated_from_default_model(tmp_path) -> None:
    default = _DefaultModel()
    seed = _SeedModel()
    result = NexusRuntime(model_routes=NexusModelRoutes(default_model=default, seed_model=seed), output_dir=tmp_path).run_text(
        "route test",
        max_rounds=2,
        min_rounds_before_stop=2,
        min_population_size=2,
        branch_factor=2,
    )

    assert result.final_answer
    assert seed.calls and set(seed.calls) == {"seed_population"}
    assert "seed_population" not in default.calls
    for expected in {"build_objective_contract", "build_evolution_policy", "relative_rank", "diagnose_search_state", "plan_mutations", "generate_offspring", "synthesize_result"}:
        assert expected in default.calls
    summary = result.evolution["runtime_metadata"]["model_routes"]
    assert summary["seed_uses_default"] is False
    assert "api_key" not in str(summary).lower()


def test_seed_route_failure_does_not_fallback_to_default_seed(tmp_path) -> None:
    default = _DefaultModel()
    seed = _FailingSeedModel()
    result = NexusRuntime(model_routes=NexusModelRoutes(default_model=default, seed_model=seed), output_dir=tmp_path).run_text(
        "route failure test",
        max_rounds=2,
        min_rounds_before_stop=2,
        min_population_size=2,
        branch_factor=2,
    )

    assert result.final_answer
    assert seed.calls >= 1
    assert "seed_population" not in default.calls


def test_model_and_routes_conflict_is_rejected() -> None:
    default = _DefaultModel()
    other = _DefaultModel()
    with pytest.raises(ValueError):
        NexusRuntime(model=other, model_routes=NexusModelRoutes(default_model=default))


def test_configured_adapter_passes_model_spec_to_transport(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_llm_json(request_type: str, payload: dict[str, Any], *, system: str, schema_hint: dict[str, Any], model_spec: LLMModelSpec | None = None, provider: Any | None = None) -> dict[str, Any]:
        captured["model_spec"] = model_spec
        return {"candidates": [{"id": "X", "artifact": "x", "artifact_type": "answer", "concise_claim": "x", "core_mechanism": "x"}]}

    monkeypatch.setattr("cognitive_evolve_runtime.llm.transport.llm_json", fake_llm_json)
    spec = LLMModelSpec(model="frontier-exploration-model", provider="openai")
    adapter = StructuredModelAdapter.from_configured_llm(model_spec=spec)
    raw = adapter.seed_population(contract={}, world={}, policy={})

    assert raw[0]["id"] == "X"
    assert captured["model_spec"] == spec
    assert adapter.metadata["model_spec"]["model"] == "frontier-exploration-model"


def test_model_routes_public_summary_redacts_fixture_path() -> None:
    adapter = StructuredModelAdapter.from_configured_llm(model_spec=LLMModelSpec(fixture="fixtures/private.json", model="fixture"))
    routes = NexusModelRoutes(default_model=adapter)
    assert routes.model_for(NexusModelRole.SEED) is adapter
    assert routes.public_summary()["default"]["model_spec"]["fixture"] == "configured"
