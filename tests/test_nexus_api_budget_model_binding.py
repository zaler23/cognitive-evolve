from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.api.profiles import _temporary_model_runtime
from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.engine.orchestrator import EngineOrchestrator
from cognitive_evolve_runtime.nexus.model_adapter import StructuredModelAdapter


class FakeNexusModel:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.offspring_counter = 0

    def build_objective_contract(self, *, user_goal: str, world: Any) -> dict[str, Any]:
        self.calls.append("build_objective_contract")
        return {
            "original_user_goal": user_goal,
            "normalized_goal": "model-normalized goal",
            "input_constraints": [],
            "allowed_evidence_sources": ["Input Evidence", "Tool Evidence", "Model Hypothesis"],
            "disallowed_goal_mutations": ["do not replace the task with a router"],
            "expected_output_forms": ["final answer"],
            "uncertainty_policy": "one-shot bounded uncertainty",
            "verification_preferences": [],
            "success_dimensions": ["objective_alignment"],
            "failure_dimensions": ["semantic_drift"],
        }

    def build_evolution_policy(self, *, contract: Any, world: Any) -> dict[str, Any]:
        self.calls.append("build_evolution_policy")
        return {
            "candidate_niches": ["direct", "rare"],
            "fitness_axes": ["objective_alignment", "answer_likelihood", "rarity", "verifiability"],
            "mutation_operators": ["Deepen", "RareInject"],
            "archive_schema": {"AnswerArchive": {"enabled": True}, "RarityArchive": {"enabled": True}},
            "parent_selection_preferences": {},
            "culling_principles": [],
            "rarity_budget": 0.3,
            "tool_preferences": [],
            "stagnation_actions": ["rare_inject"],
            "synthesis_policy": {"avoid_seed_echo": True},
        }

    def seed_population(self, *, contract: Any, world: Any, policy: Any) -> list[dict[str, Any]]:
        self.calls.append("seed_population")
        return [
            {
                "id": "M0",
                "generation": 0,
                "artifact": "LLM seed answer candidate, not static Direct Solver Seed.",
                "artifact_type": "answer",
                "concise_claim": "LLM seed",
                "core_mechanism": "model generated mechanism",
                "multihead_scores": {"objective_alignment": 0.7, "answer_likelihood": 0.7, "verifiability": 0.5},
            },
            {
                "id": "M1",
                "generation": 0,
                "artifact": "Rare model seed candidate.",
                "artifact_type": "answer",
                "concise_claim": "rare model seed",
                "core_mechanism": "rare mechanism",
                "edge_knowledge_seeds": ["edge"],
                "multihead_scores": {"objective_alignment": 0.55, "answer_likelihood": 0.45, "rarity": 0.9},
            },
        ]

    def relative_rank(self, *, candidates: list[CandidateGenome], contract: Any, policy: Any, archives: Any) -> dict[str, Any]:
        self.calls.append("relative_rank")
        best = max(candidates, key=lambda c: c.generation)
        return {
            "best_final_answer_id": best.id,
            "strongest_mechanism_id": best.id,
            "mutation_worthy_ids": [best.id],
            "edge_value_ids": [c.id for c in candidates if c.edge_knowledge_seeds],
            "auxiliary_ids": [],
            "dormant_ids": [],
            "dominated_pairs": [],
            "crossover_pairs": [],
            "preserve_incomplete_ids": [],
            "pairwise_preferences": [],
            "multihead_observations": {best.id: best.multihead_scores},
            "raw_notes": "fake model rank",
        }

    def diagnose_search_state(self, *, population: list[CandidateGenome], archives: Any, history: list[dict[str, Any]], contract: Any, policy: Any) -> dict[str, Any]:
        self.calls.append("diagnose_search_state")
        return {"stagnation_detected": False, "stagnation_type": "None", "recommended_actions": ["continue"], "notes": "fake model diagnosis"}

    def update_policy(self, *, policy: Any, diagnosis: Any) -> dict[str, Any]:
        self.calls.append("update_policy")
        return policy.to_dict()

    def plan_mutations(self, *, parents: list[CandidateGenome], actions: list[str], archives: Any, diagnosis: Any, policy: Any) -> list[dict[str, Any]]:
        self.calls.append("plan_mutations")
        return [{"operator": "Deepen", "parent_ids": [parent.id], "instruction": "model planned deepening"} for parent in parents]

    def generate_offspring(self, *, plans: list[Any], parents: list[CandidateGenome], world: Any, contract: Any, policy: Any) -> list[dict[str, Any]]:
        self.calls.append("generate_offspring")
        offspring: list[dict[str, Any]] = []
        for parent in parents:
            self.offspring_counter += 1
            offspring.append(
                {
                    "id": f"MO{self.offspring_counter}",
                    "parent_ids": [parent.id],
                    "generation": parent.generation + 1,
                    "artifact": f"LLM offspring generation {parent.generation + 1}",
                    "artifact_type": "answer",
                    "concise_claim": "LLM offspring",
                    "core_mechanism": "model evolved mechanism",
                    "multihead_scores": {"objective_alignment": 0.9, "answer_likelihood": 0.9, "verifiability": 0.6},
                }
            )
        return offspring

    def synthesize_result(self, *, population: list[CandidateGenome], archives: Any, contract: Any, world: Any) -> dict[str, Any]:
        self.calls.append("synthesize_result")
        return {
            "status": "model_synthesized",
            "final_answer": f"MODEL BACKED FINAL after {len(population)} candidates",
            "best_candidate_id": population[-1].id if population else "",
            "warnings": [],
        }


def test_exhaustive_model_tier_controls_nexus_rounds_without_explicit_rounds(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("COGEV_NEXUS_PROFILE_EXHAUSTIVE_SAFETY_ROUNDS", "3")
    fake = FakeNexusModel()
    with _temporary_model_runtime("cognitive-evolve-one-shot-exhaustive"):
        result = EngineOrchestrator(model=fake).run(
            "Hard problem",
            context={
                "task_dir": str(tmp_path),
                "interface": "openai_compatible_api",
                "openai_compatible_model": "cognitive-evolve-one-shot-exhaustive",
            },
        )

    assert result.evolution["progress_events"][-1]["max_rounds"] == 3
    assert result.evolution["runtime_metadata"]["round_budget"]["profile"] == "exhaustive"
    assert result.final_answer.startswith("MODEL BACKED FINAL")
    assert "seed_population" in fake.calls
    assert "relative_rank" in fake.calls
    assert "generate_offspring" in fake.calls
    assert "synthesize_result" in fake.calls


def test_api_context_uses_configured_nexus_model_adapter_without_explicit_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("COGEV_NEXUS_PROFILE_EXHAUSTIVE_SAFETY_ROUNDS", "2")
    fake = FakeNexusModel()
    monkeypatch.setattr(StructuredModelAdapter, "from_configured_llm", lambda: fake)

    with _temporary_model_runtime("cognitive-evolve-one-shot-exhaustive"):
        result = EngineOrchestrator().run(
            "Hard problem",
            context={
                "task_dir": str(tmp_path),
                "api_request_id": "chatcmpl-test-model-binding",
                "interface": "openai_compatible_api",
                "openai_compatible_model": "cognitive-evolve-one-shot-exhaustive",
                "raw_request": {
                    "model": "cognitive-evolve-one-shot-exhaustive",
                    "messages": [{"role": "user", "content": "Hard problem"}],
                    "stream": True,
                },
            },
        )

    assert result.final_answer.startswith("MODEL BACKED FINAL")
    assert result.evolution["progress_events"][-1]["max_rounds"] == 2
    assert result.evolution["runtime_metadata"]["model_backed"] is True
    assert "build_objective_contract" in fake.calls
