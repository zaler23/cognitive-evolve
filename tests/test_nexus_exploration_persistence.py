from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.engine.orchestrator import EngineOrchestrator
from cognitive_evolve_runtime.llm.env import LLMResponseError
from cognitive_evolve_runtime.nexus.runtime import NexusRuntime


class NarrowSeedModel:
    def seed_population(self, **_: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": "M0",
                "generation": 0,
                "artifact": "model seed",
                "artifact_type": "answer",
                "concise_claim": "model seed",
                "core_mechanism": "model_route",
                "multihead_scores": {"objective_alignment": 0.7, "answer_likelihood": 0.7},
            }
        ]

    def relative_rank(self, *, candidates: list[CandidateGenome], **_: Any) -> dict[str, Any]:
        return {
            "best_final_answer_id": "M0",
            "strongest_mechanism_id": "M0",
            "mutation_worthy_ids": ["M0"],
            "edge_value_ids": [candidate.id for candidate in candidates if candidate.edge_knowledge_seeds],
            "auxiliary_ids": [],
            "dormant_ids": [],
            "dominated_pairs": [],
            "crossover_pairs": [],
            "preserve_incomplete_ids": [],
            "pairwise_preferences": [],
            "multihead_observations": {},
        }

    def diagnose_search_state(self, **_: Any) -> dict[str, Any]:
        return {"stagnation_detected": False, "stagnation_type": "None", "recommended_actions": ["continue"]}

    def update_policy(self, *, policy: Any, **_: Any) -> dict[str, Any]:
        return policy.to_dict()

    def synthesize_result(self, *, population: list[CandidateGenome], **_: Any) -> dict[str, Any]:
        return {"status": "ok", "final_answer": f"synthesized from {len(population)} candidates"}


class BadOffspringModel(NarrowSeedModel):
    def __init__(self) -> None:
        self.offspring_attempted = False

    def generate_offspring(self, **_: Any) -> list[dict[str, Any]]:
        self.offspring_attempted = True
        raise LLMResponseError("offspring schema drift")


class InterruptingModel(NarrowSeedModel):
    def relative_rank(self, **_: Any) -> dict[str, Any]:
        raise LLMResponseError("quota exhausted during relative rank")


class DiagnoseInterruptsSecondRoundModel(NarrowSeedModel):
    def __init__(self) -> None:
        self.diagnose_calls = 0

    def diagnose_search_state(self, **_: Any) -> dict[str, Any]:
        self.diagnose_calls += 1
        if self.diagnose_calls >= 2:
            raise LLMResponseError("provider 5xx during nexus_diagnose_search_state")
        return super().diagnose_search_state(**_)


def test_nexus_amplifies_narrow_model_seed_pool_and_marks_search_seeds(tmp_path: Path) -> None:
    result = NexusRuntime(model=NarrowSeedModel(), output_dir=tmp_path).run_text(
        "Solve a hard math problem.",
        max_rounds=1,
        min_population_size=12,
    )

    candidates = result.evolution["population"]["candidates"]
    assert len(candidates) >= 12
    assert candidates[0]["id"] == "M0"
    assert any(candidate.get("metadata", {}).get("search_seed_not_final") for candidate in candidates)
    assert any("negative" in candidate.get("core_mechanism", "") or "dual" in candidate.get("core_mechanism", "") for candidate in candidates)


def test_model_offspring_failure_falls_back_to_deterministic_mutation(tmp_path: Path) -> None:
    model = BadOffspringModel()
    result = NexusRuntime(model=model, output_dir=tmp_path).run_text(
        "Keep evolving even if model offspring response is malformed.",
        max_rounds=2,
        min_population_size=10,
    )

    assert result.evolution.get("interrupted") is not True
    assert "interrupted before final convergence" not in result.final_answer
    assert model.offspring_attempted is True
    assert result.evolution["progress_events"][-1]["round"] == 2


def test_nexus_live_persistence_survives_mid_round_model_interruption(tmp_path: Path) -> None:
    result = NexusRuntime(model=InterruptingModel(), output_dir=tmp_path).run_text(
        "Hard problem that will hit quota.",
        max_rounds=4,
        min_population_size=10,
    )

    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert result.evolution["interrupted"] is True
    assert result.evolution["completion_status"] == "paused_quota"
    assert "paused on provider quota" in result.final_answer
    assert result.final_answer != "model seed"
    assert checkpoint["round"] == 1
    assert checkpoint["max_rounds"] == 4
    assert checkpoint["budget"]["current_round"] == 1
    assert len(checkpoint["population"]["candidates"]) >= 10
    assert (tmp_path / "candidate-journal.jsonl").exists()
    assert any(path.name.startswith("round-0001-error_checkpoint") for path in (tmp_path / "rounds").iterdir())


def test_error_checkpoint_reconciles_previous_progress_round_on_second_round_failure(tmp_path: Path) -> None:
    result = NexusRuntime(model=DiagnoseInterruptsSecondRoundModel(), output_dir=tmp_path).run_text(
        "Hard problem with provider 5xx during second-round diagnosis.",
        max_rounds=4,
        min_population_size=10,
    )

    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert result.evolution["interrupted"] is True
    assert result.evolution["completion_status"] == "interrupted_checkpointed"
    assert checkpoint["round"] == 2
    assert checkpoint["progress_event"]["round"] == 2
    assert checkpoint["progress_event"]["metadata"]["previous_progress_round"] == 1
    assert any(path.name.startswith("round-0002-error_checkpoint") for path in (tmp_path / "rounds").iterdir())
    assert (tmp_path / "run-result.json").exists()
    assert (tmp_path / "final-answer.md").exists()


def test_exhaustive_profile_carries_old_branch_and_candidate_width(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COGEV_NEXUS_PROFILE_EXHAUSTIVE_SAFETY_ROUNDS", "2")
    monkeypatch.setenv("COGEV_NEXUS_PROFILE_EXHAUSTIVE_MIN_CANDIDATES", "14")
    monkeypatch.setenv("COGEV_NEXUS_PROFILE_EXHAUSTIVE_BRANCH_FACTOR", "4")

    result = EngineOrchestrator(model=NarrowSeedModel()).run(
        "Difficult exhaustive task",
        context={
            "task_dir": str(tmp_path),
            "interface": "openai_compatible_api",
            "openai_compatible_model": "cognitive-evolve-one-shot-exhaustive",
            "evolution_profile": "exhaustive",
        },
    )

    budget = result.evolution["runtime_metadata"]["round_budget"]
    assert budget["initial_candidate_count"] == 14
    assert budget["mutation_branches_per_round"] == 4
    assert result.evolution["progress_events"][-1]["max_rounds"] == 2
    assert len(result.evolution["population"]["candidates"]) >= 14
