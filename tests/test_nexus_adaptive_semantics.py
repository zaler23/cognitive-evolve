from __future__ import annotations

import pytest

from typing import Any

from cognitive_evolve_runtime.api.config import DEFAULT_MODELS, round_cap_for_model
from cognitive_evolve_runtime.api.jobs import _status_from_nexus_data
from cognitive_evolve_runtime.api.profiles import _temporary_model_runtime
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.engine.orchestrator import EngineOrchestrator
from cognitive_evolve_runtime.inputs.text_packet import TextInputPacket, TextWorldModel
from cognitive_evolve_runtime.nexus.loop import seed_population
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.synthesis import synthesize_result


class BatchSeedModel:
    def __init__(self) -> None:
        self.calls = 0

    def seed_population(self, *, contract: Any, world: Any, policy: Any) -> list[dict[str, Any]]:
        self.calls += 1
        return [
            _candidate(f"B{self.calls}", f"mechanism-{self.calls}"),
            _candidate(f"DUP{self.calls}", "duplicate-mechanism"),
        ]


class NoStopModel(BatchSeedModel):
    pass


def _candidate(candidate_id: str, mechanism: str) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "artifact": f"candidate artifact {mechanism}",
        "artifact_type": "answer",
        "concise_claim": f"claim {mechanism}",
        "core_mechanism": mechanism,
        "assumptions": ["test assumption"],
        "missing_parts": ["needs verifier closure"],
        "uncertainty_notes": ["not solved yet"],
        "multihead_scores": {"objective_alignment": 0.7, "answer_likelihood": 0.6, "verifiability": 0.4},
    }


def test_model_seed_generation_runs_multiple_batches_and_dedupes(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_SEED_BATCH_LIMIT", "6")
    model = BatchSeedModel()
    packet = TextInputPacket.from_text("hard task")
    world = TextWorldModel.from_packet(packet)
    contract = NexusObjectiveContract(original_user_goal="hard task", normalized_goal="hard task")
    policy = EvolutionPolicy(candidate_niches=["a", "b", "c", "d"])

    population = seed_population(contract=contract, world=world, policy=policy, model=model, min_population_size=5)

    mechanisms = [candidate.core_mechanism for candidate in population.candidates]
    assert model.calls >= 3
    assert len({m for m in mechanisms if m.startswith("mechanism-")}) >= 4
    assert mechanisms.count("duplicate-mechanism") == 1


def test_adaptive_safety_checkpoint_is_advisory_completed_answer_first(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("COGEV_NEXUS_PROFILE_EXHAUSTIVE_SAFETY_ROUNDS", "2")
    with _temporary_model_runtime("cognitive-evolve-one-shot-exhaustive"):
        result = EngineOrchestrator(model=NoStopModel()).run(
            "Prove a hard conjecture",
            context={
                "task_dir": str(tmp_path),
                "interface": "openai_compatible_api",
                "openai_compatible_model": "cognitive-evolve-one-shot-exhaustive",
            },
        )

    evolution = result.evolution
    assert evolution["stop_reason"] == "adaptive_safety_checkpoint"
    assert evolution["completion_status"] == "completed"
    assert evolution["synthesis"]["status"] == "synthesized"
    assert evolution["synthesis"]["continuation_available"] is False
    assert evolution["synthesis"]["closure_certificate"]["terminal_status"] == "completed"
    assert evolution["synthesis"]["closure_certificate"]["answer_produced"] is True
    assert evolution["synthesis"]["closure_certificate"]["objective_solved"] is False
    assert evolution["progress_events"][-1]["metadata"]["adaptive"] is True
    assert evolution["pipeline_events"][-1]["metadata"]["progress_semantics"] == "open_ended_no_percent_complete"


def test_api_model_caps_default_to_adaptive_profiles() -> None:
    assert {model: round_cap_for_model(model) for model in DEFAULT_MODELS} == {model: "0" for model in DEFAULT_MODELS}


@pytest.mark.parametrize("status", ["needs_continuation", "paused_quota"])
def test_job_status_reflects_continuation_axis_status(status: str) -> None:
    data = {"evolution": {"completion_status": status, "synthesis": {"status": status}}}
    assert _status_from_nexus_data(data, fallback="completed") == status


def test_answer_first_output_is_not_contract_blocked() -> None:
    candidate = CandidateGenome(
        id="C-route",
        generation=2,
        artifact="A useful but non-final design route.",
        concise_claim="partial design route",
        core_mechanism="bounded model-driven policy",
        missing_parts=["needs verifier closure"],
        multihead_scores={"objective_alignment": 0.8, "answer_likelihood": 0.7, "verifiability": 0.4},
        verification_result={"passed": False, "diagnostics": ["evidence_ref_absent"], "rank_eligible": False, "final_eligible": False},
    )
    population = CandidatePopulation([candidate])
    strict_contract = NexusObjectiveContract(
        original_user_goal="design task",
        normalized_goal="design task",
        outcome_policy={"accepts_answer_first_output": False, "requires_verified_solution": True},
    )
    flexible_contract = NexusObjectiveContract(
        original_user_goal="design task",
        normalized_goal="design task",
        outcome_policy={"accepts_answer_first_output": True, "requires_verified_solution": False},
    )

    strict = synthesize_result(population=population, archives=ArchiveManager(), contract=strict_contract)
    flexible = synthesize_result(population=population, archives=ArchiveManager(), contract=flexible_contract)

    assert strict.status == "synthesized"
    assert flexible.status == "synthesized"
    assert flexible.answer_produced is True
    assert flexible.objective_solved is False
    assert strict.best_candidate_id == "C-route"
    assert flexible.best_candidate_id == "C-route"
    assert strict.final_answer == "A useful but non-final design route."
    assert flexible.final_answer == "A useful but non-final design route."


def test_answer_candidate_is_displayed_when_not_project_certified() -> None:
    candidate = CandidateGenome(
        id="C-reference",
        generation=3,
        artifact="Useful repair direction: persist incubating lane state across bootstrap.",
        artifact_type="hybrid",
        concise_claim="repair lane persistence seed",
        core_mechanism="state serialization sketch",
        missing_parts=["machine-applicable project patch"],
        multihead_scores={"objective_alignment": 0.84, "answer_likelihood": 0.65, "verifiability": 0.35},
        verification_result={
            "passed": True,
            "rank_eligible": True,
            "final_eligible": False,
            "diagnostics": [],
            "final_gate": {"diagnostics": ["final_update_artifact_absent", "final_artifact_type_not_publishable"]},
        },
    )
    strict_contract = NexusObjectiveContract(
        original_user_goal="ship verified runtime patch",
        normalized_goal="ship verified runtime patch",
        outcome_policy={"accepts_answer_first_output": False, "requires_verified_solution": True},
    )

    synthesis = synthesize_result(population=CandidatePopulation([candidate]), archives=ArchiveManager(), contract=strict_contract)

    assert synthesis.status == "synthesized"
    assert synthesis.best_candidate_id == "C-reference"
    assert synthesis.answer_produced is True
    assert synthesis.objective_solved is False
    assert synthesis.final_answer == "Useful repair direction: persist incubating lane state across bootstrap."


def test_model_semantic_assessment_overrides_conservative_fallbacks() -> None:
    from cognitive_evolve_runtime.nexus.semantics import assess

    class Classifier:
        def classify_task(self, *, prompt: str) -> dict[str, Any]:
            return {
                "level": "L2_structured",
                "profile": "balanced",
                "task_type": "model_defined_strategy",
                "semantic": {
                    "task_type": "model_defined_strategy",
                    "weak_signals": {"math": False, "research": False, "project": False, "risk": False, "evolve": True, "architecture": False},
                    "real_objective": "model picked a strategy task despite proof words",
                    "evidence_needs": ["model_selected_evidence"],
                    "capability_hints": ["model_selected_capability"],
                    "complexity_assessment": {"semantic_complexity": 0.42},
                    "hypotheses": [{"hypothesis": "model_defined_strategy", "confidence": 0.9}],
                },
            }

    result = assess("prove theorem proof conjecture", model=Classifier())

    assert result.task_type == "model_defined_strategy"
    assert result.real_objective == "model picked a strategy task despite proof words"
    assert result.weak_signals["math"] is False
    assert result.evidence_needs == ["model_selected_evidence"]
    assert result.capability_hints == ["model_selected_capability"]
    assert result.complexity_assessment == {"semantic_complexity": 0.42}


def test_model_semantics_routes_erdos_unit_distance_as_deep_math() -> None:
    from cognitive_evolve_runtime.nexus.semantics import assess, classify

    class Classifier:
        def classify_task(self, *, prompt: str) -> dict[str, Any]:
            return {
                "level": "L4_evolutionary",
                "profile": "exhaustive",
                "task_type": "open_conjecture",
                "search": True,
                "checkmodel": True,
                "artifacts": True,
                "semantic": {
                    "task_type": "open_conjecture",
                    "weak_signals": {"math": True, "research": False, "project": False, "risk": False, "evolve": False, "architecture": False},
                    "evidence_needs": ["formal_or_symbolic_checks_when_available", "counterexample_search"],
                    "real_objective": "Resolve the Erdős unit-distance problem completely without treating a route as solved.",
                },
            }

    prompt = "Prompt. Let P ⊂ R^2 be finite. Resolve the Erdős unit-distance problem completely."

    route = classify(prompt, model=Classifier())
    assessment = assess(prompt, model=Classifier(), context={"route": route})

    assert route.level == "L4_evolutionary"
    assert route.profile == "exhaustive"
    assert route.search is True
    assert route.checkmodel is True
    assert route.artifacts is True
    assert assessment.task_type == "open_conjecture"
    assert assessment.weak_signals["math"] is True
    assert "counterexample_search" in assessment.evidence_needs


def test_missing_model_does_not_hardcode_erdos_unit_distance_domain() -> None:
    from cognitive_evolve_runtime.nexus.semantics import assess, classify

    prompt = "Prompt. Let P ⊂ R^2 be finite. Resolve the Erdős unit-distance problem completely."

    route = classify(prompt)
    assessment = assess(prompt, context={"route": route})

    assert route.level == "L4_evolutionary"
    assert route.profile == "deep"
    assert route.search is True
    assert route.semantic["router_source"] == "model_unavailable_conservative"
    assert assessment.task_type == "model_unavailable_unclassified"
    assert assessment.weak_signals["math"] is False


def test_capability_selection_uses_model_semantic_signals() -> None:
    from cognitive_evolve_runtime.nexus.semantics import select_capability_ids

    class Classifier:
        def classify_task(self, *, prompt: str) -> dict[str, Any]:
            return {
                "level": "L4_evolutionary",
                "profile": "deep",
                "task_type": "technical_execution_or_codebase_task",
                "search": True,
                "checkmodel": True,
                "artifacts": True,
                "semantic": {
                    "task_type": "technical_execution_or_codebase_task",
                    "weak_signals": {"project": True, "architecture": True, "research": False, "math": False, "risk": False, "evolve": False},
                },
            }

    selected = select_capability_ids("opaque prompt", model=Classifier())

    assert "local_execution" in selected
    assert "tool_boundary" in selected
    assert "project_governance" in selected


def test_model_defined_nexus_contract_task_type_is_preserved() -> None:
    from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract

    contract = NexusObjectiveContract.from_dict(
        {
            "original_user_goal": "goal",
            "normalized_goal": "goal",
            "task_type": "model_defined_non_registry_task",
            "outcome_policy": {"accepts_answer_first_output": True},
        }
    )

    assert contract.task_type == "model_defined_non_registry_task"


def test_profile_branch_factor_is_policy_derived_when_not_explicit(tmp_path) -> None:
    from cognitive_evolve_runtime.engine.orchestrator import EngineOrchestrator

    class PolicyWidthModel(NoStopModel):
        def build_evolution_policy(self, *, contract: Any, world: Any) -> dict[str, Any]:
            return {
                "candidate_niches": ["a", "b", "c", "d", "e"],
                "metadata": {"mutation_branches_per_round": 4},
            }

    result = EngineOrchestrator(model=PolicyWidthModel()).run(
        "adaptive width task",
        context={"task_dir": str(tmp_path), "rounds": 1, "evolution_profile": "exhaustive"},
    )

    budget = result.evolution["runtime_metadata"]["round_budget"]
    assert budget["mutation_branches_per_round"] == 4
    assert budget["branch_factor_source"] == "explicit_or_policy_derived"
