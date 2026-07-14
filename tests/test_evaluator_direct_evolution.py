from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.adaptive import AdaptiveConfig, AdaptiveRuntimeController
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.loop.budget import EvolutionBudget
from cognitive_evolve_runtime.nexus.loop.controller import EvolutionLoopController
from cognitive_evolve_runtime.nexus.loop.round import EvolutionRound
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.search_kernel.branch_allocator import BranchSlot, ProductiveBranchAllocation


class _DirectOnlyModel:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.metadata: dict[str, Any] = {}
        self.provided_contexts: list[dict[str, Any] | None] = []

    def _forbidden(self, name: str) -> None:
        self.calls.append(name)
        raise AssertionError(f"model-backed reproduction must not call {name}")

    def critique_candidates(self, **_: Any):
        self._forbidden("critique_candidates")

    def relative_rank(self, **_: Any):
        self._forbidden("relative_rank")

    def diagnose_search_state(self, **_: Any):
        self._forbidden("diagnose_search_state")

    def update_policy(self, **_: Any):
        self._forbidden("update_policy")

    def plan_mutations(self, **_: Any):
        self._forbidden("plan_mutations")

    def should_stop(self, **_: Any):
        self._forbidden("should_stop")

    def synthesize_result(self, **_: Any):
        self._forbidden("synthesize_result")

    def generate_offspring(self, *, plans, parents, world, contract, policy, provided_context=None):
        self.calls.append("generate_offspring")
        self.provided_contexts.append(provided_context)
        assert len(plans) == 1
        assert plans[0].operator == "ModelDirected"
        assert plans[0].metadata["plan_source"] == "runtime_lineage_envelope"
        assert plans[0].metadata["completion_mode"] == "complete_task_artifact_only"
        requested = policy.metadata["requested_candidate_count"]
        assert requested == 1
        assert len(plans[0].metadata["branch_slots"]) == requested
        slot = plans[0].metadata["branch_slots"][0]
        index = int(slot["variation_index"])
        assert plans[0].parent_ids == [slot["parent_id"]]
        return [
            CandidateGenome(
                id=f"child-{index}",
                parent_ids=[parents[0].id],
                artifact={"answer": index, "source": parents[0].id},
                artifact_type="machine",
                concise_claim=f"candidate {index}",
                core_mechanism=f"direct semantic mutation {index}",
            )
        ]


class _AbstainingDirectModel(_DirectOnlyModel):
    def generate_offspring(self, *, plans, parents, world, contract, policy, provided_context=None):
        self.calls.append("generate_offspring")
        return []


class _SpoofingDirectModel(_DirectOnlyModel):
    def generate_offspring(self, *, plans, parents, world, contract, policy, provided_context=None):
        self.calls.append("generate_offspring")
        return [
            CandidateGenome(
                id=parents[0].id,
                parent_ids=[parents[0].id],
                artifact={"answer": "mutated"},
                artifact_type="machine",
                concise_claim="changed",
                core_mechanism="direct mutation",
                current_fate=CandidateFate.FAILED.value,
                metadata={"structural_failure": True},
            )
        ]


class _CopyingDirectModel(_DirectOnlyModel):
    def generate_offspring(self, *, plans, parents, world, contract, policy, provided_context=None):
        self.calls.append("generate_offspring")
        parent = parents[0]
        return [
            CandidateGenome(
                id="model-copy",
                parent_ids=[parent.id],
                artifact=parent.artifact,
                artifact_type=parent.artifact_type,
                concise_claim=parent.concise_claim,
                core_mechanism=parent.core_mechanism,
            )
        ]


def _evaluator_controller() -> AdaptiveRuntimeController:
    return AdaptiveRuntimeController(
        config=AdaptiveConfig.from_sources(
            explicit={
                "enabled": True,
                "evaluator": {"enabled": True, "command": "unit-test-evaluator"},
            }
        )
    )


def _apply_fake_external_evaluation(candidates: list[CandidateGenome], **_: Any) -> list[Any]:
    results: list[Any] = []
    for index, candidate in enumerate(candidates):
        candidate.metadata["evaluator"] = {
            "status": "passed",
            "passed": True,
            "metrics": {"score": 1.0 - index / 10},
        }
        candidate.multihead_scores["evaluator_score"] = 1.0 - index / 10
        results.append(SimpleNamespace(passed=True))
    return results


def _apply_failed_external_evaluation(candidates: list[CandidateGenome], **_: Any) -> list[Any]:
    for candidate in candidates:
        candidate.metadata["evaluator"] = {
            "status": "failed",
            "passed": False,
            "metrics": {"score": 0.0},
        }
        candidate.multihead_scores["evaluator_score"] = 0.0
    return [SimpleNamespace(passed=False) for _candidate in candidates]


def test_external_evaluator_path_uses_one_direct_model_call_per_slot(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "1")
    monkeypatch.setenv("COGEV_NEXUS_OFFSPRING_BATCH_LIMIT", "4")
    monkeypatch.setenv("COGEV_NEXUS_OFFSPRING_MIN_BATCHES", "4")
    model = _DirectOnlyModel()
    budget = EvolutionBudget(max_rounds=2, branch_factor=3, stop_policy="llm_after_minimum")
    pipeline = EvolutionRound(model=model, budget=budget, adaptive=_evaluator_controller())
    monkeypatch.setattr(pipeline.evaluator_runner, "evaluate_population_if_configured", _apply_fake_external_evaluation)
    population = CandidatePopulation(
        [
            CandidateGenome(
                id="parent",
                artifact={"answer": "incumbent"},
                artifact_type="machine",
                concise_claim="incumbent",
                core_mechanism="baseline",
            )
        ]
    )
    archives = ArchiveManager()
    policy = EvolutionPolicy()
    contract = NexusObjectiveContract(
        original_user_goal="improve the machine artifact",
        normalized_goal="improve the machine artifact",
        frozen_spec={"problem_text": "Return the best machine artifact."},
    )

    evaluation = pipeline.evaluate(
        current_round=1,
        population=population,
        archives=archives,
        policy=policy,
        contract=contract,
    )

    assert evaluation.critiques == []
    assert model.calls == []
    context_calls: list[list[str]] = []
    context_instructions: list[str] = []

    def context_provider(parents: list[CandidateGenome], instruction: str) -> dict[str, Any]:
        context_calls.append([parent.id for parent in parents])
        context_instructions.append(instruction)
        return {"selected_files": ["selected.py"], "slices": [{"path": "selected.py", "text": "ROUND-CONTEXT"}]}

    stop_reason, _, _ = pipeline.reproduce(
        current_round=1,
        population=population,
        archives=archives,
        policy=evaluation.policy,
        contract=contract,
        world={},
        rankings=evaluation.rankings,
        diagnosis=evaluation.diagnosis,
        critiques=evaluation.critiques,
        offspring_verifier=None,
        repair_parent_candidates=evaluation.repair_parent_candidates,
        provided_context={"selected_files": ["stale.py"], "slices": []},
        context_provider=context_provider,
    )

    assert stop_reason == ""
    assert model.calls == ["generate_offspring"] * 3
    assert context_calls == [["parent"]]
    assert context_instructions[0].count("Branch intent:") == 3
    assert all(context["slices"][0]["text"] == "ROUND-CONTEXT" for context in model.provided_contexts)
    children = [candidate for candidate in population.candidates if candidate.artifact.get("source") == "parent"]
    assert len(children) == 3
    assert {candidate.metadata["model_claimed_candidate_id"] for candidate in children} == {"child-0", "child-1", "child-2"}
    assert all(candidate.id not in {"child-0", "child-1", "child-2"} for candidate in children)
    assert len({candidate.metadata["branch_slot_id"] for candidate in children}) == 3
    assert all(candidate.metadata["branch_slot_binding_status"] == "bound" for candidate in children)
    assert all("branch_slots" not in candidate.metadata for candidate in children)
    assert pipeline.last_generation_plan["mutation_plan_source"] == "runtime_lineage_envelope"
    assert pipeline.last_completed_stage_ops[-4:] == [
        "select_parents",
        "plan_mutations",
        "generate_offspring",
        "verify_offspring",
    ]


def test_evaluator_flag_without_command_does_not_disable_model_control() -> None:
    controller = AdaptiveRuntimeController(
        config=AdaptiveConfig.from_sources(
            explicit={"enabled": True, "evaluator": {"enabled": True}}
        )
    )
    pipeline = EvolutionRound(model=_DirectOnlyModel(), budget=EvolutionBudget(), adaptive=controller)

    assert pipeline._evaluator_led() is False


def test_lineage_envelope_scopes_policy_directives_and_completion_mode() -> None:
    parent = CandidateGenome(id="P", lineage=["P"], artifact={"answer": "parent"})
    allocation = ProductiveBranchAllocation(
        slots=(
            BranchSlot(
                slot_id="slot-p",
                arm_id="P",
                parent_id="P",
                intent="explore_fresh",
                variation_index=0,
                ucb_score=1.0,
            ),
        ),
        arms=(),
        credit_summary={},
    )
    pipeline = EvolutionRound(
        model=_DirectOnlyModel(),
        budget=EvolutionBudget(),
        adaptive=AdaptiveRuntimeController(
            config=AdaptiveConfig.from_sources(explicit={"enabled": False})
        ),
    )
    common = {
        "parents": [parent],
        "current_round": 1,
        "branch_allocation": allocation,
        "actions": ["representation_shift"],
        "diagnosis": SearchDiagnosis(),
        "contract": NexusObjectiveContract(original_user_goal="improve", normalized_goal="improve"),
        "archives": ArchiveManager(),
        "population": [parent],
        "latent_exploration_plan": {},
        "policy": EvolutionPolicy(metadata={"mandatory_actions": ["counterexample_probe"]}),
    }

    strict, _ = pipeline._direct_model_plan(**common, evaluator_led=True)
    exploratory, _ = pipeline._direct_model_plan(**common, evaluator_led=False)
    directive = strict.metadata["branch_slots"][0]["directive"]

    assert strict.metadata["completion_mode"] == "complete_task_artifact_only"
    assert exploratory.metadata["completion_mode"] == "concrete_progress_allowed"
    assert directive["policy_directives"]["mandatory_actions"] == ["counterexample_probe"]


def test_full_evaluator_led_loop_calls_only_direct_offspring_model(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "1")
    monkeypatch.setenv("COGEV_NEXUS_OFFSPRING_BATCH_LIMIT", "4")
    monkeypatch.setenv("COGEV_NEXUS_OFFSPRING_MIN_BATCHES", "4")
    model = _DirectOnlyModel()
    population = CandidatePopulation(
        [
            CandidateGenome(
                id="parent",
                artifact={"answer": "incumbent"},
                artifact_type="machine",
                concise_claim="incumbent",
                core_mechanism="baseline",
                metadata={"operator_provided_incumbent": True},
            )
        ]
    )
    controller = EvolutionLoopController(
        population=population,
        archives=ArchiveManager(),
        policy=EvolutionPolicy(
            rarity_budget=0.0,
            metadata={
                "seed_harvest": {
                    "accepted_count": 0,
                    "fatal_model_error": "LLMResponseError: seed transport failure",
                }
            },
        ),
        contract=NexusObjectiveContract(
            original_user_goal="improve the machine artifact",
            normalized_goal="improve the machine artifact",
            frozen_spec={"problem_text": "Return the best machine artifact."},
        ),
        world={},
        budget=EvolutionBudget(max_rounds=2, branch_factor=3, stop_policy="llm_after_minimum"),
        model=model,
        adaptive_config={
            "enabled": True,
            "evaluator": {"enabled": True, "command": "unit-test-evaluator"},
        },
    )
    monkeypatch.setattr(
        controller.round_pipeline.evaluator_runner,
        "evaluate_population_if_configured",
        _apply_fake_external_evaluation,
    )

    result = controller.run()

    assert model.calls == ["generate_offspring"] * 3
    assert result.interrupted is False
    children = sorted(
        (candidate for candidate in result.population.candidates if candidate.metadata.get("model_claimed_candidate_id", "").startswith("child-")),
        key=lambda candidate: candidate.metadata["model_claimed_candidate_id"],
    )
    assert [candidate.metadata["model_claimed_candidate_id"] for candidate in children] == ["child-0", "child-1", "child-2"]
    assert len({candidate.id for candidate in children}) == 3
    assert all(not candidate.id.startswith("child-") for candidate in children)
    assert all(candidate.parent_ids == ["parent"] for candidate in children)
    assert all(candidate.generation == 1 for candidate in children)
    assert all(candidate.lineage == ["parent", candidate.id] for candidate in children)
    generation_plan = result.budget_history[0]["generation_plan"]
    assert generation_plan["mutation_plan_source"] == "runtime_lineage_envelope"
    assert generation_plan["completed_stage_ops"][-4:] == [
        "select_parents",
        "plan_mutations",
        "generate_offspring",
        "verify_offspring",
    ]


def test_schema_valid_empty_direct_offspring_continues_to_second_attempt(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "1")
    monkeypatch.setenv("COGEV_OFFSPRING_PARALLEL_MODE", "single_batch")
    model = _AbstainingDirectModel()
    controller = EvolutionLoopController(
        population=CandidatePopulation(
            [
                CandidateGenome(
                    id="parent",
                    artifact={"answer": "incumbent"},
                    artifact_type="machine",
                    concise_claim="incumbent",
                    core_mechanism="baseline",
                    metadata={"operator_provided_incumbent": True},
                )
            ]
        ),
        archives=ArchiveManager(),
        policy=EvolutionPolicy(rarity_budget=0.0),
        contract=NexusObjectiveContract(
            original_user_goal="improve the machine artifact",
            normalized_goal="improve the machine artifact",
            frozen_spec={"problem_text": "Return the best machine artifact."},
        ),
        world={},
        budget=EvolutionBudget(max_rounds=3, branch_factor=1, stop_policy="max_rounds"),
        model=model,
        adaptive_config={
            "enabled": True,
            "evaluator": {"enabled": True, "command": "unit-test-evaluator"},
        },
    )
    monkeypatch.setattr(
        controller.round_pipeline.evaluator_runner,
        "evaluate_population_if_configured",
        _apply_fake_external_evaluation,
    )

    result = controller.run()

    assert model.calls == ["generate_offspring", "generate_offspring"]
    assert result.interrupted is False
    assert result.current_round == 3
    assert result.stop_reason == "max_rounds"
    assert result.completion_status == "completed"
    assert [item["generation_plan"]["offspring_outcome"] for item in result.budget_history[:2]] == [
        "model_abstained_empty_batch",
        "model_abstained_empty_batch",
    ]


def test_schema_valid_empty_direct_offspring_without_incumbent_checkpoints(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "1")
    monkeypatch.setenv("COGEV_OFFSPRING_PARALLEL_MODE", "single_batch")
    model = _AbstainingDirectModel()
    controller = EvolutionLoopController(
        population=CandidatePopulation(
            [
                CandidateGenome(
                    id="failed-parent",
                    artifact={"answer": "invalid"},
                    artifact_type="machine",
                    concise_claim="invalid",
                    core_mechanism="baseline",
                )
            ]
        ),
        archives=ArchiveManager(),
        policy=EvolutionPolicy(rarity_budget=0.0),
        contract=NexusObjectiveContract(
            original_user_goal="improve the machine artifact",
            normalized_goal="improve the machine artifact",
            frozen_spec={"problem_text": "Return the best machine artifact."},
        ),
        world={},
        budget=EvolutionBudget(max_rounds=3, branch_factor=1, stop_policy="max_rounds"),
        model=model,
        adaptive_config={
            "enabled": True,
            "evaluator": {"enabled": True, "command": "unit-test-evaluator"},
        },
    )
    monkeypatch.setattr(
        controller.round_pipeline.evaluator_runner,
        "evaluate_population_if_configured",
        _apply_failed_external_evaluation,
    )

    result = controller.run()

    assert model.calls == ["generate_offspring"]
    assert result.interrupted is True
    assert result.current_round == 1
    assert result.error["type"] == "ModelResponseSchemaError"
    assert result.stop_reason == "model_schema_repair_checkpointed"
    assert result.completion_status == "interrupted_checkpointed"


def test_deduped_direct_output_is_not_an_empty_abstention(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "1")
    monkeypatch.setenv("COGEV_OFFSPRING_PARALLEL_MODE", "single_batch")
    model = _CopyingDirectModel()
    controller = EvolutionLoopController(
        population=CandidatePopulation(
            [
                CandidateGenome(
                    id="parent",
                    artifact={"answer": "incumbent"},
                    artifact_type="machine",
                    concise_claim="incumbent",
                    core_mechanism="baseline",
                    metadata={"operator_provided_incumbent": True},
                )
            ]
        ),
        archives=ArchiveManager(),
        policy=EvolutionPolicy(rarity_budget=0.0),
        contract=NexusObjectiveContract(
            original_user_goal="improve the machine artifact",
            normalized_goal="improve the machine artifact",
            frozen_spec={"problem_text": "Return the best machine artifact."},
        ),
        world={},
        budget=EvolutionBudget(max_rounds=3, branch_factor=1, stop_policy="max_rounds"),
        model=model,
        adaptive_config={
            "enabled": True,
            "evaluator": {"enabled": True, "command": "unit-test-evaluator"},
        },
    )
    monkeypatch.setattr(
        controller.round_pipeline.evaluator_runner,
        "evaluate_population_if_configured",
        _apply_fake_external_evaluation,
    )

    result = controller.run()

    assert model.calls == ["generate_offspring"]
    assert result.interrupted is False
    assert result.current_round == 1
    assert result.stop_reason == "no_new_unique_offspring"


def test_model_claimed_identity_and_fate_are_normalized_before_next_evaluator(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "1")
    monkeypatch.setenv("COGEV_OFFSPRING_PARALLEL_MODE", "single_batch")
    model = _SpoofingDirectModel()
    budget = EvolutionBudget(max_rounds=2, branch_factor=1, stop_policy="max_rounds")
    pipeline = EvolutionRound(model=model, budget=budget, adaptive=_evaluator_controller())
    seen_by_round: dict[int, list[dict[str, Any]]] = {}

    def _recording_evaluator(candidates: list[CandidateGenome], *, round_index: int, **kwargs: Any) -> list[Any]:
        seen_by_round[round_index] = [candidate.to_dict() for candidate in candidates]
        return _apply_fake_external_evaluation(candidates, round_index=round_index, **kwargs)

    monkeypatch.setattr(pipeline.evaluator_runner, "evaluate_population_if_configured", _recording_evaluator)
    population = CandidatePopulation(
        [
            CandidateGenome(
                id="parent",
                artifact={"answer": "incumbent"},
                artifact_type="machine",
                concise_claim="incumbent",
                core_mechanism="baseline",
            )
        ]
    )
    archives = ArchiveManager()
    policy = EvolutionPolicy(
        metadata={
            "live_bin_capacity": 1,
            "quality_diversity_rare_reserve_per_bin": 0,
        }
    )
    contract = NexusObjectiveContract(
        original_user_goal="improve the machine artifact",
        normalized_goal="improve the machine artifact",
        frozen_spec={"problem_text": "Return the best machine artifact."},
    )

    evaluation = pipeline.evaluate(
        current_round=1,
        population=population,
        archives=archives,
        policy=policy,
        contract=contract,
    )
    stop_reason, _, compaction = pipeline.reproduce(
        current_round=1,
        population=population,
        archives=archives,
        policy=evaluation.policy,
        contract=contract,
        world={},
        rankings=evaluation.rankings,
        diagnosis=evaluation.diagnosis,
        critiques=evaluation.critiques,
        offspring_verifier=None,
        repair_parent_candidates=evaluation.repair_parent_candidates,
    )

    assert stop_reason == ""
    assert population.by_id()["parent"].artifact == {"answer": "incumbent"}
    [child] = [candidate for candidate in population.candidates if candidate.artifact == {"answer": "mutated"}]
    assert child.id != "parent"
    assert child.metadata["model_claimed_candidate_id"] == "parent"
    assert child.parent_ids == ["parent"]
    assert child.lineage == ["parent", child.id]
    assert child.current_fate == CandidateFate.ACTIVE.value
    assert child.metadata.get("structural_failure") is not True
    assert child.metadata["model_claimed_runtime_controls"]["structural_failure"] is True
    assert child.metadata["created_in_round"] == 1
    assert child.id not in {*compaction["removed_terminal_ids"], *compaction["compacted_clone_ids"]}

    pipeline.evaluate(
        current_round=2,
        population=population,
        archives=archives,
        policy=evaluation.policy,
        contract=contract,
    )

    assert child.id in {item["id"] for item in seen_by_round[2]}
