from __future__ import annotations

import json

import pytest

from cognitive_evolve_runtime.archives.manager import ArchiveManager, FateAssignment
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.generation_plan import (
    GenerationPlan,
    GenerationPlanError,
    apply_generation_plan,
    assert_stage_ready,
    build_generation_plan,
    expected_generation_plan_id,
    validate_generation_plan_history,
)
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, EvolutionRound
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.runtime import NexusRuntime
from cognitive_evolve_runtime.persistence.checkpoint import CheckpointStore
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult


def test_generation_plan_applies_rank_archive_transition_once() -> None:
    winner = CandidateGenome(id="winner", artifact="answer", multihead_scores={"answer_likelihood": 0.9})
    helper = CandidateGenome(id="helper", artifact="validator", multihead_scores={"auxiliary_value": 0.95})
    ranking = RelativeRankingResult(best_final_answer_id="winner", auxiliary_ids=["helper"])
    assignments = [
        FateAssignment("winner", CandidateFate.ELITE.value),
        FateAssignment("helper", CandidateFate.AUXILIARY.value),
    ]
    plan = build_generation_plan(
        round_index=1,
        candidates=[winner, helper],
        fate_assignments=assignments,
        ranking=ranking,
        stage_graph=[
            {"op": "critique_and_verify"},
            {"op": "rank"},
            {"op": "archive_assign"},
            {"op": "generation_plan_validate"},
            {"op": "archive_update"},
        ],
        source="unit_test",
    )
    archives = ArchiveManager()

    applied = apply_generation_plan(plan, [winner, helper], archives)

    assert [item.candidate_id for item in applied] == ["winner", "helper"]
    assert winner.current_fate == CandidateFate.ELITE.value
    assert helper.current_fate == CandidateFate.AUXILIARY.value
    assert "winner" in archives.answer_archive
    assert "helper" in archives.auxiliary_archive.candidates
    assert winner.metadata["generation_plan_id"] == plan.plan_id
    assert helper.metadata["generation_plan_fate"] == CandidateFate.AUXILIARY.value
    assert archives.history[-1]["generation_plan_id"] == plan.plan_id
    assert plan.to_dict()["ranking_summary"]["best_final_answer_id"] == "winner"


def test_generation_plan_rejects_missing_duplicate_unknown_and_unregistered_transitions() -> None:
    candidate = CandidateGenome(id="known")
    ranking = RelativeRankingResult(best_final_answer_id="known")

    with pytest.raises(GenerationPlanError, match="missing fate assignment"):
        build_generation_plan(round_index=1, candidates=[candidate], fate_assignments=[], ranking=ranking)

    with pytest.raises(GenerationPlanError, match="duplicate fate assignment"):
        build_generation_plan(
            round_index=1,
            candidates=[candidate],
            fate_assignments=[
                FateAssignment("known", CandidateFate.ACTIVE.value),
                FateAssignment("known", CandidateFate.DORMANT.value),
            ],
            ranking=ranking,
        )

    with pytest.raises(GenerationPlanError, match="unknown candidate"):
        build_generation_plan(
            round_index=1,
            candidates=[candidate],
            fate_assignments=[FateAssignment("ghost", CandidateFate.ACTIVE.value)],
            ranking=ranking,
        )

    with pytest.raises(GenerationPlanError, match="stage op is not registered"):
        build_generation_plan(
            round_index=1,
            candidates=[candidate],
            fate_assignments=[FateAssignment("known", CandidateFate.ACTIVE.value)],
            ranking=ranking,
            stage_graph=[{"op": "invent_new_stage"}],
        )

    with pytest.raises(GenerationPlanError, match="missing prerequisite"):
        build_generation_plan(
            round_index=1,
            candidates=[candidate],
            fate_assignments=[FateAssignment("known", CandidateFate.ACTIVE.value)],
            ranking=ranking,
            stage_graph=[{"op": "archive_update"}],
        )

    with pytest.raises(GenerationPlanError, match="out of order"):
        build_generation_plan(
            round_index=1,
            candidates=[candidate],
            fate_assignments=[FateAssignment("known", CandidateFate.ACTIVE.value)],
            ranking=ranking,
            stage_graph=[
                {"op": "critique_and_verify"},
                {"op": "rank"},
                {"op": "archive_assign"},
                {"op": "generation_plan_validate"},
                {"op": "archive_update"},
                {"op": "compact"},
                {"op": "diagnose"},
                {"op": "stop_check"},
                {"op": "synthesize"},
                {"op": "select_parents"},
            ],
        )


def test_evolution_round_persists_generation_plan_in_round_history_path() -> None:
    winner = CandidateGenome(id="winner", artifact="answer", multihead_scores={"answer_likelihood": 0.9})
    helper = CandidateGenome(id="helper", artifact="validator", multihead_scores={"auxiliary_value": 0.95})
    population = CandidatePopulation([winner, helper])
    archives = ArchiveManager()
    budget = EvolutionBudget(max_rounds=1, branch_factor=2)
    round_pipeline = EvolutionRound(model=None, budget=budget)

    evaluation = round_pipeline.evaluate(
        current_round=1,
        population=population,
        archives=archives,
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="answer", normalized_goal="answer"),
    )

    assert evaluation.generation_plan["plan_id"]
    assert [item["op"] for item in evaluation.generation_plan["stage_graph"]] == [
        "critique_and_verify",
        "rank",
        "archive_assign",
        "generation_plan_validate",
        "archive_update",
        "compact",
        "diagnose",
        "stop_check",
        "select_parents",
        "plan_mutations",
        "generate_offspring",
        "verify_offspring",
    ]
    assert evaluation.generation_plan["completed_stage_ops"] == [
        "critique_and_verify",
        "rank",
        "archive_assign",
        "generation_plan_validate",
        "archive_update",
        "compact",
        "diagnose",
        "stop_check",
    ]
    assert evaluation.progress_event["metadata"]["generation_plan_id"] == evaluation.generation_plan["plan_id"]
    assert archives.history[-1]["generation_plan_id"] == evaluation.generation_plan["plan_id"]


def test_generation_plan_stage_gate_requires_authorized_completed_prerequisites() -> None:
    candidate = CandidateGenome(id="known")
    plan = build_generation_plan(
        round_index=1,
        candidates=[candidate],
        fate_assignments=[FateAssignment("known", CandidateFate.ACTIVE.value)],
        ranking=RelativeRankingResult(best_final_answer_id="known"),
        stage_graph=[
            {"op": "critique_and_verify"},
            {"op": "rank"},
            {"op": "archive_assign"},
            {"op": "generation_plan_validate"},
            {"op": "archive_update"},
            {"op": "compact"},
        ],
    )

    with pytest.raises(GenerationPlanError, match="missing completed prerequisite"):
        assert_stage_ready(plan, "compact", ["critique_and_verify", "rank"])

    with pytest.raises(GenerationPlanError, match="does not authorize"):
        assert_stage_ready(plan, "diagnose", ["critique_and_verify", "rank", "archive_assign", "generation_plan_validate", "archive_update", "compact"])

    assert_stage_ready(plan, "compact", ["critique_and_verify", "rank", "archive_assign", "generation_plan_validate", "archive_update"])


def test_generation_plan_stage_gate_covers_reproduction_prerequisites() -> None:
    candidate = CandidateGenome(id="known")
    plan = build_generation_plan(
        round_index=1,
        candidates=[candidate],
        fate_assignments=[FateAssignment("known", CandidateFate.ACTIVE.value)],
        ranking=RelativeRankingResult(best_final_answer_id="known"),
        stage_graph=[
            {"op": "critique_and_verify"},
            {"op": "rank"},
            {"op": "archive_assign"},
            {"op": "generation_plan_validate"},
            {"op": "archive_update"},
            {"op": "compact"},
            {"op": "diagnose"},
            {"op": "stop_check"},
            {"op": "select_parents"},
            {"op": "plan_mutations"},
            {"op": "generate_offspring"},
            {"op": "verify_offspring"},
        ],
    )
    through_stop = [
        "critique_and_verify",
        "rank",
        "archive_assign",
        "generation_plan_validate",
        "archive_update",
        "compact",
        "diagnose",
        "stop_check",
    ]

    with pytest.raises(GenerationPlanError, match="missing completed prerequisite"):
        assert_stage_ready(plan, "select_parents", through_stop[:-1])
    assert_stage_ready(plan, "select_parents", through_stop)

    with pytest.raises(GenerationPlanError, match="missing completed prerequisite"):
        assert_stage_ready(plan, "plan_mutations", through_stop)
    assert_stage_ready(plan, "plan_mutations", [*through_stop, "select_parents"])


def test_reproduction_advances_generation_plan_and_records_offspring_archive_updates() -> None:
    parents = CandidatePopulation(
        [
            CandidateGenome(id="p1", artifact="answer 1", core_mechanism="m1", multihead_scores={"answer_likelihood": 0.9, "objective_alignment": 0.8}),
            CandidateGenome(id="p2", artifact="answer 2", core_mechanism="m2", multihead_scores={"answer_likelihood": 0.8, "objective_alignment": 0.7}),
        ]
    )
    archives = ArchiveManager()
    budget = EvolutionBudget(max_rounds=2, branch_factor=2)
    round_pipeline = EvolutionRound(model=None, budget=budget)
    policy = EvolutionPolicy()
    contract = NexusObjectiveContract(original_user_goal="answer", normalized_goal="answer")
    evaluation = round_pipeline.evaluate(
        current_round=1,
        population=parents,
        archives=archives,
        policy=policy,
        contract=contract,
    )

    def fail_all(offspring: list[CandidateGenome]) -> list[dict[str, object]]:
        return [{"candidate_id": candidate.id, "passed": False, "diagnostics": ["seed_note_only_patch"]} for candidate in offspring]

    stop_reason, offspring_verification, _compaction = round_pipeline.reproduce(
        current_round=1,
        population=parents,
        archives=archives,
        policy=policy,
        contract=contract,
        world=object(),
        rankings=evaluation.rankings,
        diagnosis=evaluation.diagnosis,
        critiques=evaluation.critiques,
        offspring_verifier=fail_all,
        repair_parent_candidates=evaluation.repair_parent_candidates,
    )
    plan = round_pipeline.last_generation_plan

    assert stop_reason == ""
    assert offspring_verification
    assert plan["completed_stage_ops"][-4:] == ["select_parents", "plan_mutations", "generate_offspring", "verify_offspring"]
    assert plan["parent_ids"]
    assert plan["mutation_objectives"]
    assert plan["offspring_ids"]
    assert plan["reproduction_archive_updates"]
    assert {item["fate"] for item in plan["reproduction_archive_updates"]} == {CandidateFate.FAILED.value}


def test_generation_plan_history_accepts_completed_plan_with_archive_update_witness() -> None:
    candidate = CandidateGenome(id="known")
    ranking = RelativeRankingResult(best_final_answer_id="known")
    archive_plan = build_generation_plan(
        round_index=1,
        candidates=[candidate],
        fate_assignments=[FateAssignment("known", CandidateFate.ACTIVE.value)],
        ranking=ranking,
        stage_graph=[
            {"op": "critique_and_verify"},
            {"op": "rank"},
            {"op": "archive_assign"},
            {"op": "generation_plan_validate"},
            {"op": "archive_update"},
        ],
        source="runtime_rank_archive_transition",
    )
    archives = ArchiveManager()
    apply_generation_plan(archive_plan, [candidate], archives)
    completed_plan = GenerationPlan.from_dict(
        {
            **archive_plan.to_dict(),
            "parent_ids": ["known"],
            "mutation_objectives": ["continue_search"],
            "stage_graph": [
                *archive_plan.stage_graph,
                {"op": "compact"},
                {"op": "diagnose"},
                {"op": "stop_check"},
                {"op": "select_parents"},
                {"op": "plan_mutations"},
                {"op": "generate_offspring"},
                {"op": "verify_offspring"},
            ],
        }
    )
    completed_payload = completed_plan.to_dict()
    completed_payload["plan_id"] = expected_generation_plan_id(completed_plan)
    completed_payload["completed_stage_ops"] = [
        "critique_and_verify",
        "rank",
        "archive_assign",
        "generation_plan_validate",
        "archive_update",
        "compact",
        "diagnose",
        "stop_check",
        "select_parents",
        "plan_mutations",
        "generate_offspring",
        "verify_offspring",
    ]

    assert completed_payload["plan_id"] != archives.history[-1]["generation_plan_id"]

    validate_generation_plan_history(
        [{"generation_plan": completed_payload}],
        archive_history=archives.history,
    )


def test_checkpoint_restore_allows_legacy_history_without_generation_plan(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "checkpoint.json")
    store.save_state(
        round=1,
        max_rounds=1,
        population=CandidatePopulation([CandidateGenome(id="legacy")]),
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="legacy", normalized_goal="legacy"),
        budget_history=[{"round": 1, "ranking": {}}],
    )

    restored = store.restore_state()

    assert restored is not None
    assert restored["budget_history"][0]["round"] == 1


def test_checkpoint_restore_rejects_corrupt_generation_plan_stage_replay(tmp_path) -> None:
    candidate = CandidateGenome(id="known")
    plan = build_generation_plan(
        round_index=1,
        candidates=[candidate],
        fate_assignments=[FateAssignment("known", CandidateFate.ACTIVE.value)],
        ranking=RelativeRankingResult(best_final_answer_id="known"),
        stage_graph=[
            {"op": "critique_and_verify"},
            {"op": "rank"},
            {"op": "archive_assign"},
            {"op": "generation_plan_validate"},
            {"op": "archive_update"},
        ],
    ).to_dict()
    plan["completed_stage_ops"] = ["rank", "critique_and_verify"]
    store = CheckpointStore(tmp_path / "checkpoint.json")
    store.save_state(
        round=1,
        max_rounds=1,
        population=CandidatePopulation([candidate]),
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        contract=NexusObjectiveContract(original_user_goal="goal", normalized_goal="goal"),
        budget_history=[{"round": 1, "generation_plan": plan}],
    )

    with pytest.raises(GenerationPlanError, match="missing completed prerequisite"):
        store.restore_state()


def test_runtime_resume_rejects_tampered_persisted_generation_plan(tmp_path) -> None:
    NexusRuntime(output_dir=tmp_path).run_text("Answer with a rare analogy route.", max_rounds=1)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["budget_history"][-1]["generation_plan"]["completed_stage_ops"] = ["rank"]
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(GenerationPlanError, match="missing completed prerequisite"):
        NexusRuntime(output_dir=tmp_path).resume_from_checkpoint(max_rounds=2)
