from __future__ import annotations

import copy

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationOperator, MutationPlan, MutationPlanner
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus import receipts as receipt_module
from cognitive_evolve_runtime.nexus.diagnosis import SearchStateDiagnoser
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget
from cognitive_evolve_runtime.nexus.loop import reproduce_stage as reproduce_stage_module
from cognitive_evolve_runtime.nexus.loop.offspring import _merge_plan_metadata_into_model_offspring, _slot_sampling_policy
from cognitive_evolve_runtime.nexus.loop.round import EvolutionRound
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.search_kernel.branch_allocator import allocate_productive_branches
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector


def _candidate(
    candidate_id: str,
    *,
    family: str = "family-a",
    axis: str = "direct_mainstream",
    root: str | None = None,
    passed: bool = True,
    cell: str = "cell-a",
) -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        lineage=[root or candidate_id, candidate_id],
        artifact=f"artifact-{candidate_id}",
        core_mechanism=family,
        niche_memberships=[cell],
        multihead_scores={"answer_likelihood": 0.8, "objective_alignment": 0.8},
        metadata={
            "search_space": {"family_id": family, "seed_axis": axis},
            "evaluator": {"status": "passed" if passed else "failed", "passed": passed},
        },
    )


def _coverage_policy() -> EvolutionPolicy:
    return EvolutionPolicy(
        search_space={
            "candidate_families": [
                {"id": "family-a"},
                {"id": "family-b"},
            ]
        },
        metadata={
            "seed_portfolio_contract": {
                "mode": "family_x_cognitive_axis",
                "required_axes": ["direct_mainstream", "edge_knowledge"],
            },
            "axis_family_floor_slots": 1,
        },
    )


def test_zero_occupancy_axis_family_reserves_one_audited_slot_then_yields() -> None:
    parent = _candidate("parent-a")
    policy = _coverage_policy()
    diagnosis = SearchStateDiagnoser().diagnose(
        population=[parent],
        archives=ArchiveManager(),
        policy=policy,
    )

    floor = diagnosis.metadata["axis_family_coverage_floor"]
    assert floor["missing_axes"] == ["edge_knowledge"]
    assert floor["missing_families"] == ["family-b"]
    assert floor["floor_slots"] == 1

    allocation = allocate_productive_branches(
        parents=[parent],
        candidates=[parent],
        total_slots=2,
        coverage_floor_targets=floor["targets"],
        coverage_floor_slots=floor["floor_slots"],
    )

    floor_slots = [slot for slot in allocation.slots if slot.intent == "axis_family_coverage_floor"]
    assert len(floor_slots) == 1
    assert floor_slots[0].coverage_target == {"axis": "edge_knowledge", "family": "family-b"}
    assert allocation.to_dict()["coverage_floor"]["reserved_slot_ids"] == [floor_slots[0].slot_id]

    restored = [
        parent,
        _candidate("parent-b", family="family-b", axis="edge_knowledge"),
    ]
    restored_diagnosis = SearchStateDiagnoser().diagnose(
        population=restored,
        archives=ArchiveManager(),
        policy=policy,
    )
    restored_floor = restored_diagnosis.metadata["axis_family_coverage_floor"]
    restored_allocation = allocate_productive_branches(
        parents=restored,
        candidates=restored,
        total_slots=2,
        coverage_floor_targets=restored_floor["targets"],
        coverage_floor_slots=restored_floor["floor_slots"],
    )
    assert restored_floor["targets"] == []
    assert all(slot.intent != "axis_family_coverage_floor" for slot in restored_allocation.slots)


def test_archive_elite_reentry_is_bounded_and_remains_inside_parent_caps() -> None:
    live = _candidate("live", root="live-root", cell="live-cell")
    archives = ArchiveManager()
    for candidate in (
        _candidate("elite-a", root="archive-root", cell="archive-cell-a"),
        _candidate("elite-b", root="archive-root", cell="archive-cell-b"),
        _candidate("elite-c", root="archive-other", passed=False, cell="archive-other"),
    ):
        candidate.mark_fate(CandidateFate.DORMANT.value)
        archives.quality_diversity.update(candidate)
    archive_before = copy.deepcopy(archives.quality_diversity.to_dict())
    population_before = [candidate.to_dict() for candidate in [live]]

    admitted, audit = reproduce_stage_module._archive_elite_admission_view(
        population=[live],
        archives=archives,
        limit=2,
        current_round=4,
    )

    assert len(admitted) == 2
    assert audit["limit"] == 2
    assert audit["admitted_candidate_ids"] == [candidate.id for candidate in admitted]
    assert archives.quality_diversity.to_dict() == archive_before
    assert [candidate.to_dict() for candidate in [live]] == population_before
    selected = ParentSelector().select(
        [live, *admitted],
        archives,
        limit=2,
        eligibility_policy={"max_per_lineage": 1, "max_per_descriptor_cell": 1},
    )
    assert sum(bool(candidate.metadata.get("archive_elite_reentry")) for candidate in selected) <= 1
    assert archives.quality_diversity.to_dict() == archive_before
    assert live.id == "live"
    assert live.current_fate == CandidateFate.ACTIVE.value

    poor_archives = ArchiveManager()
    poor_archive = _candidate("elite-failed", root="failed-root", passed=False, cell="failed-cell")
    poor_archive.mark_fate(CandidateFate.DORMANT.value)
    poor_archives.quality_diversity.update(poor_archive)
    poor_view, _ = reproduce_stage_module._archive_elite_admission_view(
        population=[live],
        archives=poor_archives,
        limit=10,
        current_round=5,
    )
    [tier_winner] = ParentSelector().select([live, *poor_view], poor_archives, limit=1)
    assert tier_winner.id == live.id


def test_reproduce_wires_floor_and_archive_admission_into_generation_plan() -> None:
    population = CandidatePopulation([_candidate("parent-a"), _candidate("parent-b")])
    archives = ArchiveManager()
    archived = _candidate("archived-elite", family="family-b", axis="edge_knowledge", root="archive-root")
    archived.mark_fate(CandidateFate.DORMANT.value)
    archives.quality_diversity.update(archived)
    policy = _coverage_policy()
    policy.metadata["archive_elite_reentry_limit"] = 1
    policy.metadata["islands"] = {"count": 1}
    budget = EvolutionBudget(max_rounds=2, branch_factor=2)
    stage = EvolutionRound(model=None, budget=budget)
    contract = NexusObjectiveContract(original_user_goal="answer", normalized_goal="answer")
    evaluation = stage.evaluate(
        current_round=1,
        population=population,
        archives=archives,
        policy=policy,
        contract=contract,
    )
    diagnosis = SearchStateDiagnoser().diagnose(
        population=population.candidates,
        archives=archives,
        policy=policy,
    )

    stage.reproduce(
        current_round=1,
        population=population,
        archives=archives,
        policy=policy,
        contract=contract,
        world=object(),
        rankings=evaluation.rankings,
        diagnosis=diagnosis,
        critiques=evaluation.critiques,
        offspring_verifier=None,
        repair_parent_candidates=evaluation.repair_parent_candidates,
    )

    plan = stage.last_generation_plan
    assert len(plan["archive_elite_reentry"]["admitted_candidate_ids"]) == 1
    assert len(plan["axis_family_coverage_floor"]["reserved_slot_ids"]) == 1
    slots = plan["productive_branch_allocation"]["slots"]
    [floor_slot] = [slot for slot in slots if slot["intent"] == "axis_family_coverage_floor"]
    assert floor_slot["coverage_target"] == {"axis": "edge_knowledge", "family": "family-b"}


class _DiagnosisModel:
    def __init__(self, actions: list[str] | None = None) -> None:
        self.actions = actions or ["rare_inject", "continue"]

    def diagnose_search_state(self, **_: object) -> dict[str, object]:
        return {
            "stagnation_detected": True,
            "stagnation_type": "DiversityCollapse",
            "recommended_actions": self.actions,
        }


def _receipt_history(*, passed: bool, receipt_id: str, action: str = "rare_inject") -> list[dict[str, object]]:
    outcome = {"candidate_id": "child-a", "passed": passed, "probe_ref": "probe-child-a"}
    return [
        {
            "round": 1,
            "generation_plan": {
                "intervention_receipts": [
                    {
                        "receipt_id": receipt_id,
                        "diagnosed_pressure": {
                            "stagnation_type": "DiversityCollapse",
                            "diagnosis_ref": "diagnosis-a",
                        },
                        "intervention_type": "diagnosis_guided_reproduction",
                        "target": {
                            "axis": "edge_knowledge",
                            "family": "family-b",
                            "action": action,
                            "slot": "branch-a",
                        },
                        "recipient_branch_slot_ids": ["branch-a"],
                        "produced_candidate_ids": ["child-a"],
                        "outcome_refs": ["probe-child-a"],
                    }
                ]
            },
            "offspring_verification": [outcome],
        }
    ]


def test_intervention_receipt_credit_attenuates_success_and_escalates_failure() -> None:
    parent = _candidate("parent")
    success = SearchStateDiagnoser(model=_DiagnosisModel(["strategy_restart", "continue"])).diagnose(
        population=[parent],
        archives=ArchiveManager(),
        history=_receipt_history(passed=True, receipt_id="receipt-success", action="LineageRestart"),
    )
    success_credit = success.metadata["intervention_credit"][-1]
    assert success_credit["receipt_id"] == "receipt-success"
    assert success_credit["decision"] == "attenuate_pressure"
    assert success_credit["pressure_scale"] == 0.5
    assert success.recommended_actions == ["continue"]

    failure_history = _receipt_history(passed=False, receipt_id="receipt-failure")
    failure = SearchStateDiagnoser(model=_DiagnosisModel()).diagnose(
        population=[parent],
        archives=ArchiveManager(),
        history=failure_history,
    )
    failure_credit = failure.metadata["intervention_credit"][-1]
    assert failure_credit["receipt_id"] == "receipt-failure"
    assert failure_credit["decision"] == "escalate_executor"
    assert failure_credit["next_action"] == "strategy_restart"
    assert failure.recommended_actions[0] == "strategy_restart"
    assert receipt_module.intervention_credit_records(failure_history)[-1]["receipt_id"] == "receipt-failure"


def test_strategy_restart_creates_new_lineage_root_and_unknown_action_is_audited() -> None:
    parent = _candidate("failed-parent", root="failed-root")
    parent.core_mechanism = "failed route that must not be retried"
    planner = MutationPlanner()
    [restart_plan] = planner.plan_from_actions([parent], ["strategy_restart"])

    assert restart_plan.operator == MutationOperator.LINEAGE_RESTART
    child = MutationEngine().mutate(parent, restart_plan)
    assert child.parent_ids == []
    assert child.generation == 0
    assert child.lineage == [child.id]
    restart = child.metadata["strategy_restart"]
    assert restart["source_parent_id"] == parent.id
    assert restart["source_lineage_root"] == "failed-root"
    assert restart["temperature"] == "high"
    assert restart["negated_failed_route"] == parent.core_mechanism
    assert "Explicit negation constraint" in str(child.artifact)
    sampling = _slot_sampling_policy(
        EvolutionPolicy(),
        {"directive": {"action_hint": "strategy_restart"}},
    )
    assert sampling is not None
    assert sampling.temperature == 1.0

    model_child = CandidateGenome(
        parent_ids=[parent.id],
        artifact="fresh model restart artifact",
        metadata={"branch_slot_id": "restart-slot", "plan_id": "restart-plan"},
    )
    model_plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=[parent.id],
        metadata={
            "plan_id": "restart-plan",
            "plan_source": "runtime_lineage_envelope",
            "branch_slots": [
                {
                    "slot_id": "restart-slot",
                    "parent_id": parent.id,
                    "arm_id": "failed-root",
                    "intent": "standard_variation",
                    "directive": {"action_hint": "strategy_restart"},
                }
            ],
        },
    )
    _merge_plan_metadata_into_model_offspring([model_child], [model_plan], [parent])
    assert model_child.parent_ids == []
    assert model_child.lineage == [model_child.id]
    assert model_child.metadata["strategy_restart"]["temperature"] == "high"
    assert model_child.metadata["branch_slot_binding_status"] == "bound"

    [fallback] = planner.plan_from_actions([parent], ["unknown_palette_action"])
    assert fallback.operator == MutationOperator.DEEPEN
    assert fallback.metadata["action_fallback"] == {
        "raw_action": "unknown_palette_action",
        "fallback_operator": MutationOperator.DEEPEN,
        "reason": "unknown_action",
    }
