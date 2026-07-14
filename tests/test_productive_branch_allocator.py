from __future__ import annotations

import pytest

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationPlan
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.loop.offspring import _candidate_from_model_offspring, _generate_offspring, _merge_plan_metadata_into_model_offspring
from cognitive_evolve_runtime.nexus.loop.round import _attach_branch_allocation_to_plans
from cognitive_evolve_runtime.nexus.model_adapter import ModelResponseSchemaError
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.search_kernel.branch_allocator import (
    BranchSlot,
    ProductiveBranchAllocation,
    allocate_productive_branches,
    productive_outcomes,
)
from cognitive_evolve_runtime.theory.bandit import OperatorArmStats


def _seed(candidate_id: str, *, score: float = 0.0) -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        artifact={"value": candidate_id},
        metadata={"evaluator": {"status": "passed", "passed": True, "metrics": {"score": score}}},
    )


def _child(candidate_id: str, root: str, *, score: float | None, value: str | None = None) -> CandidateGenome:
    metadata = {}
    if score is not None:
        metadata["evaluator"] = {"status": "passed", "passed": True, "metrics": {"score": score}}
    return CandidateGenome(
        id=candidate_id,
        parent_ids=[root],
        generation=1,
        lineage=[root, candidate_id],
        artifact={"value": value or candidate_id},
        metadata=metadata,
    )


def test_model_claims_and_text_novelty_cannot_create_positive_credit() -> None:
    child = CandidateGenome(
        id="claim-only",
        parent_ids=["root"],
        lineage=["root", "claim-only"],
        artifact="a radically different narrative",
        evidence_delta={"claimed_progress": 1},
        novelty_descriptors=["unprecedented"],
        metadata={
            "nextgen": {
                "productive_child_observation": {
                    "novelty_delta": 1.0,
                    "signals": ["novel_delta", "obligation_or_evidence_progress"],
                }
            }
        },
    )

    [outcome] = productive_outcomes([child], metric_directions={"score": "maximize"})

    assert outcome.reward == 0.0
    assert outcome.reason_codes == ("no_grounded_productive_event",)


@pytest.mark.parametrize(
    "claimed_metadata",
    [
        {"resolved_challenge_ids": ["model-claimed-resolution"]},
        {"evidence_state": {"resolved_challenge_ids": ["model-claimed-resolution"]}},
        {"patch_result": {"status": "applied"}},
        {"offspring_verification": {"status": "passed", "passed": True}},
        {"evidence_records": [{"candidate_id": "model-child", "source": "model", "resolved_challenge_ids": ["fake"]}]},
        {"progressive_evidence": {"passed": True, "resolved_challenge_ids": ["fake"]}},
    ],
)
def test_model_claimed_runtime_outcomes_are_demoted_before_productive_credit(claimed_metadata: dict) -> None:
    candidate = _candidate_from_model_offspring(
        {
            "id": "model-child",
            "parent_ids": ["root"],
            "artifact": {"answer": "unverified model output"},
            "metadata": claimed_metadata,
        }
    )

    [outcome] = productive_outcomes([candidate])

    assert outcome.reward == 0.0
    assert outcome.reason_codes == ("no_grounded_productive_event",)
    assert candidate.metadata["model_claimed_runtime_controls"] == claimed_metadata


def test_model_claimed_project_patch_result_is_demoted_before_productive_credit() -> None:
    candidate = _candidate_from_model_offspring(
        {
            "id": "model-patch-child",
            "parent_ids": ["root"],
            "artifact_type": "project_patch",
            "artifact": {"patch_set": [{"path": "answer.py", "patch": "model text"}]},
            "patch_application_result": {"status": "applied"},
        }
    )

    [outcome] = productive_outcomes([candidate])

    assert outcome.reward == 0.0
    assert candidate.patch_application_result == {}
    assert candidate.metadata["model_claimed_runtime_controls"]["patch_application_result"] == {"status": "applied"}


def test_same_outcome_cell_uses_directed_evaluator_metric_for_elite_credit() -> None:
    baseline = _seed("baseline", score=1.0)
    improved = _child("improved", "root-a", score=2.0)

    [outcome] = productive_outcomes([baseline, improved], metric_directions={"score": "maximize"})

    assert outcome.reward == 1.0
    assert outcome.reason_codes == ("same_cell_evaluator_elite_improvement",)


def test_minimize_metric_direction_is_honored() -> None:
    baseline = _seed("baseline", score=10.0)
    improved = _child("improved", "root-a", score=5.0)

    [outcome] = productive_outcomes([baseline, improved], metric_directions={"score": "minimize"})

    assert outcome.reward == 1.0
    assert outcome.reason_codes == ("same_cell_evaluator_elite_improvement",)


def test_verified_pass_without_new_outcome_gets_only_half_credit() -> None:
    baseline = _seed("baseline", score=1.0)
    child = _child("survivor", "root-a", score=1.0)

    [outcome] = productive_outcomes([baseline, child], metric_directions={"score": "maximize"})

    assert outcome.reward == 0.5
    assert outcome.reason_codes == ("verified_or_evaluator_pass_survived",)


def test_boolean_metric_combinations_do_not_create_new_outcome_cells() -> None:
    baseline = _seed("baseline")
    baseline.metadata["evaluator"]["metrics"] = {"check_a": False, "check_b": False}
    child = _child("different-bools", "root-a", score=None)
    child.metadata["evaluator"] = {
        "status": "passed",
        "passed": True,
        "metrics": {"check_a": True, "check_b": True},
    }

    [outcome] = productive_outcomes([baseline, child])

    assert outcome.reward == 0.5
    assert outcome.reason_codes == ("verified_or_evaluator_pass_survived",)


@pytest.mark.parametrize("failure_source", ["evaluator", "patch"])
def test_observed_evaluator_or_patch_failure_cannot_receive_credit(failure_source: str) -> None:
    child = _child("failed", "root-a", score=None)
    if failure_source == "evaluator":
        child.metadata["evaluator"] = {"status": "failed", "passed": False}
    else:
        child.patch_application_result = {"status": "failed"}

    [outcome] = productive_outcomes([child])

    assert outcome.reward == 0.0
    assert outcome.risk == 1.0
    assert outcome.reason_codes == ("observed_evaluator_or_patch_failure",)


def test_ineligible_candidate_cannot_consume_challenge_or_pareto_baseline() -> None:
    baseline = _seed("baseline", score=1.0)
    failed = _child("failed", "root-a", score=10.0)
    failed.metadata.update(
        {
            "created_in_round": 1,
            "evaluator": {"status": "failed", "passed": False, "metrics": {"score": 10.0}},
            "evidence_state": {"resolved_challenge_ids": ["challenge-a"]},
        }
    )
    improved = _child("improved", "root-b", score=2.0)
    improved.metadata["created_in_round"] = 2
    resolved = _child("resolved", "root-c", score=None)
    resolved.metadata.update(
        {
            "created_in_round": 3,
            "evaluator": {"status": "passed", "passed": True},
            "evidence_state": {"resolved_challenge_ids": ["challenge-a"]},
        }
    )

    outcomes = productive_outcomes([baseline, failed, improved, resolved], metric_directions={"score": "maximize"})
    by_id = {outcome.candidate_id: outcome for outcome in outcomes}

    assert by_id["failed"].reason_codes == ("observed_evaluator_or_patch_failure",)
    assert by_id["improved"].reward == 1.0
    assert by_id["improved"].reason_codes == ("same_cell_evaluator_elite_improvement",)
    assert by_id["resolved"].reward == 1.0
    assert by_id["resolved"].reason_codes == ("new_challenge_resolution",)


def test_same_cohort_stable_passes_share_one_half_credit_event() -> None:
    baseline = _seed("baseline", score=1.0)
    first = _child("first", "root-a", score=1.0)
    second = _child("second", "root-b", score=1.0)
    first.metadata["created_in_round"] = 1
    second.metadata["created_in_round"] = 1

    outcomes = productive_outcomes([baseline, first, second], metric_directions={"score": "maximize"})

    assert {outcome.candidate_id: outcome.reward for outcome in outcomes} == {"first": 0.25, "second": 0.25}
    assert {outcome.reason_codes for outcome in outcomes} == {("verified_or_evaluator_pass_survived",)}


def test_same_cohort_new_outcome_credit_is_independent_of_model_ordering() -> None:
    first = _child("first", "root-a", score=1.0)
    second = _child("second", "root-b", score=1.0)
    first.metadata["created_in_round"] = 1
    second.metadata["created_in_round"] = 1
    first.created_at = "9999-model-late"
    second.created_at = "0000-model-early"

    outcomes = productive_outcomes([first, second], metric_directions={"score": "maximize"})

    assert {outcome.candidate_id: outcome.reward for outcome in outcomes} == {"first": 0.5, "second": 0.5}
    assert {outcome.reason_codes for outcome in outcomes} == {("new_grounded_outcome_cell",)}


def test_unbound_children_neither_dilute_nor_consume_grounded_events() -> None:
    unbound_cell = _child("unbound-cell", "root-a", score=1.0)
    bound_cell = _child("bound-cell", "root-b", score=1.0)
    unbound_cell.metadata.update({"created_in_round": 1, "branch_slot_binding_status": "unbound_parent_mismatch"})
    bound_cell.metadata.update({"created_in_round": 1, "branch_slot_binding_status": "bound"})

    unbound_challenge = _child("unbound-challenge", "root-c", score=None)
    unbound_challenge.metadata.update(
        {
            "created_in_round": 2,
            "branch_slot_binding_status": "unbound_parent_mismatch",
            "evidence_state": {"resolved_challenge_ids": ["challenge-a"]},
        }
    )
    bound_challenge = _child("bound-challenge", "root-d", score=None)
    bound_challenge.metadata.update(
        {
            "created_in_round": 3,
            "branch_slot_binding_status": "bound",
            "evidence_state": {"resolved_challenge_ids": ["challenge-a"]},
        }
    )
    unbound_challenge.artifact = dict(bound_challenge.artifact)

    outcomes = productive_outcomes([unbound_cell, bound_cell, unbound_challenge, bound_challenge])
    by_id = {outcome.candidate_id: outcome for outcome in outcomes}

    assert by_id["unbound-cell"].reward == 0.0
    assert by_id["unbound-cell"].risk == 1.0
    assert by_id["unbound-cell"].reason_codes == ("unbound_branch_slot",)
    assert by_id["bound-cell"].reward == 1.0
    assert by_id["bound-cell"].reason_codes == ("new_grounded_outcome_cell",)
    assert by_id["unbound-challenge"].reward == 0.0
    assert by_id["unbound-challenge"].reason_codes == ("unbound_branch_slot",)
    assert by_id["bound-challenge"].reward == 1.0
    assert by_id["bound-challenge"].reason_codes == ("new_challenge_resolution",)


def test_unbound_same_artifact_does_not_mark_bound_sibling_duplicate() -> None:
    unbound = _child("unbound", "root-a", score=1.0, value="shared")
    bound = _child("bound", "root-b", score=1.0, value="shared")
    unbound.metadata.update({"created_in_round": 1, "branch_slot_binding_status": "unbound_parent_mismatch"})
    bound.metadata.update({"created_in_round": 1, "branch_slot_binding_status": "bound"})

    outcomes = productive_outcomes([unbound, bound])
    by_id = {outcome.candidate_id: outcome for outcome in outcomes}

    assert by_id["unbound"].reason_codes == ("unbound_branch_slot",)
    assert by_id["bound"].reward == 1.0
    assert by_id["bound"].reason_codes == ("new_grounded_outcome_cell",)


def test_unbound_candidate_does_not_set_the_pareto_baseline() -> None:
    baseline = _seed("baseline", score=1.0)
    unbound = _child("unbound", "root-a", score=10.0)
    bound = _child("bound", "root-b", score=2.0)
    unbound.metadata.update({"created_in_round": 1, "branch_slot_binding_status": "unbound_parent_mismatch"})
    bound.metadata.update({"created_in_round": 2, "branch_slot_binding_status": "bound"})

    outcomes = productive_outcomes([baseline, unbound, bound], metric_directions={"score": "maximize"})
    by_id = {outcome.candidate_id: outcome for outcome in outcomes}

    assert by_id["unbound"].reason_codes == ("unbound_branch_slot",)
    assert by_id["bound"].reward == 1.0
    assert by_id["bound"].reason_codes == ("same_cell_evaluator_elite_improvement",)


def test_resolved_challenge_combinations_do_not_create_exponential_outcome_cells() -> None:
    first = _child("first", "root-a", score=None)
    second = _child("second", "root-b", score=None)
    combined = _child("combined", "root-c", score=None)
    first.metadata["evidence_state"] = {"resolved_challenge_ids": ["a"]}
    second.metadata["evidence_state"] = {"resolved_challenge_ids": ["b"]}
    combined.metadata["evidence_state"] = {"resolved_challenge_ids": ["a", "b"]}
    first.metadata["created_in_round"] = 1
    second.metadata["created_in_round"] = 2
    combined.metadata["created_in_round"] = 3

    outcomes = productive_outcomes([first, second, combined])

    assert [outcome.reason_codes for outcome in outcomes] == [
        ("new_challenge_resolution",),
        ("new_challenge_resolution",),
        ("no_grounded_productive_event",),
    ]
    assert [outcome.reward for outcome in outcomes] == [1.0, 1.0, 0.0]


def test_exact_artifact_duplicate_gets_zero_even_if_metric_improves() -> None:
    baseline = _seed("baseline", score=1.0)
    duplicate = _child("duplicate", "root-a", score=9.0, value="baseline")
    duplicate.artifact = dict(baseline.artifact)

    [outcome] = productive_outcomes([baseline, duplicate], metric_directions={"score": "maximize"})

    assert outcome.reward == 0.0
    assert outcome.risk == 1.0
    assert outcome.reason_codes == ("exact_phenotype_duplicate",)


def test_allocator_is_nonuniform_but_preserves_an_untried_lineage_slot() -> None:
    baseline = _seed("baseline", score=0.0)
    parent_a = CandidateGenome(id="A", artifact={"parent": "A"})
    parent_b = CandidateGenome(id="B", artifact={"parent": "B"})
    parent_c = CandidateGenome(id="C", artifact={"parent": "C"})
    a1 = _child("a1", "A", score=1.0)
    a2 = _child("a2", "A", score=2.0)
    b1 = _child("b1", "B", score=None)

    first = allocate_productive_branches(
        parents=[parent_a, parent_b, parent_c],
        candidates=[baseline, parent_a, parent_b, parent_c, a1, a2, b1],
        metric_directions={"score": "maximize"},
        total_slots=4,
    )
    second = allocate_productive_branches(
        parents=[parent_a, parent_b, parent_c],
        candidates=[baseline, parent_a, parent_b, parent_c, a1, a2, b1],
        metric_directions={"score": "maximize"},
        total_slots=4,
    )

    arm_ids = [slot.arm_id for slot in first.slots]

    assert len(arm_ids) == 4
    assert arm_ids[0] == "C"
    assert first.slots[0].intent == "explore_fresh"
    assert first.slots[0].coverage_bonus == 0.0
    assert first.slots[0].coverage_scale == 0.0
    assert first.slots[0].allocation_score == 0.0
    assert set(arm_ids) == {"A", "B", "C"}
    assert first.to_dict() == second.to_dict()


def test_family_coverage_bonus_is_scaled_by_the_current_finite_ucb_span() -> None:
    parent_dense = CandidateGenome(
        id="dense-root",
        lineage=["dense-root"],
        artifact={"value": "dense-root"},
        metadata={"search_space": {"family_id": "dense"}},
    )
    parent_sparse = CandidateGenome(
        id="sparse-root",
        lineage=["sparse-root"],
        artifact={"value": "sparse-root"},
        metadata={"search_space": {"family_id": "sparse"}},
    )
    dense_history = [
        CandidateGenome(
            id=f"dense-{index}",
            lineage=["dense-root", f"dense-{index}"],
            artifact={"value": f"dense-{index}"},
            metadata={"search_space": {"family_id": "dense"}},
        )
        for index in range(2)
    ]
    budget_history = [
        {
            "round": 0,
            "generation_plan": {
                "productive_branch_allocation": {
                    "slots": [
                        {"slot_id": "dense-0", "arm_id": "dense-root"},
                        {"slot_id": "dense-1", "arm_id": "dense-root"},
                        {"slot_id": "sparse-0", "arm_id": "sparse-root"},
                    ]
                }
            },
        }
    ]

    allocation = allocate_productive_branches(
        parents=[parent_dense, parent_sparse],
        candidates=[parent_dense, parent_sparse, *dense_history],
        budget_history=budget_history,
        total_slots=1,
    )

    [slot] = allocation.slots
    assert slot.arm_id == "sparse-root"
    assert allocation.observed_family_counts == {"dense": 3, "sparse": 1}
    assert slot.coverage_bonus == pytest.approx(2 / 3, abs=1e-6)
    assert slot.coverage_scale > 0.0
    assert slot.allocation_score - slot.ucb_score == pytest.approx(
        slot.coverage_bonus * slot.coverage_scale,
        abs=2e-6,
    )
    assert allocation.to_dict()["observed_family_counts"] == {"dense": 3, "sparse": 1}


def test_general_family_sets_density_scale_but_never_receives_coverage_bonus() -> None:
    general = CandidateGenome(
        id="z-general-root",
        lineage=["z-general-root"],
        artifact={"value": "general"},
        metadata={"search_space": {"family_id": "general"}},
    )
    rare = CandidateGenome(
        id="a-rare-root",
        lineage=["a-rare-root"],
        artifact={"value": "rare"},
        metadata={"search_space": {"family_id": "rare"}},
    )
    general_history = [
        CandidateGenome(
            id=f"general-{index}",
            artifact={"value": index},
            metadata={"search_space": {"family_id": "general"}},
        )
        for index in range(2)
    ]
    budget_history = [
        {
            "round": 0,
            "generation_plan": {
                "productive_branch_allocation": {
                    "slots": [
                        {"slot_id": "general-0", "arm_id": "z-general-root"},
                        {"slot_id": "rare-0", "arm_id": "a-rare-root"},
                    ]
                }
            },
        }
    ]

    allocation = allocate_productive_branches(
        parents=[general, rare],
        candidates=[general, rare, *general_history],
        budget_history=budget_history,
        total_slots=1,
    )

    [slot] = allocation.slots
    assert allocation.observed_family_counts == {"general": 3, "rare": 1}
    assert slot.arm_id == "a-rare-root"
    assert slot.ucb_score == slot.allocation_score
    assert slot.coverage_scale == 0.0
    assert slot.coverage_bonus == pytest.approx(2 / 3, abs=1e-6)


def test_single_plan_binds_each_child_to_one_distinct_branch_slot() -> None:
    parent = CandidateGenome(id="P", artifact={"value": 0})
    slots = [
        {
            "slot_id": f"slot-{index}",
            "arm_id": "P",
            "parent_id": "P",
            "intent": "standard_variation",
            "variation_index": index,
            "ucb_score": 1.0,
            "directive": {
                "instruction": f"trusted instruction {index}",
                "search_pressure": {"slot": index},
                "search_pressure_id": f"pressure-{index}",
                "target_challenge_ids": [f"challenge-{index}"],
                "artifact_policy": {"slot": index},
                "latent_exploration_action": {"action_id": f"latent-{index}"},
                "problem_model_discrimination_action": {"action_id": f"problem-{index}"},
                "policy_directives": {"mandatory_actions": [f"probe-{index}"]},
            },
        }
        for index in range(3)
    ]
    plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=["P"],
        metadata={
            "plan_id": "direct",
            "plan_source": "runtime_lineage_envelope",
            "branch_slots": slots,
            "action_palette": ["envelope-wide"],
            "archive_search_memory": {"all": "archives"},
            "latent_exploration": {"all": "latent"},
            "search_diagnosis": {"all": "diagnosis"},
        },
    )
    children = [
        CandidateGenome(
            id=f"C{index}",
            parent_ids=["P"],
            artifact={"value": index + 1},
            metadata={
                "search_pressure_id": "model-spoof",
                "target_challenge_ids": ["model-spoof"],
                "artifact_policy": {"model": "spoof"},
                "latent_exploration_action": {"action_id": "model-spoof"},
                "problem_model_discrimination_action": {"action_id": "model-spoof"},
                "action_palette": ["model-spoof"],
            },
        )
        for index in range(3)
    ]

    _merge_plan_metadata_into_model_offspring(children, [plan], [parent])

    assert {child.metadata["branch_slot_id"] for child in children} == {"slot-0", "slot-1", "slot-2"}
    assert all(child.metadata["branch_slot_binding_status"] == "bound" for child in children)
    assert all("branch_slots" not in child.metadata for child in children)
    for child in children:
        slot_index = child.metadata["branch_slot_id"].removeprefix("slot-")
        assert child.metadata["search_pressure_id"] == f"pressure-{slot_index}"
        assert child.metadata["target_challenge_ids"] == [f"challenge-{slot_index}"]
        assert child.metadata["artifact_policy"] == {"slot": int(slot_index)}
        assert child.metadata["latent_exploration_action"]["action_id"] == f"latent-{slot_index}"
        assert child.metadata["problem_model_discrimination_action"]["action_id"] == f"problem-{slot_index}"
        assert child.metadata["branch_slot_directive"]["policy_directives"] == {
            "mandatory_actions": [f"probe-{slot_index}"]
        }
        assert child.metadata["model_claimed_runtime_controls"]["search_pressure_id"] == "model-spoof"
        assert "action_palette" not in child.metadata
        assert "archive_search_memory" not in child.metadata
        assert "latent_exploration" not in child.metadata
        assert "search_diagnosis" not in child.metadata


def test_claimed_slot_with_wrong_parent_is_kept_but_cannot_receive_credit() -> None:
    parent_a = CandidateGenome(id="A", artifact={"value": "a"})
    parent_b = CandidateGenome(id="B", artifact={"value": "b"})
    plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=["A", "B"],
        metadata={
            "plan_id": "direct",
            "plan_source": "runtime_lineage_envelope",
            "branch_slots": [
                {"slot_id": "slot-a", "arm_id": "A", "parent_id": "A", "intent": "standard_variation", "variation_index": 0, "ucb_score": 1.0}
            ],
        },
    )
    child = CandidateGenome(id="C", parent_ids=["B"], artifact={"value": "changed"}, metadata={"branch_slot_id": "slot-a"})

    _merge_plan_metadata_into_model_offspring([child], [plan], [parent_a, parent_b])

    assert child.metadata["branch_slot_binding_status"] == "unbound_parent_mismatch"
    assert child.metadata["model_claimed_branch_slot_id"] == "slot-a"
    assert "branch_arm_id" not in child.metadata


def test_runtime_lineage_envelope_resolves_a_unique_model_parent_alias() -> None:
    parent_a = CandidateGenome(
        id="A",
        lineage=["A"],
        artifact={"value": "a"},
        metadata={"model_claimed_candidate_id": "FX0"},
    )
    parent_b = CandidateGenome(
        id="B",
        lineage=["B"],
        artifact={"value": "b"},
        metadata={"model_claimed_candidate_id": "FX1"},
    )
    plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=["A", "B"],
        metadata={
            "plan_id": "direct",
            "plan_source": "runtime_lineage_envelope",
            "branch_slots": [
                {"slot_id": "slot-b", "arm_id": "B", "parent_id": "B", "intent": "standard_variation", "variation_index": 0, "ucb_score": 1.0},
                {"slot_id": "slot-a", "arm_id": "A", "parent_id": "A", "intent": "standard_variation", "variation_index": 0, "ucb_score": 1.0},
            ],
        },
    )
    child = CandidateGenome(
        id="C",
        parent_ids=["FX0"],
        artifact={"value": "changed"},
    )

    _merge_plan_metadata_into_model_offspring([child], [plan], [parent_a, parent_b])

    assert child.parent_ids == ["A"]
    assert child.lineage == ["A", "C"]
    assert child.metadata["branch_slot_id"] == "slot-a"
    assert child.metadata["branch_slot_binding_status"] == "bound"
    assert child.metadata["model_claimed_parent_ids"] == ["FX0"]
    assert productive_outcomes([child])[0].arm_id == "A"


def test_runtime_lineage_envelope_rejects_an_ambiguous_model_parent_alias() -> None:
    parents = [
        CandidateGenome(
            id=parent_id,
            lineage=[parent_id],
            artifact={"value": parent_id},
            metadata={"model_claimed_candidate_id": "duplicate"},
        )
        for parent_id in ("A", "B")
    ]
    plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=["A", "B"],
        metadata={
            "plan_id": "direct",
            "plan_source": "runtime_lineage_envelope",
            "branch_slots": [
                {"slot_id": "slot-a", "arm_id": "A", "parent_id": "A", "intent": "standard_variation", "variation_index": 0, "ucb_score": 1.0},
                {"slot_id": "slot-b", "arm_id": "B", "parent_id": "B", "intent": "standard_variation", "variation_index": 0, "ucb_score": 1.0},
            ],
        },
    )
    child = CandidateGenome(id="C", parent_ids=["duplicate"], artifact={"value": "changed"})

    with pytest.raises(ModelResponseSchemaError, match="parent_ids outside its mutation plan"):
        _merge_plan_metadata_into_model_offspring([child], [plan], parents)


def test_multi_parent_child_cannot_bind_a_non_primary_lineage_slot() -> None:
    parent_a = CandidateGenome(id="A", lineage=["A"], artifact={"value": "a"})
    parent_b = CandidateGenome(id="B", lineage=["B"], artifact={"value": "b"})
    plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=["A", "B"],
        metadata={
            "plan_id": "direct",
            "branch_slots": [
                {"slot_id": "slot-b", "arm_id": "B", "parent_id": "B", "intent": "standard_variation", "variation_index": 0, "ucb_score": 1.0}
            ],
        },
    )
    child = CandidateGenome(
        id="C",
        parent_ids=["A", "B"],
        artifact={"value": "changed"},
        metadata={"branch_slot_id": "slot-b"},
    )

    _merge_plan_metadata_into_model_offspring([child], [plan], [parent_a, parent_b])

    assert child.lineage[:2] == ["A", "B"]
    assert child.metadata["branch_slot_binding_status"] == "unbound_parent_mismatch"
    assert "branch_arm_id" not in child.metadata


def test_allocator_never_lets_branch_metadata_override_lineage_root() -> None:
    parent_a = CandidateGenome(id="A", lineage=["A"], artifact={"value": "a"})
    parent_b = CandidateGenome(id="B", lineage=["B"], artifact={"value": "b"})
    child = _child("child", "A", score=1.0)
    child.metadata.update(
        {
            "branch_slot_id": "claimed-slot",
            "branch_slot_binding_status": "bound",
            "branch_arm_id": "B",
        }
    )

    allocation = allocate_productive_branches(
        parents=[parent_a, parent_b],
        candidates=[parent_a, parent_b, child],
        metric_directions={"score": "maximize"},
        total_slots=2,
    )

    assert all(arm.reward_sum == 0.0 for arm in allocation.arms)


def test_plan_slot_quota_never_falls_across_parent_lineages() -> None:
    slots = tuple(
        [
            BranchSlot(slot_id=f"a-{index}", arm_id="A", parent_id="A", intent="standard_variation", variation_index=index, ucb_score=1.0)
            for index in range(3)
        ]
        + [BranchSlot(slot_id="b-0", arm_id="B", parent_id="B", intent="standard_variation", variation_index=0, ucb_score=1.0)]
    )
    allocation = ProductiveBranchAllocation(
        slots=slots,
        arms=(OperatorArmStats(arm_id="A"), OperatorArmStats(arm_id="B")),
        credit_summary={},
    )
    plans = [
        MutationPlan(operator="Deepen", parent_ids=[parent_id], metadata={"plan_id": f"plan-{index}"})
        for index, parent_id in enumerate(["A", "A", "B", "B"])
    ]
    rejected: list[dict] = []

    attached = _attach_branch_allocation_to_plans(plans, allocation, include_manifest=True, rejected_out=rejected)

    assert [plan.parent_ids for plan in attached] == [["A"], ["A"], ["B"]]
    assert [plan.metadata["branch_arm_id"] for plan in attached] == ["A", "A", "B"]
    assert rejected == [{"reason": "parent_slot_quota_exceeded", "plan_id": "plan-3", "parent_ids": ["B"]}]


def test_allocated_direct_generation_never_spends_a_duplicate_refill_call() -> None:
    parent = CandidateGenome(id="P", artifact={"value": "parent"})
    prior = CandidateGenome(id="OLD", artifact={"value": "duplicate"}, concise_claim="duplicate", core_mechanism="duplicate")
    plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=["P"],
        metadata={
            "branch_slots": [
                {"slot_id": "slot-p", "arm_id": "P", "parent_id": "P", "intent": "standard_variation", "variation_index": 0, "ucb_score": 1.0}
            ]
        },
    )

    class Model:
        calls = 0

        def generate_offspring(self, **_: object) -> list[CandidateGenome]:
            self.calls += 1
            return [CandidateGenome(parent_ids=["P"], artifact={"value": "duplicate"}, concise_claim="duplicate", core_mechanism="duplicate")]

    model = Model()
    offspring = _generate_offspring(
        model=model,
        mutation_engine=MutationEngine(),
        parents=[parent],
        plans=[plan],
        world={},
        contract=NexusObjectiveContract(original_user_goal="test", normalized_goal="test"),
        policy=EvolutionPolicy(),
        candidate_pool=[parent, prior],
        target_size=1,
    )

    assert model.calls == 1
    assert offspring == []
