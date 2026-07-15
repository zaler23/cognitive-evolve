from __future__ import annotations

import inspect
import json
from pathlib import Path

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.candidates import mutation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, EvolutionRound
from cognitive_evolve_runtime.nexus.move_replay import (
    derive_move_replay_view,
    replay_query_for_parent,
    select_contextual_replay,
)
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.runtime import NexusRuntime
from cognitive_evolve_runtime.persistence.checkpoint import CheckpointStore
from cognitive_evolve_runtime.nexus.search_kernel.branch_allocator import (
    BranchSlot,
    ProductiveBranchAllocation,
)


def _parent(candidate_id: str, family: str) -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        artifact={"parent": candidate_id},
        artifact_type="structured_config",
        core_mechanism=family,
        niche_memberships=[family],
        metadata={"search_space": {"family_id": family}},
    )


def _receipt_record(
    *,
    round_index: int,
    child: CandidateGenome,
    stagnation_type: str,
    move_kind: str,
    sampling_profile: str = "explore:default:0",
) -> dict[str, object]:
    slot_id = str(child.metadata["branch_slot_id"])
    receipt_id = f"intervention-{round_index}-{child.id}"
    return {
        "round": round_index,
        "generation_plan": {
            "round_index": round_index,
            "productive_branch_allocation": {
                "slots": [
                    {
                        "slot_id": slot_id,
                        "arm_id": child.lineage[0],
                        "parent_id": child.parent_ids[0],
                    }
                ]
            },
            "slot_sampling_profiles": [
                {
                    "slot_id": slot_id,
                    "sampling_profile_id": sampling_profile,
                }
            ],
            "intervention_receipts": [
                {
                    "receipt_id": receipt_id,
                    "diagnosed_pressure": {
                        "stagnation_type": stagnation_type,
                        "diagnosis_ref": f"diagnosis-{round_index}",
                    },
                    "intervention_type": "diagnosis_guided_reproduction",
                    "target": {
                        "axis": "",
                        "family": "",
                        "action": move_kind,
                        "slot": slot_id,
                    },
                    "recipient_branch_slot_ids": [slot_id],
                    "produced_candidate_ids": [child.id],
                    "outcome_refs": [f"outcome-{child.id}"],
                }
            ],
        },
    }


def _observed_child(
    *,
    round_index: int,
    parent: CandidateGenome,
    move_kind: str,
    success: bool,
    score: float,
    secret: str = "",
) -> CandidateGenome:
    child_id = f"{parent.id}-{move_kind}-{round_index}"
    metadata: dict[str, object] = {
        "created_in_round": round_index,
        "branch_slot_id": f"slot-{round_index}",
        "branch_slot_binding_status": "bound",
        "evaluator": {
            "status": "passed" if success else "failed",
            "passed": success,
            "metrics": {"score": score},
        },
    }
    if secret:
        metadata["model_chain_of_thought"] = secret
    return CandidateGenome(
        id=child_id,
        parent_ids=[parent.id],
        generation=round_index,
        lineage=[parent.id, child_id],
        artifact={"result": child_id, "private_reasoning": secret},
        artifact_type=parent.artifact_type,
        core_mechanism=parent.core_mechanism,
        niche_memberships=list(parent.niche_memberships),
        metadata=metadata,
    )


def test_contextual_emitter_walk_forward_beats_uniform_random_regret() -> None:
    parents = {
        "LocalOptimum": _parent("context-a", "family-a"),
        "SemanticDrift": _parent("context-b", "family-b"),
    }
    optimal = {"LocalOptimum": "repair", "SemanticDrift": "transfer"}
    candidates = list(parents.values())
    history: list[dict[str, object]] = []
    regret = 0.0
    success_score = 0.0
    rounds = 40

    for round_index in range(1, rounds + 1):
        context = "LocalOptimum" if round_index % 2 else "SemanticDrift"
        parent = parents[context]
        view = derive_move_replay_view(
            budget_history=history,
            candidates=candidates,
            metric_directions={"score": "maximize"},
            before_round=round_index,
        )
        query = replay_query_for_parent(
            parent,
            diagnosis=SearchDiagnosis(stagnation_detected=True, stagnation_type=context),
        )
        selection = select_contextual_replay(
            view,
            query,
            available_move_kinds=("repair", "transfer"),
            donor_role="none",
            default_sampling_profile="explore:default:0",
        )
        chosen = selection.preferred_emitter.move_kind
        succeeded = chosen == optimal[context]
        regret += float(not succeeded)
        if succeeded:
            success_score += 1.0
        child = _observed_child(
            round_index=round_index,
            parent=parent,
            move_kind=chosen,
            success=succeeded,
            score=success_score,
        )
        candidates.append(child)
        history.append(
            _receipt_record(
                round_index=round_index,
                child=child,
                stagnation_type=context,
                move_kind=chosen,
            )
        )

    uniform_random_expected_regret = rounds * 0.5
    assert regret < uniform_random_expected_regret

    final_view = derive_move_replay_view(
        budget_history=history,
        candidates=candidates,
        metric_directions={"score": "maximize"},
        before_round=rounds + 1,
    )
    learned = {}
    for context, parent in parents.items():
        query = replay_query_for_parent(
            parent,
            diagnosis=SearchDiagnosis(stagnation_detected=True, stagnation_type=context),
        )
        learned[context] = select_contextual_replay(
            final_view,
            query,
            available_move_kinds=("repair", "transfer"),
            donor_role="none",
            default_sampling_profile="explore:default:0",
        ).preferred_emitter.move_kind
    assert learned == optimal


def test_grounded_receipt_and_direction_aware_improvement_are_required_for_credit() -> None:
    parent = _parent("min-parent", "min-family")
    baseline = CandidateGenome(
        id="baseline",
        artifact={"result": "baseline"},
        artifact_type=parent.artifact_type,
        metadata={
            "evaluator": {
                "status": "measured",
                "metrics": {"loss": 10.0},
                "selection_binding": {
                    "metric": "loss",
                    "direction": "minimize",
                    "value_type": "number",
                },
            }
        },
    )
    improved = _observed_child(
        round_index=1,
        parent=parent,
        move_kind="repair",
        success=True,
        score=0.0,
    )
    improved.metadata["evaluator"] = {
        "status": "measured",
        "metrics": {"loss": 5.0},
        "selection_binding": {
            "metric": "loss",
            "direction": "minimize",
            "value_type": "number",
        },
    }
    self_report_only = _observed_child(
        round_index=2,
        parent=parent,
        move_kind="transfer",
        success=True,
        score=999.0,
    )
    self_report_only.metadata["model_claimed_success"] = True
    history = [
        _receipt_record(
            round_index=1,
            child=improved,
            stagnation_type="LocalOptimum",
            move_kind="repair",
        )
    ]

    view = derive_move_replay_view(
        budget_history=history,
        candidates=[parent, baseline, improved, self_report_only],
        before_round=3,
    )

    assert [entry.candidate_id for entry in view.entries] == [improved.id]
    assert view.entries[0].reward == 1.0
    assert view.entries[0].reason_codes == ("same_cell_evaluator_elite_improvement",)
    assert self_report_only.id not in {entry.candidate_id for entry in view.entries}


def test_replay_injection_is_capped_receipted_and_excludes_artifact_or_cot_text() -> None:
    parent = _parent("cap-parent", "cap-family")
    candidates = [parent]
    history: list[dict[str, object]] = []
    secret = "LONG PRIVATE CHAIN OF THOUGHT MUST NEVER BE INJECTED"
    score = 0.0
    for round_index, succeeded in enumerate([True, True, True, False, False], start=1):
        if succeeded:
            score += 1.0
        child = _observed_child(
            round_index=round_index,
            parent=parent,
            move_kind="repair",
            success=succeeded,
            score=score,
            secret=secret,
        )
        candidates.append(child)
        history.append(
            _receipt_record(
                round_index=round_index,
                child=child,
                stagnation_type="LocalOptimum",
                move_kind="repair",
            )
        )
    view = derive_move_replay_view(
        budget_history=history,
        candidates=candidates,
        metric_directions={"score": "maximize"},
        before_round=10,
    )
    query = replay_query_for_parent(
        parent,
        diagnosis=SearchDiagnosis(stagnation_detected=True, stagnation_type="LocalOptimum"),
    )
    selection = select_contextual_replay(
        view,
        query,
        available_move_kinds=("repair", "transfer"),
        donor_role="none",
        default_sampling_profile="explore:default:0",
    )
    directive = selection.to_directive()
    serialized = json.dumps(directive, sort_keys=True)

    assert len(directive["successful_moves"]) == 2
    assert len(directive["failed_counterexamples"]) == 1
    assert all(item["receipt_refs"] for item in [*directive["successful_moves"], *directive["failed_counterexamples"]])
    assert secret not in serialized
    assert "private_reasoning" not in serialized
    assert selection.selection_basis["similarity"] == "descriptor_token_jaccard"


def test_current_child_is_excluded_and_checkpoint_round_trip_is_identical(tmp_path: Path) -> None:
    parent = _parent("resume-parent", "resume-family")
    prior = _observed_child(
        round_index=1,
        parent=parent,
        move_kind="repair",
        success=True,
        score=1.0,
    )
    current = _observed_child(
        round_index=2,
        parent=parent,
        move_kind="transfer",
        success=True,
        score=2.0,
    )
    history = [
        _receipt_record(
            round_index=1,
            child=prior,
            stagnation_type="LocalOptimum",
            move_kind="repair",
        ),
        _receipt_record(
            round_index=2,
            child=current,
            stagnation_type="LocalOptimum",
            move_kind="transfer",
        ),
    ]
    candidates = [parent, prior, current]
    before = derive_move_replay_view(
        budget_history=history,
        candidates=candidates,
        metric_directions={"score": "maximize"},
        before_round=2,
    )

    assert {entry.candidate_id for entry in before.entries} == {prior.id}

    store = CheckpointStore(tmp_path / "checkpoint.json")
    store.save_state(
        round=2,
        max_rounds=3,
        population=CandidatePopulation(candidates),
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        diagnosis=SearchDiagnosis(stagnation_detected=True, stagnation_type="LocalOptimum"),
        contract=NexusObjectiveContract(original_user_goal="improve", normalized_goal="improve"),
        budget_history=history,
    )
    checkpoint = store.load()
    assert checkpoint is not None
    restored_history = checkpoint.budget_history
    restored_candidates = CandidatePopulation.from_dict(checkpoint.population).candidates
    after = derive_move_replay_view(
        budget_history=restored_history,
        candidates=restored_candidates,
        metric_directions={"score": "maximize"},
        before_round=2,
    )
    query_before = replay_query_for_parent(
        parent,
        diagnosis=SearchDiagnosis(stagnation_detected=True, stagnation_type="LocalOptimum"),
    )
    query_after = replay_query_for_parent(
        restored_candidates[0],
        diagnosis=SearchDiagnosis(stagnation_detected=True, stagnation_type="LocalOptimum"),
    )
    selection_before = select_contextual_replay(
        before,
        query_before,
        available_move_kinds=("repair", "transfer"),
        donor_role="none",
        default_sampling_profile="explore:default:0",
    )
    selection_after = select_contextual_replay(
        after,
        query_after,
        available_move_kinds=("repair", "transfer"),
        donor_role="none",
        default_sampling_profile="explore:default:0",
    )

    assert before.to_dict() == after.to_dict()
    assert selection_before.to_dict() == selection_after.to_dict()


def test_blend_obligations_force_next_round_repair_targeting() -> None:
    parent = _parent("blend-child", "hybrid-family")
    parent.proof_obligations = [
        {
            "id": "blend-obligation-1",
            "status": "pending",
            "description": "reconcile incompatible mechanism domains",
            "source": "blend_receipt",
        }
    ]
    query = replay_query_for_parent(
        parent,
        diagnosis=SearchDiagnosis(stagnation_detected=True, stagnation_type="LocalOptimum"),
    )
    view = derive_move_replay_view(budget_history=[], candidates=[parent], before_round=2)
    selection = select_contextual_replay(
        view,
        query,
        available_move_kinds=("transfer", "repair"),
        donor_role="none",
        default_sampling_profile="explore:default:0",
    )

    assert selection.preferred_emitter.move_kind == "repair"
    assert selection.to_directive()["target_obligation_ids"] == ["blend-obligation-1"]
    assert selection.selection_basis["obligation_category_boost"] is True


def test_direct_model_replay_is_injected_only_into_the_matching_slot_directive() -> None:
    parent = _parent("directive-parent", "directive-family")
    other_parent = _parent("other-parent", "other-family")
    child = _observed_child(
        round_index=1,
        parent=parent,
        move_kind="repair",
        success=True,
        score=1.0,
    )
    history = [
        _receipt_record(
            round_index=1,
            child=child,
            stagnation_type="LocalOptimum",
            move_kind="repair",
        )
    ]
    view = derive_move_replay_view(
        budget_history=history,
        candidates=[parent, child],
        metric_directions={"score": "maximize"},
        before_round=2,
    )
    query = replay_query_for_parent(
        parent,
        diagnosis=SearchDiagnosis(stagnation_detected=True, stagnation_type="LocalOptimum"),
    )
    selection = select_contextual_replay(
        view,
        query,
        available_move_kinds=("repair", "transfer"),
        donor_role="none",
        default_sampling_profile="explore:default:0",
    )
    other_selection = select_contextual_replay(
        view,
        replay_query_for_parent(
            other_parent,
            diagnosis=SearchDiagnosis(stagnation_detected=True, stagnation_type="SemanticDrift"),
        ),
        available_move_kinds=("transfer", "repair"),
        donor_role="none",
        default_sampling_profile="explore:default:0",
    )
    allocation = ProductiveBranchAllocation(
        slots=(
            BranchSlot(
                slot_id="slot-directive",
                arm_id=parent.id,
                parent_id=parent.id,
                intent="standard_variation",
                variation_index=0,
                ucb_score=1.0,
            ),
            BranchSlot(
                slot_id="slot-other",
                arm_id=other_parent.id,
                parent_id=other_parent.id,
                intent="standard_variation",
                variation_index=0,
                ucb_score=1.0,
            ),
        ),
        arms=(),
        credit_summary={},
    )
    pipeline = EvolutionRound(model=None, budget=EvolutionBudget())

    plan, _latent = pipeline._direct_model_plan(
        parents=[parent, other_parent],
        current_round=2,
        branch_allocation=allocation,
        actions=["transfer"],
        diagnosis=SearchDiagnosis(stagnation_detected=True, stagnation_type="LocalOptimum"),
        contract=NexusObjectiveContract(original_user_goal="improve", normalized_goal="improve"),
        archives=ArchiveManager(),
        population=[parent, other_parent, child],
        latent_exploration_plan={},
        policy=EvolutionPolicy(),
        evaluator_led=False,
        crossover_slots={},
        replay_selections={
            "slot-directive": selection,
            "slot-other": other_selection,
        },
    )
    slots = {item["slot_id"]: item for item in plan.metadata["branch_slots"]}
    slot = slots["slot-directive"]

    assert "move_replay" not in plan.metadata
    assert slot["directive"]["move_replay"]["receipt_refs"]
    assert slot["directive"]["action_hint"] == "repair"
    assert "receipt-grounded replay" in slot["directive"]["instruction"]
    assert slots["slot-other"]["directive"]["action_hint"] == "transfer"
    assert slots["slot-other"]["directive"]["move_replay"]["preferred_emitter"]["move_kind"] == "transfer"


def test_generation_plan_and_progress_event_audit_replay_selection(tmp_path: Path) -> None:
    result = NexusRuntime(output_dir=tmp_path).run_text(
        "Return a concise grounded answer.",
        max_rounds=2,
        stop_policy="max_rounds",
    )
    first_round = result.evolution["budget_history"][0]
    replay_audit = first_round["generation_plan"]["move_replay_audit"]
    event_metadata = first_round["progress_event"]["metadata"]

    assert replay_audit["authority"] == "read_only_budget_history_generation_plan_receipts_lineage"
    assert replay_audit["source_cutoff_round_exclusive"] == 1
    assert replay_audit["slots"]
    assert all(item["selection_basis"] for item in replay_audit["slots"])
    assert event_metadata["move_replay_view_id"] == replay_audit["view_id"]
    assert event_metadata["move_replay_selections"]


def test_fate_based_shadow_reward_path_is_removed() -> None:
    source = inspect.getsource(mutation)

    assert "_shadow_action_reward" not in source
    assert "shadow-action-palette-bandit" not in source
