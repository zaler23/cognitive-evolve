from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.crossover import crossover
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.candidates.mutation import (
    MoveContractError,
    MutationEngine,
    MutationPlan,
    MutationPlanner,
    recompute_move_delta,
)
from cognitive_evolve_runtime.candidates.project_candidate import PatchOperation, ProjectCandidateGenome
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.loop.controller import EvolutionLoopController
from cognitive_evolve_runtime.nexus.loop.offspring import _generate_offspring, _plan_mutations
from cognitive_evolve_runtime.nexus.loop.reproduce_stage import _role_crossover_slots
from cognitive_evolve_runtime.nexus.model_adapter import ModelResponseSchemaError
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.receipts import DONOR_ROLES, record_reproduction_receipts
from cognitive_evolve_runtime.nexus.search_kernel.branch_allocator import BranchSlot, ProductiveBranchAllocation
from cognitive_evolve_runtime.persistence.checkpoint import CheckpointStore


class _World:
    kind = "text"


class _Contract:
    objective = "offline cross-move fixture"

    def to_dict(self) -> dict[str, Any]:
        return {"objective": self.objective}


def _parents() -> tuple[CandidateGenome, CandidateGenome]:
    primary = CandidateGenome(
        id="primary",
        generation=2,
        lineage=["root-primary", "primary"],
        artifact={"kept": "primary-anchor", "mechanism": "primary-mechanism"},
        verification_result={"passed": True},
        verification_trace=[{"passed": True, "result_ref": "parent-pass"}],
    )
    donor = CandidateGenome(
        id="donor",
        generation=5,
        lineage=["root-donor", "donor"],
        artifact={"mechanism": "donor-fragment"},
        verification_result={"passed": True},
    )
    return primary, donor


def _crossover_slot(role: str) -> dict[str, Any]:
    return {
        "slot_id": f"slot-{role}",
        "arm_id": "root-primary",
        "parent_id": "primary",
        "primary_parent_id": "primary",
        "donor_parent_id": "donor",
        "donor_role": role,
        "intent": "escape_stagnation",
        "required_contribution_map": {
            "generic_space_mapping": "element-level shared structural role mapping",
            "retained_from_primary": "primary elements retained by the child",
            "borrowed_from_donor": "donor elements materially present in the child",
            "structural_correspondence": "cross-parent structural relation",
            "emergent_delta": "child-only structure",
            "incompatibilities": "mapping conflicts",
            "unresolved_obligations": "open merge obligations",
        },
    }


def _blend_receipt(*, mapping: bool = True, borrowed: bool = True, incompatible: bool = False) -> dict[str, Any]:
    receipt = {
        "retained_from_primary": ["primary-anchor"],
        "borrowed_from_donor": ["donor-fragment"] if borrowed else [],
        "structural_correspondence": ["mechanism -> mechanism"],
        "emergent_delta": ["role-specific synthesis"],
        "incompatibilities": ["mechanism domains require reconciliation"] if incompatible else [],
        "unresolved_obligations": [],
    }
    if mapping:
        receipt["generic_space_mapping"] = [
            {
                "primary": "mechanism",
                "donor": "mechanism",
                "child": "mechanism",
            }
        ]
    return receipt


class _CrossoverModel:
    def __init__(
        self,
        *,
        mapping: bool = True,
        borrowed: bool = True,
        material: bool | None = None,
        incompatible: bool = False,
        outsider: bool = False,
    ) -> None:
        self.mapping = mapping
        self.borrowed = borrowed
        self.material = borrowed if material is None else material
        self.incompatible = incompatible
        self.outsider = outsider
        self.parent_calls: list[list[str]] = []

    def generate_offspring(
        self,
        *,
        plans: list[MutationPlan],
        parents: list[CandidateGenome],
        world: Any,
        contract: Any,
        policy: EvolutionPolicy,
    ) -> list[dict[str, Any]]:
        self.parent_calls.append([parent.id for parent in parents])
        slot = plans[0].metadata["branch_slots"][0]
        mechanism = "donor-fragment" if self.material else "primary-mechanism"
        return [
            {
                "id": "model-child",
                "parent_ids": ["primary", "outside"] if self.outsider else ["primary", "donor"],
                "generation": 99,
                "artifact": {
                    "kept": "primary-anchor",
                    "mechanism": mechanism,
                    "emergent": slot["donor_role"],
                },
                "concise_claim": "role crossover child",
                "core_mechanism": mechanism,
                "verification_result": {"passed": True},
                "verification_trace": [{"passed": True, "result_ref": "model-claimed-pass"}],
                "metadata": {
                    "branch_slot_id": slot["slot_id"],
                    "blend_receipt": _blend_receipt(
                        mapping=self.mapping,
                        borrowed=self.borrowed,
                        incompatible=self.incompatible,
                    ),
                },
            }
        ]


def _generate_crossover(role: str, model: _CrossoverModel) -> CandidateGenome:
    primary, donor = _parents()
    slot = _crossover_slot(role)
    plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=[primary.id, donor.id],
        metadata={
            "plan_id": "runtime-crossover-plan",
            "plan_source": "runtime_lineage_envelope",
            "branch_slots": [slot],
        },
    )
    [child] = _generate_offspring(
        model=model,
        mutation_engine=MutationEngine(),
        parents=[primary, donor],
        plans=[plan],
        world=_World(),
        contract=_Contract(),
        policy=EvolutionPolicy(),
        candidate_pool=[primary, donor],
        target_size=1,
    )
    return child


@pytest.mark.parametrize("role", sorted(DONOR_ROLES))
def test_role_crossover_slot_passes_both_parents_and_binds_truthful_lineage(role: str) -> None:
    model = _CrossoverModel()

    child = _generate_crossover(role, model)

    assert model.parent_calls == [["primary", "donor"]]
    assert child.parent_ids == ["primary", "donor"]
    assert child.generation == 6
    assert child.lineage == ["root-primary", "primary", "root-donor", "donor", child.id]
    assert child.metadata["donor_role"] == role
    assert child.metadata["blend_receipt"]["generic_space_mapping"]
    assert child.verification_result == {}
    assert child.verification_trace == []


@pytest.mark.parametrize(
    ("mapping", "borrowed", "material", "reason"),
    [
        (False, True, True, "generic_space_mapping"),
        (True, False, False, "borrowed_from_donor"),
        (True, True, False, "no engine-observable material donor contribution"),
    ],
)
def test_unreceipted_or_zero_contribution_crossover_is_honestly_single_parent(
    mapping: bool,
    borrowed: bool,
    material: bool,
    reason: str,
) -> None:
    child = _generate_crossover(
        "representation_donor",
        _CrossoverModel(mapping=mapping, borrowed=borrowed, material=material),
    )

    assert child.parent_ids == ["primary"]
    assert "blend_receipt" not in child.metadata
    assert reason in child.metadata["blend_receipt_rejection"]


def test_crossover_rejects_parent_outside_slot_plan() -> None:
    with pytest.raises(ModelResponseSchemaError, match="parent_ids outside its mutation plan"):
        _generate_crossover("repair_pattern_donor", _CrossoverModel(outsider=True))


def test_incompatible_mapping_and_patch_conflict_create_obligations() -> None:
    child = _generate_crossover(
        "mechanism_fragment_donor",
        _CrossoverModel(incompatible=True),
    )

    introduced = child.obligation_delta["introduced"]
    assert introduced
    assert any(item.get("id") in introduced for item in child.proof_obligations if isinstance(item, dict))

    left = ProjectCandidateGenome(
        id="left",
        patch_set=[PatchOperation(path="same.py", operation="write", content="left")],
    )
    right = ProjectCandidateGenome(
        id="right",
        patch_set=[PatchOperation(path="same.py", operation="write", content="right")],
    )
    patch_child = crossover(left, right)
    assert patch_child.metadata["patch_merge_conflicts"]
    assert len(patch_child.patch_set) == 1
    assert patch_child.patch_set[0].content == "left"
    assert patch_child.obligation_delta["introduced"]


def test_blend_receipt_enters_existing_generation_plan_receipt_channel() -> None:
    child = _generate_crossover("representation_donor", _CrossoverModel())
    plan: dict[str, Any] = {
        "productive_branch_allocation": {
            "slots": [{"slot_id": child.metadata["branch_slot_id"]}],
        }
    }

    record_reproduction_receipts(
        plan,
        diagnosis=SearchDiagnosis(),
        mutation_plans=[],
        offspring=[child],
        outcomes=[],
    )

    assert plan["blend_receipts"][0]["candidate_id"] == child.id
    assert plan["blend_receipts"][0]["donor_role"] == "representation_donor"


def test_stagnation_policy_quota_can_role_three_crossover_slots_without_allocator_changes() -> None:
    slots = tuple(
        BranchSlot(
            slot_id=f"slot-{index}",
            arm_id=f"P{index}",
            parent_id=f"P{index}",
            intent="escape_stagnation",
            variation_index=0,
            ucb_score=1.0,
        )
        for index in range(3)
    )
    allocation = ProductiveBranchAllocation(slots=slots, arms=(), credit_summary={})

    assigned = _role_crossover_slots(
        allocation,
        parent_ids=["P0", "P1", "P2"],
        quota=3,
        roles=[
            "representation_donor",
            "repair_pattern_donor",
            "mechanism_fragment_donor",
        ],
    )

    assert {item["donor_role"] for item in assigned.values()} == DONOR_ROLES
    assert all(item["primary_parent_id"] != item["donor_parent_id"] for item in assigned.values())
    assert all(set(item["required_contribution_map"]) >= {"generic_space_mapping", "borrowed_from_donor"} for item in assigned.values())


def _dict_parent() -> CandidateGenome:
    return CandidateGenome(
        id="dict-parent",
        artifact={"params": {"width": 1, "depth": 3}, "constraints": ["width > 0"], "mode": "single"},
        artifact_type="structured_config",
    )


def _path_assignment_plan(*, operator: str = "", move_kind: str = "path_assignment", value: Any = 2) -> MutationPlan:
    return MutationPlan(
        operator=operator,
        move_kind=move_kind,
        input_domain="dict",
        declared_invariant_refs=["inv-depth"],
        expected_delta={"operation": "assign", "path": "params.width", "after": value},
        metadata={
            "move_invariants": [
                {"ref": "inv-depth", "path": "params.depth", "operator": "equal", "expected": 3},
            ]
        },
    )


def test_structured_move_delta_and_invariant_are_independently_recomputed() -> None:
    parent = _dict_parent()
    plan = _path_assignment_plan()

    child = MutationEngine().mutate(parent, plan)

    assert child.artifact["params"]["width"] == 2
    assert plan.actual_delta == recompute_move_delta(parent, child, plan)
    assert plan.actual_delta["invariants"] == {"inv-depth": True}
    assert plan.receipt_refs == [child.metadata["move_receipt"]["receipt_id"]]
    assert child.metadata["move_contract"]["move_kind"] == "path_assignment"


def test_move_label_ablation_keeps_same_structural_transform() -> None:
    parent = _dict_parent()
    named = _path_assignment_plan(operator="PathAssignment")
    ablated = _path_assignment_plan(operator="", move_kind="")

    named_child = MutationEngine().mutate(parent, named)
    ablated_child = MutationEngine().mutate(parent, ablated)

    assert named_child.artifact == ablated_child.artifact
    assert named.actual_delta == ablated.actual_delta
    assert ablated.move_kind == "path_assignment"


@pytest.mark.parametrize(
    "plan",
    [
        MutationPlan(
            operator="",
            move_kind="path_assignment",
            input_domain="dict",
            expected_delta={"operation": "assign", "path": "params.missing", "after": 2},
        ),
        _path_assignment_plan(value="wrong-type"),
    ],
)
def test_invalid_typed_move_is_rejected_before_child_generation(plan: MutationPlan) -> None:
    with pytest.raises(MoveContractError):
        MutationEngine().mutate(_dict_parent(), plan)
    assert plan.actual_delta == {}
    assert plan.receipt_refs == []


def test_invalid_model_move_rejection_is_kept_in_plan_harvest_audit() -> None:
    class PlanningModel:
        def plan_mutations(self, **_kwargs: Any) -> list[dict[str, Any]]:
            return [
                {
                    "operator": "",
                    "parent_ids": ["dict-parent"],
                    "move_kind": "path_assignment",
                    "input_domain": "dict",
                    "expected_delta": {"operation": "assign", "path": "params.missing", "after": 2},
                },
                {
                    "operator": "ModelDirected",
                    "parent_ids": ["dict-parent"],
                    "instruction": "material semantic variation",
                },
            ]

    [accepted] = _plan_mutations(
        model=PlanningModel(),
        mutation_planner=MutationPlanner(),
        parents=[_dict_parent()],
        actions=[],
        archives=ArchiveManager(),
        diagnosis=SearchDiagnosis(),
        policy=EvolutionPolicy(),
        target_count=1,
    )

    rejected = accepted.metadata["search_kernel_plan_harvest"]["rejected"]
    assert rejected[0]["reason"] == "invalid_move_contract"
    assert "does not exist" in rejected[0]["error"]


@pytest.mark.parametrize(
    ("move_kind", "expected", "assertion"),
    [
        (
            "field_split",
            {"operation": "split", "path": "mode", "fields": {"train": "single", "serve": "single"}},
            lambda artifact: artifact["mode"] == {"train": "single", "serve": "single"},
        ),
        (
            "constraint_strengthening",
            {"operation": "strengthen", "path": "constraints", "add": "depth <= 8"},
            lambda artifact: artifact["constraints"][-1] == "depth <= 8",
        ),
    ],
)
def test_minimal_dict_move_family_is_engine_executed(
    move_kind: str,
    expected: dict[str, Any],
    assertion: Any,
) -> None:
    plan = MutationPlan(operator="", move_kind=move_kind, input_domain="dict", expected_delta=expected)
    child = MutationEngine().mutate(_dict_parent(), plan)
    assert assertion(child.artifact)
    assert plan.actual_delta


@pytest.mark.parametrize(
    ("move_kind", "expected_delta"),
    [
        ("patch.add", {"path": "new.py", "content": "new"}),
        ("patch.append", {"path": "append.py", "content": "more"}),
        ("patch.replace", {"path": "replace.py", "old_text": "beta", "new_text": "gamma"}),
        ("patch.delete", {"path": "delete.py"}),
    ],
)
def test_minimal_project_patch_move_family_uses_conflict_aware_merge(
    move_kind: str,
    expected_delta: dict[str, Any],
) -> None:
    parent = ProjectCandidateGenome(
        id="patch-parent",
        patch_set=[
            PatchOperation(path="append.py", operation="append", content="base"),
            PatchOperation(path="replace.py", operation="replace", old_text="alpha", new_text="omega"),
        ],
    )
    plan = MutationPlan(
        operator="",
        move_kind=move_kind,
        input_domain="project_patch",
        expected_delta=expected_delta,
    )

    child = MutationEngine().mutate(parent, plan)

    assert isinstance(child, ProjectCandidateGenome)
    assert plan.actual_delta["conflicts"] == []
    assert len(child.patch_set) == len(parent.patch_set) + 1


@pytest.mark.parametrize(
    ("move_kind", "expected_delta"),
    [
        (
            "proof.instantiate_object",
            {"object": {"kind": "proof_object", "id": "lemma-1", "statement": "A implies A"}},
        ),
        (
            "proof.split_obligation",
            {
                "obligation_id": "obl-1",
                "child_obligations": [
                    {"id": "obl-1.a", "description": "case a"},
                    {"id": "obl-1.b", "description": "case b"},
                ],
            },
        ),
        (
            "proof.construct_witness",
            {"obligation_id": "obl-1", "witness": {"kind": "witness", "id": "wit-1", "value": 3}},
        ),
    ],
)
def test_minimal_proof_move_family_is_engine_observable(
    move_kind: str,
    expected_delta: dict[str, Any],
) -> None:
    parent = CandidateGenome(
        id="proof-parent",
        artifact="proof route",
        artifact_type="proof",
        proof_obligations=[{"id": "obl-1", "status": "pending", "description": "prove the case"}],
    )
    plan = MutationPlan(
        operator="",
        move_kind=move_kind,
        input_domain="proof",
        expected_delta=expected_delta,
    )

    child = MutationEngine().mutate(parent, plan)

    assert plan.actual_delta == recompute_move_delta(parent, child, plan)
    assert plan.receipt_refs


def test_free_text_candidate_stays_semantic_hint() -> None:
    parent = CandidateGenome(id="text-parent", artifact="free text", artifact_type="text")
    plan = MutationPlan(
        operator="",
        move_kind="path_assignment",
        input_domain="dict",
        expected_delta={"operation": "assign", "path": "answer", "after": "changed"},
    )

    child = MutationEngine().mutate(parent, plan)

    assert plan.move_kind == "semantic_hint"
    assert plan.actual_delta == {}
    assert plan.receipt_refs == []
    assert "move_receipt" not in child.metadata
    assert child.metadata["move_contract"] == {
        "move_kind": "semantic_hint",
        "input_domain": "text",
        "declared_invariant_refs": [],
        "expected_delta": {},
        "actual_delta": {},
        "receipt_refs": [],
    }


def test_move_receipt_and_delta_survive_checkpoint_replay(tmp_path) -> None:
    parent = _dict_parent()
    move_plan = _path_assignment_plan()
    child = MutationEngine().mutate(parent, move_plan)
    generation_plan: dict[str, Any] = {}
    record_reproduction_receipts(
        generation_plan,
        diagnosis=SearchDiagnosis(),
        mutation_plans=[move_plan],
        offspring=[child],
        outcomes=[],
    )
    store = CheckpointStore(tmp_path / "checkpoint.json")
    store.save_state(
        round=1,
        max_rounds=2,
        population=CandidatePopulation([parent, child]),
        archives=ArchiveManager(),
        budget_history=[{"round": 1, "generation_plan": generation_plan}],
        progress_event={"round": 1},
    )

    loaded = store.load()
    assert loaded is not None
    replayed = CandidatePopulation.from_dict(loaded.population).by_id()[child.id]
    assert replayed.metadata["move_receipt"]["receipt_id"] == move_plan.receipt_refs[0]
    assert replayed.metadata["move_contract"]["actual_delta"] == move_plan.actual_delta
    persisted = loaded.budget_history[0]["generation_plan"]
    assert persisted["move_receipts"][0]["receipt_id"] == move_plan.receipt_refs[0]


def test_generation_event_exposes_complete_move_contract() -> None:
    parent = _dict_parent()
    move_plan = _path_assignment_plan()
    child = MutationEngine().mutate(parent, move_plan)
    generation_plan: dict[str, Any] = {}
    record_reproduction_receipts(
        generation_plan,
        diagnosis=SearchDiagnosis(),
        mutation_plans=[move_plan],
        offspring=[child],
        outcomes=[],
    )
    controller = object.__new__(EvolutionLoopController)
    controller.round_pipeline = SimpleNamespace(last_generation_plan=generation_plan)
    controller.budget = SimpleNamespace(history=[{}], stop_reason="")
    evaluation = SimpleNamespace(progress_event={"metadata": {}})

    controller._record_reproduction_result(1, evaluation, "stop", [], {}, None)

    [contract] = evaluation.progress_event["metadata"]["move_contracts"]
    assert set(contract) == {
        "move_kind",
        "input_domain",
        "declared_invariant_refs",
        "expected_delta",
        "actual_delta",
        "receipt_refs",
    }
