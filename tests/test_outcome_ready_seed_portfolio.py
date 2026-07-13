from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import candidate_from_dict
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.loop.seeding import (
    SEED_COGNITIVE_AXES,
    _allocate_seed_slots,
    _order_seed_batch_for_portfolio,
    _seed_target_size,
    seed_population,
)
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy, EvolutionPolicyBuilder
from cognitive_evolve_runtime.nexus.prompt_view import build_prompt_view


class _World:
    kind = "text"


def _policy() -> EvolutionPolicy:
    return EvolutionPolicy(
        search_space={
            "source": "model_authored_search_space",
            "candidate_families": [
                {"id": "family_a", "description": "first model-defined mechanism family", "quota_min": 2},
                {"id": "family_b", "description": "second model-defined mechanism family", "quota_min": 8},
            ],
        },
        metadata={"initial_candidate_count": 1},
    )


def _candidate_for_slot(
    slot: dict[str, Any],
    *,
    suffix: str = "",
    axis_claim: bool = True,
    edge_ready: bool = True,
    outcome_ready: bool = True,
) -> dict[str, Any]:
    slot_id = str(slot["slot_id"])
    family_id = str(slot["family_id"])
    seed_axis = str(slot["seed_axis"])
    evaluation_dimensions = [f"observable-{slot_id}"] if outcome_ready else []
    search_space = {
        "family_id": family_id,
        "seed_axis": seed_axis,
        "seed_axis_claim": f"The {seed_axis} operation changes the mechanism for {family_id}." if axis_claim else "",
    }
    if seed_axis == "cross_domain_transfer":
        search_space["transfer_source_domain"] = "model-chosen source discipline"
    elif seed_axis == "counterexample_probe":
        search_space["probe_target_assumption"] = "the dominant mechanism's hidden closure assumption"
    elif seed_axis == "representation_shift":
        search_space["representation_shift"] = {"from": "direct object", "to": "causal graph"}
    elif seed_axis == "tool_probe":
        search_space["tool_probe_plan"] = {"tool": "small simulation", "observable": "failure boundary"}
    return {
        "id": f"candidate-{slot_id}{suffix}",
        "artifact": {"answer": f"materialized-{slot_id}{suffix}"},
        "artifact_type": "answer",
        "concise_claim": f"candidate for {slot_id}{suffix}",
        "core_mechanism": f"mechanism {slot_id}{suffix}",
        "assumptions": [],
        "missing_parts": [],
        "uncertainty_notes": [],
        "edge_knowledge_seeds": [f"edge-{slot_id}"] if seed_axis == "edge_knowledge" and edge_ready else [],
        "niche_memberships": [f"model-chosen-lens-{slot_id}"],
        "evaluation_dimensions": evaluation_dimensions,
        "metadata": {
            "seed_type": slot_id,
            "search_space": search_space,
            "structured_output_fields": {"evaluation_dimensions": evaluation_dimensions},
        },
    }


def test_seed_portfolio_crosses_every_model_family_with_all_cognitive_axes_and_honors_quota() -> None:
    policy = _policy()

    slots = _allocate_seed_slots(policy, requested_count=1)

    axes = {axis["id"] for axis in SEED_COGNITIVE_AXES}
    by_family = {
        family_id: [slot for slot in slots if slot["family_id"] == family_id]
        for family_id in ("family_a", "family_b")
    }
    assert len(by_family["family_a"]) == 6
    assert len(by_family["family_b"]) == 8
    assert {slot["seed_axis"] for slot in by_family["family_a"]} == axes
    assert {slot["seed_axis"] for slot in by_family["family_b"]} == axes
    assert _seed_target_size(policy=policy, world=_World(), requested_minimum=2) == 14


def test_model_policy_omission_cannot_disable_contract_search_families() -> None:
    contract = NexusObjectiveContract(
        original_user_goal="explore an unknown mechanism",
        normalized_goal="explore unknown mechanism",
        search_space_plan={
            "source": "model_authored_search_space",
            "candidate_families": [{"id": "contract_family", "quota_min": 2}],
        },
    )

    class Model:
        def build_evolution_policy(self, **_: Any) -> dict[str, Any]:
            return {"candidate_niches": ["wide"]}

    policy = EvolutionPolicyBuilder().build(contract=contract, world=_World(), model=Model())
    slots = _allocate_seed_slots(policy, requested_count=1)

    assert policy.search_space["candidate_families"] == [{"id": "contract_family", "quota_min": 2}]
    assert len(slots) == len(SEED_COGNITIVE_AXES)


def test_model_policy_string_families_are_normalized_before_seed_allocation() -> None:
    class Model:
        def build_evolution_policy(self, **_: Any) -> dict[str, Any]:
            return {
                "search_space": {
                    "source": "model_authored_search_space",
                    "candidate_families": ["mechanism A", "mechanism B"],
                }
            }

    policy = EvolutionPolicyBuilder().build(
        contract=NexusObjectiveContract(original_user_goal="unknown", normalized_goal="unknown"),
        world=_World(),
        model=Model(),
    )
    slots = _allocate_seed_slots(policy, requested_count=1)
    policy.metadata["seed_portfolio"] = slots
    prompt = build_prompt_view(
        "nexus_seed_population",
        {"contract": {}, "world": {}, "policy": policy, "requested_candidate_count": len(slots)},
    ).payload
    contract_family_ids = [item["id"] for item in prompt["search_space_contract"]["candidate_families"]]
    portfolio_family_ids = list(dict.fromkeys(slot["family_id"] for slot in prompt["policy"]["seed_portfolio"]))

    assert policy.search_space["candidate_families"] == [{"id": "mechanism A"}, {"id": "mechanism B"}]
    assert {slot["family_id"] for slot in slots} == {"mechanism A", "mechanism B"}
    assert contract_family_ids == portfolio_family_ids == ["mechanism A", "mechanism B"]
    assert len(slots) == 2 * len(SEED_COGNITIVE_AXES)


def test_missing_search_plan_still_gets_domain_neutral_six_axis_coverage() -> None:
    class Model:
        def build_evolution_policy(self, **_: Any) -> dict[str, Any]:
            return {"candidate_niches": ["wide"]}

    policy = EvolutionPolicyBuilder().build(
        contract=NexusObjectiveContract(original_user_goal="unknown", normalized_goal="unknown"),
        world=_World(),
        model=Model(),
    )
    slots = _allocate_seed_slots(policy, requested_count=1)

    assert policy.search_space["source"] == "objective_derived_placeholder_search_space"
    axes = {axis["id"] for axis in SEED_COGNITIVE_AXES}
    family_ids = {slot["family_id"] for slot in slots}
    assert len(family_ids) >= 3
    assert all({slot["seed_axis"] for slot in slots if slot["family_id"] == family_id} == axes for family_id in family_ids)


def test_default_seed_run_passes_full_portfolio_in_one_model_call(monkeypatch: Any) -> None:
    policy = _policy()
    seen_portfolios: list[list[dict[str, Any]]] = []
    monkeypatch.setenv("COGEV_NEXUS_SEED_BATCH_LIMIT", "3")
    monkeypatch.setenv("COGEV_NEXUS_SEED_MIN_BATCHES", "3")
    monkeypatch.setenv("COGEV_NEXUS_SEED_FANOUT_WORKERS", "3")

    class Model:
        def seed_population(self, *, policy: EvolutionPolicy, **_: Any) -> list[dict[str, Any]]:
            portfolio = [dict(slot) for slot in policy.metadata["seed_portfolio"]]
            seen_portfolios.append(portfolio)
            candidates = [_candidate_for_slot(slot) for slot in reversed(portfolio)]
            for candidate in candidates:
                candidate["parent_ids"] = ["model-invented-shared-seed-root"]
                candidate["lineage"] = ["model-invented-shared-seed-root", candidate["id"]]
            return candidates

    population = seed_population(
        contract=NexusObjectiveContract(original_user_goal="explore an unknown mechanism", normalized_goal="explore unknown mechanism"),
        world=_World(),
        policy=policy,
        model=Model(),
    )

    assert len(seen_portfolios) == 1
    assert len(seen_portfolios[0]) == 14
    assert len(population.candidates) == 14
    assert all(candidate.parent_ids == [] for candidate in population.candidates)
    assert all(candidate.lineage == [candidate.id] for candidate in population.candidates)
    coverage = policy.metadata["seed_coverage"]
    assert coverage["contract_coverage_status"] == "complete"
    assert coverage["missing_slot_ids"] == []
    assert coverage["missing_edge_slot_ids"] == []
    assert coverage["missing_outcome_slot_ids"] == []
    assert coverage["capability_status"] == "pending_pba_outcome"


def test_model_claimed_duplicate_seed_ids_cannot_collapse_lineage_arms() -> None:
    policy = EvolutionPolicy(
        search_space={
            "source": "model_authored_search_space",
            "candidate_families": [{"id": "family", "description": "one model-defined family"}],
        }
    )

    class Model:
        def seed_population(self, *, policy: EvolutionPolicy, **_: Any) -> list[dict[str, Any]]:
            candidates = [_candidate_for_slot(slot) for slot in policy.metadata["seed_portfolio"]]
            for candidate in candidates:
                candidate["id"] = "model-reused-id"
                candidate["metadata"]["evidence_records"] = [
                    {"candidate_id": "model-reused-id", "source": "model", "resolved_challenge_ids": ["fake"]}
                ]
            return candidates

    population = seed_population(
        contract=NexusObjectiveContract(original_user_goal="unknown", normalized_goal="unknown"),
        world=_World(),
        policy=policy,
        model=Model(),
    )

    assert len({candidate.id for candidate in population.candidates}) == len(SEED_COGNITIVE_AXES)
    assert len({candidate.lineage[0] for candidate in population.candidates}) == len(SEED_COGNITIVE_AXES)
    assert {candidate.metadata["model_claimed_candidate_id"] for candidate in population.candidates} == {"model-reused-id"}
    assert all("evidence_records" not in candidate.metadata for candidate in population.candidates)


def test_seed_batch_order_covers_families_before_repeating_one_family() -> None:
    slots = [
        {"slot_id": "A-direct", "family_id": "A", "seed_axis": "direct_mainstream"},
        {"slot_id": "B-direct", "family_id": "B", "seed_axis": "direct_mainstream"},
    ]
    crowded = [
        candidate_from_dict(_candidate_for_slot({**slots[0], "slot_id": "A-direct"})),
        candidate_from_dict(_candidate_for_slot({**slots[0], "slot_id": "A-extra"}, suffix="-extra")),
        candidate_from_dict(_candidate_for_slot({**slots[1], "slot_id": "B-direct"})),
    ]

    ordered = _order_seed_batch_for_portfolio(crowded, slots)

    assert [candidate.metadata["search_space"]["family_id"] for candidate in ordered[:2]] == ["A", "B"]


def test_missing_axis_edge_and_outcome_declarations_are_reported_without_rejecting_candidates() -> None:
    policy = EvolutionPolicy(
        search_space={
            "source": "model_authored_search_space",
            "candidate_families": [{"id": "family", "description": "one model-defined family", "quota_min": 1}],
        }
    )

    class Model:
        def seed_population(self, *, policy: EvolutionPolicy, **_: Any) -> list[dict[str, Any]]:
            slots = list(policy.metadata["seed_portfolio"])
            out = [_candidate_for_slot(slot) for slot in slots]
            edge_index = next(index for index, slot in enumerate(slots) if slot["seed_axis"] == "edge_knowledge")
            out[edge_index] = _candidate_for_slot(slots[edge_index], edge_ready=False)
            out[1] = _candidate_for_slot(slots[1], outcome_ready=False)
            out[0] = _candidate_for_slot(slots[0], axis_claim=False)
            return out

    population = seed_population(
        contract=NexusObjectiveContract(original_user_goal="unknown", normalized_goal="unknown"),
        world=_World(),
        policy=policy,
        model=Model(),
    )

    assert len(population.candidates) == 6
    coverage = policy.metadata["seed_coverage"]
    planned = policy.metadata["seed_portfolio"]
    edge_slot = next(slot for slot in planned if slot["seed_axis"] == "edge_knowledge")
    assert coverage["contract_coverage_status"] == "shortfall"
    assert edge_slot["slot_id"] in coverage["missing_edge_slot_ids"]
    assert planned[1]["slot_id"] in coverage["missing_outcome_slot_ids"]
    assert planned[0]["slot_id"] in coverage["missing_slot_ids"]
    direct_candidate = next(candidate for candidate in population.candidates if candidate.metadata["seed_type"] == planned[0]["slot_id"])
    assert direct_candidate.id in coverage["missing_axis_declaration_candidate_ids"]
    assert coverage["capability_status"] == "pending_pba_outcome"


def test_axis_specific_receipt_cannot_be_replaced_by_generic_axis_claim() -> None:
    policy = EvolutionPolicy(
        search_space={
            "source": "model_authored_search_space",
            "candidate_families": [{"id": "family", "description": "one model-defined family"}],
        }
    )

    class Model:
        def seed_population(self, *, policy: EvolutionPolicy, **_: Any) -> list[dict[str, Any]]:
            slots = list(policy.metadata["seed_portfolio"])
            out = [_candidate_for_slot(slot) for slot in slots]
            cross_index = next(index for index, slot in enumerate(slots) if slot["seed_axis"] == "cross_domain_transfer")
            out[cross_index]["metadata"]["search_space"].pop("transfer_source_domain")
            return out

    population = seed_population(
        contract=NexusObjectiveContract(original_user_goal="unknown", normalized_goal="unknown"),
        world=_World(),
        policy=policy,
        model=Model(),
    )

    cross_candidate = next(
        candidate
        for candidate in population.candidates
        if candidate.metadata["search_space"]["seed_axis"] == "cross_domain_transfer"
    )
    coverage = policy.metadata["seed_coverage"]
    assert coverage["contract_coverage_status"] == "shortfall"
    assert cross_candidate.id in coverage["missing_axis_specific_claim_candidate_ids"]
    assert cross_candidate.metadata["seed_type"] in coverage["missing_slot_ids"]


def test_reusing_one_lens_across_axes_is_reported_as_coverage_shortfall() -> None:
    policy = EvolutionPolicy(
        search_space={
            "source": "model_authored_search_space",
            "candidate_families": [{"id": "family", "description": "one model-defined family"}],
        }
    )

    class Model:
        def seed_population(self, *, policy: EvolutionPolicy, **_: Any) -> list[dict[str, Any]]:
            candidates = [_candidate_for_slot(slot) for slot in policy.metadata["seed_portfolio"]]
            for candidate in candidates:
                candidate["niche_memberships"] = ["same familiar discipline"]
            return candidates

    seed_population(
        contract=NexusObjectiveContract(original_user_goal="unknown", normalized_goal="unknown"),
        world=_World(),
        policy=policy,
        model=Model(),
    )

    coverage = policy.metadata["seed_coverage"]
    assert coverage["contract_coverage_status"] == "shortfall"
    assert len(coverage["redundant_lens_slot_ids"]) == len(SEED_COGNITIVE_AXES) - 1


def test_seed_prompt_uses_requested_count_and_keeps_axis_contract_and_all_slots() -> None:
    policy = _policy()
    slots = _allocate_seed_slots(policy, requested_count=1)
    policy.metadata["seed_portfolio"] = slots
    policy.metadata["seed_portfolio_contract"] = {"mode": "family_x_cognitive_axis"}

    view = build_prompt_view(
        "nexus_seed_population",
        {"contract": {}, "world": {}, "policy": policy, "requested_candidate_count": len(slots)},
    ).payload

    assert view["search_space_contract"]["candidate_target_count"] == len(slots)
    assert view["policy"]["seed_portfolio"] == slots
    assert view["policy"]["seed_portfolio_contract"]["mode"] == "family_x_cognitive_axis"
