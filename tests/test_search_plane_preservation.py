from __future__ import annotations

import inspect
import json

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy, EvolutionPolicyBuilder
from cognitive_evolve_runtime.nexus.prompt_view import build_prompt_view, contract_prompt_view, policy_prompt_view
from cognitive_evolve_runtime.nexus.search_space import build_search_space_map
from cognitive_evolve_runtime.persistence.checkpoint import CheckpointStore


CONTRACT_FAMILY_IDS = [
    "global_relaxation_and_exact_structure",
    "incumbent_chain_and_exchange_search",
    "exact_window_and_large_neighborhood_repair",
    "verification_and_admission_pipeline",
    "diversity_and_plateau_navigation",
    "lower_bound_and_bottleneck_diagnosis",
]
POLICY_FAMILY_IDS = [
    *CONTRACT_FAMILY_IDS,
    "bounded_neighborhood_certification",
    "independent_rematerialization_and_review",
]


def _families(ids: list[str], *, detail_key: str) -> list[dict[str, object]]:
    return [
        {
            "id": family_id,
            "plane": f"plane-{index}",
            "quota_min": 1,
            detail_key: f"{family_id}-DESCRIPTION-CANARY",
        }
        for index, family_id in enumerate(ids, start=1)
    ]


def test_v5_model_authored_search_planes_survive_typed_roundtrip_into_offspring_prompt(tmp_path) -> None:
    raw_contract = {
        "original_user_goal": "Improve the exact supplied incumbent.",
        "normalized_goal": "improve the exact supplied incumbent",
        "search_space_plan": {"families": _families(CONTRACT_FAMILY_IDS, detail_key="description")},
    }
    raw_policy = {
        "search_space": {
            "distinct_from_recent_candidates": "replace objective-word placeholders",
            "families": _families(POLICY_FAMILY_IDS, detail_key="purpose"),
        }
    }

    contract = NexusObjectiveContract.from_dict(json.loads(json.dumps(NexusObjectiveContract.from_dict(raw_contract).to_dict())))
    policy = EvolutionPolicy.from_json(EvolutionPolicy.from_dict(raw_policy).to_json())
    contract_view = contract_prompt_view(contract)
    policy_view = policy_prompt_view(policy)
    prompt = build_prompt_view("nexus_generate_offspring", {"contract": contract, "policy": policy}).payload

    contract_families = contract_view["search_space_plan"]["candidate_families"]
    policy_families = policy_view["search_space"]["candidate_families"]
    assert contract_families == raw_contract["search_space_plan"]["families"]
    assert policy_families == raw_policy["search_space"]["families"]
    active_families = prompt["search_space_contract"]["candidate_families"]
    assert [item["id"] for item in active_families] == POLICY_FAMILY_IDS
    for expected, actual in zip(raw_policy["search_space"]["families"], active_families, strict=True):
        assert all(actual[key] == value for key, value in expected.items())
    assert prompt["search_space_contract"]["source"] == "model_authored_search_space"
    assert prompt["search_space_contract"]["needs_model_authored_search_space"] is False
    assert not any(item["id"].startswith("model_defined_focus_") for item in active_families)
    assert all(isinstance(item, dict) for item in active_families)

    store = CheckpointStore(tmp_path / "checkpoint.json")
    store.save_state(
        round=0,
        max_rounds=1,
        population=CandidatePopulation(),
        archives=ArchiveManager(),
        contract=contract,
        policy=policy,
    )
    restored = store.restore_state()
    assert restored is not None
    assert restored["contract"]["search_space_plan"] == contract.search_space_plan
    assert restored["policy"].search_space == policy.search_space


def test_empty_search_space_mapping_does_not_hide_later_model_plan() -> None:
    search_map = build_search_space_map(
        {
            "candidate_families": [],
            "search_space": {},
            "search_space_plan": {"families": _families(["later_model_family"], detail_key="description")},
        }
    )

    assert search_map["source"] == "model_authored_search_space"
    assert search_map["route_family"] == ["later_model_family"]


def test_contract_plan_beats_id_only_fallback_policy_metadata() -> None:
    family = {
        "id": "contract_family",
        "plane": "objective",
        "quota_min": 2,
        "description": "CONTRACT-DESCRIPTION-CANARY",
    }
    contract = NexusObjectiveContract(
        original_user_goal="study the objective",
        normalized_goal="study the objective",
        search_space_plan={"families": [family]},
    )
    policy = EvolutionPolicyBuilder().build(contract=contract, world={"kind": "text"}, model=None)

    active = build_prompt_view("nexus_generate_offspring", {"contract": contract, "policy": policy}).payload["search_space_contract"]

    assert active["candidate_families"] == [family]


def test_malformed_policy_search_space_does_not_hide_valid_contract_plan() -> None:
    contract = NexusObjectiveContract(
        original_user_goal="study the objective",
        normalized_goal="study the objective",
        search_space_plan={"families": [{"id": "contract_real", "plane": "objective", "quota_min": 2}]},
    )

    for malformed in (
        {"coverage_gate": {"min_family_count": 1}},
        {"families": [], "note": "empty"},
        {"families": [{"description": "missing id"}]},
        {"candidate_families": ["not a family mapping"]},
    ):
        active = build_prompt_view(
            "nexus_generate_offspring",
            {"contract": contract, "policy": EvolutionPolicy(search_space=malformed)},
        ).payload["search_space_contract"]

        assert active["source"] == "model_authored_search_space"
        assert [item["id"] for item in active["candidate_families"]] == ["contract_real"]


def test_model_authored_family_count_is_not_capped_without_budget_pressure() -> None:
    family_ids = [f"family_{index:02d}" for index in range(13)]
    policy = EvolutionPolicy(search_space={"families": _families(family_ids, detail_key="purpose")})

    view = build_prompt_view(
        "nexus_generate_offspring",
        {
            "contract": NexusObjectiveContract(original_user_goal="wide search", normalized_goal="wide search"),
            "policy": policy,
        },
        max_chars=120_000,
    )

    assert view.metadata["truncated"] is False
    assert [item["id"] for item in view.payload["search_space_contract"]["candidate_families"]] == family_ids

    constrained = build_prompt_view(
        "nexus_generate_offspring",
        {
            "contract": NexusObjectiveContract(original_user_goal="wide search", normalized_goal="wide search"),
            "policy": policy,
        },
        max_chars=8_000,
    )
    assert constrained.metadata["sent_payload_chars"] <= 8_000
    assert [item["id"] for item in constrained.payload["search_space_contract"]["candidate_families"]] == family_ids


def test_minimal_prompt_keeps_all_search_families_when_they_fit() -> None:
    contract_families = _families(CONTRACT_FAMILY_IDS, detail_key="description")
    policy_families = _families(POLICY_FAMILY_IDS, detail_key="purpose")
    for item in contract_families:
        item["description"] += " contract-detail" * 90
    for item in policy_families:
        item["purpose"] += " policy-detail" * 45
    contract = NexusObjectiveContract(
        original_user_goal="improve an exact incumbent",
        normalized_goal="improve an exact incumbent",
        search_space_plan={"families": contract_families},
    )
    policy = EvolutionPolicy(search_space={"families": policy_families})

    view = build_prompt_view(
        "nexus_generate_offspring",
        {"contract": contract, "policy": policy},
        max_chars=20_000,
    )

    assert view.metadata["truncated"] is True
    assert view.metadata["sent_payload_chars"] <= 20_000
    assert [item["id"] for item in view.payload["search_space_contract"]["candidate_families"]] == POLICY_FAMILY_IDS


def test_pathological_prompt_truncation_is_explicit_about_omitted_families() -> None:
    family_ids = [f"pathological_family_{index:03d}" for index in range(200)]
    families = _families(family_ids, detail_key="description")
    for item in families:
        item["description"] += " very-long-description" * 80
    policy = EvolutionPolicy(search_space={"families": families})

    view = build_prompt_view(
        "nexus_generate_offspring",
        {
            "contract": NexusObjectiveContract(original_user_goal="wide search", normalized_goal="wide search"),
            "policy": policy,
        },
        max_chars=4_000,
    )

    search_contract = view.payload["search_space_contract"]
    kept_ids = [item["id"] for item in search_contract["candidate_families"]]
    omitted = int(search_contract.get("families_omitted") or 0)
    assert kept_ids == family_ids[: len(kept_ids)]
    assert len(kept_ids) + omitted == len(family_ids)
    assert omitted > 0
    assert search_contract["family_details_omitted"] == len(family_ids)
    assert view.metadata["sent_payload_chars"] <= 4_000


def test_compaction_preserves_unknown_family_fields_before_ids_only_fallback() -> None:
    family_ids = [f"rich_family_{index:02d}" for index in range(8)]
    families = [
        {
            "id": family_id,
            "description": "D" * 1_000,
            "axes": {"novelty": "N" * 1_000, "proof": "P" * 1_000},
            "operator": {"name": "mutate", "instruction": "M" * 1_000},
            "invariants": ["I" * 500, "J" * 500],
            "custom_kernel": {"rule": "K" * 1_000},
        }
        for family_id in family_ids
    ]

    view = build_prompt_view(
        "nexus_generate_offspring",
        {
            "contract": NexusObjectiveContract(original_user_goal="rich search", normalized_goal="rich search"),
            "policy": EvolutionPolicy(search_space={"families": families}),
        },
        max_chars=30_000,
    )

    search_contract = view.payload["search_space_contract"]
    assert [item["id"] for item in search_contract["candidate_families"]] == family_ids
    assert all(
        all(key in item for key in ("axes", "operator", "invariants", "custom_kernel"))
        for item in search_contract["candidate_families"]
    )
    assert search_contract["family_details_compacted"] > 0
    assert "family_details_omitted" not in search_contract
    assert view.metadata["sent_payload_chars"] <= 30_000


def test_extreme_budget_uses_explicit_search_omission_view_and_stays_bounded() -> None:
    family_ids = [f"budget_family_{index:02d}" for index in range(30)]
    policy = EvolutionPolicy(search_space={"families": _families(family_ids, detail_key="description")})
    kept_counts: list[int] = []

    for limit in (1_000, 1_500, 2_000, 3_000, 4_000):
        view = build_prompt_view(
            "nexus_generate_offspring",
            {
                "contract": NexusObjectiveContract(original_user_goal="wide search", normalized_goal="wide search"),
                "policy": policy,
            },
            max_chars=limit,
        )
        search_contract = view.payload["search_space_contract"]
        kept = len(search_contract["candidate_families"])
        kept_counts.append(kept)
        assert kept + int(search_contract.get("families_omitted") or 0) == len(family_ids)
        assert view.metadata["sent_payload_chars"] <= limit

    assert kept_counts == sorted(kept_counts)


def test_protected_contract_over_budget_reports_exact_shortfall() -> None:
    contract = NexusObjectiveContract(
        original_user_goal="protected task",
        normalized_goal="protected task",
        frozen_spec={"problem_text": "X" * 5_000},
    )

    view = build_prompt_view(
        "nexus_generate_offspring",
        {"contract": contract, "policy": EvolutionPolicy()},
        max_chars=500,
    )

    assert view.payload["contract"]["frozen_spec"] == contract.frozen_spec
    assert view.metadata["sent_payload_chars"] > 500
    assert view.metadata["protected_over_budget"] is True
    assert view.metadata["budget_shortfall_chars"] == view.metadata["sent_request_chars"] - 500


def test_missing_model_plan_keeps_explicit_placeholder_fallback() -> None:
    prompt = build_prompt_view(
        "nexus_generate_offspring",
        {
            "contract": NexusObjectiveContract(original_user_goal="study a hard objective", normalized_goal="study a hard objective"),
            "policy": EvolutionPolicy(),
        },
    ).payload

    search_contract = prompt["search_space_contract"]
    assert search_contract["needs_model_authored_search_space"] is True
    assert all(item["id"].startswith("model_defined_focus_") for item in search_contract["candidate_families"])


def test_empty_new_fields_preserve_legacy_contract_and_policy_hashes() -> None:
    contract = NexusObjectiveContract(
        original_user_goal="legacy goal",
        normalized_goal="legacy goal",
        created_at="2026-01-01T00:00:00+00:00",
    )
    policy = EvolutionPolicy(created_at="2026-01-01T00:00:00+00:00")

    assert "search_space_plan" not in contract.to_dict()
    assert "search_space" not in policy.to_dict()
    assert contract.contract_hash() == "7d620a4cba419fe74e672eadc8124012ed24ff4cc522f50c53055a6e2a3d4e63"
    assert policy.policy_hash == "a59e6ea3f48be785cf1f85afcb05ae54ac686b95ce68f22c64cf61651bd8d3aa"


def test_contract_and_policy_hashes_survive_json_roundtrip() -> None:
    contract = NexusObjectiveContract(
        original_user_goal="roundtrip goal",
        normalized_goal="roundtrip goal",
        search_space_plan={"families": _families(["roundtrip_family"], detail_key="description")},
        created_at="2026-01-01T00:00:00+00:00",
    )
    policy = EvolutionPolicy(
        search_space={"families": _families(["roundtrip_family"], detail_key="purpose")},
        created_at="2026-01-01T00:00:00+00:00",
    )

    restored_contract = NexusObjectiveContract.from_dict(json.loads(json.dumps(contract.to_dict())))
    restored_policy = EvolutionPolicy.from_json(policy.to_json())

    assert restored_contract.dynamic_artifact_contract == {}
    assert restored_contract.contract_hash() == contract.contract_hash()
    assert restored_policy.policy_hash == policy.policy_hash
    assert inspect.signature(NexusObjectiveContract).parameters["search_space_plan"].kind is inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(EvolutionPolicy).parameters["search_space"].kind is inspect.Parameter.KEYWORD_ONLY
