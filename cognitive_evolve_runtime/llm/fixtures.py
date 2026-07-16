from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .env import LLMConfigurationError, LLMResponseError


def _seed_candidate(slot: dict[str, Any]) -> dict[str, Any]:
    slot_id = str(slot["slot_id"])
    family_id = str(slot["family_id"])
    seed_axis = str(slot["seed_axis"])
    search_space: dict[str, Any] = {"family_id": family_id, "seed_axis": seed_axis}
    if seed_axis == "direct_mainstream":
        search_space["seed_axis_claim"] = f"Fixture mainstream mechanism for {family_id}."
    elif seed_axis == "cross_domain_transfer":
        search_space["transfer_source_domain"] = f"fixture transfer source for {slot_id}"
    elif seed_axis == "counterexample_probe":
        search_space["probe_target_assumption"] = f"fixture assumption for {slot_id}"
    elif seed_axis == "representation_shift":
        search_space["representation_shift"] = {
            "from": f"fixture direct representation for {slot_id}",
            "to": f"fixture causal representation for {slot_id}",
        }
    elif seed_axis == "tool_probe":
        search_space["tool_probe_plan"] = {
            "tool": "fixture deterministic probe",
            "observable": f"fixture observable for {slot_id}",
        }
    evaluation_dimensions = [f"fixture-observable-{slot_id}"]
    return {
        "id": f"fixture-seed-{slot_id}",
        "artifact": f"Fixture seed artifact for {slot_id}.",
        "artifact_type": "answer",
        "concise_claim": f"Fixture seed claim for {slot_id}.",
        "core_mechanism": f"Fixture seed mechanism for {slot_id}.",
        "assumptions": [],
        "missing_parts": [],
        "uncertainty_notes": [],
        "edge_knowledge_seeds": [f"fixture-edge-{slot_id}"] if seed_axis == "edge_knowledge" else [],
        "niche_memberships": [f"fixture-lens-{slot_id}"],
        "evaluation_dimensions": evaluation_dimensions,
        "metadata": {
            "seed_type": slot_id,
            "search_space": search_space,
            "structured_output_fields": {"evaluation_dimensions": evaluation_dimensions},
        },
        "multihead_scores": {
            "answer_likelihood": 0.7,
            "objective_alignment": 0.7,
            "verifiability": 0.5,
        },
    }


def _bind_seed_response(response: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    policy = payload.get("policy")
    portfolio = policy.get("seed_portfolio") if isinstance(policy, dict) else None
    if not isinstance(portfolio, list) or not portfolio:
        raise LLMResponseError("Fixture seed response requires policy.seed_portfolio")
    response["candidates"] = [_seed_candidate(slot) for slot in portfolio]
    return response


def _bind_offspring_response(response: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    plans = payload.get("plans")
    slots: list[dict[str, Any]] = []
    for plan in plans if isinstance(plans, list) else []:
        metadata = plan.get("metadata") if isinstance(plan, dict) else None
        branch_slots = metadata.get("branch_slots") if isinstance(metadata, dict) else None
        if isinstance(branch_slots, list):
            slots.extend(slot for slot in branch_slots if isinstance(slot, dict))
    offspring = response.get("offspring")
    if not slots or not isinstance(offspring, list) or not offspring:
        raise LLMResponseError("Fixture offspring response requires a runtime lineage envelope")
    template = offspring[0]
    bound: list[dict[str, Any]] = []
    for slot in slots:
        slot_id = str(slot["slot_id"])
        parent_id = str(slot["parent_id"])
        item = deepcopy(template)
        item["id"] = f"fixture-offspring-{slot_id}"
        item["artifact"] = f"Fixture offspring artifact for {slot_id} from {parent_id}."
        item["concise_claim"] = f"Fixture offspring claim for {slot_id}."
        item["core_mechanism"] = f"Fixture offspring mechanism for {slot_id}."
        item["parent_ids"] = [parent_id]
        item["niche_memberships"] = [f"fixture-offspring-lens-{slot_id}"]
        item["evaluation_dimensions"] = [f"fixture-offspring-observable-{slot_id}"]
        metadata = dict(item.get("metadata") or {})
        metadata["branch_slot_id"] = slot_id
        item["metadata"] = metadata
        bound.append(item)
    response["offspring"] = bound
    return response


def load_fixture_response(request_type: str, payload: dict[str, Any], fixture_path: str) -> dict[str, Any]:
    try:
        fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise LLMConfigurationError(f"Invalid LLM fixture {fixture_path}: {exc}") from exc
    if not isinstance(fixture, dict):
        raise LLMConfigurationError(f"Invalid LLM fixture {fixture_path}: top-level JSON must be an object")
    responses = fixture.get("responses", fixture)
    if not isinstance(responses, dict):
        raise LLMConfigurationError(f"Invalid LLM fixture {fixture_path}: responses must be an object")
    if request_type == "classify_route":
        prompt = str(payload.get("prompt", "")).lower()
        for case in responses.get("classify_route_cases", []):
            contains = [str(term).lower() for term in case.get("contains", [])]
            if contains and all(term in prompt for term in contains):
                response = case.get("response", {})
                if isinstance(response, dict):
                    return response
    response = responses.get(request_type)
    if response is None:
        response = responses.get("default", {}).get(request_type)
    if not isinstance(response, dict):
        raise LLMResponseError(f"Fixture has no response for request_type={request_type}")
    if request_type == "nexus_seed_population":
        return _bind_seed_response(response, payload)
    if request_type == "nexus_generate_offspring":
        return _bind_offspring_response(response, payload)
    return response
