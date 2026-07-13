"""Bounded model-facing prompt views for Nexus state.

The Nexus runtime keeps full genomes, archives, journals, and checkpoints on
local disk.  Model calls should see compact *views* of that state: enough signal
for search control, ranking, critique, mutation, and synthesis without echoing
full artifacts, repeated archive copies, raw tool output, or historical JSON.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

from cognitive_evolve_runtime.candidates.genome import CandidateGenome, candidate_from_dict
from cognitive_evolve_runtime.candidates.project_candidate import ProjectCandidateGenome
from cognitive_evolve_runtime.llm.env import prompt_char_limit_details
from cognitive_evolve_runtime.nexus.activation import ACTIVATION_REQUESTS, activation_prompt_contract
from cognitive_evolve_runtime.nexus.search_space import build_search_space_map
from cognitive_evolve_runtime.nexus.prompt_profiles import apply_prompt_profile
from cognitive_evolve_runtime.nexus.nextgen import false_cull_monitor
from cognitive_evolve_runtime.nexus.strategy_comparison import strategy_comparison_context

NEXUS_PROMPT_MAX_CHARS_ENV = "COGEV_NEXUS_PROMPT_MAX_CHARS"
NEXUS_LONG_CONTEXT_MAX_CHARS_ENV = "COGEV_NEXUS_LONG_CONTEXT_MAX_CHARS"

DEFAULT_MAX_PROMPT_CHARS = 250_000
DEFAULT_LONG_CONTEXT_MAX_CHARS = 500_000

HEAVY_CANDIDATE_FIELDS = {
    "artifact",
    "tool_results",
    "verification_trace",
    "mutation_history",
    "failure_lessons",
    "inherited_genes",
}


def prompt_char_budget(*, long_context: bool = False) -> int:
    """Return the model-facing prompt budget in characters.

    Nexus settings choose the requested view size; ``COGEV_LLM_MAX_PROMPT_CHARS``
    remains the provider-facing upper bound.  Long-context calls may request a
    higher ceiling via ``COGEV_NEXUS_LONG_CONTEXT_MAX_CHARS`` only when the
    transport cap also permits it.
    """

    return int(prompt_char_budget_details(long_context=long_context)["effective"])


def prompt_char_budget_details(*, long_context: bool = False) -> dict[str, int | bool | None]:
    requested = _positive_int(os.environ.get(NEXUS_LONG_CONTEXT_MAX_CHARS_ENV)) if long_context else None
    requested = requested or _positive_int(os.environ.get(NEXUS_PROMPT_MAX_CHARS_ENV))
    requested = requested or (DEFAULT_LONG_CONTEXT_MAX_CHARS if long_context else DEFAULT_MAX_PROMPT_CHARS)
    return prompt_char_limit_details(requested)


@dataclass(frozen=True)
class PromptView:
    """A bounded payload plus accounting metadata."""

    payload: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"payload": self.payload, "metadata": self.metadata}


def build_prompt_view(
    request_type: str,
    payload: dict[str, Any],
    *,
    max_chars: int | None = None,
    schema_hint: dict[str, Any] | None = None,
) -> PromptView:
    """Build a compact, bounded prompt view for a Nexus model request."""

    effective_limit = max(1, int(max_chars or prompt_char_budget(long_context=is_long_context_request(request_type))))
    payload_limit = (
        _structured_payload_char_budget(request_type, schema_hint, max_chars=effective_limit)
        if schema_hint is not None
        else effective_limit
    )
    controls = _prompt_context_controls(payload)
    protected_paths = list(
        dict.fromkeys(
            [
                *_protected_paths_from_controls(controls),
                "packet.raw_text",
                "prompt",
                "user_goal",
                "contract.frozen_spec",
                "source_context.problem_spec",
                "source_context.frozen_spec",
                "source_context.initial_candidates",
                "source_context.selected_files",
                "source_context.slices",
                "source_context.context_limits",
                *(
                    ["parents", "plans"]
                    if request_type == "nexus_generate_offspring"
                    else ["parents"] if request_type == "nexus_plan_mutations" else []
                ),
                *(
                    ["candidates"]
                    if request_type in {"nexus_critique_candidates", "nexus_relative_rank", "nexus_synthesize_result"}
                    else []
                ),
                *(
                    ["policy.seed_portfolio", "policy.seed_portfolio_contract", "policy.seed_instruction"]
                    if request_type == "nexus_seed_population"
                    else []
                ),
                *(["policy.rejected_offspring_feedback", "policy.accepted_offspring_candidates"] if request_type == "nexus_generate_offspring" else []),
                *(["coverage_report", "clusters", "representatives", "instructions"] if request_type == "nexus_pool_preprocess" else []),
            ]
        )
    )
    raw_chars = _json_chars(payload)
    compressed = _apply_prompt_context_controls(_compress_payload(request_type, payload), controls)
    profiled, profile_metadata = apply_prompt_profile(request_type, compressed)
    if "requested_candidate_count" in compressed:
        profiled["requested_candidate_count"] = compressed["requested_candidate_count"]
    protected_candidate_ids = _protected_candidate_ids_from_controls(controls)
    if protected_candidate_ids and "candidates" not in profiled and isinstance(compressed.get("candidates"), list):
        profiled["candidates"] = _trimmed_with_protected(compressed.get("candidates"), max(1, len(protected_candidate_ids)), protected_candidate_ids)
        profiled["_protected_candidate_ids"] = sorted(protected_candidate_ids)
    compressed_chars = _json_chars(profiled)
    applied_protected_paths = list(_snapshot_paths(profiled, protected_paths))
    bounded = _fit_payload(profiled, max_chars=payload_limit, protected_paths=applied_protected_paths)
    sent_chars = _json_chars(bounded)
    sent_request_chars = _structured_request_chars(request_type, schema_hint, bounded) if schema_hint is not None else sent_chars
    budget_shortfall_chars = max(0, sent_request_chars - effective_limit)
    metadata = {
        "type": "nexus_prompt_view",
        "request_type": request_type,
        "raw_payload_chars": raw_chars,
        "compressed_payload_chars": compressed_chars,
        "sent_payload_chars": sent_chars,
        "max_prompt_chars": effective_limit,
        "effective_max_prompt_chars": effective_limit,
        "prompt_limit": prompt_char_limit_details(effective_limit),
        "payload_max_chars": payload_limit,
        "sent_request_chars": sent_request_chars,
        "compressed": True,
        "truncated": sent_chars < compressed_chars,
        "raw_payload_sha256": _sha256_json(payload),
        "sent_payload_sha256": _sha256_json(bounded),
        "policy": "candidate_archive_summary_then_recursive_fit",
        "omitted_heavy_fields": (
            []
            if request_type in {"nexus_critique_candidates", "nexus_relative_rank", "nexus_synthesize_result"}
            else sorted(HEAVY_CANDIDATE_FIELDS)
        ),
        "protected_paths_applied": applied_protected_paths,
        "protected_over_budget": bool(budget_shortfall_chars),
        "budget_shortfall_chars": budget_shortfall_chars,
        "context_transform_applied": bool(controls),
        "profile_applied": bool(profile_metadata.get("profile_applied")),
        "profile_name": profile_metadata.get("profile_name"),
        "removed_strength_shortcut_keys": profile_metadata.get("removed_strength_shortcut_keys", []),
    }
    return PromptView(payload=bounded, metadata=metadata)


def candidate_prompt_view(candidate: CandidateGenome | dict[str, Any], *, detail: str = "summary", max_artifact_chars: int | None = None) -> dict[str, Any]:
    """Return a compact model-facing view of a candidate genome."""

    genome = candidate if isinstance(candidate, CandidateGenome) else candidate_from_dict(candidate)
    artifact_text = _stringify(genome.artifact)
    view: dict[str, Any] = {
        "id": genome.id,
        "parent_ids": list(genome.parent_ids)[:4],
        "generation": int(genome.generation or 0),
        "fate": genome.current_fate,
        "artifact_type": genome.artifact_type,
        "concise_claim": _clip(genome.concise_claim, 900),
        "core_mechanism": _clip(genome.core_mechanism, 900),
        "assumptions": _clip_list(genome.assumptions, 5, 880),
        "missing_parts": _clip_list(genome.missing_parts, 5, 880),
        "uncertainty_notes": _clip_list(genome.uncertainty_notes, 3, 880),
        "edge_knowledge_seeds": _clip_list(genome.edge_knowledge_seeds, 5, 880),
        "novelty_descriptors": _clip_list(genome.novelty_descriptors, 5, 640),
        "niche_memberships": _clip_list(genome.niche_memberships, 5, 640),
        "failure_lessons": _clip_list(genome.failure_lessons, 5 if detail != "tiny" else 2, 900),
        "inherited_genes": _clip_list(genome.inherited_genes, 5 if detail != "tiny" else 2, 900),
        "mutation_history_tail": _clip_list(genome.mutation_history[-4:], 4, 720),
        "scores": _top_scores(genome.multihead_scores),
        "tool_feedback_summary": _feedback_summary(genome.tool_results),
        "verification_summary": _feedback_summary(genome.verification_trace),
        "formal_artifacts": [_small_mapping(item, max_items=8, string_chars=880) for item in genome.formal_artifacts[:4]],
        "proof_obligations": [_small_mapping(item, max_items=8, string_chars=880) for item in genome.proof_obligations[:6]],
        "obligation_delta": _small_mapping(genome.obligation_delta, max_items=8, string_chars=880),
        "evidence_refs": [_small_mapping(item, max_items=8, string_chars=880) for item in genome.evidence_refs[:6]],
        "source_bindings": [_small_mapping(item, max_items=8, string_chars=880) for item in genome.source_bindings[:6]],
        "evidence_delta": _small_mapping(genome.evidence_delta, max_items=8, string_chars=880),
        "verification_result": _small_mapping(genome.verification_result, max_items=10, string_chars=900),
        "artifact_summary": _artifact_summary(artifact_text, detail=detail, max_artifact_chars=max_artifact_chars),
        "artifact_sha256": _sha256_text(artifact_text) if artifact_text else "",
        "metadata": _metadata_view(genome.metadata),
    }
    nextgen = _to_mapping(_to_mapping(genome.metadata).get("nextgen"))
    if nextgen:
        view["nextgen"] = _small_mapping(nextgen, max_items=8, string_chars=180)
    repair_seed_contract = _repair_seed_contract_view(genome, detail=detail)
    if repair_seed_contract:
        view["repair_seed_contract"] = repair_seed_contract
    evaluator_feedback = _evaluator_feedback_view(genome.metadata, detail=detail)
    if evaluator_feedback:
        view["external_evaluator"] = evaluator_feedback
    if detail == "exact":
        view.update(
            {
                "parent_ids": list(genome.parent_ids),
                "concise_claim": genome.concise_claim,
                "core_mechanism": genome.core_mechanism,
                "assumptions": list(genome.assumptions),
                "missing_parts": list(genome.missing_parts),
                "uncertainty_notes": list(genome.uncertainty_notes),
                "edge_knowledge_seeds": list(genome.edge_knowledge_seeds),
                "novelty_descriptors": list(genome.novelty_descriptors),
                "niche_memberships": list(genome.niche_memberships),
                "failure_lessons": list(genome.failure_lessons),
                "inherited_genes": list(genome.inherited_genes),
                "mutation_history_tail": list(genome.mutation_history),
                "formal_artifacts": list(genome.formal_artifacts),
                "proof_obligations": list(genome.proof_obligations),
                "obligation_delta": dict(genome.obligation_delta),
                "evidence_refs": list(genome.evidence_refs),
                "source_bindings": list(genome.source_bindings),
                "evidence_delta": dict(genome.evidence_delta),
                "verification_result": dict(genome.verification_result),
                "tool_results": list(genome.tool_results),
                "verification_trace": list(genome.verification_trace),
                "scores": dict(genome.multihead_scores),
                "metadata": dict(genome.metadata),
                "artifact": genome.artifact,
            }
        )
    if isinstance(genome, ProjectCandidateGenome):
        view["patch_summary"] = [
            {
                "path": op.path,
                "operation": op.operation,
                "content_sha256": _sha256_text(op.content),
                "content_preview": _clip(op.content, 180 if detail != "tiny" else 80),
            }
            for op in genome.patch_set[:8]
        ]
        view["touched_files"] = _clip_list(genome.touched_files, 12, 220)
        view["touched_symbols"] = _clip_list(genome.touched_symbols, 12, 160)
        view["affected_tests"] = _clip_list(genome.affected_tests, 8, 180)
        view["risk_notes"] = _clip_list(genome.risk_notes, 5, 220)
        view["expected_effects"] = _clip_list(genome.expected_effects, 5, 220)
        view["patch_application_result"] = _small_mapping(genome.patch_application_result, max_items=8, string_chars=240)
        view["commands_run"] = [_small_mapping(item, max_items=8, string_chars=180) for item in genome.commands_run[-4:]]
        if detail == "exact":
            view.update(
                {
                    "patch_set": [operation.to_dict() for operation in genome.patch_set],
                    "touched_files": list(genome.touched_files),
                    "touched_symbols": list(genome.touched_symbols),
                    "affected_tests": list(genome.affected_tests),
                    "risk_notes": list(genome.risk_notes),
                    "expected_effects": list(genome.expected_effects),
                    "patch_application_result": dict(genome.patch_application_result),
                    "commands_run": [dict(item) for item in genome.commands_run],
                }
            )
    return view


def archive_prompt_view(archives: Any, *, population: list[CandidateGenome] | None = None) -> dict[str, Any]:
    """Return archive counts plus a few high-value exemplars, not full archives."""

    if archives is None:
        return {}
    summary = archives.summary() if hasattr(archives, "summary") else _small_mapping(_to_mapping(archives), max_items=20, string_chars=120)
    view: dict[str, Any] = {"summary": summary}
    seen_ids = {candidate.id for candidate in population or []}
    if population:
        view["active_ids"] = [c.id for c in population if c.current_fate == "Active"]
        view["elite_ids"] = [c.id for c in population if c.current_fate == "Elite"]
    archive_specs = (
        ("answer_elites", getattr(archives, "answer_archive", {}), 3),
        ("rarity_elites", getattr(getattr(archives, "rarity_archive", None), "candidates", {}), 3),
        ("dormant_hints", getattr(getattr(archives, "dormant_archive", None), "candidates", {}), 3),
        ("auxiliary_hints", getattr(getattr(archives, "auxiliary_archive", None), "candidates", {}), 2),
    )
    for key, store, limit in archive_specs:
        selected = _archive_candidates(store, limit=limit, detail="tiny", exclude_ids=seen_ids)
        view[key] = selected
        seen_ids.update(str(item.get("id") or "") for item in selected if item.get("id"))
    failure_archive = getattr(archives, "failure_archive", None)
    records = getattr(failure_archive, "records", {}) if failure_archive is not None else {}
    view["failure_lessons"] = _failure_records_view(records, limit=8)
    return view


def history_prompt_view(history: list[dict[str, Any]] | None, *, rounds: int | None = None) -> list[dict[str, Any]]:
    """Return round-level signals; explicit ``rounds`` keeps a recent window."""

    items = [item for item in history or [] if isinstance(item, dict)]
    recent = items if rounds is None else items[-max(1, rounds):]
    out: list[dict[str, Any]] = []
    for item in recent:
        out.append(
            {
                "round": item.get("round"),
                "diagnosis": _to_mapping(item.get("diagnosis")),
                "ranking": _ranking_view(item.get("ranking") or {}),
                "progress_event": _to_mapping(item.get("progress_event")),
                "error": _to_mapping(item.get("error")),
            }
        )
    return out


def world_prompt_view(world: Any, *, detail: str = "summary") -> dict[str, Any]:
    mapping = _to_mapping(world)
    if not mapping:
        return {}
    kind = mapping.get("kind") or mapping.get("snapshot", {}).get("kind") or "world"
    if kind == "project" or "file_manifest" in mapping or "project_world_model" in mapping or "snapshot" in mapping:
        return _project_world_view(mapping)
    return {
        "kind": kind,
        "input_packet_id": _stringify(mapping.get("input_packet_id") or ""),
        "goal_summary": _stringify(mapping.get("goal_summary") or mapping.get("summary") or mapping.get("raw_text") or mapping.get("repr") or ""),
        "likely_task_types": list(mapping.get("likely_task_types") or mapping.get("task_type_hypotheses") or []),
        "constraint_summary": list(mapping.get("constraint_summary") or mapping.get("constraints") or []),
        "evidence_boundaries": dict(mapping.get("evidence_boundaries") or {}),
        "uncertainty_zones": list(mapping.get("uncertainty_zones") or []),
        "edge_seed_pool": list(mapping.get("edge_seed_pool") or mapping.get("possible_edge_knowledge_seeds") or []),
    }


def contract_prompt_view(contract: Any) -> dict[str, Any]:
    data = _to_mapping(contract)
    if not data:
        return {}
    keys = [
        "normalized_goal",
        "input_constraints",
        "allowed_evidence_sources",
        "disallowed_goal_mutations",
        "expected_output_forms",
        "uncertainty_policy",
        "verification_preferences",
        "success_dimensions",
        "failure_dimensions",
        "allowed_patch_scope",
        "unsafe_change_patterns",
        "dynamic_artifact_contract_hash",
    ]
    view = {key: data.get(key) for key in keys if key in data}
    if data.get("normalized_goal") is not None:
        view["normalized_goal"] = _stringify(data.get("normalized_goal"))
    frozen_spec = _to_mapping(data.get("frozen_spec"))
    if frozen_spec and "problem_text" in frozen_spec:
        problem_text = _stringify(frozen_spec.get("problem_text"))
        view["frozen_spec"] = {
            "problem_text": problem_text,
            "spec_sha256": str(frozen_spec.get("spec_sha256") or _sha256_text(problem_text)),
        }
    elif data.get("original_user_goal") is not None:
        view["original_user_goal"] = _stringify(data.get("original_user_goal"))
    dac = data.get("dynamic_artifact_contract")
    outcome_policy = data.get("outcome_policy")
    if (not isinstance(dac, dict) or not dac) and isinstance(outcome_policy, dict):
        dac = outcome_policy.get("dynamic_artifact_contract")
    if isinstance(dac, dict):
        view["dynamic_artifact_contract"] = dict(dac)
    search_space_plan = _search_space_plan_prompt_view(data.get("search_space_plan"))
    search_space = _search_space_plan_prompt_view(data.get("search_space"))
    if not search_space_plan and not search_space and isinstance(outcome_policy, dict):
        search_space_plan = _search_space_plan_prompt_view(outcome_policy.get("search_space_plan") or outcome_policy.get("search_space"))
    if not search_space_plan and not search_space and isinstance(dac, dict):
        search_space_plan = _search_space_plan_prompt_view(dac.get("search_space_plan") or dac.get("search_space"))
    if search_space_plan:
        view["search_space_plan"] = search_space_plan
    if search_space:
        view["search_space"] = search_space
    return view


def policy_prompt_view(policy: Any) -> dict[str, Any]:
    data = _to_mapping(policy)
    if not data:
        return {}
    keys = [
        "candidate_niches",
        "fitness_axes",
        "mutation_operators",
        "parent_selection_preferences",
        "culling_principles",
        "rarity_budget",
        "tool_preferences",
        "stagnation_actions",
        "synthesis_policy",
        "updated_from_diagnoses",
        "search_space_plan",
        "search_space",
    ]
    view = {key: data.get(key) for key in keys if key in data}
    for key in ("search_space", "search_space_plan"):
        plan_view = _search_space_plan_prompt_view(data.get(key))
        if plan_view:
            view[key] = plan_view
    metadata = _to_mapping(data.get("metadata"))
    search_space_plan = metadata.get("search_space_plan") or metadata.get("search_space_contract")
    metadata_plan_view = _search_space_plan_prompt_view(search_space_plan)
    if metadata_plan_view:
        view["search_space_plan"] = metadata_plan_view
    if metadata.get("search_space_plan_required"):
        view["search_space_plan_required"] = metadata.get("search_space_plan_required")
    if isinstance(metadata.get("strategy_comparison"), dict):
        view["strategy_comparison"] = dict(metadata.get("strategy_comparison") or {})
    if isinstance(metadata.get("productive_branch_allocation"), dict):
        view["productive_branch_allocation"] = dict(metadata.get("productive_branch_allocation") or {})
    if isinstance(metadata.get("theory"), dict):
        view["theory"] = dict(metadata.get("theory") or {})
    seed_family_priority = metadata.get("seed_family_priority")
    if isinstance(seed_family_priority, list):
        view["seed_family_priority"] = [dict(item) for item in seed_family_priority if isinstance(item, dict)]
    if metadata.get("seed_family_priority_source"):
        view["seed_family_priority_source"] = metadata.get("seed_family_priority_source")
    if metadata.get("seed_instruction"):
        view["seed_instruction"] = metadata.get("seed_instruction")
    if isinstance(metadata.get("seed_family_coverage_snapshot"), dict):
        view["seed_family_coverage_snapshot"] = dict(metadata.get("seed_family_coverage_snapshot") or {})
    if isinstance(metadata.get("seed_portfolio"), list):
        view["seed_portfolio"] = [dict(item) for item in metadata.get("seed_portfolio") or [] if isinstance(item, dict)]
    if isinstance(metadata.get("seed_portfolio_contract"), dict):
        view["seed_portfolio_contract"] = dict(metadata.get("seed_portfolio_contract") or {})
    for kind in ("seed", "mutation_plan", "offspring"):
        for key in (
            f"{kind}_batch_index",
            f"accepted_{kind}_signatures",
            f"rejected_{kind}_count",
            f"rejected_{kind}_feedback",
            f"{kind}_instruction",
        ):
            if key in metadata:
                view[key] = metadata[key]
    if "accepted_offspring_candidates" in metadata:
        view["accepted_offspring_candidates"] = metadata["accepted_offspring_candidates"]
    if "search_kernel_skills" in metadata:
        view["search_kernel_skills"] = metadata["search_kernel_skills"]
    for key in ("seed_coverage", "target_perturb_seed_judgment", "algorithm_efficiency", "model_parallel_efficiency", "seed_active_frontier", "seed_reservoir_ref"):
        if key in metadata:
            value = metadata.get(key)
            view[key] = dict(value) if isinstance(value, dict) else value
    return view


def _compress_payload(request_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    if request_type == "nexus_synthesize_result":
        return _synthesis_evidence_manifest(payload)

    candidates = _coerce_candidates(payload.get("candidates") or payload.get("population") or payload.get("parents") or [])
    population = _coerce_candidates(payload.get("population") or payload.get("candidates") or [])
    parents = _coerce_candidates(payload.get("parents") or [])
    compressed: dict[str, Any] = {
        "request_type": request_type,
        "prompt_contract": {
            "state_is_compressed": True,
            "full_state_location": "local Nexus checkpoint/population/archive files",
            "do_not_assume_omitted_raw_outputs_are_absent": True,
            "current_state_precedence": "Exact artifacts in current parents and source_context supersede earlier packet/world absence claims about those artifacts or identifiers.",
        },
    }
    if request_type == "nexus_build_text_world_model" and "packet" in payload:
        compressed["packet"] = _text_packet_prompt_view(payload.get("packet"))
    contract_view = contract_prompt_view(payload.get("contract"))
    policy_view = policy_prompt_view(payload.get("policy"))
    world_view = world_prompt_view(payload.get("world") or payload.get("snapshot"), detail="summary")
    requested_candidate_count = _positive_int(payload.get("requested_candidate_count")) or 0
    compressed["search_space_contract"] = _search_space_contract_from_views(
        contract_view=contract_view,
        policy_view=policy_view,
        world_view=world_view,
        request_type=request_type,
        candidate_target_count=requested_candidate_count or len(candidates) or len(parents) or 0,
    )
    if request_type in {"nexus_seed_population", "nexus_generate_offspring", "nexus_plan_mutations"}:
        plans = payload.get("plans") if isinstance(payload.get("plans"), list) else []
        direct_plan_metadata = (
            _to_mapping(_to_mapping(plans[0]).get("metadata"))
            if len(plans) == 1
            else {}
        )
        direct_task_artifact = (
            request_type == "nexus_generate_offspring"
            and direct_plan_metadata.get("plan_source") == "runtime_lineage_envelope"
            and direct_plan_metadata.get("completion_mode") == "complete_task_artifact_only"
        )
        dynamic_artifact_contract = _to_mapping(contract_view.get("dynamic_artifact_contract"))
        required_work_product = _to_mapping(dynamic_artifact_contract.get("required_work_product"))
        allowed_shapes = dynamic_artifact_contract.get("allowed_artifact_shapes")
        adapter_requirements = _to_mapping(dynamic_artifact_contract.get("adapter_requirements"))
        schema_bound_seed = request_type == "nexus_seed_population" and (
            bool(required_work_product.get("required_fields"))
            or any(
                bool(_to_mapping(shape).get("required_fields"))
                for shape in (allowed_shapes if isinstance(allowed_shapes, list) else [])
            )
            or bool(adapter_requirements.get("required_fields"))
        )
        compressed["artifact_generation_contract"] = _artifact_generation_contract_from_view(
            contract_view,
            request_type=request_type,
            search_space_contract=compressed["search_space_contract"],
            complete_task_artifact_only=direct_task_artifact or schema_bound_seed,
        )
    if request_type in ACTIVATION_REQUESTS:
        compressed["activation_contract"] = activation_prompt_contract(
            request_type=request_type,
            contract_view=contract_view,
            policy_view=policy_view,
            policy_metadata=_to_mapping(getattr(payload.get("policy"), "metadata", None) or _to_mapping(payload.get("policy")).get("metadata")),
            semantic_control=_to_mapping(payload.get("semantic_control")),
        )
    if request_type == "nexus_relative_rank":
        compressed["ranking_bias_mitigation"] = {
            "position_bias": "Candidates may be presented in original and reversed order; use candidate ids and evidence, not list position.",
            "verbosity_bias": "Do not reward longer artifacts for length; compare concrete artifact delta, evidence, source binding, and obligation progress.",
            "style_bias": "Prefer verified useful progress over polished narrative.",
        }
    if "user_goal" in payload:
        compressed["user_goal"] = _stringify(payload.get("user_goal"))
    if "prompt" in payload:
        compressed["prompt"] = _stringify(payload.get("prompt"))
    if "round_index" in payload:
        compressed["round_index"] = payload.get("round_index")
    if "requested_candidate_count" in payload:
        compressed["requested_candidate_count"] = payload.get("requested_candidate_count")
    if "actions" in payload:
        compressed["actions"] = _clip_list(payload.get("actions") or [], 20, 160)
    if "source_context" in payload:
        compressed["source_context"] = _source_context_view(payload.get("source_context"))
        if request_type == "nexus_generate_offspring":
            compressed["source_context"].pop("initial_candidates", None)
    if "mutation_instruction" in payload:
        compressed["mutation_instruction"] = _stringify(payload.get("mutation_instruction"))
    if "plans" in payload:
        compressed["plans"] = (
            [_to_mapping(plan) for plan in payload.get("plans") or []]
            if request_type == "nexus_generate_offspring"
            else [_small_mapping(_to_mapping(plan), max_items=10, string_chars=1040) for plan in payload.get("plans") or []]
        )
    for key in ("coverage_report", "clusters", "representatives", "instructions", "extra"):
        if key in payload:
            compressed[key] = payload[key]
    compressed["contract"] = contract_view
    if contract_view.get("frozen_spec") and isinstance(compressed.get("source_context"), dict):
        compressed["source_context"].pop("frozen_spec", None)
    compressed["policy"] = policy_view
    compressed["world"] = world_view
    protected_candidate_ids = _protected_candidate_ids_from_controls(_prompt_context_controls(payload))
    if candidates:
        candidate_detail = "exact" if request_type in {"nexus_critique_candidates", "nexus_relative_rank", "nexus_diagnose_search_state", "nexus_should_stop"} else "summary"
        selected_candidates = (
            candidates
            if request_type in {"nexus_critique_candidates", "nexus_relative_rank"}
            else _select_candidates_for_prompt(candidates, limit=48, protected_ids=protected_candidate_ids)
        )
        compressed["candidates"] = [
            candidate_prompt_view(c, detail=candidate_detail)
            for c in selected_candidates
        ]
        if protected_candidate_ids:
            compressed["_protected_candidate_ids"] = sorted(protected_candidate_ids)
        compressed["candidate_population_stats"] = _population_stats(candidates)
    if parents:
        exact_parents = request_type in {"nexus_generate_offspring", "nexus_plan_mutations", "nexus_request_context"}
        parent_detail = "exact" if exact_parents else "summary"
        selected_parents = parents if exact_parents else parents[:16]
        compressed["parents"] = [candidate_prompt_view(c, detail=parent_detail) for c in selected_parents]
    strategy = strategy_comparison_context(payload.get("policy"), population or candidates)
    if strategy:
        compressed["strategy_comparison"] = strategy
    compressed["archives"] = archive_prompt_view(payload.get("archives"), population=population or candidates)
    compressed["history"] = history_prompt_view(payload.get("history") if isinstance(payload.get("history"), list) else [])
    diagnosis = payload.get("diagnosis")
    if diagnosis is not None:
        compressed["diagnosis"] = _to_mapping(diagnosis)
    # Keep unknown scalar/small fields, but never carry huge nested blobs by default.
    for key, value in payload.items():
        if key in {"user_goal", "prompt", "packet", "contract", "world", "snapshot", "policy", "candidates", "population", "parents", "archives", "history", "diagnosis", "plans", "actions", "source_context", "mutation_instruction", "round_index", "coverage_report", "clusters", "representatives", "instructions", "extra"}:
            continue
        if _json_chars(value) <= 16000:
            compressed[key] = value
        else:
            compressed[key] = _summarize_value(value)
    return compressed


def _text_packet_prompt_view(value: Any) -> dict[str, Any]:
    """Keep the user task visible while bounding duplicated packet evidence."""

    data = _to_mapping(value)
    raw_text = _stringify(data.get("raw_text"))
    raw_evidence = data.get("available_evidence")
    evidence = [dict(item) for item in raw_evidence if isinstance(item, dict)] if isinstance(raw_evidence, list) else []
    return {
        "packet_id": _stringify(data.get("packet_id") or ""),
        "raw_text": raw_text,
        "raw_text_chars": len(raw_text),
        "raw_text_sha256": _sha256_text(raw_text),
        "raw_text_truncated": False,
        "extracted_claims": list(data.get("extracted_claims") or []),
        "constraints": list(data.get("constraints") or []),
        "available_evidence": evidence,
        "task_type_hypotheses": list(data.get("task_type_hypotheses") or []),
        "uncertainty_zones": list(data.get("uncertainty_zones") or []),
        "possible_edge_knowledge_seeds": list(data.get("possible_edge_knowledge_seeds") or []),
    }


def _source_context_view(value: Any) -> dict[str, Any]:
    data = _to_mapping(value)
    out: dict[str, Any] = {}
    if "problem_spec" in data:
        out["problem_spec"] = _stringify(data.get("problem_spec"))
    frozen_spec = _to_mapping(data.get("frozen_spec"))
    if frozen_spec and "problem_text" in frozen_spec:
        problem_text = _stringify(frozen_spec.get("problem_text"))
        out["frozen_spec"] = {
            "problem_text": problem_text,
            "spec_sha256": str(frozen_spec.get("spec_sha256") or _sha256_text(problem_text)),
        }
    raw_initial = data.get("initial_candidates")
    if isinstance(raw_initial, list):
        out["initial_candidates"] = [_initial_candidate_context_view(item) for item in raw_initial]
    selected = data.get("selected_files")
    if isinstance(selected, list):
        out["selected_files"] = [str(item) for item in selected if str(item or "").strip()]
    elif selected:
        out["selected_files"] = _clip_list([selected], 1, 260)
    budget = data.get("budget_policy")
    if isinstance(budget, dict):
        out["budget_policy"] = _small_mapping(budget, max_items=8, string_chars=260)
    elif budget is not None:
        out["budget_policy"] = _clip(budget, 260)
    context_limits = data.get("context_limits")
    if isinstance(context_limits, dict):
        out["context_limits"] = dict(context_limits)
    slices: list[dict[str, Any]] = []
    raw_slices = data.get("slices") if isinstance(data.get("slices"), list) else []
    for item in raw_slices:
        if not isinstance(item, dict):
            continue
        view: dict[str, Any] = {}
        for key in ("path", "hash", "start", "end", "truncated", "original_chars", "selected_chars"):
            if key in item and item.get(key) is not None:
                view[key] = item.get(key)
        if "text" in item:
            view["text"] = _stringify(item.get("text"))
        if view:
            slices.append(view)
    if slices:
        out["slices"] = slices
    return out


def _initial_candidate_context_view(value: Any) -> dict[str, Any]:
    data = _to_mapping(value)
    keys = (
        "id",
        "parent_ids",
        "generation",
        "artifact_type",
        "artifact",
        "concise_claim",
        "core_mechanism",
        "assumptions",
        "missing_parts",
        "uncertainty_notes",
        "contract_hash",
    )
    view = {key: data.get(key) for key in keys if key in data}
    if "artifact" in view:
        view["artifact_sha256"] = _sha256_json(view["artifact"])
    return view


def _artifact_generation_contract(contract: Any, *, request_type: str) -> dict[str, Any]:
    contract_view = contract_prompt_view(contract)
    return _artifact_generation_contract_from_view(
        contract_view,
        request_type=request_type,
        search_space_contract=_search_space_contract_from_views(contract_view=contract_view, policy_view={}, world_view={}, request_type=request_type),
    )


def _artifact_generation_contract_from_view(
    view: dict[str, Any],
    *,
    request_type: str,
    search_space_contract: dict[str, Any] | None = None,
    complete_task_artifact_only: bool = False,
) -> dict[str, Any]:
    """Domain-neutral instruction that pushes evolution toward real artifacts.

    The runtime must not hard-code "proof", "code", "science", "article", or
    "fiction" as privileged domains.  The model-defined artifact contract tells
    the model what kind of work product exists in this run; this helper only
    states the platform invariant: every candidate or mutation must create or
    move toward an object-level artifact, not merely discuss one.
    """

    dac = view.get("dynamic_artifact_contract") if isinstance(view, dict) else {}
    required_work_product = dac.get("required_work_product") if isinstance(dac, dict) else {}
    minimum_delta = dac.get("minimum_concrete_delta") if isinstance(dac, dict) else {}
    contract = {
        "request_type": request_type,
        "model_defined_required_work_product": required_work_product or "use the dynamic_artifact_contract",
        "model_defined_minimum_delta": minimum_delta or "produce a concrete artifact delta relative to parent state",
        "non_negotiable_runtime_invariant": "Candidates must contain the actual object-level artifact or an executable repair step toward it; pure commentary, plans, labels, or promises are not progress.",
        "search_space_breadth_contract": search_space_contract or {},
        "search_space_rule": "Before deepening one surface, allocate candidates across materially different model-defined planes from the user objective; local files/tools/evidence are grounding surfaces, not the objective itself.",
        "surface_bias_guard": "Do not let an easy-to-patch or easy-to-verify local surface monopolize seed, mutation, or offspring generation when the user objective asks for higher-level mechanism, lifecycle, policy, materialization, or final-answer design.",
        "required_search_space_metadata": "Each candidate should include metadata.search_space.family_id or search_space.plane_id chosen from the model-authored search space; if none exists, author one from the objective first.",
        "project_patch_output_rule": "For project/code patch candidates, prefer artifact.patch_set with PatchOperation objects ({path, operation, content, old_text, new_text}) for simple write/append/replace/delete edits. If you use artifact.unified_diff instead, every hunk header must be standard unified diff syntax with line ranges like @@ -start,count +start,count @@; never emit bare @@ headers.",
        "examples_are_not_domain_limits": True,
        "valid_evolution_shapes": [
            "existing artifact refinement",
            "existing artifact extension",
            "new artifact materialization",
        ],
        "when_incomplete": "emit the smallest concrete partial artifact, worked example, structured object, or repair obligation that can be verified against the model-defined contract",
        "required_structured_candidate_fields": [
            "touched_files",
            "source_bindings",
            "evidence_refs",
            "evaluation_dimensions",
            "final_gate",
            "edge_knowledge_seeds",
            "formal_artifacts",
        ],
        "structured_field_rule": "Populate these fields from the actual artifact delta. Preserve or extend parent edge_knowledge_seeds and formal_artifacts when they remain relevant. Use empty arrays/objects when the model-defined contract is not file/source based; do not invent paths or evidence.",
        "duplicate_rule": "When parents or prior-attempt signatures are supplied, every materialized artifact must differ exactly from them and siblings must be pairwise distinct.",
    }
    if complete_task_artifact_only:
        contract.update(
            {
                "completion_mode": "complete_task_artifact_only",
                "model_defined_required_work_product": "a complete evaluator-visible task artifact matching contract.frozen_spec when present, otherwise dynamic_artifact_contract",
                "model_defined_minimum_delta": "return a complete artifact that differs from a truthful parent when parents are supplied; otherwise materialize a complete schema-valid seed",
                "non_negotiable_runtime_invariant": "Every candidate must contain a complete evaluator-visible task artifact; procedural substitutes, search instructions, commentary, partial artifacts, and unchanged copies are invalid.",
                "when_incomplete": "Return an empty result array under the request schema; the runtime preserves any already accepted incumbent and otherwise checkpoints. Never substitute partial work, procedures, or plans for the task artifact.",
                "duplicate_rule": "Do not return an unchanged parent, a prior-attempt artifact, or duplicate siblings. Return an empty result rather than a known exact duplicate.",
            }
        )
    return contract


def _search_space_contract_from_views(
    *,
    contract_view: dict[str, Any],
    policy_view: dict[str, Any],
    world_view: dict[str, Any],
    request_type: str,
    candidate_target_count: int = 0,
) -> dict[str, Any]:
    objective = str(contract_view.get("normalized_goal") or contract_view.get("original_user_goal") or "")
    plan: dict[str, Any] = {}
    for value in (
        policy_view.get("search_space"),
        contract_view.get("search_space_plan"),
        contract_view.get("search_space"),
        policy_view.get("search_space_plan"),
    ):
        usable = _usable_search_space_plan(value)
        if usable:
            plan = usable
            break
    assessment = {
        "task_type": world_view.get("kind") or "",
        "real_objective": objective,
        "search_space_plan": plan,
    }
    search_map = build_search_space_map(assessment, requested_candidate_count=max(0, int(candidate_target_count or 0)))
    theory = _to_mapping(policy_view.get("theory"))
    producers = _to_mapping(theory.get("producers"))
    theory_dimensions = [str(key) for key, enabled in producers.items() if enabled]
    if not theory_dimensions and theory.get("enabled"):
        theory_dimensions = ["mdl", "boed", "geometry"]
    world_structure = {
        "kind": world_view.get("kind"),
        "file_roles": list(_to_mapping(world_view.get("file_roles")).keys()),
        "hotspots": list(_to_mapping(world_view.get("hotspot_map")).keys()),
        "objective_relevant_files": list(_to_mapping(world_view.get("objective_relevance_map")).keys()),
    }
    return {
        "request_type": request_type,
        "source": search_map.get("source"),
        "model_driven": True,
        "candidate_target_count": search_map.get("candidate_target_count"),
        "needs_model_authored_search_space": bool(search_map.get("needs_model_authored_search_space")),
        "candidate_families": search_map.get("candidate_families", []),
        "coverage_gate": search_map.get("coverage_gate", {}),
        "surface_bias_guard": search_map.get("surface_bias_guard", {}),
        "theory_dimensions": {
            "source": "supplemental_prompt_pressure",
            "dimensions": theory_dimensions,
            "rule": "Use these only to diversify exploration pressure; do not treat them as eligibility, stop, or correctness gates.",
        },
        "world_structure": {k: v for k, v in world_structure.items() if v},
        "anti_narrowing_instruction": (
            "If recent candidates cluster around one implementation/detail surface, generate the next candidates from different objective-level planes rather than another same-surface patch variant."
        ),
    }


def _usable_search_space_plan(value: Any) -> dict[str, Any]:
    plan = _to_mapping(value)
    for key in ("candidate_families", "exploration_planes", "families", "planes"):
        families = plan.get(key)
        if isinstance(families, list) and any(
            isinstance(item, dict) and str(item.get("id") or item.get("name") or "").strip()
            for item in families
        ):
            return plan
    return {}


def _synthesis_evidence_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a compact final-synthesis input from evidence manifests only.

    Final synthesis is a high-failure stage for OpenAI-compatible endpoints when
    it receives the full evolutionary universe.  The model does not need raw
    checkpoints or every artifact to decide how to present the final/reference
    result; it needs the contract, archive summary, candidate status, and the
    strongest evidence/blocker deltas.  This manifest keeps the stage generic:
    no code/math/science domain branch is assumed.
    """

    candidates = _coerce_candidates(payload.get("population") or payload.get("candidates") or [])
    selected = _select_candidates_for_prompt(candidates, limit=16, protected_ids=_protected_candidate_ids_from_controls(_prompt_context_controls(payload)))
    return {
        "request_type": "nexus_synthesize_result",
        "prompt_contract": {
            "state_is_compressed": True,
            "full_state_location": "local Nexus checkpoint/population/archive files",
            "final_synthesis_must_not_claim_unverified_completion": True,
            "if_no_contract_valid_final_exists_return_reference_or_route_incomplete": True,
            "best_direction_must_directly_answer_frozen_user_goal": True,
            "supporting_material_must_not_replace_the_explored_object": True,
            "do_not_use_fixed_target_categories": True,
        },
        "contract": contract_prompt_view(payload.get("contract")),
        "world": world_prompt_view(payload.get("world"), detail="tiny"),
        "candidate_population_stats": _population_stats(candidates),
        "candidates": [candidate_prompt_view(c, detail="exact") for c in selected],
        "archives": archive_prompt_view(payload.get("archives"), population=candidates),
        "strategy_comparison": strategy_comparison_context(payload.get("policy"), candidates),
        "synthesis_requirements": {
            "return_non_empty_json": True,
            "required_fields": ["status", "final_answer"],
            "do_not_treat_reference_material_as_solved": True,
            "surface_answer_candidates_without_project_certification": True,
            "best_candidate_id_rule": "Choose the candidate whose main claim directly answers the frozen goal. Put records, verification wrappers, or audit scaffolds in supporting material unless the frozen goal itself asks for them.",
            "intent_binding_output": "If useful, include free-text intent_alignment_rationale and a continuous intent_directness score; do not emit enum target kinds.",
        },
    }


def _prompt_context_controls(payload: dict[str, Any]) -> dict[str, Any]:
    controls = payload.get("_prompt_context_controls") if isinstance(payload, dict) else None
    return dict(controls) if isinstance(controls, dict) else {}


def _protected_paths_from_controls(controls: dict[str, Any]) -> list[str]:
    refs = [str(item) for item in controls.get("protect_refs", []) if item] if isinstance(controls.get("protect_refs"), list) else []
    mapping = {
        "problem_spec": ["contract", "world", "policy"],
        "verification_plan": ["verification_plan"],
        "verification_regime": ["verification_regime"],
        "honesty_invariant": ["prompt_contract", "synthesis_requirements"],
    }
    out: list[str] = []
    for ref in refs:
        out.extend(mapping.get(ref, [ref] if ref in {"contract", "world", "policy", "prompt_contract", "verification_regime"} else []))
    if controls.get("verification_plan"):
        out.append("verification_plan")
    if controls.get("verification_regime"):
        out.append("verification_regime")
    return list(dict.fromkeys(out))


def _protected_candidate_ids_from_controls(controls: dict[str, Any]) -> set[str]:
    raw = controls.get("protect_candidate_ids") or controls.get("protected_candidate_ids") or []
    if not isinstance(raw, list):
        return set()
    return {str(item) for item in raw if str(item or "").strip()}


def _apply_prompt_context_controls(compressed: dict[str, Any], controls: dict[str, Any]) -> dict[str, Any]:
    if not controls:
        return compressed
    out = json.loads(json.dumps(compressed, ensure_ascii=False, default=str))
    if "verification_plan" in controls and "verification_plan" not in out:
        out["verification_plan"] = _to_mapping(controls.get("verification_plan"))
    if "verification_regime" in controls:
        regime = _verification_regime_view(controls.get("verification_regime"))
        out = _insert_after_key(out, "candidates" if "candidates" in out else "candidate_population_stats", "verification_regime", regime)
    drop_refs = [str(item) for item in controls.get("drop_refs", []) if item] if isinstance(controls.get("drop_refs"), list) else []
    for ref in drop_refs:
        if ref == "drop:history":
            out.pop("history", None)
        elif ref == "drop:archive_elites" and isinstance(out.get("archives"), dict):
            for key in ("answer_elites", "rarity_elites", "dormant_hints"):
                out["archives"].pop(key, None)
        elif ref == "drop:failure_lessons":
            if isinstance(out.get("archives"), dict):
                out["archives"].pop("failure_lessons", None)
            for key in ("candidates", "parents"):
                if isinstance(out.get(key), list):
                    for item in out[key]:
                        if isinstance(item, dict):
                            item.pop("failure_lessons", None)
    out["_prompt_context_controls_applied"] = {
        "protect_refs": controls.get("protect_refs", []),
        "drop_refs": drop_refs,
        "view_hash": str(controls.get("view_hash") or ""),
        "verification_regime_count": len(out.get("verification_regime", []) if isinstance(out.get("verification_regime"), list) else []),
    }
    return out


def _verification_regime_view(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    allowed = {
        "id",
        "origin",
        "must_pass",
        "exogeneity_probe",
        "variety_probe",
        "falsification_budget",
        "replay_record",
    }
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        cleaned = {key: item.get(key) for key in allowed if key in item}
        if cleaned:
            out.append(cleaned)
    return out


def _insert_after_key(mapping: dict[str, Any], after_key: str, key: str, value: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    inserted = False
    for current_key, current_value in mapping.items():
        if current_key == key:
            continue
        out[current_key] = current_value
        if current_key == after_key:
            out[key] = value
            inserted = True
    if not inserted:
        out[key] = value
    return out


def _shrink_verification_regime(regime: Any, *, max_chars: int) -> list[dict[str, Any]]:
    items = [dict(item) for item in regime if isinstance(item, dict)] if isinstance(regime, list) else []
    items.sort(key=lambda item: (not bool(item.get("must_pass")), str(item.get("id") or "")))
    out: list[dict[str, Any]] = []
    for item in items[-8:]:
        cleaned = dict(item)
        if isinstance(cleaned.get("replay_record"), dict):
            cleaned["replay_record"] = _small_mapping(cleaned["replay_record"], max_items=4, string_chars=80)
        for probe_key in ("exogeneity_probe", "variety_probe"):
            if isinstance(cleaned.get(probe_key), dict):
                probe = dict(cleaned[probe_key])
                for field in ("context", "content"):
                    if field in probe:
                        probe[field] = _clip(probe[field], 160)
                cleaned[probe_key] = _small_mapping(probe, max_items=6, string_chars=160)
        candidate = [*out, cleaned]
        if _json_chars(candidate) > max_chars and out:
            continue
        out = candidate
        if _json_chars(out) > max_chars:
            break
    return out


def _snapshot_paths(payload: dict[str, Any], paths: list[str]) -> dict[str, Any]:
    return {path: _get_path(payload, path) for path in paths if _get_path(payload, path) is not None}


def _restore_paths(payload: dict[str, Any], snapshot: dict[str, Any]) -> None:
    for path, value in snapshot.items():
        _set_path(payload, path, value)


def _get_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in str(path).split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _set_path(payload: dict[str, Any], path: str, value: Any) -> None:
    current = payload
    parts = [part for part in str(path).split(".") if part]
    for part in parts[:-1]:
        if not isinstance(current.get(part), dict):
            current[part] = {}
        current = current[part]
    if parts:
        current[parts[-1]] = value


def _structured_request_chars(request_type: str, schema_hint: dict[str, Any], payload: dict[str, Any]) -> int:
    return _json_chars({"request_type": request_type, "schema_hint": schema_hint, "payload": payload})


def _structured_payload_char_budget(request_type: str, schema_hint: dict[str, Any], *, max_chars: int) -> int:
    envelope_chars = _structured_request_chars(request_type, schema_hint, {}) - _json_chars({})
    return max(1, int(max_chars) - envelope_chars)


def _fit_payload(payload: dict[str, Any], *, max_chars: int, protected_paths: list[str] | None = None) -> dict[str, Any]:
    """Recursively shrink a compressed payload until its JSON fits the budget."""

    fitted = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    protected_snapshot = _snapshot_paths(fitted, protected_paths or [])
    if isinstance(fitted.get("verification_regime"), list) and "verification_regime" not in protected_snapshot:
        fitted["verification_regime"] = _shrink_verification_regime(fitted["verification_regime"], max_chars=max(1000, max_chars // 4))
    if isinstance(payload.get("search_space_contract"), dict):
        protected_snapshot["search_space_contract"] = payload["search_space_contract"]
    if _json_chars(fitted) <= max_chars:
        return fitted
    # First reduce candidate and archive exemplars; these dominate most payloads.
    for limit in (32, 24, 16, 12, 8, 4):
        _trim_sequence(fitted, "candidates", limit, protected_ids=set(fitted.get("_protected_candidate_ids", []) if isinstance(fitted.get("_protected_candidate_ids"), list) else []))
        _trim_sequence(fitted, "parents", min(limit, 8))
        for key in ("answer_elites", "rarity_elites", "dormant_hints", "auxiliary_hints", "failure_lessons"):
            if isinstance(fitted.get("archives"), dict):
                _trim_sequence(fitted["archives"], key, min(5, max(2, limit // 4)))
        _restore_paths(fitted, protected_snapshot)
        if _json_chars(fitted) <= max_chars:
            return fitted
    # Then halve large strings/lists until fit.  This preserves schema shape.
    for string_limit in (600, 360, 220, 120, 60):
        fitted = _recursive_clip(fitted, string_limit=string_limit, list_limit=8)
        _restore_paths(fitted, protected_snapshot)
        if _json_chars(fitted) <= max_chars:
            return fitted
    minimal: dict[str, Any] = {
        "request_type": fitted.get("request_type"),
        "prompt_contract": fitted.get("prompt_contract"),
    }
    if fitted.get("packet"):
        minimal["packet"] = fitted.get("packet")
    minimal.update({
        "contract": fitted.get("contract"),
        "policy": fitted.get("policy"),
        "candidate_population_stats": fitted.get("candidate_population_stats"),
        "candidates": _trimmed_with_protected(fitted.get("candidates", []), 3, set(fitted.get("_protected_candidate_ids", []) if isinstance(fitted.get("_protected_candidate_ids"), list) else [])),
        "archives": fitted.get("archives", {}).get("summary", {}) if isinstance(fitted.get("archives"), dict) else {},
        "_fit_warning": "payload was reduced to minimal Nexus prompt view",
    })
    if payload.get("search_space_contract"):
        minimal["search_space_contract"] = payload.get("search_space_contract")
    if fitted.get("history"):
        minimal["history"] = fitted.get("history", [])[-1:]
    _restore_paths(minimal, protected_snapshot)
    if _json_chars(minimal) <= max_chars:
        return minimal

    search_space = _to_mapping(payload.get("search_space_contract"))
    families = [dict(item) for item in search_space.get("candidate_families", []) if isinstance(item, dict)]
    non_search_protected = {path: value for path, value in protected_snapshot.items() if path != "search_space_contract"}
    clipped = _recursive_clip(minimal, string_limit=120, list_limit=4)
    _restore_paths(clipped, non_search_protected)
    clipped["search_space_contract"] = search_space
    if _json_chars(clipped) <= max_chars:
        return clipped

    compact_search_space = {
        str(key): _recursive_clip(value, string_limit=120, list_limit=4)
        for key, value in search_space.items()
        if key != "candidate_families"
    }
    compact_families: list[dict[str, Any]] = []
    compacted_count = 0
    for item in families:
        family = {
            str(key): value if key == "id" else _recursive_clip(value, string_limit=120, list_limit=4)
            for key, value in item.items()
        }
        family["id"] = str(item.get("id") or item.get("name") or "")
        compacted_count += int(family != item)
        compact_families.append(family)
    compact_search_space["candidate_families"] = compact_families
    if compacted_count:
        compact_search_space["family_details_compacted"] = compacted_count
    if compact_search_space != search_space:
        compact_search_space["search_space_compacted"] = True
    clipped["search_space_contract"] = compact_search_space
    if _json_chars(clipped) <= max_chars:
        return clipped

    id_families = [{"id": str(item.get("id") or item.get("name") or "")} for item in families]
    id_search_space = {
        **{
            key: value
            for key, value in compact_search_space.items()
            if key not in {"candidate_families", "family_details_compacted"}
        },
        "candidate_families": id_families,
        "family_details_omitted": len(id_families),
    }
    clipped["search_space_contract"] = id_search_space
    if _json_chars(clipped) <= max_chars:
        return clipped

    base_search_space = {
        **{key: value for key, value in id_search_space.items() if key != "candidate_families"},
        "candidate_families": [],
        "families_omitted": len(id_families),
    }
    clipped["search_space_contract"] = base_search_space
    for index, family in enumerate(id_families):
        candidate_search_space = {
            **base_search_space,
            "candidate_families": [*clipped["search_space_contract"]["candidate_families"], family],
            "families_omitted": len(id_families) - index - 1,
        }
        candidate_payload = {**clipped, "search_space_contract": candidate_search_space}
        if _json_chars(candidate_payload) > max_chars:
            break
        clipped = candidate_payload
    emergency_search_space = {
        "source": search_space.get("source"),
        "candidate_families": [],
        "family_details_omitted": len(id_families),
        "families_omitted": len(id_families),
    }
    emergency = {
        "request_type": minimal.get("request_type"),
        "_fit_warning": "payload was reduced to search-space omission view",
        "search_space_contract": emergency_search_space,
    }
    _restore_paths(emergency, non_search_protected)
    if _json_chars(emergency) <= max_chars:
        for index, family in enumerate(id_families):
            candidate_search_space = {
                **emergency_search_space,
                "candidate_families": [*emergency["search_space_contract"]["candidate_families"], family],
                "families_omitted": len(id_families) - index - 1,
            }
            candidate_payload = {**emergency, "search_space_contract": candidate_search_space}
            if _json_chars(candidate_payload) > max_chars:
                break
            emergency = candidate_payload
    clipped_count = len(_to_mapping(clipped.get("search_space_contract")).get("candidate_families") or [])
    emergency_count = len(_to_mapping(emergency.get("search_space_contract")).get("candidate_families") or [])
    return emergency if emergency_count > clipped_count or _json_chars(clipped) > max_chars else clipped


def _select_candidates_for_prompt(candidates: list[CandidateGenome], *, limit: int, protected_ids: set[str] | None = None) -> list[CandidateGenome]:
    protected_ids = set(protected_ids or set())
    if len(candidates) <= limit:
        return candidates
    buckets: list[CandidateGenome] = []
    seen: set[str] = set()

    def add(items: Iterable[CandidateGenome], n: int) -> None:
        for candidate in items:
            if len(buckets) >= limit:
                return
            if candidate.id in seen:
                continue
            seen.add(candidate.id)
            buckets.append(candidate)
            n -= 1
            if n <= 0:
                return

    by_quality = sorted(candidates, key=_candidate_prompt_priority, reverse=True)
    protected = [c for c in candidates if c.id in protected_ids]
    active = [c for c in by_quality if c.current_fate == "Active"]
    elite = [c for c in by_quality if c.current_fate == "Elite"]
    incubating = [c for c in by_quality if c.current_fate == "Incubating"]
    rare = [c for c in by_quality if c.edge_knowledge_seeds or c.multihead_scores.get("rarity", 0.0) > 0.3]
    dormant = [c for c in by_quality if c.current_fate == "Dormant"]
    recent = sorted(candidates, key=lambda c: (int(c.generation or 0), c.created_at), reverse=True)
    add(protected, len(protected))
    for semantic_bucket in (elite, active, incubating, rare, dormant, recent):
        add(semantic_bucket, 1)
    add(elite, max(4, limit // 6))
    add(active, max(8, limit // 3))
    add(incubating, max(4, limit // 5))
    add(rare, max(4, limit // 6))
    add(dormant, max(3, limit // 8))
    add(recent, limit - len(buckets))
    return buckets[:limit]


def _population_stats(candidates: list[CandidateGenome]) -> dict[str, Any]:
    fates: dict[str, int] = {}
    generations: dict[str, int] = {}
    niches: dict[str, int] = {}
    search_planes: dict[str, int] = {}
    source_surfaces: dict[str, int] = {}
    for c in candidates:
        fates[c.current_fate] = fates.get(c.current_fate, 0) + 1
        generations[str(c.generation)] = generations.get(str(c.generation), 0) + 1
        for niche in c.niche_memberships or ([c.core_mechanism] if c.core_mechanism else []):
            niches[niche] = niches.get(niche, 0) + 1
        metadata = c.metadata if isinstance(c.metadata, dict) else {}
        search_space = c.search_space if hasattr(c, "search_space") else metadata.get("search_space")
        if isinstance(search_space, dict):
            plane = str(search_space.get("family_id") or search_space.get("plane_id") or "").strip()
            if plane:
                search_planes[plane] = search_planes.get(plane, 0) + 1
        for binding in c.source_bindings:
            if not isinstance(binding, dict):
                continue
            path = str(binding.get("path") or binding.get("file") or binding.get("source_path") or "").strip()
            if path:
                source_surfaces[path] = source_surfaces.get(path, 0) + 1
    return {
        "count": len(candidates),
        "fates": fates,
        "generations": generations,
        "top_niches": sorted(niches.items(), key=lambda item: item[1], reverse=True),
        "top_search_planes": sorted(search_planes.items(), key=lambda item: item[1], reverse=True),
        "top_source_surfaces": sorted(source_surfaces.items(), key=lambda item: item[1], reverse=True),
        "surface_concentration_warning": _surface_concentration_warning(candidates, source_surfaces),
        "nextgen_false_cull_monitor": false_cull_monitor(candidates),
    }


def _surface_concentration_warning(candidates: list[CandidateGenome], source_surfaces: dict[str, int]) -> str:
    if not candidates or not source_surfaces:
        return ""
    top_count = max(source_surfaces.values())
    if top_count * 2 < len(candidates):
        return ""
    return "candidate search is clustering around one local surface; ask the model to widen objective-level search planes before producing more same-surface variants"


def _candidate_prompt_priority(candidate: CandidateGenome) -> float:
    scores = candidate.multihead_scores
    return (
        2.0 * scores.get("answer_likelihood", 0.0)
        + 1.6 * scores.get("objective_alignment", 0.0)
        + 1.2 * scores.get("core_mechanism_strength", 0.0)
        + 0.8 * scores.get("verifiability", 0.0)
        + 0.6 * scores.get("rarity", 0.0)
        + 0.4 * scores.get("novelty", 0.0)
        + 0.1 * int(candidate.generation or 0)
    )


def _archive_candidates(
    store: Any,
    *,
    limit: int,
    detail: str = "summary",
    max_artifact_chars: int | None = None,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    data = _to_mapping(store)
    excluded = set(exclude_ids or set())
    candidates: list[CandidateGenome] = []
    for item in data.values():
        if isinstance(item, CandidateGenome):
            candidate = item
        elif isinstance(item, dict):
            try:
                candidate = candidate_from_dict(item)
            except Exception:
                continue
        else:
            continue
        if candidate.id not in excluded:
            candidates.append(candidate)
    selected = sorted(candidates, key=_candidate_prompt_priority, reverse=True)[:limit]
    return [candidate_prompt_view(candidate, detail=detail, max_artifact_chars=max_artifact_chars) for candidate in selected]


def _failure_records_view(records: Any, *, limit: int) -> list[dict[str, Any]]:
    mapping = _to_mapping(records)
    out: list[dict[str, Any]] = []
    for value in list(mapping.values())[-limit:]:
        data = _to_mapping(value)
        out.append(
            {
                "candidate_id": data.get("candidate_id"),
                "failure_signature": data.get("failure_signature") or data.get("signature") or "",
                "inherited_gene_summary": data.get("inherited_gene_summary") or "",
                "future_reactivation_condition": data.get("future_reactivation_condition") or "",
            }
        )
    return out


def _feedback_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    diagnostics: list[str] = []
    counterexamples: list[str] = []
    for item in items:
        status = str(item.get("status") or item.get("result") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        diag = item.get("diagnostics") or item.get("message") or item.get("raw_output_ref") or item.get("flaws")
        if diag:
            diagnostics.append(_clip(diag, 180))
        cex = item.get("counterexamples") or item.get("counterexample")
        if cex:
            counterexamples.append(_clip(cex, 180))
    return {
        "count": len(items),
        "recent_status_counts": statuses,
        "recent_diagnostics": diagnostics[-4:],
        "recent_counterexamples": counterexamples[-3:],
    }


def _artifact_summary(text: str, *, detail: str, max_artifact_chars: int | None = None) -> dict[str, Any]:
    chars = len(text)
    if not text:
        return {"chars": 0, "preview": ""}
    preview_chars = int(max_artifact_chars) if max_artifact_chars is not None else (900 if detail == "summary" else 300)
    return {
        "chars": chars,
        "preview": _hard_clip(text, preview_chars) if max_artifact_chars is not None else _clip(text, preview_chars),
        "tail": _hard_clip(text[-400:], 400) if chars > preview_chars * 2 and detail != "tiny" else "",
    }


def _project_world_view(mapping: dict[str, Any]) -> dict[str, Any]:
    snapshot = mapping.get("snapshot") if isinstance(mapping.get("snapshot"), dict) else mapping
    world = mapping.get("project_world_model") if isinstance(mapping.get("project_world_model"), dict) else mapping
    manifest = snapshot.get("file_manifest") or snapshot.get("manifest") or []
    if isinstance(manifest, dict):
        manifest_items = list(manifest.keys())
    else:
        manifest_items = [str(item.get("path") if isinstance(item, dict) else item) for item in manifest] if isinstance(manifest, list) else []
    return {
        "kind": "project",
        "snapshot_id": snapshot.get("snapshot_id"),
        "root_hash": snapshot.get("root_hash"),
        "language_profile": dict(snapshot.get("language_profile") or {}),
        "package_managers": list(snapshot.get("package_managers") or []),
        "detected_commands": list(snapshot.get("detected_commands") or []),
        "file_manifest": manifest_items,
        "file_count": len(manifest) if isinstance(manifest, (list, dict)) else snapshot.get("file_count"),
        "file_roles": _to_mapping(world.get("file_roles")),
        "test_map": _to_mapping(world.get("test_map")),
        "config_map": _to_mapping(world.get("config_map")),
        "hotspot_map": _to_mapping(world.get("hotspot_map")),
        "objective_relevance_map": _ranked_score_map(world.get("objective_relevance_map") or {}),
    }



def _ranked_score_map(mapping: Any) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        return {}
    return {str(k): v for k, v in sorted(mapping.items(), key=lambda item: (-float(item[1] or 0.0), str(item[0])))}

def _ranking_view(data: Any) -> dict[str, Any]:
    mapping = _to_mapping(data)
    return dict(mapping)


def _coerce_candidates(values: Any) -> list[CandidateGenome]:
    if not isinstance(values, list):
        return []
    candidates: list[CandidateGenome] = []
    for item in values:
        if isinstance(item, CandidateGenome):
            candidates.append(item)
        elif isinstance(item, dict):
            try:
                candidates.append(candidate_from_dict(item))
            except Exception:
                continue
    return candidates


def _metadata_view(metadata: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "seed_type",
        "search_seed_not_final",
        "exploration_source",
        "mutation_operator",
        "reactivated_in_round",
        "model_seed_error",
        "model_claimed_verification",
        "patch_merge_conflict",
        "search_space",
    }
    view = _small_mapping({k: v for k, v in (metadata or {}).items() if k in keep}, max_items=12, string_chars=220)
    claimed_verification = (metadata or {}).get("model_claimed_verification")
    if isinstance(claimed_verification, dict):
        view["model_claimed_verification"] = dict(claimed_verification)
    return view


def _evaluator_feedback_view(metadata: dict[str, Any], *, detail: str = "summary") -> dict[str, Any]:
    metadata = _to_mapping(metadata)
    evaluator = _to_mapping(metadata.get("evaluator"))
    evidence_state = _to_mapping(metadata.get("evidence_state"))
    records = metadata.get("evidence_records") if isinstance(metadata.get("evidence_records"), list) else []
    latest = _to_mapping(records[-1]) if records else {}
    if not evaluator and not evidence_state and not latest:
        return {}
    out: dict[str, Any] = {}
    for key in ("status", "passed"):
        if key in evaluator:
            out[key] = evaluator.get(key)
    metrics = _to_mapping(evaluator.get("metrics")) or _to_mapping(_to_mapping(latest.get("metadata")).get("metrics"))
    if metrics:
        out["metrics"] = dict(metrics) if detail == "exact" else _small_mapping(metrics, max_items=20, string_chars=260)
    diagnostics = evaluator.get("diagnostics") if isinstance(evaluator.get("diagnostics"), list) else latest.get("diagnostics")
    hints = latest.get("hints")
    if isinstance(diagnostics, list) and diagnostics:
        out["diagnostics"] = list(diagnostics) if detail == "exact" else _clip_list(diagnostics, 8, 320)
    details = evaluator.get("details")
    if isinstance(details, dict) and details:
        out["details"] = dict(details) if detail == "exact" else _small_mapping(details, max_items=20, string_chars=1000)
    if isinstance(hints, list) and hints:
        out["hints"] = list(hints) if detail == "exact" else _clip_list(hints, 6, 260)
    state_keys = (
        "search_score",
        "final_score",
        "repair_value",
        "continuation_value",
        "final_blocked",
        "parent_blocked",
        "terminal_reject",
        "target_challenge_ids",
        "resolved_challenge_ids",
    )
    state = {key: evidence_state.get(key) for key in state_keys if key in evidence_state}
    if state:
        out["evidence_state"] = state if detail == "exact" else _small_mapping(state, max_items=len(state_keys), string_chars=220)
    return out


def _repair_seed_contract_view(candidate: CandidateGenome, *, detail: str = "summary") -> dict[str, Any]:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    seed = metadata.get("repair_seed")
    repair = metadata.get("repair_required")
    guidance = metadata.get("failure_micro_guidance")
    if not any((isinstance(seed, dict), isinstance(repair, dict), isinstance(guidance, list))):
        return {}
    seed_data = seed if isinstance(seed, dict) else {}
    repair_data = repair if isinstance(repair, dict) else {}
    guidance_items = [dict(item) for item in guidance or [] if isinstance(item, dict)]
    disallowed = list(seed_data.get("disallowed_repeat_patterns", []) or [])
    if not disallowed:
        disallowed = [str(item.get("disallowed_repeat_pattern")) for item in guidance_items if item.get("disallowed_repeat_pattern")]
    if detail == "exact":
        return {
            "candidate_id": candidate.id,
            "source": seed_data.get("source") or repair_data.get("source") or "",
            "category": seed_data.get("category") or "",
            "target_files": list(seed_data.get("target_files") or _paths_from_repair(repair_data)),
            "blockers": list(seed_data.get("blockers") or repair_data.get("blockers") or []),
            "required_evidence": list(seed_data.get("required_evidence") or repair_data.get("evidence_needed") or []),
            "disallowed_repeat_patterns": disallowed,
            "next_actions": list(repair_data.get("next_actions") or [item.get("next_action") for item in guidance_items]),
            "contract": "answer-first exploration: legacy verifier/source/proof blockers are advisory only; emit a bold direct answer, mechanism, theorem, algorithm variant, or cross-domain hypothesis",
        }
    return {
        "candidate_id": candidate.id,
        "source": _clip(seed_data.get("source") or repair_data.get("source") or "", 80),
        "category": _clip(seed_data.get("category") or "", 100),
        "target_files": _clip_list(seed_data.get("target_files") or _paths_from_repair(repair_data), 6, 180),
        "blockers": _clip_list(seed_data.get("blockers") or repair_data.get("blockers") or [], 6, 220),
        "required_evidence": _clip_list(seed_data.get("required_evidence") or repair_data.get("evidence_needed") or [], 6, 120),
        "disallowed_repeat_patterns": _clip_list(disallowed, 4, 160),
        "next_actions": _clip_list(repair_data.get("next_actions") or [item.get("next_action") for item in guidance_items], 4, 220),
        "contract": "answer-first exploration: legacy verifier/source/proof blockers are advisory only; emit a bold direct answer, mechanism, theorem, algorithm variant, or cross-domain hypothesis",
    }


def _paths_from_repair(repair: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for binding in repair.get("source_bindings", []) or []:
        if isinstance(binding, dict) and binding.get("path"):
            out.append(str(binding.get("path")))
    return out


def _small_mapping(mapping: Any, *, max_items: int, string_chars: int) -> dict[str, Any]:
    data = _to_mapping(mapping)
    out: dict[str, Any] = {}
    for index, (key, value) in enumerate(data.items()):
        if index >= max_items:
            out["_omitted_items"] = len(data) - max_items
            break
        if isinstance(value, str):
            out[str(key)] = _clip(value, string_chars)
        elif isinstance(value, (int, float, bool)) or value is None:
            out[str(key)] = value
        elif isinstance(value, list):
            out[str(key)] = _clip_list(value, 8, string_chars)
        elif isinstance(value, dict):
            out[str(key)] = _small_mapping(value, max_items=8, string_chars=string_chars)
        else:
            out[str(key)] = _clip(value, string_chars)
    return out


def _search_space_plan_prompt_view(value: Any) -> dict[str, Any]:
    plan = _to_mapping(value)
    if not plan:
        return {}
    view = dict(plan)
    families = plan.get("candidate_families")
    if not isinstance(families, list) or not families:
        families = plan.get("families")
    if isinstance(families, list):
        view["candidate_families"] = [dict(item) for item in families if isinstance(item, dict)]
        view.pop("families", None)
    return view


def _top_scores(scores: dict[str, float]) -> dict[str, float]:
    return {str(key): round(float(value), 4) for key, value in scores.items()}


def _recursive_clip(value: Any, *, string_limit: int, list_limit: int) -> Any:
    if isinstance(value, str):
        return _hard_clip(value, string_limit)
    if isinstance(value, list):
        clipped = [_recursive_clip(item, string_limit=string_limit, list_limit=list_limit) for item in value[:list_limit]]
        if len(value) > list_limit:
            clipped.append({"_omitted_items": len(value) - list_limit})
        return clipped
    if isinstance(value, dict):
        return {str(k): _recursive_clip(v, string_limit=string_limit, list_limit=list_limit) for k, v in list(value.items())[: max(4, list_limit * 2)]}
    return value


def _trim_sequence(mapping: dict[str, Any], key: str, limit: int, *, protected_ids: set[str] | None = None) -> None:
    value = mapping.get(key)
    if isinstance(value, list) and len(value) > limit:
        mapping[key] = _trimmed_with_protected(value, limit, protected_ids or set()) + [{"_omitted_items": max(0, len(value) - limit)}]


def _trimmed_with_protected(value: Any, limit: int, protected_ids: set[str]) -> list[Any]:
    if not isinstance(value, list):
        return []
    if len(value) <= limit:
        return list(value)
    protected: list[Any] = []
    rest: list[Any] = []
    for item in value:
        item_id = str(item.get("id") or "") if isinstance(item, dict) else ""
        (protected if item_id in protected_ids else rest).append(item)
    return [*protected, *rest][: max(0, int(limit or 0))]


def _summarize_value(value: Any) -> dict[str, Any]:
    return {"type": type(value).__name__, "chars": _json_chars(value), "sha256": _sha256_json(value)}


def _clip_list(values: Any, limit: int, chars: int) -> list[str]:
    if not isinstance(values, list):
        return []
    clipped = [_clip(item, chars) for item in values[:limit]]
    if len(values) > limit:
        clipped.append(f"...{len(values) - limit} omitted")
    return clipped


def _clip(value: Any, chars: int) -> str:
    return _hard_clip(value, max(1, int(chars)))


def _hard_clip(value: Any, chars: int) -> str:
    text = _stringify(value)
    if len(text) <= chars:
        return text
    marker = "...[truncated]..."
    keep = max(1, chars - len(marker))
    head = max(1, int(keep * 0.72))
    tail = max(1, keep - head)
    return text[:head] + marker + text[-tail:]


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        try:
            data = value.to_dict()
            return dict(data) if isinstance(data, dict) else {"repr": repr(value)}
        except Exception:
            return {"repr": repr(value)}
    return {"repr": repr(value)}


def _json_chars(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def is_long_context_request(request_type: str) -> bool:
    return request_type in {
        "nexus_classify_task",
        "nexus_build_objective_contract",
        "nexus_build_text_world_model",
        "nexus_build_project_objective_contract",
        "nexus_build_evolution_policy",
        "nexus_request_context",
        "nexus_seed_population",
        "nexus_critique_candidates",
        "nexus_plan_mutations",
        "nexus_generate_offspring",
        "nexus_synthesize_result",
        "nexus_diagnose_search_state",
        "nexus_relative_rank",
        "nexus_pool_preprocess",
        "nexus_should_stop",
    }


__all__ = [
    "NEXUS_PROMPT_MAX_CHARS_ENV",
    "NEXUS_LONG_CONTEXT_MAX_CHARS_ENV",
    "PromptView",
    "archive_prompt_view",
    "build_prompt_view",
    "candidate_prompt_view",
    "contract_prompt_view",
    "history_prompt_view",
    "is_long_context_request",
    "policy_prompt_view",
    "prompt_char_budget",
    "prompt_char_budget_details",
    "world_prompt_view",
]
