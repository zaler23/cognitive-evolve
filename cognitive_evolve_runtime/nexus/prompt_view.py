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
from cognitive_evolve_runtime.llm.env import LLM_MAX_PROMPT_CHARS_ENV
from cognitive_evolve_runtime.nexus.activation import ACTIVATION_REQUESTS, activation_prompt_contract
from cognitive_evolve_runtime.nexus.factor_resurrection import failure_factor_hints, resurrect_factor_trace
from cognitive_evolve_runtime.nexus.search_space import build_search_space_map
from cognitive_evolve_runtime.nexus.prompt_profiles import apply_prompt_profile
from cognitive_evolve_runtime.nexus.nextgen import false_cull_monitor
from cognitive_evolve_runtime.nexus.strategy_comparison import strategy_comparison_context

NEXUS_PROMPT_MAX_CHARS_ENV = "COGEV_NEXUS_PROMPT_MAX_CHARS"
NEXUS_LONG_CONTEXT_MAX_CHARS_ENV = "COGEV_NEXUS_LONG_CONTEXT_MAX_CHARS"

DEFAULT_MAX_PROMPT_CHARS = 120_000
DEFAULT_LONG_CONTEXT_MAX_CHARS = 256_000

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

    ``COGEV_NEXUS_PROMPT_MAX_CHARS`` is the primary Nexus budget.  If unset, we
    honor the existing LLM max prompt env.  Long-context calls may opt into a
    higher ceiling via ``COGEV_NEXUS_LONG_CONTEXT_MAX_CHARS``; this is still far
    below the accidental million-character payloads that prompted this guard.
    """

    if long_context:
        explicit_long = _positive_int(os.environ.get(NEXUS_LONG_CONTEXT_MAX_CHARS_ENV))
        if explicit_long:
            return explicit_long
    explicit = _positive_int(os.environ.get(NEXUS_PROMPT_MAX_CHARS_ENV))
    if explicit:
        return explicit
    llm_limit = _positive_int(os.environ.get(LLM_MAX_PROMPT_CHARS_ENV))
    if llm_limit:
        return llm_limit
    return DEFAULT_LONG_CONTEXT_MAX_CHARS if long_context else DEFAULT_MAX_PROMPT_CHARS


@dataclass(frozen=True)
class PromptView:
    """A bounded payload plus accounting metadata."""

    payload: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"payload": self.payload, "metadata": self.metadata}


def build_prompt_view(request_type: str, payload: dict[str, Any], *, max_chars: int | None = None) -> PromptView:
    """Build a compact, bounded prompt view for a Nexus model request."""

    limit = max(1, int(max_chars or prompt_char_budget(long_context=_is_long_context_request(request_type))))
    controls = _prompt_context_controls(payload)
    protected_paths = _protected_paths_from_controls(controls)
    raw_chars = _json_chars(payload)
    compressed = _apply_prompt_context_controls(_compress_payload(request_type, payload), controls)
    profiled, profile_metadata = apply_prompt_profile(request_type, compressed)
    protected_candidate_ids = _protected_candidate_ids_from_controls(controls)
    if protected_candidate_ids and "candidates" not in profiled and isinstance(compressed.get("candidates"), list):
        profiled["candidates"] = _trimmed_with_protected(compressed.get("candidates"), max(1, len(protected_candidate_ids)), protected_candidate_ids)
        profiled["_protected_candidate_ids"] = sorted(protected_candidate_ids)
    compressed_chars = _json_chars(profiled)
    bounded = _fit_payload(profiled, max_chars=limit, protected_paths=protected_paths)
    sent_chars = _json_chars(bounded)
    metadata = {
        "type": "nexus_prompt_view",
        "request_type": request_type,
        "raw_payload_chars": raw_chars,
        "compressed_payload_chars": compressed_chars,
        "sent_payload_chars": sent_chars,
        "max_prompt_chars": limit,
        "compressed": True,
        "truncated": sent_chars > limit,
        "raw_payload_sha256": _sha256_json(payload),
        "sent_payload_sha256": _sha256_json(bounded),
        "policy": "candidate_archive_summary_then_recursive_fit",
        "omitted_heavy_fields": sorted(HEAVY_CANDIDATE_FIELDS),
        "protected_paths_applied": protected_paths,
        "protected_over_budget": bool(protected_paths and sent_chars > limit),
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
        "concise_claim": _clip(genome.concise_claim, 320),
        "core_mechanism": _clip(genome.core_mechanism, 500),
        "assumptions": _clip_list(genome.assumptions, 5, 220),
        "missing_parts": _clip_list(genome.missing_parts, 5, 220),
        "uncertainty_notes": _clip_list(genome.uncertainty_notes, 3, 220),
        "edge_knowledge_seeds": _clip_list(genome.edge_knowledge_seeds, 5, 220),
        "novelty_descriptors": _clip_list(genome.novelty_descriptors, 5, 160),
        "niche_memberships": _clip_list(genome.niche_memberships, 5, 160),
        "failure_lessons": _clip_list(genome.failure_lessons, 5 if detail != "tiny" else 2, 260),
        "inherited_genes": _clip_list(genome.inherited_genes, 5 if detail != "tiny" else 2, 260),
        "mutation_history_tail": _clip_list(genome.mutation_history[-4:], 4, 180),
        "scores": _top_scores(genome.multihead_scores),
        "tool_feedback_summary": _feedback_summary(genome.tool_results),
        "verification_summary": _feedback_summary(genome.verification_trace),
        "formal_artifacts": [_small_mapping(item, max_items=8, string_chars=220) for item in genome.formal_artifacts[:4]],
        "proof_obligations": [_small_mapping(item, max_items=8, string_chars=220) for item in genome.proof_obligations[:6]],
        "obligation_delta": _small_mapping(genome.obligation_delta, max_items=8, string_chars=220),
        "evidence_refs": [_small_mapping(item, max_items=8, string_chars=220) for item in genome.evidence_refs[:6]],
        "source_bindings": [_small_mapping(item, max_items=8, string_chars=220) for item in genome.source_bindings[:6]],
        "evidence_delta": _small_mapping(genome.evidence_delta, max_items=8, string_chars=220),
        "verification_result": _small_mapping(genome.verification_result, max_items=10, string_chars=260),
        "artifact_summary": _artifact_summary(artifact_text, detail=detail, max_artifact_chars=max_artifact_chars),
        "artifact_sha256": _sha256_text(artifact_text) if artifact_text else "",
        "metadata": _metadata_view(genome.metadata),
    }
    nextgen = _to_mapping(_to_mapping(genome.metadata).get("nextgen"))
    if nextgen:
        view["nextgen"] = _small_mapping(nextgen, max_items=8, string_chars=180)
    repair_seed_contract = _repair_seed_contract_view(genome)
    if repair_seed_contract:
        view["repair_seed_contract"] = repair_seed_contract
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
    return view


def archive_prompt_view(archives: Any, *, population: list[CandidateGenome] | None = None) -> dict[str, Any]:
    """Return archive counts plus a few high-value exemplars, not full archives."""

    if archives is None:
        return {}
    summary = archives.summary() if hasattr(archives, "summary") else _small_mapping(_to_mapping(archives), max_items=20, string_chars=120)
    view: dict[str, Any] = {"summary": summary}
    if population:
        view["active_ids"] = [c.id for c in population if c.current_fate == "Active"][:16]
        view["elite_ids"] = [c.id for c in population if c.current_fate == "Elite"][:8]
    view["answer_elites"] = _archive_candidates(getattr(archives, "answer_archive", {}), limit=5)
    view["rarity_elites"] = _archive_candidates(getattr(getattr(archives, "rarity_archive", None), "candidates", {}), limit=5)
    view["dormant_hints"] = _archive_candidates(getattr(getattr(archives, "dormant_archive", None), "candidates", {}), limit=5, detail="tiny")
    view["auxiliary_hints"] = _archive_candidates(getattr(getattr(archives, "auxiliary_archive", None), "candidates", {}), limit=4, detail="tiny")
    failure_archive = getattr(archives, "failure_archive", None)
    records = getattr(failure_archive, "records", {}) if failure_archive is not None else {}
    view["failure_lessons"] = _failure_records_view(records, limit=8)
    hints = failure_factor_hints(archives, population=population, limit=8)
    if hints:
        view["failure_factor_hints"] = hints
    return view


def history_prompt_view(history: list[dict[str, Any]] | None, *, rounds: int = 2) -> list[dict[str, Any]]:
    """Return only recent round-level signals from budget history."""

    items = [item for item in history or [] if isinstance(item, dict)]
    recent = items[-max(1, rounds):]
    out: list[dict[str, Any]] = []
    for item in recent:
        out.append(
            {
                "round": item.get("round"),
                "diagnosis": _small_mapping(item.get("diagnosis") or {}, max_items=8, string_chars=220),
                "ranking": _ranking_view(item.get("ranking") or {}),
                "progress_event": _small_mapping(item.get("progress_event") or {}, max_items=12, string_chars=120),
                "error": _small_mapping(item.get("error") or {}, max_items=8, string_chars=220),
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
        "summary": _clip(mapping.get("summary") or mapping.get("raw_text") or mapping.get("repr") or "", 4000 if detail != "tiny" else 1200),
        "task_type_hypotheses": _clip_list(mapping.get("task_type_hypotheses") or [], 8, 180),
        "constraints": _clip_list(mapping.get("constraints") or [], 12, 220),
        "available_evidence": _clip_list(mapping.get("available_evidence") or [], 12, 260),
        "uncertainty_zones": _clip_list(mapping.get("uncertainty_zones") or [], 8, 220),
        "edge_seed_pool": _clip_list(mapping.get("edge_seed_pool") or mapping.get("possible_edge_knowledge_seeds") or [], 12, 220),
    }


def contract_prompt_view(contract: Any) -> dict[str, Any]:
    data = _to_mapping(contract)
    if not data:
        return {}
    keys = [
        "original_user_goal",
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
    view = _small_mapping({key: data.get(key) for key in keys if key in data}, max_items=18, string_chars=800)
    dac = data.get("dynamic_artifact_contract")
    outcome_policy = data.get("outcome_policy")
    if (not isinstance(dac, dict) or not dac) and isinstance(outcome_policy, dict):
        dac = outcome_policy.get("dynamic_artifact_contract")
    if isinstance(dac, dict):
        view["dynamic_artifact_contract"] = _small_mapping(dac, max_items=10, string_chars=500)
    search_space_plan = data.get("search_space_plan")
    if (not isinstance(search_space_plan, dict) or not search_space_plan) and isinstance(outcome_policy, dict):
        search_space_plan = outcome_policy.get("search_space_plan") or outcome_policy.get("search_space")
    if (not isinstance(search_space_plan, dict) or not search_space_plan) and isinstance(dac, dict):
        search_space_plan = dac.get("search_space_plan") or dac.get("search_space")
    if isinstance(search_space_plan, dict):
        view["search_space_plan"] = _small_mapping(search_space_plan, max_items=10, string_chars=500)
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
    ]
    view = _small_mapping({key: data.get(key) for key in keys if key in data}, max_items=18, string_chars=500)
    metadata = _to_mapping(data.get("metadata"))
    search_space_plan = metadata.get("search_space_plan") or metadata.get("search_space_contract")
    if isinstance(search_space_plan, dict):
        view["search_space_plan"] = _small_mapping(search_space_plan, max_items=10, string_chars=500)
    if metadata.get("search_space_plan_required"):
        view["search_space_plan_required"] = _clip(metadata.get("search_space_plan_required"), 500)
    if isinstance(metadata.get("strategy_comparison"), dict):
        view["strategy_comparison"] = _small_mapping(metadata.get("strategy_comparison"), max_items=8, string_chars=260)
    for key in ("seed_coverage", "target_perturb_seed_judgment", "factor_resurrection_summary", "algorithm_efficiency", "model_parallel_efficiency", "minimal_core_ablation", "seed_active_frontier", "seed_reservoir_ref"):
        if key in metadata:
            view[key] = _small_mapping(metadata.get(key), max_items=10, string_chars=260)
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
        },
    }
    contract_view = contract_prompt_view(payload.get("contract"))
    policy_view = policy_prompt_view(payload.get("policy"))
    world_view = world_prompt_view(payload.get("world") or payload.get("snapshot"), detail="summary")
    compressed["search_space_contract"] = _search_space_contract_from_views(
        contract_view=contract_view,
        policy_view=policy_view,
        world_view=world_view,
        request_type=request_type,
        candidate_target_count=len(candidates) or len(parents) or 0,
    )
    if request_type in {"nexus_seed_population", "nexus_generate_offspring", "nexus_plan_mutations"}:
        compressed["artifact_generation_contract"] = _artifact_generation_contract_from_view(
            contract_view,
            request_type=request_type,
            search_space_contract=compressed["search_space_contract"],
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
        compressed["user_goal"] = _clip(payload.get("user_goal"), 6000)
    if "round_index" in payload:
        compressed["round_index"] = payload.get("round_index")
    if "actions" in payload:
        compressed["actions"] = _clip_list(payload.get("actions") or [], 20, 160)
    if "source_context" in payload:
        compressed["source_context"] = _source_context_view(payload.get("source_context"))
    if "mutation_instruction" in payload:
        compressed["mutation_instruction"] = _clip(payload.get("mutation_instruction"), 1000)
    if "plans" in payload:
        compressed["plans"] = [_small_mapping(_to_mapping(plan), max_items=10, string_chars=260) for plan in payload.get("plans") or []]
    compressed["contract"] = contract_view
    compressed["policy"] = policy_view
    compressed["world"] = world_view
    protected_candidate_ids = _protected_candidate_ids_from_controls(_prompt_context_controls(payload))
    if candidates:
        compressed["candidates"] = [
            candidate_prompt_view(c, detail="summary")
            for c in _select_candidates_for_prompt(candidates, limit=48, protected_ids=protected_candidate_ids)
        ]
        if protected_candidate_ids:
            compressed["_protected_candidate_ids"] = sorted(protected_candidate_ids)
        compressed["candidate_population_stats"] = _population_stats(candidates)
    if parents:
        compressed["parents"] = [candidate_prompt_view(c, detail="summary") for c in parents[:16]]
    factors = resurrect_factor_trace(population or candidates, limit=10)
    if factors:
        compressed["resurrected_factor_trace"] = factors
    strategy = strategy_comparison_context(payload.get("policy"), population or candidates)
    if strategy:
        compressed["strategy_comparison"] = strategy
    compressed["archives"] = archive_prompt_view(payload.get("archives"), population=population or candidates)
    compressed["history"] = history_prompt_view(payload.get("history") if isinstance(payload.get("history"), list) else [])
    diagnosis = payload.get("diagnosis")
    if diagnosis is not None:
        compressed["diagnosis"] = _small_mapping(_to_mapping(diagnosis), max_items=14, string_chars=400)
    # Keep unknown scalar/small fields, but never carry huge nested blobs by default.
    for key, value in payload.items():
        if key in {"user_goal", "contract", "world", "snapshot", "policy", "candidates", "population", "parents", "archives", "history", "diagnosis", "plans", "actions", "source_context", "mutation_instruction", "round_index"}:
            continue
        if _json_chars(value) <= 4000:
            compressed[key] = value
        else:
            compressed[key] = _summarize_value(value)
    return compressed


def _source_context_view(value: Any) -> dict[str, Any]:
    data = _to_mapping(value)
    out: dict[str, Any] = {}
    selected = data.get("selected_files")
    if isinstance(selected, list):
        out["selected_files"] = [str(item) for item in selected[:12] if str(item or "").strip()]
    elif selected:
        out["selected_files"] = _clip_list([selected], 1, 260)
    budget = data.get("budget_policy")
    if isinstance(budget, dict):
        out["budget_policy"] = _small_mapping(budget, max_items=8, string_chars=260)
    elif budget is not None:
        out["budget_policy"] = _clip(budget, 260)
    slices: list[dict[str, Any]] = []
    raw_slices = data.get("slices") if isinstance(data.get("slices"), list) else []
    for item in raw_slices[:8]:
        if not isinstance(item, dict):
            continue
        view: dict[str, Any] = {}
        for key in ("path", "hash", "start", "end"):
            if key in item and item.get(key) is not None:
                view[key] = item.get(key)
        if "text" in item:
            view["text"] = _stringify(item.get("text"))
        if view:
            slices.append(view)
    if slices:
        out["slices"] = slices
    return out


def _artifact_generation_contract(contract: Any, *, request_type: str) -> dict[str, Any]:
    contract_view = contract_prompt_view(contract)
    return _artifact_generation_contract_from_view(
        contract_view,
        request_type=request_type,
        search_space_contract=_search_space_contract_from_views(contract_view=contract_view, policy_view={}, world_view={}, request_type=request_type),
    )


def _artifact_generation_contract_from_view(view: dict[str, Any], *, request_type: str, search_space_contract: dict[str, Any] | None = None) -> dict[str, Any]:
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
    return {
        "request_type": request_type,
        "model_defined_required_work_product": required_work_product or "use the dynamic_artifact_contract",
        "model_defined_minimum_delta": minimum_delta or "produce a concrete artifact delta relative to parent state",
        "non_negotiable_runtime_invariant": "Candidates must contain the actual object-level artifact or an executable repair step toward it; pure commentary, plans, labels, or promises are not progress.",
        "search_space_breadth_contract": search_space_contract or {},
        "search_space_rule": "Before deepening one surface, allocate candidates across materially different model-defined planes from the user objective; local files/tools/evidence are grounding surfaces, not the objective itself.",
        "surface_bias_guard": "Do not let an easy-to-patch or easy-to-verify local surface monopolize seed, mutation, or offspring generation when the user objective asks for higher-level mechanism, lifecycle, policy, materialization, or final-answer design.",
        "required_search_space_metadata": "Each candidate should include metadata.search_space.family_id or search_space.plane_id chosen from the model-authored search space; if none exists, author one from the objective first.",
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
    }


def _search_space_contract_from_views(
    *,
    contract_view: dict[str, Any],
    policy_view: dict[str, Any],
    world_view: dict[str, Any],
    request_type: str,
    candidate_target_count: int = 0,
) -> dict[str, Any]:
    objective = str(contract_view.get("normalized_goal") or contract_view.get("original_user_goal") or "")
    plan = contract_view.get("search_space_plan") if isinstance(contract_view.get("search_space_plan"), dict) else {}
    if not plan and isinstance(policy_view.get("search_space_plan"), dict):
        plan = policy_view["search_space_plan"]
    assessment = {
        "task_type": world_view.get("kind") or "",
        "real_objective": objective,
        "search_space_plan": plan,
    }
    search_map = build_search_space_map(assessment, requested_candidate_count=max(0, int(candidate_target_count or 0)))
    return {
        "request_type": request_type,
        "source": search_map.get("source"),
        "model_driven": True,
        "needs_model_authored_search_space": bool(search_map.get("needs_model_authored_search_space")),
        "candidate_families": search_map.get("candidate_families", [])[:12],
        "coverage_gate": search_map.get("coverage_gate", {}),
        "surface_bias_guard": search_map.get("surface_bias_guard", {}),
        "anti_narrowing_instruction": (
            "If recent candidates cluster around one implementation/detail surface, generate the next candidates from different objective-level planes rather than another same-surface patch variant."
        ),
    }


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
        "candidates": [candidate_prompt_view(c, detail="tiny", max_artifact_chars=240) for c in selected],
        "archives": archive_prompt_view(payload.get("archives"), population=candidates),
        "resurrected_factor_trace": resurrect_factor_trace(candidates, limit=8),
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
        out["verification_plan"] = _small_mapping(_to_mapping(controls.get("verification_plan")), max_items=10, string_chars=300)
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
        for key in ("exogeneity_probe", "variety_probe", "falsification_budget", "replay_record"):
            if isinstance(cleaned.get(key), dict):
                cleaned[key] = _small_mapping(cleaned[key], max_items=8, string_chars=280)
        if cleaned:
            out.append(cleaned)
    return _shrink_verification_regime(out, max_chars=6000)


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
    for item in items:
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


def _fit_payload(payload: dict[str, Any], *, max_chars: int, protected_paths: list[str] | None = None) -> dict[str, Any]:
    """Recursively shrink a compressed payload until its JSON fits the budget."""

    fitted = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    if isinstance(fitted.get("verification_regime"), list):
        fitted["verification_regime"] = _shrink_verification_regime(fitted["verification_regime"], max_chars=max(1000, max_chars // 4))
    protected_snapshot = _snapshot_paths(fitted, protected_paths or [])
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
        _restore_paths(fitted, protected_snapshot)
        if _json_chars(fitted) <= max_chars:
            return fitted
    # Then halve large strings/lists until fit.  This preserves schema shape.
    for string_limit in (600, 360, 220, 120, 60):
        fitted = _recursive_clip(fitted, string_limit=string_limit, list_limit=8)
        if _json_chars(fitted) <= max_chars:
            return fitted
    minimal = {
        "request_type": fitted.get("request_type"),
        "prompt_contract": fitted.get("prompt_contract"),
        "contract": fitted.get("contract"),
        "policy": fitted.get("policy"),
        "candidate_population_stats": fitted.get("candidate_population_stats"),
        "candidates": _trimmed_with_protected(fitted.get("candidates", []), 3, set(fitted.get("_protected_candidate_ids", []) if isinstance(fitted.get("_protected_candidate_ids"), list) else [])),
        "archives": fitted.get("archives", {}).get("summary", {}) if isinstance(fitted.get("archives"), dict) else {},
        "_fit_warning": "payload was reduced to minimal Nexus prompt view",
    }
    if fitted.get("history"):
        minimal["history"] = fitted.get("history", [])[-1:]
    _restore_paths(minimal, protected_snapshot)
    clipped = _recursive_clip(minimal, string_limit=120, list_limit=4)
    _restore_paths(clipped, protected_snapshot)
    return clipped


def _select_candidates_for_prompt(candidates: list[CandidateGenome], *, limit: int, protected_ids: set[str] | None = None) -> list[CandidateGenome]:
    protected_ids = set(protected_ids or set())
    if len(candidates) <= limit:
        return candidates
    buckets: list[CandidateGenome] = []
    seen: set[str] = set()

    def add(items: Iterable[CandidateGenome], n: int) -> None:
        for candidate in items:
            if len([c for c in buckets if c.id not in seen]) >= limit:
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
        for niche in c.niche_memberships[:3] or ([c.core_mechanism] if c.core_mechanism else []):
            niches[niche] = niches.get(niche, 0) + 1
        metadata = c.metadata if isinstance(c.metadata, dict) else {}
        search_space = c.search_space if hasattr(c, "search_space") else metadata.get("search_space")
        if isinstance(search_space, dict):
            plane = str(search_space.get("family_id") or search_space.get("plane_id") or "").strip()
            if plane:
                search_planes[plane] = search_planes.get(plane, 0) + 1
        for binding in c.source_bindings[:8]:
            if not isinstance(binding, dict):
                continue
            path = str(binding.get("path") or binding.get("file") or binding.get("source_path") or "").strip()
            if path:
                source_surfaces[path] = source_surfaces.get(path, 0) + 1
    return {
        "count": len(candidates),
        "fates": fates,
        "generations": generations,
        "top_niches": sorted(niches.items(), key=lambda item: item[1], reverse=True)[:12],
        "top_search_planes": sorted(search_planes.items(), key=lambda item: item[1], reverse=True)[:12],
        "top_source_surfaces": sorted(source_surfaces.items(), key=lambda item: item[1], reverse=True)[:12],
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


def _archive_candidates(store: Any, *, limit: int, detail: str = "summary") -> list[dict[str, Any]]:
    data = _to_mapping(store)
    candidates: list[CandidateGenome] = []
    for item in list(data.values())[: max(limit * 4, limit)]:
        if isinstance(item, CandidateGenome):
            candidates.append(item)
        elif isinstance(item, dict):
            try:
                candidates.append(candidate_from_dict(item))
            except Exception:
                continue
    selected = sorted(candidates, key=_candidate_prompt_priority, reverse=True)[:limit]
    return [candidate_prompt_view(candidate, detail=detail) for candidate in selected]


def _failure_records_view(records: Any, *, limit: int) -> list[dict[str, Any]]:
    mapping = _to_mapping(records)
    out: list[dict[str, Any]] = []
    for value in list(mapping.values())[-limit:]:
        data = _to_mapping(value)
        out.append(
            {
                "candidate_id": data.get("candidate_id"),
                "failure_signature": _clip(data.get("failure_signature") or data.get("signature") or "", 260),
                "inherited_gene_summary": _clip(data.get("inherited_gene_summary") or "", 260),
                "future_reactivation_condition": _clip(data.get("future_reactivation_condition") or "", 220),
            }
        )
    return out


def _feedback_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    diagnostics: list[str] = []
    counterexamples: list[str] = []
    for item in items[-8:]:
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
    preview_chars = int(max_artifact_chars) if max_artifact_chars is not None else (1200 if detail == "summary" else 300)
    return {
        "chars": chars,
        "preview": _clip(text, preview_chars),
        "tail": _clip(text[-400:], 400) if chars > preview_chars * 2 and detail != "tiny" else "",
    }


def _project_world_view(mapping: dict[str, Any]) -> dict[str, Any]:
    snapshot = mapping.get("snapshot") if isinstance(mapping.get("snapshot"), dict) else mapping
    world = mapping.get("project_world_model") if isinstance(mapping.get("project_world_model"), dict) else mapping
    manifest = snapshot.get("file_manifest") or snapshot.get("manifest") or []
    if isinstance(manifest, dict):
        manifest_items = list(manifest.keys())[:80]
    else:
        manifest_items = [str(item.get("path") if isinstance(item, dict) else item) for item in manifest[:80]] if isinstance(manifest, list) else []
    return {
        "kind": "project",
        "snapshot_id": snapshot.get("snapshot_id"),
        "root_hash": snapshot.get("root_hash"),
        "language_profile": _small_mapping(snapshot.get("language_profile") or {}, max_items=12, string_chars=120),
        "package_managers": _clip_list(snapshot.get("package_managers") or [], 8, 120),
        "detected_commands": _clip_list(snapshot.get("detected_commands") or [], 12, 180),
        "file_manifest_sample": manifest_items[:80],
        "file_count": len(manifest) if isinstance(manifest, (list, dict)) else snapshot.get("file_count"),
        "file_roles": _small_mapping(world.get("file_roles") or {}, max_items=40, string_chars=160),
        "test_map": _small_mapping(world.get("test_map") or {}, max_items=30, string_chars=160),
        "config_map": _small_mapping(world.get("config_map") or {}, max_items=30, string_chars=160),
        "hotspot_map": _small_mapping(world.get("hotspot_map") or {}, max_items=30, string_chars=160),
        "objective_relevance_map": _small_mapping(world.get("objective_relevance_map") or {}, max_items=30, string_chars=180),
    }


def _ranking_view(data: Any) -> dict[str, Any]:
    mapping = _to_mapping(data)
    return {
        "best_final_answer_id": mapping.get("best_final_answer_id"),
        "strongest_mechanism_id": mapping.get("strongest_mechanism_id"),
        "mutation_worthy_ids": _clip_list(mapping.get("mutation_worthy_ids") or [], 8, 80),
        "edge_value_ids": _clip_list(mapping.get("edge_value_ids") or [], 8, 80),
        "auxiliary_ids": _clip_list(mapping.get("auxiliary_ids") or [], 8, 80),
        "dormant_ids": _clip_list(mapping.get("dormant_ids") or [], 8, 80),
        "raw_notes": _clip(mapping.get("raw_notes") or "", 240),
    }


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
        "patch_merge_conflict",
        "search_space",
    }
    return _small_mapping({k: v for k, v in (metadata or {}).items() if k in keep}, max_items=12, string_chars=220)


def _repair_seed_contract_view(candidate: CandidateGenome) -> dict[str, Any]:
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


def _top_scores(scores: dict[str, float]) -> dict[str, float]:
    keys = [
        "objective_alignment",
        "answer_likelihood",
        "core_mechanism_strength",
        "novelty",
        "rarity",
        "verifiability",
        "tool_progress",
        "robustness",
        "auxiliary_value",
        "deferral_risk",
    ]
    out = {key: round(float(scores.get(key, 0.0)), 4) for key in keys if key in scores}
    extras = sorted(((k, v) for k, v in scores.items() if k not in out), key=lambda item: abs(float(item[1])), reverse=True)[:4]
    for key, value in extras:
        out[str(key)] = round(float(value), 4)
    return out


def _recursive_clip(value: Any, *, string_limit: int, list_limit: int) -> Any:
    if isinstance(value, str):
        return _clip(value, string_limit)
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


def _is_long_context_request(request_type: str) -> bool:
    return request_type in {
        "nexus_seed_population",
        "nexus_plan_mutations",
        "nexus_generate_offspring",
        "nexus_synthesize_result",
        "nexus_diagnose_search_state",
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
    "policy_prompt_view",
    "prompt_char_budget",
    "world_prompt_view",
]
