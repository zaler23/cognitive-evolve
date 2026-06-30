"""Nexus evolution loop skeleton with deterministic fake-model support."""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.crossover import crossover
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation, candidate_from_dict
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationOperator, MutationPlan, MutationPlanner
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.events.progress import EvolutionProgressEvent, PipelineProgressEvent
from cognitive_evolve_runtime.nexus.critique import CandidateCritique, CritiqueEngine
from cognitive_evolve_runtime.nexus.activation_reseed import emergency_activation_reseed
from cognitive_evolve_runtime.nexus._serde import utc_now
from cognitive_evolve_runtime.nexus.exploration import action_palette_for_round, amplify_population
from cognitive_evolve_runtime.nexus.diagnosis import PolicyUpdater, SearchDiagnosis, SearchStateDiagnoser
from cognitive_evolve_runtime.nexus.generation_plan import GenerationPlan, apply_generation_plan, assert_stage_ready, build_generation_plan, expected_generation_plan_id
from cognitive_evolve_runtime.nexus.model_errors import is_quota_error
from cognitive_evolve_runtime.nexus.obligations import candidate_obligation_delta, candidate_source_bindings
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.population_vitality import vitality_snapshot
from cognitive_evolve_runtime.nexus.population_control import compact_live_population
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike, NexusMutationPlannerModelProtocol, NexusOffspringModelProtocol, NexusSeedModelProtocol, NexusStopModelProtocol
from cognitive_evolve_runtime.nexus.repair_reactivation import recover_failure_archive_repair_seeds, recover_repairable_dormant_seeds
from cognitive_evolve_runtime.nexus.synthesis import SynthesizedResult, synthesize_result
from cognitive_evolve_runtime.nexus.stop_decision import StopDecisionEngine
from cognitive_evolve_runtime.nexus.semantic_dedupe import CandidateDeduper
from cognitive_evolve_runtime.outcomes.latent_audit import audit_latent_replay_bundle
from cognitive_evolve_runtime.outcomes.runtime_bridge import (
    annotate_candidates_with_latent_signals,
    apply_latent_exploration_to_mutation_plans,
    best_candidate_m5_certificate,
    improvement_certificate_from_any,
    ingest_latent_feedback,
    ingest_runtime_trial_feedback,
    latent_completion_override,
    latent_exploration_plan_for_contract,
    m5_certificate_summary,
    requires_verified_improvement,
)
from cognitive_evolve_runtime.nexus.reproduction import (
    dedupe_offspring_against_population,
    elite_gap_merge_offspring,
    parents_for_crossover,
    ranked_repair_fallback_parents,
    sync_repair_parent_attempts_to_dormant_archive,
    verify_offspring,
)
from cognitive_evolve_runtime.theory import TheoryConfig, TheoryLayer, build_population_representation
from cognitive_evolve_runtime.nexus._shared import MODEL_BOUNDARY_ERRORS, positive_int
from cognitive_evolve_runtime.llm.retry import provider_error_category
from cognitive_evolve_runtime.ranking.multihead_elo import MultiHeadElo
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult, RelativeRater

def _repair_seed_for_parent(parent: CandidateGenome | None) -> dict[str, Any]:
    if parent is None or not isinstance(parent.metadata, dict):
        return {}
    seed = parent.metadata.get("repair_seed")
    return dict(seed) if isinstance(seed, dict) else {}


def _source_integration_points_for_parent(parent: CandidateGenome | None) -> list[dict[str, Any]]:
    if parent is None:
        return []
    points: list[dict[str, Any]] = []
    for binding in candidate_source_bindings(parent):
        path = binding.get("path")
        if path:
            points.append({"path": str(path), "kind": str(binding.get("kind") or "source_binding"), "evidence_need": "post-pass verification"})
    repair = parent.metadata.get("repair_required") if isinstance(parent.metadata, dict) else None
    if isinstance(repair, dict):
        for binding in repair.get("source_bindings", []) or []:
            if not isinstance(binding, dict):
                continue
            path = binding.get("path")
            if path:
                points.append({"path": str(path), "kind": str(binding.get("kind") or "repair_source_binding"), "evidence_need": "repair-target post-pass verification"})
    repair_seed = _repair_seed_for_parent(parent)
    for path in repair_seed.get("target_files", [])[:8] if repair_seed else []:
        if path:
            points.append({"path": str(path), "kind": "repair_seed_target", "evidence_need": "pre-fail/post-pass verification"})
    for attr, kind in (("affected_tests", "test"), ("touched_symbols", "symbol")):
        for value in getattr(parent, attr, []) or []:
            points.append({"ref": str(value), "kind": kind, "evidence_need": "pre-fail/post-pass expectation"})
    for ref in parent.evidence_refs[:8]:
        if isinstance(ref, dict):
            points.append({"ref": str(ref.get("id") or ref.get("path") or ref.get("kind") or ""), "kind": str(ref.get("kind") or "evidence_ref"), "evidence_need": "preserve-or-improve"})
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for point in points:
        key = (str(point.get("path") or point.get("ref") or ""), str(point.get("kind") or ""))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        deduped.append(point)
    return deduped[:12]


def _failure_micro_guidance_for_parent(parent: CandidateGenome | None) -> list[dict[str, Any]]:
    if parent is None:
        return []
    raw = parent.metadata.get("failure_micro_guidance") if isinstance(parent.metadata, dict) else None
    if raw is None and isinstance(parent.verification_result, dict):
        raw = parent.verification_result.get("failure_guidance")
    directives: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        blocker = _clip(item.get("blocker"), 80)
        next_action = _clip(item.get("next_action"), 180)
        if not blocker or not next_action:
            continue
        directives.append(
            {
                "blocker": blocker,
                "next_action": next_action,
                "evidence_needed": [_clip(value, 60) for value in item.get("evidence_needed", [])[:5] if value],
                "source_bindings": [dict(binding) for binding in item.get("source_bindings", [])[:5] if isinstance(binding, dict)],
                "disallowed_repeat_pattern": _clip(item.get("disallowed_repeat_pattern"), 120),
                "severity": _clip(item.get("severity"), 20) or "warning",
            }
        )
        if len(directives) >= 3:
            break
    return directives


def _repair_requirement_for_parent(parent: CandidateGenome | None) -> dict[str, Any]:
    if parent is None or not isinstance(parent.metadata, dict):
        return {}
    repair = parent.metadata.get("repair_required")
    if isinstance(repair, dict) and repair.get("blockers"):
        return dict(repair)
    decision = parent.metadata.get("stage_eligibility")
    if isinstance(decision, dict) and decision.get("repair_required") and decision.get("repair_blockers"):
        return {
            "blockers": [str(item) for item in decision.get("repair_blockers", []) if item],
            "evidence_needed": [],
            "source_bindings": [],
            "next_actions": [],
            "stage": str(decision.get("stage") or ""),
        }
    guidance = _failure_micro_guidance_for_parent(parent)
    if guidance:
        blockers = [str(item.get("blocker") or "") for item in guidance if item.get("blocker")]
        evidence_needed: list[str] = []
        source_bindings: list[dict[str, Any]] = []
        next_actions: list[str] = []
        for item in guidance:
            evidence_needed.extend(str(value) for value in item.get("evidence_needed", []) if value)
            source_bindings.extend(dict(value) for value in item.get("source_bindings", []) if isinstance(value, dict))
            action = str(item.get("next_action") or "").strip()
            if action:
                next_actions.append(action)
        return {
            "blockers": list(dict.fromkeys(blockers)),
            "evidence_needed": list(dict.fromkeys(evidence_needed)),
            "source_bindings": source_bindings[:5],
            "next_actions": list(dict.fromkeys(next_actions)),
            "stage": "repair_guidance",
        }
    return {}


def _repair_operator_for_requirement(repair: dict[str, Any]) -> str:
    blockers = {str(item) for item in repair.get("blockers", []) if item}
    evidence_needed = {str(item) for item in repair.get("evidence_needed", []) if item}
    tokens = blockers | evidence_needed
    if tokens.intersection({"proof_object_absent", "proof_object_structurally_weak", "formal_artifact", "structural_check"}):
        return MutationOperator.INSTANTIATE_FORMAL_ARTIFACT
    if tokens.intersection({"ledger_non_progressing", "obligation_delta_absent", "blocking_obligation_not_targeted", "obligation_delta", "targeted_obligation_id"}):
        return MutationOperator.DISCHARGE_OBLIGATION
    if tokens.intersection({"evidence_ref_absent", "evidence_ref_unverified", "verified_evidence_ref", "source_binding_absent", "source_binding"}):
        return MutationOperator.TOOL_GROUND
    return MutationOperator.REPAIR


def _clip(value: Any, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


__all__ = ["_failure_micro_guidance_for_parent", "_repair_operator_for_requirement", "_repair_requirement_for_parent", "_repair_seed_for_parent", "_source_integration_points_for_parent"]
