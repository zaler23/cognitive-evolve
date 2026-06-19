"""Nexus runtime-state projections.

This module contains only current Nexus schemas.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.validation.result import aggregate_verification_results, verification_result_from_mapping

from cognitive_evolve_runtime.nexus.state_contract import (
    EXTERNAL_QUESTIONS_ALLOWED,
    FINAL_ANSWER_MAY_REQUEST_CLARIFICATION,
    INTERACTION_MODE,
    RUNTIME_PATH,
    RUNTIME_VERSION,
    normalize_runtime_state,
)


def nexus_search_state(run_data: dict[str, Any]) -> dict[str, Any]:
    """Return a compact structured search view from a Nexus run."""

    evolution = dict(run_data.get("evolution") or {})
    population = dict(evolution.get("population") or {})
    candidates = [dict(item) for item in population.get("candidates", []) if isinstance(item, dict)]
    archives = dict(evolution.get("archives") or {})
    diagnosis = dict(evolution.get("diagnosis") or {})
    synthesis = dict(evolution.get("synthesis") or {})
    selected_id = str(synthesis.get("best_candidate_id") or "") or None
    reference_id = str(synthesis.get("reference_candidate_id") or "") or None
    completion_status = str(evolution.get("completion_status") or synthesis.get("completion_status") or synthesis.get("status") or "completed")
    return {
        "status": completion_status,
        "runtime_architecture": "nexus",
        "candidate_genomes": candidates,
        "answer_archive": list(dict(archives.get("answer_archive") or {}).values()),
        "selected_candidate": {"id": selected_id or "", "selection_method": "nexus_relative_multihead_archive"},
        "reference_candidate": {
            "id": reference_id or "",
            "note": str(synthesis.get("reference_note") or ""),
            "semantics": "displayed as candidate material only; correctness requires human or external verifier judgment",
        },
        "selection": {
            "selected_id": selected_id,
            "reference_id": reference_id,
            "selection_method": "nexus_relative_multihead_archive",
            "frontier_ids": list(dict(archives.get("answer_archive") or {}).keys()),
        },
        "archive_summary": synthesis.get("archives_summary") if isinstance(synthesis.get("archives_summary"), dict) else {},
        "search_diagnosis": diagnosis,
        "round_artifacts": evolution.get("budget_history") or [],
        "quality_diversity_archive": dict(archives.get("quality_diversity") or {}),
        "multihead_elo": dict(evolution.get("elo") or {}),
        "progress_events": evolution.get("progress_events") or [],
        "checkpoint_policy": "nexus_checkpoint_store",
    }


def nexus_evolution_summary(run_data: dict[str, Any]) -> dict[str, Any]:
    evolution = dict(run_data.get("evolution") or {})
    progress_events = [dict(item) for item in evolution.get("progress_events", []) if isinstance(item, dict)]
    final_progress = progress_events[-1] if progress_events else {}
    population = dict(evolution.get("population") or {})
    candidates = population.get("candidates") if isinstance(population.get("candidates"), list) else []
    return {
        "enabled": True,
        "runtime_architecture": "nexus",
        "budget_policy": "nexus_model_driven_evolution_policy",
        "candidate_count": len(candidates),
        "initial_rounds": 1 if candidates else 0,
        "max_rounds": int(final_progress.get("max_rounds", len(evolution.get("budget_history") or []) or 1) or 1),
        "actual_rounds": int(final_progress.get("round", len(evolution.get("budget_history") or []) or 0) or 0),
        "completion_status": evolution.get("completion_status") or dict(evolution.get("synthesis") or {}).get("completion_status"),
        "stop_reason": evolution.get("stop_reason") or ("interrupted" if evolution.get("interrupted") else "nexus_budget_or_return_policy_completed"),
        "verifier_depth": "nexus_local_verification_trace",
        "tool_verification": "structured_tool_feedback",
        "progress_event": final_progress,
    }


def nexus_verification_results(run_data: dict[str, Any]) -> dict[str, Any]:
    """Derive the public verification gate from persisted Nexus evidence."""

    evolution = dict(run_data.get("evolution") or {})
    summaries = [dict(item) for item in run_data.get("verification_summaries", []) if isinstance(item, dict)]
    project_failed = [item for item in summaries if item.get("passed") is False]
    round_records = [dict(item) for item in evolution.get("budget_history", []) if isinstance(item, dict)]
    candidate_failures = 0
    candidate_warnings = 0
    for record in round_records:
        for item in record.get("verification", []) or []:
            if not isinstance(item, dict):
                continue
            if item.get("passed") is False:
                candidate_failures += 1
            elif item.get("status") == "warning":
                candidate_warnings += 1
        for item in record.get("offspring_verification", []) or []:
            if isinstance(item, dict) and item.get("passed") is False:
                project_failed.append(item)
    interrupted = bool(evolution.get("interrupted"))
    synthesis = dict(evolution.get("synthesis") or {})
    synthesis_status = str(synthesis.get("status") or "")
    completion_status = str(evolution.get("completion_status") or synthesis.get("completion_status") or "").lower()
    incomplete = completion_status in {"needs_continuation", "interrupted_checkpointed", "paused_quota", "route_incomplete"}
    closure = synthesis.get("closure_certificate") if isinstance(synthesis.get("closure_certificate"), dict) else {}
    graded = synthesis.get("graded_output") if isinstance(synthesis.get("graded_output"), dict) else {}
    if not graded and isinstance(closure.get("graded_output"), dict):
        graded = closure.get("graded_output") or {}
    objective_solved = bool(synthesis.get("objective_solved") or closure.get("objective_solved")) and dict(graded or {}).get("mode") == "verified_result"
    # ``passed`` is the artifact/runtime integrity gate used by the local
    # validation suite. Objective closure is reported separately as
    # ``objective_solved`` and by API payloads so a valid checkpoint/failure
    # report does not masquerade as a solved answer.
    passed = not interrupted and not project_failed and "interrupted" not in synthesis_status.lower()
    canonical_inputs = summaries + project_failed + [
        {"passed": passed, "status": "pass" if passed else "fail", "source": "nexus_integrity_gate", "confidence": 1.0}
    ]
    canonical = aggregate_verification_results(
        [verification_result_from_mapping(item, source="nexus_verification_results") for item in canonical_inputs],
        source="nexus_verification_results",
    ).to_dict()
    return {
        "passed": passed,
        "objective_solved": objective_solved,
        "completion_status": completion_status or synthesis_status,
        "runtime_architecture": "nexus",
        "source": "nexus_verification_results",
        "canonical_result": canonical,
        "verification_summaries": summaries,
        "project_failed_count": len(project_failed),
        "candidate_failure_count": candidate_failures,
        "candidate_warning_count": candidate_warnings,
        "interrupted": interrupted,
        "synthesis_status": synthesis_status,
    }


def nexus_runtime_state(
    *,
    task_dir: Path,
    prompt: str,
    run_data: dict[str, Any],
    selected_capabilities: list[str] | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    """Build the canonical runtime-state.json payload for Nexus runs."""

    artifacts = dict(run_data.get("artifacts") or {})
    nodes = [
        {
            "id": "nexus_runtime",
            "capability": "cognitive_search",
            "status": status,
            "outputs": ["nexus-runtime/run-result.json", "nexus-runtime/final-answer.md"],
            "runtime_role": "canonical_entry",
        },
        {
            "id": "candidate_evolution",
            "capability": "cognitive_search",
            "status": status,
            "outputs": ["nexus-runtime/population.json", "nexus-runtime/archives.json"],
            "runtime_role": "structured_genome_loop",
        },
        {
            "id": "verification_feedback",
            "capability": "independent_review",
            "status": "completed",
            "outputs": ["nexus-runtime/events.jsonl", "nexus-runtime/checkpoint.json"],
            "runtime_role": "structured_tool_feedback",
        },
        {
            "id": "synthesis",
            "capability": "user_cognition",
            "status": status,
            "outputs": ["nexus-runtime/final-answer.md"],
            "runtime_role": "nexus_synthesis",
        },
    ]
    state = {
        "version": RUNTIME_VERSION,
        "task": task_dir.name,
        "status": status,
        "prompt": prompt,
        "runtime_path": RUNTIME_PATH,
        "runtime_architecture": "nexus",
        "interaction_mode": INTERACTION_MODE,
        "external_questions_allowed": EXTERNAL_QUESTIONS_ALLOWED,
        "final_answer_may_request_clarification": FINAL_ANSWER_MAY_REQUEST_CLARIFICATION,
        "active_capabilities": list(selected_capabilities or []),
        "nodes": nodes,
        "nexus_runtime": run_data,
        "single_runtime": {
            "enforced": True,
            "source_of_truth": "NexusRuntime",
        },
        "nexus_evolution": nexus_evolution_summary(run_data),
        "nexus_search": nexus_search_state(run_data),
        "evidence_plan": {
            "runtime_architecture": "nexus",
            "allowed_evidence_sources": dict(run_data.get("contract") or {}).get("allowed_evidence_sources") or [],
            "tool_preferences": dict(run_data.get("policy") or {}).get("tool_preferences") or [],
        },
        "evidence_artifacts": artifacts,
        "objective_contract": dict(run_data.get("contract") or {}),
        "final_answer": str(dict(dict(run_data.get("evolution") or {}).get("synthesis") or {}).get("final_answer") or ""),
        "final_answer_artifact": "nexus-runtime/final-answer.md",
        "verification_results": nexus_verification_results(run_data),
        "uncertainty_policy": dict(run_data.get("contract") or {}).get("uncertainty_policy") or {},
        "uncertainty_fuse": {"tripped": False, "runtime_architecture": "nexus"},
    }
    return normalize_runtime_state(state)


__all__ = ["nexus_evolution_summary", "nexus_runtime_state", "nexus_search_state", "nexus_verification_results"]
