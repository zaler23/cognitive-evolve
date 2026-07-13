"""Nexus runtime-state projections.

This module contains only current Nexus schemas.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.candidates.genome import _producer_safe_candidate_payload
from cognitive_evolve_runtime.validation.result import aggregate_verification_results, verification_result_from_mapping
from cognitive_evolve_runtime.verification.grading import certificate_allows_verified_result
from cognitive_evolve_runtime.verification.types import graded_output_from_dict

from cognitive_evolve_runtime.nexus.state_contract import (
    EXTERNAL_QUESTIONS_ALLOWED,
    FINAL_ANSWER_MAY_REQUEST_CLARIFICATION,
    INTERACTION_MODE,
    RUNTIME_PATH,
    RUNTIME_VERSION,
    normalize_runtime_state,
)
from cognitive_evolve_runtime.nexus.stop_reasons import (
    CANDIDATE_READY_FOR_EXTERNAL_REVIEW,
    is_solved_stop_reason,
    normalize_external_review_stop_reason,
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
    raw_completion_status = str(evolution.get("completion_status") or synthesis.get("completion_status") or synthesis.get("status") or "completed")
    completion_status = _canonical_completion_status(raw_completion_status)
    return {
        "status": completion_status,
        "raw_completion_status": raw_completion_status if raw_completion_status != completion_status else "",
        "completion_status_note": str(evolution.get("completion_status_note") or _completion_status_note(raw_completion_status)),
        "runtime_architecture": "nexus",
        "candidate_genomes": candidates,
        "answer_archive": list(dict(archives.get("answer_archive") or {}).values()),
        "selected_candidate": {"id": selected_id or "", "selection_method": "nexus_relative_multihead_archive"},
        "selection": {
            "selected_id": selected_id,
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
        "completion_status": _canonical_completion_status(evolution.get("completion_status") or dict(evolution.get("synthesis") or {}).get("completion_status")),
        "raw_completion_status": str(evolution.get("completion_status") or dict(evolution.get("synthesis") or {}).get("completion_status") or ""),
        "completion_status_note": str(
            evolution.get("completion_status_note")
            or dict(evolution.get("synthesis") or {}).get("completion_status_note")
            or _completion_status_note(evolution.get("completion_status") or dict(evolution.get("synthesis") or {}).get("completion_status"))
        ),
        "stop_reason": evolution.get("stop_reason") or ("interrupted" if evolution.get("interrupted") else "nexus_budget_or_return_policy_completed"),
        "verifier_depth": "nexus_preliminary_validation_trace",
        "tool_verification": "structured_preliminary_tool_feedback",
        "ready_for_external_review": _ready_for_external_review(evolution),
        "validation_semantics": "preliminary_only_external_review_required",
        "progress_event": final_progress,
    }


def nexus_verification_results(run_data: dict[str, Any]) -> dict[str, Any]:
    """Derive preliminary check state without claiming objective closure."""

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
    completion_status = _canonical_completion_status(evolution.get("completion_status") or synthesis.get("completion_status") or "").lower()
    closure = synthesis.get("closure_certificate") if isinstance(synthesis.get("closure_certificate"), dict) else {}
    critical_failures = [str(item) for item in closure.get("critical_failures", []) if item]
    graded = synthesis.get("graded_output") if isinstance(synthesis.get("graded_output"), dict) else {}
    if not graded and isinstance(closure.get("graded_output"), dict):
        graded = closure.get("graded_output") or {}
    answer_produced = bool(synthesis.get("answer_produced") or closure.get("answer_produced") or str(synthesis.get("final_answer") or "").strip())
    runtime_integrity_passed = bool(
        not interrupted
        and not project_failed
        and not critical_failures
        and "interrupted" not in synthesis_status.lower()
        and synthesis_status.lower() not in {"failed", "failed_verification"}
        and completion_status not in {"failed", "failed_verification"}
    )
    try:
        canonical_graded = graded_output_from_dict(graded).to_dict() if graded else {}
    except AssertionError:
        canonical_graded = {}
    replay = canonical_graded.get("replay_certificate") if isinstance(canonical_graded.get("replay_certificate"), dict) else {}
    preliminary_result_passed = bool(
        canonical_graded.get("mode") == "preliminary_result"
        and canonical_graded.get("result")
        and certificate_allows_verified_result(replay)
    )
    canonical_inputs = list(summaries)
    if preliminary_result_passed:
        canonical_inputs.append({"passed": True, "status": "pass", "source": "preliminary_result", "confidence": 1.0})
    if not runtime_integrity_passed:
        canonical_inputs.append(
            {
                "passed": False,
                "status": "fail",
                "source": "runtime_integrity",
                "confidence": 1.0,
                "reason": "; ".join(critical_failures) or synthesis_status or "interrupted_or_project_check_failed",
            }
        )
    canonical = aggregate_verification_results(
        [verification_result_from_mapping(item, source="nexus_verification_results") for item in canonical_inputs],
        source="nexus_verification_results",
    ).to_dict()
    verdict = str(canonical.get("verdict") or "inconclusive")
    if verdict == "inconclusive":
        canonical["passed"] = None
    validation_status = (
        "not_run"
        if not canonical_inputs
        else "preliminary_passed"
        if verdict == "pass"
        else "preliminary_failed"
        if verdict == "fail"
        else "inconclusive"
    )
    canonical["validation_status"] = validation_status
    preliminary_validation_passed = True if validation_status == "preliminary_passed" else (False if validation_status == "preliminary_failed" else None)
    return {
        "preliminary_validation_passed": preliminary_validation_passed,
        "validation_status": validation_status,
        "validation_semantics": "preliminary_only_external_review_required",
        "runtime_integrity_passed": runtime_integrity_passed,
        "objective_solved": False,
        "objective_solved_semantics": "producer_never_solves_external_review_required",
        "ready_for_external_review": _ready_for_external_review(evolution),
        "answer_produced": answer_produced,
        "completion_status": completion_status or synthesis_status,
        "runtime_architecture": "nexus",
        "source": "nexus_verification_results",
        "canonical_result": canonical,
        "verification_summaries": summaries,
        "project_failed_count": len(project_failed),
        "candidate_failure_count": candidate_failures,
        "candidate_warning_count": candidate_warnings,
        "critical_failure_count": len(critical_failures),
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

    run_data = _producer_safe_run_data(run_data)
    artifacts = dict(run_data.get("artifacts") or {})
    verification_results = nexus_verification_results(run_data)
    validation_status = str(verification_results.get("validation_status") or "not_run")
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
            "capability": "preliminary_validation",
            "status": validation_status,
            "outputs": ["nexus-runtime/events.jsonl", "nexus-runtime/checkpoint.json"],
            "runtime_role": "producer_preliminary_tool_feedback",
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
        "verification_results": verification_results,
        "uncertainty_policy": dict(run_data.get("contract") or {}).get("uncertainty_policy") or {},
        "uncertainty_fuse": {"tripped": False, "runtime_architecture": "nexus"},
    }
    return normalize_runtime_state(state)


__all__ = ["nexus_evolution_summary", "nexus_runtime_state", "nexus_search_state", "nexus_verification_results"]


def _canonical_completion_status(status: Any) -> str:
    text = str(status or "completed").strip()
    if text.lower() == "solved":
        return "completed"
    return text or "completed"


def _completion_status_note(status: Any) -> str:
    return "legacy_solved_status_downgraded_to_completed" if str(status or "").strip().lower() == "solved" else ""


def _ready_for_external_review(evolution: dict[str, Any]) -> bool:
    synthesis = dict(evolution.get("synthesis") or {})
    closure = dict(synthesis.get("closure_certificate") or {})
    has_candidate_material = bool(
        synthesis.get("best_candidate_id")
        or synthesis.get("answer_produced")
        or closure.get("answer_produced")
        or str(synthesis.get("final_answer") or "").strip()
    )
    if not has_candidate_material:
        return False
    reason = evolution.get("stop_reason") or closure.get("stop_reason")
    normalized = normalize_external_review_stop_reason(reason)
    if normalized == CANDIDATE_READY_FOR_EXTERNAL_REVIEW:
        return True
    # Old solved checkpoints are read as review-ready, never as producer-owned
    # objective closure.
    return bool(is_solved_stop_reason(reason) or synthesis.get("objective_solved") or closure.get("objective_solved"))


def _producer_safe_run_data(run_data: dict[str, Any]) -> dict[str, Any]:
    """Downgrade legacy closure claims at the public runtime-state boundary."""

    original_evolution = dict(dict(run_data or {}).get("evolution") or {})
    original_synthesis = dict(original_evolution.get("synthesis") or {})
    safe = dict(_producer_safe_candidate_payload(run_data or {}) or {})
    evolution = dict(safe.get("evolution") or {})
    synthesis = dict(evolution.get("synthesis") or {})
    closure = dict(synthesis.get("closure_certificate") or {})
    for owner, original in ((evolution, original_evolution), (synthesis, original_synthesis)):
        raw_status = original.get("completion_status")
        if str(raw_status or "").strip().lower() == "solved":
            owner["completion_status"] = "completed"
            owner["completion_status_note"] = "legacy_solved_status_downgraded_to_completed"
    if str(synthesis.get("status") or "").strip().lower() == "solved":
        synthesis["status"] = "completed"
    for owner in (evolution, closure):
        if is_solved_stop_reason(owner.get("stop_reason")):
            owner["stop_reason"] = CANDIDATE_READY_FOR_EXTERNAL_REVIEW
    synthesis["objective_solved"] = False
    closure["objective_solved"] = False
    for owner, key in ((synthesis, "graded_output"), (closure, "graded_output"), (evolution, "graded_output")):
        raw = owner.get(key)
        if isinstance(raw, dict):
            try:
                owner[key] = graded_output_from_dict(raw).to_dict()
            except AssertionError:
                owner.pop(key, None)
    final_certificate = closure.get("final_certificate")
    if isinstance(final_certificate, dict):
        closure["final_certificate"] = {**final_certificate, "objective_solved": False}
    best_current = synthesis.get("best_current_direction")
    if isinstance(best_current, dict):
        best_current = dict(best_current)
        best_current["route"] = "best_current"
        if best_current.pop("verification_status", None) == "verified":
            best_current["validation_status"] = "preliminary_passed"
        best_current.pop("blocked_from_verified_claim_reason", None)
        synthesis["best_current_direction"] = best_current
    adaptive_state = evolution.get("adaptive_state")
    if isinstance(adaptive_state, dict):
        adaptive_state = dict(adaptive_state)
        adaptive_certificate = adaptive_state.get("final_certificate")
        if isinstance(adaptive_certificate, dict):
            adaptive_state["final_certificate"] = {**adaptive_certificate, "objective_solved": False}
        adaptive_state["events"] = [
            {**event, "objective_solved": False}
            if isinstance(event, dict) and event.get("type") == "final_certificate"
            else event
            for event in adaptive_state.get("events", [])
        ]
        evolution["adaptive_state"] = adaptive_state
    synthesis["closure_certificate"] = closure
    evolution["synthesis"] = synthesis
    safe["evolution"] = evolution
    return safe
