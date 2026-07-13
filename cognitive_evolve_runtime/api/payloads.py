from __future__ import annotations

import time
from typing import Any

from cognitive_evolve_runtime.nexus.state import nexus_verification_results

from .usage import _usage


def _nexus_actual_rounds(nexus_data: dict[str, Any]) -> int | None:
    events = ((nexus_data.get("evolution") or {}).get("progress_events") or []) if isinstance(nexus_data, dict) else []
    if events and isinstance(events[-1], dict):
        try:
            return int(events[-1].get("round"))
        except (TypeError, ValueError):
            return None
    summary = nexus_data.get("nexus_evolution") if isinstance(nexus_data, dict) else {}
    if isinstance(summary, dict):
        try:
            return int(summary.get("actual_rounds"))
        except (TypeError, ValueError):
            return None
    return None


def _nexus_verification_passed(nexus_data: dict[str, Any]) -> bool | None:
    """Compatibility helper returning the preliminary check tri-state."""

    return _nexus_validation_state(nexus_data).get("preliminary_validation_passed")


def _nexus_closure_certificate(nexus_data: dict[str, Any]) -> dict[str, Any]:
    evolution = nexus_data.get("evolution") if isinstance(nexus_data.get("evolution"), dict) else {}
    synthesis = evolution.get("synthesis") if isinstance(evolution.get("synthesis"), dict) else {}
    closure = synthesis.get("closure_certificate") if isinstance(synthesis.get("closure_certificate"), dict) else {}
    return closure


def _nexus_objective_solved(nexus_data: dict[str, Any]) -> bool:
    return False


def _nexus_answer_produced(nexus_data: dict[str, Any]) -> bool:
    if not isinstance(nexus_data, dict):
        return False
    evolution = nexus_data.get("evolution") if isinstance(nexus_data.get("evolution"), dict) else {}
    synthesis = evolution.get("synthesis") if isinstance(evolution.get("synthesis"), dict) else {}
    closure = _nexus_closure_certificate(nexus_data)
    return bool(synthesis.get("answer_produced") or closure.get("answer_produced") or str(synthesis.get("final_answer") or "").strip())


def _nexus_completion_status(nexus_data: dict[str, Any]) -> str:
    if not isinstance(nexus_data, dict):
        return "unknown"
    evolution = nexus_data.get("evolution") if isinstance(nexus_data.get("evolution"), dict) else {}
    synthesis = evolution.get("synthesis") if isinstance(evolution.get("synthesis"), dict) else {}
    raw = str(evolution.get("completion_status") or synthesis.get("completion_status") or synthesis.get("status") or "completed")
    return "completed" if raw.strip().lower() == "solved" else raw


def _nexus_validation_state(nexus_data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(nexus_data, dict):
        return {
            "preliminary_validation_passed": None,
            "validation_status": "not_run",
            "ready_for_external_review": False,
        }
    return nexus_verification_results(nexus_data)


def _nexus_validation_status(nexus_data: dict[str, Any]) -> str:
    return str(_nexus_validation_state(nexus_data).get("validation_status") or "not_run")


def _nexus_ready_for_external_review(nexus_data: dict[str, Any]) -> bool:
    return bool(_nexus_validation_state(nexus_data).get("ready_for_external_review"))


def _nexus_verification_summary(nexus_data: dict[str, Any]) -> dict[str, Any]:
    canonical = _nexus_validation_state(nexus_data).get("canonical_result")
    return dict(canonical) if isinstance(canonical, dict) else {
        "verdict": "inconclusive",
        "passed": None,
        "validation_status": "not_run",
        "source": "api_payload",
        "confidence": 0.0,
        "reason": "no preliminary validation result",
    }


def _completion_payload(*, request_id: str, model: str, prompt: str, answer: str, nexus_data: dict[str, Any]) -> dict[str, Any]:
    synthesis = ((nexus_data.get("evolution") or {}).get("synthesis") or {}) if isinstance(nexus_data, dict) else {}
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage(prompt, answer, model=model),
        "system_fingerprint": "cogev-v2-nexus",
        "cognitive_evolve": {
            "runtime_path": "nexus",
            "mode": nexus_data.get("mode") if isinstance(nexus_data, dict) else None,
            "actual_rounds": _nexus_actual_rounds(nexus_data),
            "preliminary_validation_passed": _nexus_verification_passed(nexus_data),
            "validation_status": _nexus_validation_status(nexus_data),
            "validation_semantics": "preliminary_only_external_review_required",
            "objective_solved": _nexus_objective_solved(nexus_data),
            "objective_solved_semantics": "producer_never_solves_external_review_required",
            "ready_for_external_review": _nexus_ready_for_external_review(nexus_data),
            "answer_produced": _nexus_answer_produced(nexus_data),
            "preliminary_validation_summary": _nexus_verification_summary(nexus_data),
            "completion_status": _nexus_completion_status(nexus_data),
            "answer_candidate_id": str(synthesis.get("best_candidate_id") or synthesis.get("candidate_id") or ""),
            "answer_semantics": "best-current candidate material; external review owns correctness after the run",
            "completion_semantics": "adaptive multi-round candidate evolution; completed means best-current material was produced, not externally certified",
            "streaming_semantics": "progress events plus final answer chunks; not provider token streaming",
        },
    }


__all__ = ["_completion_payload", "_nexus_actual_rounds", "_nexus_answer_produced", "_nexus_completion_status", "_nexus_objective_solved", "_nexus_ready_for_external_review", "_nexus_validation_status", "_nexus_verification_passed", "_nexus_verification_summary"]
