from __future__ import annotations

import time
from typing import Any

from cognitive_evolve_runtime.validation.result import aggregate_verification_results, verification_result_from_mapping

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


def _nexus_verification_passed(nexus_data: dict[str, Any]) -> bool:
    if not isinstance(nexus_data, dict):
        return False
    completion = _nexus_completion_status(nexus_data)
    if completion in {"completed", "best_current_route", "needs_continuation", "interrupted_checkpointed", "paused_quota", "route_incomplete", "failed_verification", "failed", "unknown"}:
        return False
    if not _nexus_objective_solved(nexus_data):
        return False
    closure = _nexus_closure_certificate(nexus_data)
    if closure and closure.get("critical_failures"):
        return False
    graded = _nexus_graded_output(nexus_data)
    if graded.get("mode") != "verified_result":
        return False
    try:
        if int(graded.get("verification_strength") or 0) < 4:
            return False
    except (TypeError, ValueError):
        return False
    result = graded.get("result") if isinstance(graded.get("result"), dict) else {}
    replay = graded.get("replay_certificate") if isinstance(graded.get("replay_certificate"), dict) else {}
    if not result or not bool(result.get("replayable")) or not replay:
        return False
    summaries = nexus_data.get("verification_summaries")
    if isinstance(summaries, list) and summaries:
        return all(bool((item or {}).get("passed", (item or {}).get("verification_result", {}).get("passed", True))) for item in summaries if isinstance(item, dict))
    verification = nexus_data.get("verification_results")
    if isinstance(verification, dict) and "passed" in verification:
        return bool(verification.get("passed"))
    return True


def _nexus_closure_certificate(nexus_data: dict[str, Any]) -> dict[str, Any]:
    evolution = nexus_data.get("evolution") if isinstance(nexus_data.get("evolution"), dict) else {}
    synthesis = evolution.get("synthesis") if isinstance(evolution.get("synthesis"), dict) else {}
    closure = synthesis.get("closure_certificate") if isinstance(synthesis.get("closure_certificate"), dict) else {}
    return closure


def _nexus_graded_output(nexus_data: dict[str, Any]) -> dict[str, Any]:
    evolution = nexus_data.get("evolution") if isinstance(nexus_data.get("evolution"), dict) else {}
    synthesis = evolution.get("synthesis") if isinstance(evolution.get("synthesis"), dict) else {}
    closure = _nexus_closure_certificate(nexus_data)
    graded = synthesis.get("graded_output") if isinstance(synthesis.get("graded_output"), dict) else {}
    if not graded and isinstance(closure.get("graded_output"), dict):
        graded = closure.get("graded_output") or {}
    return dict(graded or {})


def _nexus_objective_solved(nexus_data: dict[str, Any]) -> bool:
    if not isinstance(nexus_data, dict):
        return False
    evolution = nexus_data.get("evolution") if isinstance(nexus_data.get("evolution"), dict) else {}
    synthesis = evolution.get("synthesis") if isinstance(evolution.get("synthesis"), dict) else {}
    closure = _nexus_closure_certificate(nexus_data)
    closure_solved = bool(synthesis.get("objective_solved") or closure.get("objective_solved"))
    if not closure_solved:
        return False
    return _nexus_graded_output(nexus_data).get("mode") == "verified_result"


def _nexus_completion_status(nexus_data: dict[str, Any]) -> str:
    if not isinstance(nexus_data, dict):
        return "unknown"
    evolution = nexus_data.get("evolution") if isinstance(nexus_data.get("evolution"), dict) else {}
    synthesis = evolution.get("synthesis") if isinstance(evolution.get("synthesis"), dict) else {}
    return str(evolution.get("completion_status") or synthesis.get("completion_status") or synthesis.get("status") or "completed")


def _nexus_verification_summary(nexus_data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(nexus_data, dict):
        return verification_result_from_mapping({"status": "unknown"}, source="api_payload").to_dict()
    raw_items: list[dict[str, Any]] = []
    summaries = nexus_data.get("verification_summaries")
    if isinstance(summaries, list):
        raw_items.extend(item for item in summaries if isinstance(item, dict))
    verification = nexus_data.get("verification_results")
    if isinstance(verification, dict):
        raw_items.append(verification)
    if not raw_items:
        status = "pass" if _nexus_verification_passed(nexus_data) else "inconclusive"
        raw_items.append({"status": status, "source": "nexus_completion_status"})
    return aggregate_verification_results([verification_result_from_mapping(item, source="api_payload") for item in raw_items], source="api_payload").to_dict()


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
                "finish_reason": "length" if _nexus_completion_status(nexus_data) in {"needs_continuation", "paused_quota"} else "stop",
            }
        ],
        "usage": _usage(prompt, answer, model=model),
        "system_fingerprint": "cogev-v2-nexus",
        "cognitive_evolve": {
            "runtime_path": "nexus",
            "mode": nexus_data.get("mode") if isinstance(nexus_data, dict) else None,
            "actual_rounds": _nexus_actual_rounds(nexus_data),
            "verification_passed": _nexus_verification_passed(nexus_data),
            "objective_solved": _nexus_objective_solved(nexus_data),
            "verification_summary": _nexus_verification_summary(nexus_data),
            "completion_status": _nexus_completion_status(nexus_data),
            "reference_candidate_id": str(synthesis.get("reference_candidate_id") or ""),
            "reference_semantics": "candidate output only; correctness requires human or external verifier judgment",
            "completion_semantics": "adaptive multi-round candidate evolution; safety checkpoints are needs_continuation, not solved",
            "streaming_semantics": "progress events plus final answer chunks; not provider token streaming",
        },
    }


__all__ = ["_completion_payload", "_nexus_actual_rounds", "_nexus_completion_status", "_nexus_objective_solved", "_nexus_verification_passed", "_nexus_verification_summary"]
