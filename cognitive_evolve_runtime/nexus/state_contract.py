"""Canonical runtime-state contract for the Nexus-only architecture."""
from __future__ import annotations

from typing import Any

RUNTIME_VERSION = "2.0"
INTERACTION_MODE = "one_shot"
EXTERNAL_QUESTIONS_ALLOWED = False
FINAL_ANSWER_MAY_REQUEST_CLARIFICATION = False
RUNTIME_PATH = "nexus"


def normalize_runtime_state(state: dict[str, Any]) -> dict[str, Any]:
    """Fill required Nexus runtime fields without reviving legacy schemas."""

    normalized = dict(state)
    normalized["version"] = RUNTIME_VERSION
    normalized.setdefault("interaction_mode", INTERACTION_MODE)
    normalized.setdefault("external_questions_allowed", EXTERNAL_QUESTIONS_ALLOWED)
    normalized.setdefault("final_answer_may_request_clarification", FINAL_ANSWER_MAY_REQUEST_CLARIFICATION)
    normalized.setdefault("evidence_plan", {})
    normalized.setdefault("evidence_artifacts", {})
    normalized.setdefault("uncertainty_policy", {})
    normalized.setdefault("uncertainty_fuse", {})
    normalized.setdefault("verification_results", {})
    normalized.setdefault("runtime_path", RUNTIME_PATH)
    normalized.setdefault(
        "architecture_contract",
        {
            "external_interface": "one_shot",
            "runtime": "nexus_structured_genome_evolution",
            "task_semantics": "nexus_semantic_profile_objective_contract_and_policy",
            "tool_evidence_scaffold": True,
        },
    )
    return normalized


__all__ = [
    "RUNTIME_VERSION",
    "INTERACTION_MODE",
    "EXTERNAL_QUESTIONS_ALLOWED",
    "FINAL_ANSWER_MAY_REQUEST_CLARIFICATION",
    "RUNTIME_PATH",
    "normalize_runtime_state",
]
