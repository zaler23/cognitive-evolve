#!/usr/bin/env python3
"""Canonical task-type registry for CognitiveEvolve runtime routing.

The runtime previously repeated the same task-type vocabulary in the semantic
controller, prompt schema hints, budget profiles, and tests.  This module is the
single source of truth used by the contract compiler and bounded local gates.
"""
from __future__ import annotations

from typing import Any

ALLOWED_TASK_TYPES: tuple[str, ...] = (
    "governed_safe_plan",
    "architecture_refactor_or_migration",
    "research_or_evidence_dependent_plan",
    "technical_execution_or_codebase_task",
    "structured_decision_or_design",
    "direct_answer_or_small_edit",
    "proof_resolution",
    "open_conjecture",
    "mechanism_discovery",
    "novel_algorithm_design",
)

DEFAULT_TASK_TYPE = "structured_decision_or_design"

FRONTIER_TASK_TYPES: frozenset[str] = frozenset(
    {
        "proof_resolution",
        "open_conjecture",
        "mechanism_discovery",
        "novel_algorithm_design",
    }
)

RESEARCH_TASK_TYPES: frozenset[str] = frozenset(
    {
        "research_or_evidence_dependent_plan",
        "proof_resolution",
        "open_conjecture",
        "mechanism_discovery",
        "novel_algorithm_design",
    }
)

CODE_TASK_TYPES: frozenset[str] = frozenset(
    {
        "architecture_refactor_or_migration",
        "technical_execution_or_codebase_task",
    }
)

EVIDENCE_REQUIRED_TASK_TYPES: frozenset[str] = frozenset(
    {
        "governed_safe_plan",
        "architecture_refactor_or_migration",
        "research_or_evidence_dependent_plan",
        "technical_execution_or_codebase_task",
        "proof_resolution",
        "open_conjecture",
        "mechanism_discovery",
        "novel_algorithm_design",
    }
)


def normalize_task_type(value: Any, *, default: str = DEFAULT_TASK_TYPE) -> str:
    """Return a stable task label without treating the legacy list as authority.

    CognitiveEvolve is model-driven: task labels are descriptive hints, not a
    finite admission taxonomy.  The legacy constants remain only for backward
    compatibility and fallback budgeting.
    """

    candidate = str(value or "").strip()
    return candidate or default


def task_type_schema_hint() -> str:
    """Human-readable schema hint for LLM routing prompts."""

    return "model-defined free-text task label; legacy examples: " + " | ".join(ALLOWED_TASK_TYPES)


def task_type_registry() -> dict[str, Any]:
    """Return the public task-type registry for artifacts and tests."""

    return {
        "allowed_task_types": "model_defined_free_text",
        "legacy_task_type_examples": list(ALLOWED_TASK_TYPES),
        "default": DEFAULT_TASK_TYPE,
        "frontier": sorted(FRONTIER_TASK_TYPES),
        "research": sorted(RESEARCH_TASK_TYPES),
        "code": sorted(CODE_TASK_TYPES),
        "evidence_required": sorted(EVIDENCE_REQUIRED_TASK_TYPES),
        "single_source_of_truth": "model_generated_dynamic_artifact_contract",
    }


def is_frontier_task_type(task_type: str | None) -> bool:
    return normalize_task_type(task_type) in FRONTIER_TASK_TYPES


def is_research_task_type(task_type: str | None) -> bool:
    return normalize_task_type(task_type) in RESEARCH_TASK_TYPES


def is_code_task_type(task_type: str | None) -> bool:
    return normalize_task_type(task_type) in CODE_TASK_TYPES


def evidence_required_for_task_type(task_type: str | None) -> bool:
    return normalize_task_type(task_type) in EVIDENCE_REQUIRED_TASK_TYPES


__all__ = [
    "ALLOWED_TASK_TYPES",
    "DEFAULT_TASK_TYPE",
    "FRONTIER_TASK_TYPES",
    "RESEARCH_TASK_TYPES",
    "CODE_TASK_TYPES",
    "EVIDENCE_REQUIRED_TASK_TYPES",
    "normalize_task_type",
    "task_type_schema_hint",
    "task_type_registry",
    "is_frontier_task_type",
    "is_research_task_type",
    "is_code_task_type",
    "evidence_required_for_task_type",
]
