"""JSON schemas for the structured Nexus model adapter."""
from __future__ import annotations

from typing import Any

def _objective_contract_schema(*, project: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "original_user_goal": {"type": "string"},
        "normalized_goal": {"type": "string"},
        "task_type": {"type": "string"},
        "outcome_policy": {"type": "object"},
        "dynamic_artifact_contract": {"type": "object"},
        "search_space_plan": {
            "type": "object",
            "description": "Model-authored exploration planes/families for this objective. Do not copy a finite runtime taxonomy; derive planes from the user objective and expected artifact.",
        },
        "input_constraints": _string_array(),
        "allowed_evidence_sources": _string_array(),
        "disallowed_goal_mutations": _string_array(),
        "expected_output_forms": _string_array(),
        "uncertainty_policy": {"type": "string"},
        "verification_preferences": _string_array(),
        "success_dimensions": _string_array(),
        "failure_dimensions": _string_array(),
    }
    if project:
        properties.update({
            "frozen_regions": _string_array(),
            "mutable_regions": _string_array(),
            "contract_files": _string_array(),
            "implementation_files": _string_array(),
            "test_contracts": _string_array(),
            "allowed_patch_scope": _string_array(),
            "unsafe_change_patterns": _string_array(),
        })
    return {"type": "object", "required": ["original_user_goal", "normalized_goal"], "properties": properties, "additionalProperties": True}

def _text_world_model_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "input_packet_id": {"type": "string"},
            "goal_summary": {"type": "string"},
            "evidence_boundaries": {"type": "object"},
            "likely_task_types": _string_array(),
            "constraint_summary": _string_array(),
            "uncertainty_zones": _string_array(),
            "edge_seed_pool": _string_array(),
        },
        "additionalProperties": True,
    }

def _task_classification_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "level": {"type": "string"},
            "profile": {"type": "string"},
            "search": {"type": "boolean"},
            "checkmodel": {"type": "boolean"},
            "artifacts": {"type": "boolean"},
            "reason": {"type": "string"},
            "task_type": {"type": "string"},
            "difficulty": {
                "type": "string",
                "description": "Model-authored problem difficulty label, e.g. easy, standard, hard, CCF-A hard, research, frontier.",
            },
            "difficulty_score": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Optional normalized difficulty/complexity score where 0 is easy and 1 is frontier-level.",
            },
            "suggested_profile": {
                "type": "string",
                "description": "Optional Nexus budget profile suggested by difficulty: one-shot, balanced, deep, ultra, exhaustive, frontier_proof, or breakthrough.",
            },
            "suggested_rounds": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional positive initial checkpoint suggestion after estimating task difficulty, desired output level, complexity dimensions, and your own model capability. This is not a correctness guarantee. Do not use a single benchmark example; generalize by task class. Explicit operator --rounds still has priority.",
            },
            "model_self_assessment": {
                "type": "object",
                "description": "Self-estimate your capability and the task's effort shape before round planning. Use generalized task classes, not provider-specific secrets or a single named benchmark. Suggested fields: capability_tier, capability_score in [0,1], round_multiplier, target_output_level, effort_class, complexity_dimensions, expected_round_range, rationale. selected rounds are an initial evolution checkpoint, not a solve guarantee.",
                "properties": {
                    "capability_tier": {"type": "string"},
                    "capability_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "round_multiplier": {"type": "number", "minimum": 0.5, "maximum": 6.0},
                    "target_output_level": {
                        "type": "string",
                        "description": "Examples: idea_seed, direction, concrete_direction, research_seed, candidate_solution, proof_attempt, paper_level.",
                    },
                    "effort_class": {
                        "type": "string",
                        "description": "Examples: routine, contest_exact_algorithm, hard_algorithm_design, research_direction, open_research, frontier_research, multi_month.",
                    },
                    "complexity_dimensions": {
                        "type": "object",
                        "description": "Optional normalized scores such as novelty, proof_burden, search_space_width, verification_difficulty, ambiguity, implementation_burden, external_knowledge_dependency.",
                        "additionalProperties": {"type": "number"},
                    },
                    "expected_round_range": {
                        "type": "object",
                        "description": "Optional model-authored checkpoint band; use lower_bound_rounds, checkpoint_rounds, stretch_rounds.",
                        "additionalProperties": True,
                    },
                    "rationale": {"type": "string"},
                },
                "additionalProperties": True,
            },
            "semantic": {"type": "object"},
        },
        "additionalProperties": True,
    }


def _pool_preprocess_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "schedule_hints": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
            },
            "source_gap_requests": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
            },
            "diagnostics": _string_array(),
        },
        "additionalProperties": True,
    }

def _evolution_policy_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "candidate_niches": _string_array(),
            "fitness_axes": _string_array(),
            "mutation_operators": _string_array(),
            "archive_schema": {"type": "object"},
            "parent_selection_preferences": {"type": "object"},
            "culling_principles": _string_array(),
            "rarity_budget": {"type": "number", "minimum": 0},
            "tool_preferences": _string_array(),
            "stagnation_actions": _string_array(),
            "synthesis_policy": {"type": "object"},
            "metadata": {"type": "object"},
            "search_space": {
                "type": "object",
                "description": "Model-authored search-space placement for this candidate. Include family_id or plane_id from the search_space_contract, plus why this plane is distinct from recent candidates.",
                "additionalProperties": True,
            },
        },
        "additionalProperties": True,
    }

def _candidate_population_schema() -> dict[str, Any]:
    candidate_item = _candidate_item_schema()
    return {
        "type": "object",
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "items": candidate_item,
            }
        },
        "additionalProperties": True,
    }

def _candidate_critiques_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["critiques"],
        "properties": {
            "critiques": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "round": {"type": "integer"},
                        "strengths": _string_array(),
                        "flaws": _string_array(),
                        "missing_evidence": _string_array(),
                        "proposed_mutations": _string_array(),
                        "reusable_genes": _string_array(),
                        "severity": {"type": "number"},
                    },
                    "additionalProperties": True,
                },
            }
        },
        "additionalProperties": True,
    }

def _mutation_plan_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["plans"],
        "properties": {
            "plans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "operator": {"type": "string"},
                        "parent_ids": _string_array(),
                        "instruction": {"type": "string"},
                        "rarity_seed": {"type": "string"},
                        "expected_gene_effects": _string_array(),
                        "metadata": {"type": "object"},
                    },
                    "additionalProperties": True,
                },
            }
        },
        "additionalProperties": True,
    }

def _offspring_population_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["offspring"],
        "properties": {
            "offspring": {
                "type": "array",
                "items": _candidate_item_schema(
                    required=(
                        "artifact_type",
                        "concise_claim",
                        "core_mechanism",
                        "touched_files",
                        "source_bindings",
                        "evidence_refs",
                        "evaluation_dimensions",
                        "final_gate",
                    )
                ),
            }
        },
        "additionalProperties": True,
    }

def _synthesis_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["status", "final_answer"],
        "properties": {
            "status": {"type": "string"},
            "final_answer": {"type": "string"},
            "best_candidate_id": {"type": "string"},
            "best_auxiliary_candidate_id": {"type": "string"},
            "warnings": _string_array(),
            "failure_analysis": {"type": "string"},
        },
        "additionalProperties": True,
    }

def _candidate_item_schema(required: tuple[str, ...] | None = None) -> dict[str, Any]:
    required_fields = list(required or ("artifact", "artifact_type", "concise_claim", "core_mechanism", "assumptions", "missing_parts", "uncertainty_notes"))
    return {
        "type": "object",
        "required": required_fields,
        "properties": {
            "artifact": {},
            "artifact_type": {"type": "string"},
            "concise_claim": {"type": "string"},
            "core_mechanism": {"type": "string"},
            "assumptions": _string_array(),
            "missing_parts": _string_array(),
            "uncertainty_notes": _string_array(),
            "verification_trace": {"type": "array", "items": {"type": "object"}},
            "formal_artifacts": {
                "type": "array",
                "description": "Concrete verifier-readable or contract-readable artifact objects. The model-defined artifact contract decides the artifact shape; use proof/check/source objects only when that contract explicitly requires those adapters. Do not use descriptive placeholder prose.",
                "items": {"type": "object", "additionalProperties": True},
            },
            "proof_obligations": {
                "type": "array",
                "description": "Named obligations with id/status/description. Use status introduced, blocked, decomposed, discharged, refuted, or open.",
                "items": {"type": "object", "additionalProperties": True},
            },
            "obligation_delta": {
                "type": "object",
                "description": "Ledger delta for this candidate. Include model-defined artifact_delta/progress actions, and use targeted/discharged/decomposed/refuted/introduced only when the contract defines obligations that need them.",
                "additionalProperties": True,
            },
            "evidence_refs": {
                "type": "array",
                "description": "Evidence or evaluation references bound to the model-defined artifact contract. Source/test/patch/formal refs are required only when the contract activates those adapters; generic artifact/rubric/comparison refs are valid for non-code tasks.",
                "items": {"type": "object", "additionalProperties": True},
            },
            "source_bindings": {
                "type": "array",
                "description": "Exact source/materialization binding points only for candidates whose dynamic artifact contract requires source, patch, file, or tool adapters. Leave empty for pure in-memory artifact evolution.",
                "items": {"type": "object", "additionalProperties": True},
            },
            "touched_files": {
                "type": "array",
                "description": "Project/file/material paths actually touched or materialized by the artifact. For non-file tasks this may be empty; do not invent file paths.",
                "items": {"type": "string"},
            },
            "evaluation_dimensions": {
                "type": "array",
                "description": "Model-authored dimensions the runtime should use to judge this candidate's concrete artifact delta.",
                "items": {"type": "string"},
            },
            "final_gate": {
                "type": "object",
                "description": "Candidate-local view of what still blocks final acceptance under the model-defined artifact contract.",
                "additionalProperties": True,
            },
            "evidence_delta": {
                "type": "object",
                "description": "Delta describing new, verified, refuted, planned, or blocked evidence since the parent candidate.",
                "additionalProperties": True,
            },
            "verification_result": {"type": "object", "additionalProperties": True},
            "failure_lessons": _string_array(),
            "multihead_scores": {"type": "object"},
            "metadata": {"type": "object"},
        },
        "additionalProperties": True,
    }

def _stop_decision_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["stop", "reason", "solved"],
        "properties": {
            "stop": {"type": "boolean"},
            "reason": {"type": "string"},
            "solved": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "continuation_needed": {"type": "boolean"},
            "stop_kind": {
                "type": "string",
                "enum": [
                    "",
                    "candidate_ready_for_external_review",
                    "diminishing_returns_checkpoint",
                    "objective_solved",
                    "needs_continuation",
                ],
            },
            "open_gaps": _string_array(),
        },
        "additionalProperties": True,
    }

def _search_diagnosis_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "stagnation_detected": {"type": "boolean"},
            "stagnation_type": {"type": "string"},
            "over_explored_families": _string_array(),
            "under_explored_families": _string_array(),
            "prematurely_culled_genes": _string_array(),
            "auxiliary_collapse_risk": {"type": "number"},
            "semantic_drift_risk": {"type": "number"},
            "recommended_actions": _string_array(),
            "notes": {"type": "string"},
        },
        "additionalProperties": True,
    }

def _context_request_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "need_files": _string_array(),
            "need_symbols": _string_array(),
            "need_tests": _string_array(),
            "target_obligation_ids": _string_array(),
            "evidence_need": {"type": "string"},
            "reason": {"type": "string"},
        },
        "additionalProperties": True,
    }

def _string_array() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}

def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return dict(value)
    return {"repr": repr(value)}

__all__ = [
    "_candidate_critiques_schema", "_candidate_population_schema", "_context_request_schema",
    "_evolution_policy_schema", "_mutation_plan_schema", "_pool_preprocess_schema", "_objective_contract_schema",
    "_offspring_population_schema", "_search_diagnosis_schema", "_stop_decision_schema",
    "_synthesis_schema", "_task_classification_schema", "_text_world_model_schema",
]
