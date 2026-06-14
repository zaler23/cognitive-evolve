"""Difficulty and round-checkpoint estimation for CLI-facing Nexus runs.

This module owns the model-authored difficulty/self-capability mapping used by
``cognitive_evolve_runtime.runtime``.  The selected rounds are initial evolution
checkpoints, not solve guarantees.
"""
from __future__ import annotations

import math
from dataclasses import replace

from cognitive_evolve_runtime.nexus.budgeting import resolve_nexus_round_budget

_PROFILE_ORDER = ("one-shot", "balanced", "deep", "ultra", "exhaustive", "frontier_proof", "breakthrough")
_DIFFICULTY_PROFILE_FLOOR = {
    "easy": "one-shot",
    "standard": "balanced",
    "hard": "deep",
    "research": "exhaustive",
    "frontier": "exhaustive",
    "unknown": "balanced",
}
_DIFFICULTY_ROUND_BANDS = {
    # These are not "solve in N rounds" promises.  They are generalized
    # checkpoint bands.  The entrypoint selects an initial checkpoint inside or
    # above the band after reading the model's self-assessed capability, desired
    # output level, and task-specific complexity dimensions.
    "easy": {"lower_bound_rounds": 2, "checkpoint_rounds": 4, "stretch_rounds": 12},
    "standard": {"lower_bound_rounds": 6, "checkpoint_rounds": 12, "stretch_rounds": 36},
    "hard": {"lower_bound_rounds": 16, "checkpoint_rounds": 48, "stretch_rounds": 120},
    "research": {"lower_bound_rounds": 36, "checkpoint_rounds": 120, "stretch_rounds": 360},
    "frontier": {"lower_bound_rounds": 64, "checkpoint_rounds": 240, "stretch_rounds": 720},
    "unknown": {"lower_bound_rounds": 12, "checkpoint_rounds": 48, "stretch_rounds": 180},
}
_MODEL_CAPABILITY_SCORE_MULTIPLIERS: tuple[tuple[float, float, str], ...] = (
    (0.90, 0.60, "frontier"),
    (0.75, 0.75, "strong"),
    (0.55, 1.00, "standard"),
    (0.35, 1.35, "limited"),
    (0.20, 1.70, "weak"),
    (0.00, 2.50, "very_weak"),
)
_OUTPUT_LEVEL_MULTIPLIERS = {
    "idea_seed": 0.35,
    "inspiration": 0.35,
    "direction": 0.60,
    "concrete_direction": 0.75,
    "research_seed": 0.85,
    "candidate_solution": 1.00,
    "proof_attempt": 1.35,
    "paper_level": 1.80,
}
_EFFORT_CLASS_MULTIPLIERS = {
    "routine": 0.75,
    "contest_exact_algorithm": 1.00,
    "hard_algorithm_design": 1.15,
    "research_direction": 1.35,
    "open_research": 2.00,
    "frontier_research": 3.00,
    "multi_month": 4.00,
}
_MIN_MODEL_ROUND_MULTIPLIER = 0.50
_MAX_MODEL_ROUND_MULTIPLIER = 6.00


def runtime_round_budget(*, route_profile: str, route_semantic: dict[str, object], rounds: int | None, difficulty_assessment: dict[str, object] | None = None):
    del route_semantic
    context = {"evolution_profile": route_profile, "runtime_entry": "runtime_run"}
    if rounds is not None:
        context["rounds"] = rounds
        return resolve_nexus_round_budget(context)

    estimated_rounds = _difficulty_adjusted_rounds(difficulty_assessment)
    if estimated_rounds is not None:
        budget = resolve_nexus_round_budget({**context, "rounds": estimated_rounds})
        warnings = list(budget.config_warnings or [])
        warnings.append("runtime_model_difficulty_round_estimate_applied")
        return replace(
            budget,
            source="model_difficulty_round_estimate",
            explicit_override=None,
            config_warnings=warnings,
        )

    return resolve_nexus_round_budget(context)


def runtime_entry_difficulty(route: object) -> dict[str, object]:
    """Return the entrypoint's model-authored problem-difficulty judgment.

    This intentionally avoids prompt-keyword heuristics.  It consumes only the
    model route payload already obtained by ``classify_task``: difficulty labels,
    suggested profile/round fields, complexity scores, level, and profile.
    """

    semantic = getattr(route, "semantic", {}) if isinstance(getattr(route, "semantic", {}), dict) else {}
    raw = semantic.get("raw") if isinstance(semantic.get("raw"), dict) else {}
    route_profile = _normalize_profile(getattr(route, "profile", "balanced"))
    route_level = str(getattr(route, "level", "") or raw.get("level") or "")
    task_type = str(semantic.get("task_type") or raw.get("task_type") or "")
    model_authored = semantic.get("model_route_available") is not False and semantic.get("fallback_only") is not True

    difficulty_label = _first_text(
        semantic.get("difficulty"),
        semantic.get("difficulty_level"),
        semantic.get("problem_difficulty"),
        semantic.get("complexity_label"),
        raw.get("difficulty"),
        raw.get("difficulty_level"),
        raw.get("problem_difficulty"),
        raw.get("complexity_label"),
    )
    suggested_profile = _normalize_profile(
        _first_text(
            semantic.get("suggested_profile"),
            semantic.get("difficulty_profile"),
            semantic.get("evolution_profile"),
            raw.get("suggested_profile"),
            raw.get("difficulty_profile"),
            raw.get("evolution_profile"),
            route_profile,
        )
    )
    suggested_rounds = _positive_int_or_none(
        semantic.get("suggested_rounds")
        or semantic.get("recommended_rounds")
        or raw.get("suggested_rounds")
        or raw.get("recommended_rounds")
    )
    complexity_score = _complexity_score_from_semantic(semantic) or _complexity_score_from_semantic(raw)

    source = "model_route_profile"
    if difficulty_label:
        difficulty = _normalize_difficulty_label(difficulty_label)
        source = "model_difficulty_label"
    elif complexity_score is not None:
        difficulty = _difficulty_from_score(complexity_score)
        source = "model_complexity_score"
    elif model_authored and task_type != "route_incomplete":
        difficulty = _difficulty_from_profile_or_level(suggested_profile, route_level)
    else:
        difficulty = "unknown"
        source = "model_unavailable_or_incomplete"

    self_assessment = _model_self_assessment(semantic, raw)
    round_estimate = _round_estimate_payload(
        difficulty=difficulty,
        suggested_rounds=suggested_rounds,
        self_assessment=self_assessment,
    )
    return {
        "difficulty": difficulty,
        "source": source,
        "model_authored": bool(model_authored),
        "profile": runtime_profile_from_difficulty(suggested_profile, {"difficulty": difficulty}),
        "route_profile": route_profile,
        "route_level": route_level,
        "task_type": task_type,
        "complexity_score": complexity_score,
        "suggested_rounds": suggested_rounds,
        "model_self_assessment": self_assessment,
        "round_estimate": round_estimate,
    }


def _difficulty_adjusted_rounds(difficulty_assessment: dict[str, object] | None) -> int | None:
    estimate = (difficulty_assessment or {}).get("round_estimate")
    if isinstance(estimate, dict):
        rounds = _positive_int_or_none(estimate.get("selected_rounds"))
        if rounds is not None:
            return rounds
    if difficulty_assessment:
        return _round_estimate_payload(
            difficulty=str(difficulty_assessment.get("difficulty") or "unknown"),
            suggested_rounds=_positive_int_or_none(difficulty_assessment.get("suggested_rounds")),
            self_assessment=difficulty_assessment.get("model_self_assessment") if isinstance(difficulty_assessment.get("model_self_assessment"), dict) else {},
        )["selected_rounds"]
    return None


def _round_estimate_payload(*, difficulty: str, suggested_rounds: int | None, self_assessment: dict[str, object]) -> dict[str, object]:
    normalized_difficulty = difficulty if difficulty in _DIFFICULTY_ROUND_BANDS else "unknown"
    band = dict(_DIFFICULTY_ROUND_BANDS[normalized_difficulty])
    capability = _model_capability_payload(self_assessment)
    output_level = _normalize_output_level(self_assessment.get("target_output_level") or self_assessment.get("desired_output_level"))
    effort_class = _normalize_effort_class(self_assessment.get("effort_class") or self_assessment.get("task_effort_class"))
    complexity = _complexity_multiplier_payload(self_assessment.get("complexity_dimensions"))
    model_range = _round_range_payload(self_assessment.get("expected_round_range") or self_assessment.get("round_range"))
    computed_checkpoint = max(
        1,
        int(
            math.ceil(
                float(band["checkpoint_rounds"])
                * float(capability["round_multiplier"])
                * float(_OUTPUT_LEVEL_MULTIPLIERS[output_level])
                * float(_EFFORT_CLASS_MULTIPLIERS[effort_class])
                * float(complexity["multiplier"])
            )
        ),
    )
    lower_bound_rounds = max(
        1,
        int(
            math.ceil(
                float(band["lower_bound_rounds"])
                * float(capability["round_multiplier"])
                * float(_OUTPUT_LEVEL_MULTIPLIERS[output_level])
                * float(_EFFORT_CLASS_MULTIPLIERS[effort_class])
            )
        ),
    )
    selected_rounds = max(
        computed_checkpoint,
        lower_bound_rounds,
        int(suggested_rounds or 0),
        int(model_range.get("checkpoint_rounds") or 0),
    )
    return {
        "policy": "generalized_checkpoint_band_times_model_self_capability_and_task_complexity",
        "difficulty_anchor": normalized_difficulty,
        "round_band": band,
        "lower_bound_rounds": lower_bound_rounds,
        "checkpoint_rounds": computed_checkpoint,
        "stretch_rounds": max(int(band["stretch_rounds"]), int(model_range.get("stretch_rounds") or 0)),
        "target_output_level": output_level,
        "effort_class": effort_class,
        "complexity_multiplier": complexity,
        "model_capability": capability,
        "model_expected_round_range": model_range,
        "model_suggested_rounds": suggested_rounds,
        "selected_rounds": selected_rounds,
        "selection_rule": "max(computed_checkpoint, lower_bound, model_expected_checkpoint, model_suggested_rounds)",
        "semantics": "selected_rounds is an initial evolution checkpoint, not a correctness or solve guarantee",
    }


def _model_self_assessment(semantic: dict[str, object], raw: dict[str, object]) -> dict[str, object]:
    source = _first_mapping(
        semantic.get("model_self_assessment"),
        raw.get("model_self_assessment"),
        semantic.get("self_assessed_capability"),
        raw.get("self_assessed_capability"),
        semantic.get("model_capability"),
        raw.get("model_capability"),
    )
    capability_score = _bounded_float_or_none(
        source.get("capability_score")
        or source.get("self_capability_score")
        or source.get("algorithmic_reasoning_score")
        or semantic.get("model_capability_score")
        or raw.get("model_capability_score")
        or semantic.get("capability_score")
        or raw.get("capability_score")
    )
    capability_tier = _first_text(
        source.get("capability_tier"),
        source.get("tier"),
        source.get("model_tier"),
        semantic.get("model_capability_tier"),
        raw.get("model_capability_tier"),
        semantic.get("capability_tier"),
        raw.get("capability_tier"),
    )
    if capability_score is None:
        capability_score = _capability_score_from_label(capability_tier)
    model_multiplier = _round_multiplier_or_none(
        source.get("round_multiplier")
        or source.get("expected_round_multiplier")
        or source.get("self_estimated_round_multiplier")
        or semantic.get("model_round_multiplier")
        or raw.get("model_round_multiplier")
    )
    target_output_level = _first_text(
        source.get("target_output_level"),
        source.get("desired_output_level"),
        semantic.get("target_output_level"),
        raw.get("target_output_level"),
        semantic.get("desired_output_level"),
        raw.get("desired_output_level"),
    )
    effort_class = _first_text(
        source.get("effort_class"),
        source.get("task_effort_class"),
        semantic.get("effort_class"),
        raw.get("effort_class"),
        semantic.get("task_effort_class"),
        raw.get("task_effort_class"),
    )
    complexity_dimensions = _first_mapping(
        source.get("complexity_dimensions"),
        source.get("round_complexity_dimensions"),
        semantic.get("complexity_dimensions"),
        raw.get("complexity_dimensions"),
        semantic.get("round_complexity_dimensions"),
        raw.get("round_complexity_dimensions"),
    )
    expected_round_range = _first_mapping(
        source.get("expected_round_range"),
        source.get("round_range"),
        semantic.get("expected_round_range"),
        raw.get("expected_round_range"),
        semantic.get("round_range"),
        raw.get("round_range"),
    )
    return {
        "capability_tier": _normalize_capability_tier(capability_tier, capability_score),
        "capability_score": capability_score,
        "round_multiplier": model_multiplier,
        "target_output_level": _normalize_output_level(target_output_level),
        "effort_class": _normalize_effort_class(effort_class),
        "complexity_dimensions": complexity_dimensions,
        "expected_round_range": expected_round_range,
        "source": "model_self_assessment" if source or capability_tier or capability_score is not None or model_multiplier is not None else "default_standard_capability",
    }


def _model_capability_payload(self_assessment: dict[str, object]) -> dict[str, object]:
    explicit_multiplier = _round_multiplier_or_none(self_assessment.get("round_multiplier"))
    if explicit_multiplier is not None:
        return {
            "capability_tier": str(self_assessment.get("capability_tier") or "model_supplied_multiplier"),
            "capability_score": self_assessment.get("capability_score"),
            "round_multiplier": explicit_multiplier,
            "source": str(self_assessment.get("source") or "model_round_multiplier"),
        }
    score = _bounded_float_or_none(self_assessment.get("capability_score"))
    if score is None:
        score = 0.55
    for threshold, multiplier, tier in _MODEL_CAPABILITY_SCORE_MULTIPLIERS:
        if score >= threshold:
            return {
                "capability_tier": str(self_assessment.get("capability_tier") or tier),
                "capability_score": score,
                "round_multiplier": multiplier,
                "source": str(self_assessment.get("source") or "capability_score_mapping"),
            }
    return {
        "capability_tier": str(self_assessment.get("capability_tier") or "standard"),
        "capability_score": score,
        "round_multiplier": 1.0,
        "source": str(self_assessment.get("source") or "capability_score_mapping"),
    }


def runtime_profile_from_difficulty(route_profile: object, difficulty_assessment: dict[str, object] | None) -> str:
    base = _normalize_profile(route_profile)
    difficulty = str((difficulty_assessment or {}).get("difficulty") or "unknown")
    floor = _DIFFICULTY_PROFILE_FLOOR.get(difficulty, "balanced")
    return _max_profile(base, floor)


def _normalize_profile(value: object) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "fast": "one-shot",
        "one_shot": "one-shot",
        "research": "exhaustive",
        "frontier": "frontier_proof",
        "frontier-proof": "frontier_proof",
    }
    normalized = aliases.get(text, text)
    return normalized if normalized in _PROFILE_ORDER else "balanced"


def _max_profile(left: object, right: object) -> str:
    l_profile = _normalize_profile(left)
    r_profile = _normalize_profile(right)
    return l_profile if _PROFILE_ORDER.index(l_profile) >= _PROFILE_ORDER.index(r_profile) else r_profile


def _first_text(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _first_mapping(*values: object) -> dict[str, object]:
    for value in values:
        if isinstance(value, dict):
            return dict(value)
    return {}


def _positive_int_or_none(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _complexity_score_from_semantic(semantic: dict[str, object]) -> float | None:
    for key in ("difficulty_score", "complexity_score", "problem_complexity"):
        score = _bounded_float_or_none(semantic.get(key))
        if score is not None:
            return score
    raw = semantic.get("complexity_assessment")
    if isinstance(raw, dict):
        for key in ("semantic_complexity", "overall", "difficulty", "complexity"):
            score = _bounded_float_or_none(raw.get(key))
            if score is not None:
                return score
    return None


def _bounded_float_or_none(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return max(0.0, min(1.0, parsed))


def _round_multiplier_or_none(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return max(_MIN_MODEL_ROUND_MULTIPLIER, min(_MAX_MODEL_ROUND_MULTIPLIER, parsed))


def _normalize_output_level(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "idea": "idea_seed",
        "seed": "idea_seed",
        "inspiration_seed": "idea_seed",
        "research_direction": "direction",
        "route": "direction",
        "plan": "research_seed",
        "solution": "candidate_solution",
        "answer": "candidate_solution",
        "proof": "proof_attempt",
        "paper": "paper_level",
    }
    normalized = aliases.get(text, text)
    return normalized if normalized in _OUTPUT_LEVEL_MULTIPLIERS else "candidate_solution"


def _normalize_effort_class(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "contest": "contest_exact_algorithm",
        "ccf_a": "contest_exact_algorithm",
        "ccfa": "contest_exact_algorithm",
        "algorithm": "hard_algorithm_design",
        "hard_algorithm": "hard_algorithm_design",
        "research": "research_direction",
        "open_problem": "open_research",
        "frontier": "frontier_research",
        "long_horizon": "multi_month",
    }
    normalized = aliases.get(text, text)
    return normalized if normalized in _EFFORT_CLASS_MULTIPLIERS else "contest_exact_algorithm"


def _complexity_multiplier_payload(value: object) -> dict[str, object]:
    dimensions = dict(value) if isinstance(value, dict) else {}
    normalized: dict[str, float] = {}
    for key in (
        "novelty",
        "proof_burden",
        "search_space_width",
        "verification_difficulty",
        "ambiguity",
        "implementation_burden",
        "external_knowledge_dependency",
    ):
        score = _bounded_float_or_none(dimensions.get(key))
        if score is not None:
            normalized[key] = score
    if not normalized:
        return {"dimensions": {}, "average": 0.0, "multiplier": 1.0}
    average = sum(normalized.values()) / len(normalized)
    multiplier = 1.0 + 1.75 * average
    return {"dimensions": normalized, "average": round(average, 4), "multiplier": round(multiplier, 4)}


def _round_range_payload(value: object) -> dict[str, int | None]:
    data = dict(value) if isinstance(value, dict) else {}
    lower = _positive_int_or_none(data.get("lower_bound_rounds") or data.get("min_rounds") or data.get("minimum_rounds"))
    checkpoint = _positive_int_or_none(data.get("checkpoint_rounds") or data.get("initial_checkpoint_rounds") or data.get("target_rounds"))
    stretch = _positive_int_or_none(data.get("stretch_rounds") or data.get("upper_bound_rounds") or data.get("max_rounds"))
    return {
        "lower_bound_rounds": lower,
        "checkpoint_rounds": checkpoint,
        "stretch_rounds": stretch,
    }


def _capability_score_from_label(value: object) -> float | None:
    text = str(value or "").strip().lower().replace("_", "-")
    if not text:
        return None
    if any(token in text for token in ("frontier", "top", "elite", "very-strong")):
        return 0.9
    if any(token in text for token in ("strong", "advanced", "high")):
        return 0.75
    if any(token in text for token in ("standard", "medium", "average", "normal")):
        return 0.55
    if any(token in text for token in ("limited", "small", "cheap", "flash", "fast")):
        return 0.35
    if any(token in text for token in ("weak", "low")):
        return 0.2
    return None


def _normalize_capability_tier(label: object, score: float | None) -> str:
    text = str(label or "").strip().lower().replace(" ", "_").replace("-", "_")
    if text:
        return text
    if score is None:
        return "standard"
    for threshold, _, tier in _MODEL_CAPABILITY_SCORE_MULTIPLIERS:
        if score >= threshold:
            return tier
    return "standard"


def _normalize_difficulty_label(value: object) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    compact = text.replace(" ", "-")
    if any(token in compact for token in ("frontier", "breakthrough", "unsolved", "open-conjecture")):
        return "frontier"
    if any(token in compact for token in ("research", "paper-level", "publishable")):
        return "research"
    if any(token in compact for token in ("ccf-a", "ccfa", "hard", "difficult", "expert", "advanced")):
        return "hard"
    if any(token in compact for token in ("standard", "medium", "normal", "moderate")):
        return "standard"
    if any(token in compact for token in ("easy", "simple", "trivial", "basic")):
        return "easy"
    return "unknown"


def _difficulty_from_score(score: float) -> str:
    if score >= 0.9:
        return "frontier"
    if score >= 0.75:
        return "research"
    if score >= 0.55:
        return "hard"
    if score >= 0.25:
        return "standard"
    return "easy"


def _difficulty_from_profile_or_level(profile: str, level: str) -> str:
    if profile in {"frontier_proof", "breakthrough"}:
        return "frontier"
    if profile in {"ultra", "exhaustive"}:
        return "research"
    if profile == "deep" or level in {"L4_evolutionary", "L5_longitudinal", "L6_governed"}:
        return "hard"
    if profile == "balanced" or level in {"L2_structured", "L3_comparative"}:
        return "standard"
    return "easy"




__all__ = [
    "runtime_entry_difficulty",
    "runtime_profile_from_difficulty",
    "runtime_round_budget",
]
