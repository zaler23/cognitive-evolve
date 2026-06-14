"""Nexus-native, model-defined search-space breadth map.

The runtime should not decide that a task's useful search families are "proof",
"code", "science", "article", or any other finite domain list.  It may surface
coverage pressure, but the actual families/planes should come from the model
contract or from objective-derived placeholders that explicitly ask the model to
replace them.
"""
from __future__ import annotations

import re
from typing import Any


def build_search_space_map(assessment: dict[str, Any], requested_candidate_count: int = 0) -> dict[str, Any]:
    task_type = str(assessment.get("task_type") or "")
    objective = str(assessment.get("real_objective") or assessment.get("surface_request") or "")
    model_space = _first_mapping(
        assessment.get("search_space"),
        assessment.get("search_space_plan"),
        assessment.get("exploration_plan"),
        assessment.get("outcome_policy"),
    )
    model_families = _first_list(
        assessment.get("candidate_families"),
        assessment.get("exploration_planes"),
        model_space.get("candidate_families"),
        model_space.get("exploration_planes"),
        model_space.get("families"),
        model_space.get("planes"),
    )
    if any(isinstance(item, dict) and (item.get("id") or item.get("name")) for item in model_families):
        families = _normalize_model_families(model_families)
        problem_class = str(assessment.get("problem_class") or model_space.get("problem_class") or task_type or "model_defined_task")
        source = "model_authored_search_space"
        needs_model_authored_search_space = False
    else:
        families = _objective_derived_placeholder_families(objective, requested_candidate_count)
        problem_class = str(task_type or "model_defined_task")
        source = "objective_derived_placeholder_search_space"
        needs_model_authored_search_space = True
    requested = max(1, int(requested_candidate_count or 1))
    required = _coverage_required(model_space, families=families, requested=requested)
    return {
        "version": "nexus-search-space-map-v2",
        "runtime_architecture": "nexus",
        "source": source,
        "model_driven": True,
        "needs_model_authored_search_space": needs_model_authored_search_space,
        "problem_class": problem_class,
        "task_type_hint": task_type,
        "candidate_target_count": requested,
        "candidate_families": families,
        "route_family": [str(item.get("id")) for item in families],
        "coverage_gate": {
            "min_family_count": required,
            "rule": "cover materially distinct model-defined planes before deepening a single local surface",
            "allow_budget_degraded_status": requested < required,
        },
        "surface_bias_guard": {
            "rule": "available implementation/evidence surfaces are context, not the search objective; local verifiability must not crowd out higher-level planes requested by the user",
            "examples_are_not_domain_limits": True,
        },
    }


def classify_candidate(candidate: dict[str, Any], search_space_map: dict[str, Any], *, objective: str = "") -> dict[str, Any]:
    text = " ".join(str(candidate.get(key) or "") for key in ["artifact", "concise_claim", "core_mechanism", "claim", "content"]).lower()
    families = [dict(item) for item in search_space_map.get("candidate_families", []) if isinstance(item, dict)]
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    declared = candidate.get("search_space") if isinstance(candidate.get("search_space"), dict) else metadata.get("search_space")
    family_id = str((declared or {}).get("family_id") or (declared or {}).get("plane_id") or "").strip()
    if not family_id:
        family_id = _infer_family_by_overlap(text, families)
    known = {str(item.get("id")) for item in families}
    if family_id not in known and families:
        family_id = str(families[0].get("id"))
    selected = next((item for item in families if str(item.get("id")) == family_id), {})
    return {
        "family_id": family_id,
        "search_axes": dict(selected.get("axes") or {}),
        "mechanism_specificity": _score_overlap(text, selected),
        "objective_proximity": _score_overlap((objective or "").lower() + " " + text, selected),
        "breakthrough_proximity": _score_overlap(text, selected),
        "meta_task": False,
        "objective_final_eligible": True,
        "classification_reason": "nexus_search_space_classification",
    }


def analyze_coverage(candidates: list[dict[str, Any]], search_space_map: dict[str, Any]) -> dict[str, Any]:
    families = {str(item.get("id")) for item in search_space_map.get("candidate_families", []) if isinstance(item, dict)}
    seen = {str((item.get("search_space") or item).get("family_id") or "") for item in candidates if isinstance(item, dict)}
    missing = sorted(families - seen)
    required = int((search_space_map.get("coverage_gate") or {}).get("min_family_count") or 1)
    return {
        "status": "ok" if len(seen & families) >= required else "undercovered",
        "covered_family_count": len(seen & families),
        "covered_count": len(seen & families),
        "required_family_count": required,
        "missing_families": missing,
        "source": "nexus_search_space_coverage_gate",
    }


def _normalize_model_families(values: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, item in enumerate(values, start=1):
        if not isinstance(item, dict):
            continue
        raw_id = str(item.get("id") or item.get("name") or f"model_defined_plane_{index}").strip()
        family_id = _slug(raw_id) or f"model_defined_plane_{index}"
        family = dict(item)
        family["id"] = family_id
        family.setdefault("description", str(item.get("description") or item.get("intent") or raw_id))
        family.setdefault("quota_min", 1)
        out.append(family)
    return out or _objective_derived_placeholder_families("", 0)


def _objective_derived_placeholder_families(objective: str, requested_candidate_count: int = 0) -> list[dict[str, Any]]:
    terms = _objective_terms(objective)
    if not terms:
        terms = ["objective", "artifact", "evaluation"]
    target = max(3, min(8, requested_candidate_count or len(terms) or 3))
    families: list[dict[str, Any]] = []
    for index in range(target):
        term = terms[index % len(terms)]
        families.append(
            {
                "id": f"model_defined_focus_{index + 1}_{_slug(term) or 'objective'}",
                "description": f"Placeholder plane derived from objective focus '{term}'. The model should replace this with a task-specific search plane.",
                "axes": {"objective_focus": term, "model_authored_replacement_needed": True},
                "quota_min": 1,
            }
        )
    return families


def _objective_terms(objective: str) -> list[str]:
    text = str(objective or "").strip()
    latin = re.findall(r"[A-Za-z][A-Za-z0-9_/-]{3,}", text)
    cjk = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    stop = {
        "with",
        "from",
        "that",
        "this",
        "into",
        "your",
        "have",
        "core",
        "task",
        "goal",
    }
    out: list[str] = []
    for term in [*latin, *cjk]:
        normalized = term.strip().strip("-_/").lower()
        if not normalized or normalized in stop or normalized in out:
            continue
        out.append(normalized)
    return out[:12]


def _infer_family_by_overlap(text: str, families: list[dict[str, Any]]) -> str:
    if not families:
        return ""
    scored = [(_score_overlap(text, family), str(family.get("id") or "")) for family in families]
    scored.sort(reverse=True)
    return scored[0][1]


def _score_overlap(text: str, family: dict[str, Any]) -> float:
    haystack = set(_objective_terms(text))
    family_text = " ".join(str(family.get(key) or "") for key in ("id", "name", "description", "intent"))
    axes = family.get("axes")
    if isinstance(axes, dict):
        family_text += " " + " ".join(str(value) for value in axes.values())
    needles = set(_objective_terms(family_text))
    if not needles:
        return 0.3
    return min(1.0, len(haystack & needles) / max(1, len(needles)))


def _coverage_required(model_space: dict[str, Any], *, families: list[dict[str, Any]], requested: int) -> int:
    gate = model_space.get("coverage_gate") if isinstance(model_space.get("coverage_gate"), dict) else {}
    configured = gate.get("min_family_count") or model_space.get("min_family_count")
    try:
        parsed = int(configured)
    except (TypeError, ValueError):
        parsed = 0
    if parsed > 0:
        return max(1, min(parsed, len(families)))
    quotas = 0
    for family in families:
        try:
            quotas += max(0, int(family.get("quota_min") or 0))
        except (TypeError, ValueError):
            continue
    if quotas:
        return max(1, min(quotas, len(families)))
    return max(1, min(len(families), requested))


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return dict(value)
    return {}


def _first_list(*values: Any) -> list[Any]:
    for value in values:
        if isinstance(value, list):
            return list(value)
    return []


def _slug(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:80]


__all__ = ["analyze_coverage", "build_search_space_map", "classify_candidate"]
