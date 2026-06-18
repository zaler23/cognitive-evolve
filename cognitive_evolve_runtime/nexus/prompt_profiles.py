"""Request-type prompt profiles for Nexus model calls."""
from __future__ import annotations

import json
from typing import Any

_OFFSPRING_KEEP = {
    "request_type",
    "plans",
    "parents",
    "contract",
    "policy",
    "world",
    "mutation_instruction",
    "artifact_generation_contract",
    "search_space_contract",
    "verification_regime",
    "_prompt_context_controls_applied",
}
_PLAN_KEEP = {
    "request_type",
    "parents",
    "actions",
    "diagnosis",
    "policy",
    "contract",
    "archives",
    "history",
    "search_space_contract",
    "artifact_generation_contract",
    "verification_regime",
}
_CRITIQUE_KEEP = {"request_type", "candidates", "contract", "policy", "world", "verification_plan", "verification_regime", "candidate_population_stats"}
_DIAGNOSE_KEEP = {"request_type", "candidate_population_stats", "history", "diagnosis", "policy", "contract", "archives", "verification_plan", "verification_regime"}
_POLICY_KEEP = {"request_type", "diagnosis", "policy", "history", "contract", "candidate_population_stats"}

PROFILE_KEEP: dict[str, set[str]] = {
    "nexus_generate_offspring": _OFFSPRING_KEEP,
    "nexus_plan_mutations": _PLAN_KEEP,
    "nexus_critique_candidates": _CRITIQUE_KEEP,
    "nexus_diagnose_search_state": _DIAGNOSE_KEEP,
    "nexus_update_policy": _POLICY_KEEP,
}

_FORBIDDEN_STRENGTH_KEYS = {"strength_contribution", "legacy_strength", "measured_strength", "measured_strength_value"}


def apply_prompt_profile(request_type: str, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a profiled payload plus metadata.

    The profile prunes unrelated sections and strips strength shortcuts from the
    model-facing view.  It never mutates the caller's payload.
    """

    keep = PROFILE_KEEP.get(str(request_type or ""))
    data = _json_clone(payload)
    if keep:
        kept = {key: value for key, value in data.items() if key in keep or key.startswith("_")}
        if "verification_regime" in data:
            kept["verification_regime"] = data["verification_regime"]
        data = kept
    removed_strength_keys: list[str] = []
    data = _strip_forbidden_strength_keys(data, removed=removed_strength_keys)
    metadata = {
        "profile_applied": bool(keep),
        "profile_name": str(request_type or "default"),
        "removed_strength_shortcut_keys": sorted(set(removed_strength_keys)),
        "payload_chars_after_profile": len(json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)),
    }
    return data, metadata


def _strip_forbidden_strength_keys(value: Any, *, removed: list[str]) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in _FORBIDDEN_STRENGTH_KEYS or (key == "replayable" and _looks_like_legacy_verification_dict(value)):
                removed.append(str(key))
                continue
            out[key] = _strip_forbidden_strength_keys(item, removed=removed)
        return out
    if isinstance(value, list):
        return [_strip_forbidden_strength_keys(item, removed=removed) for item in value]
    return value


def _looks_like_legacy_verification_dict(value: dict[str, Any]) -> bool:
    return any(key in value for key in ("strength", "strength_value", "measured_strength", "strength_contribution"))


def _json_clone(value: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return dict(value)


__all__ = ["apply_prompt_profile", "PROFILE_KEEP"]
