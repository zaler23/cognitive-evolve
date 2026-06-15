"""Machine artifact normalization for the Evidence Control Plane."""
from __future__ import annotations

import ast
import json
from typing import Any

from cognitive_evolve_runtime.evaluators.evidence import ArtifactPolicy
from cognitive_evolve_runtime.nexus._serde import coerce_dict
from cognitive_evolve_runtime.core.scalars import bounded_score


def normalize_artifact(candidate: Any, *, artifact_type: str = "", policy: ArtifactPolicy | None = None, machine_artifact_required: bool | None = None) -> dict[str, Any]:
    policy = policy or ArtifactPolicy(machine_readable_required=bool(machine_artifact_required))
    declared_type = str(artifact_type or policy.artifact_type or getattr(candidate, "artifact_type", "") or "").strip()
    artifact = getattr(candidate, "artifact", None)
    if _is_clean_artifact(artifact):
        normalized = artifact
        inferred_type = declared_type or _infer_type(normalized)
        return {
            "artifact_type": inferred_type,
            "artifact": artifact,
            "normalized_artifact": normalized,
            "status": "clean",
            "schema_cleanliness": 1.0,
            "probe_eligible": True,
            "final_eligible": True,
            "diagnostics": [],
        }
    if isinstance(artifact, str) and artifact.strip():
        refolded = _extract_embedded_artifact(artifact)
        if refolded is not None:
            return {
                "artifact_type": declared_type or _infer_type(refolded),
                "artifact": artifact,
                "normalized_artifact": refolded,
                "status": "refolded",
                "schema_cleanliness": 0.55,
                "probe_eligible": bool(policy.allow_refold_for_probe),
                "final_eligible": bool(policy.allow_refold_for_final),
                "diagnostics": ["artifact_refolded_from_natural_language"],
            }
        return {
            "artifact_type": declared_type,
            "artifact": artifact,
            "normalized_artifact": None,
            "status": "malformed" if policy.machine_readable_required else "refolded",
            "schema_cleanliness": 0.2 if policy.machine_readable_required else 0.35,
            "probe_eligible": bool(policy.allow_text_fallback and not policy.machine_readable_required),
            "final_eligible": bool(policy.allow_text_fallback and not policy.machine_readable_required and policy.allow_refold_for_final),
            "diagnostics": ["artifact_not_machine_parseable"],
        }
    fallback_text = str(getattr(candidate, "concise_claim", "") or getattr(candidate, "core_mechanism", "") or "").strip()
    if fallback_text and policy.allow_text_fallback and not policy.machine_readable_required:
        return {
            "artifact_type": declared_type or "natural_language",
            "artifact": fallback_text,
            "normalized_artifact": fallback_text,
            "status": "refolded",
            "schema_cleanliness": 0.4,
            "probe_eligible": True,
            "final_eligible": bool(policy.allow_refold_for_final),
            "diagnostics": ["natural_language_fallback_refolded_for_probe_only"],
        }
    return {
        "artifact_type": declared_type,
        "artifact": artifact,
        "normalized_artifact": None,
        "status": "absent",
        "schema_cleanliness": 0.0,
        "probe_eligible": False,
        "final_eligible": False,
        "diagnostics": ["artifact_absent"],
    }

def _is_clean_artifact(value: Any) -> bool:
    return isinstance(value, (dict, list)) and bool(value)


def _infer_type(value: Any) -> str:
    if isinstance(value, dict):
        if isinstance(value.get("layers"), list) or isinstance(value.get("comparators"), list):
            return "combinatorial_artifact"
        if any(key in value for key in ("patch", "diff", "unified_diff", "patch_set")):
            return "patch"
    if isinstance(value, list):
        return "array_artifact"
    return "machine_artifact"


def _extract_embedded_artifact(text: str) -> Any | None:
    raw = text.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
        except Exception:
            continue
        if _is_clean_artifact(parsed):
            return parsed
    segment = _balanced_segment(raw, "{", "}") or _balanced_segment(raw, "[", "]")
    if not segment:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(segment)
        except Exception:
            continue
        if _is_clean_artifact(parsed):
            return parsed
    return None


def _balanced_segment(text: str, left: str, right: str) -> str | None:
    start = text.find(left)
    if start < 0:
        return None
    depth = 0
    in_str = False
    quote = ""
    escaped = False
    for index, ch in enumerate(text[start:], start):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                in_str = False
            continue
        if ch in {"'", '"'}:
            in_str = True
            quote = ch
        elif ch == left:
            depth += 1
        elif ch == right:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def artifact_policy_from_config(data: dict[str, Any] | None) -> ArtifactPolicy:
    cfg = coerce_dict(data)
    evidence = coerce_dict(cfg.get("evidence"))
    merged = {**cfg, **evidence}
    if "machine_artifact_required" in merged and "machine_readable_required" not in merged:
        merged["machine_readable_required"] = merged.get("machine_artifact_required")
    return ArtifactPolicy.from_mapping(merged)


def _bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled", "required"}


__all__ = ["artifact_policy_from_config", "normalize_artifact"]
