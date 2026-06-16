"""Machine artifact normalization for the Evidence Control Plane."""
from __future__ import annotations

import ast
from copy import deepcopy
import json
import re
from typing import Any

from cognitive_evolve_runtime.evaluators.evidence import ArtifactPolicy
from cognitive_evolve_runtime.nexus._serde import coerce_dict
from cognitive_evolve_runtime.core.scalars import bounded_score


DEFAULT_FORBIDDEN_SEMANTIC_TERMS = (
    "objective_contract",
    "fallback_route",
    "route_profile",
    "semantic_intake",
    "nexus",
    "cogev",
    "runtime_state",
    "runtime-state",
    "checkpoint",
    "hardware_latency",
    "recovery_window",
    "state_integrity",
    "crc64",
)


def normalize_artifact(candidate: Any, *, artifact_type: str = "", policy: ArtifactPolicy | None = None, machine_artifact_required: bool | None = None) -> dict[str, Any]:
    policy = policy or ArtifactPolicy(machine_readable_required=bool(machine_artifact_required))
    declared_type = str(artifact_type or policy.artifact_type or getattr(candidate, "artifact_type", "") or "").strip()
    artifact = getattr(candidate, "artifact", None)
    if _is_clean_artifact(artifact):
        return _machine_artifact_view(
            original_artifact=artifact,
            parsed_artifact=artifact,
            declared_type=declared_type,
            policy=policy,
            forced_refolded=False,
            base_diagnostics=[],
        )
    if isinstance(artifact, str) and artifact.strip():
        refolded = _extract_embedded_artifact(artifact)
        if refolded is not None:
            return _machine_artifact_view(
                original_artifact=artifact,
                parsed_artifact=refolded,
                declared_type=declared_type,
                policy=policy,
                forced_refolded=True,
                base_diagnostics=["artifact_refolded_from_natural_language"],
            )
        return {
            "artifact_type": declared_type,
            "artifact": artifact,
            "normalized_artifact": None,
            "status": "malformed" if policy.machine_readable_required else "refolded",
            "schema_cleanliness": 0.2 if policy.machine_readable_required else 0.35,
            "probe_eligible": bool(policy.allow_text_fallback and not policy.machine_readable_required),
            "final_eligible": bool(policy.allow_text_fallback and not policy.machine_readable_required and policy.allow_refold_for_final),
            "diagnostics": ["machine_parse_failure", "artifact_not_machine_parseable"] if policy.machine_readable_required else ["artifact_not_machine_parseable"],
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


def semantic_drift_diagnostics(artifact_view: dict[str, Any], policy: ArtifactPolicy) -> list[str]:
    """Detect generic runtime/internal semantic drift in machine artifacts.

    This is intentionally policy-driven and domain-neutral: callers may provide
    allowed or forbidden terms through ``ArtifactPolicy.metadata``.  The default
    forbidden terms only cover runtime/control-plane vocabulary that should not
    become part of a user-facing machine artifact unless explicitly allowed.
    """

    if not policy.machine_readable_required:
        return []
    artifact = artifact_view.get("normalized_artifact")
    if artifact is None:
        artifact = artifact_view.get("artifact")
    if artifact in (None, ""):
        return []
    metadata = coerce_dict(policy.metadata)
    allowed_terms = set(_normalized_terms(metadata.get("domain_vocabulary")))
    allowed_terms.update(_normalized_terms(metadata.get("allowed_domain_terms")))
    allowed_terms.update(_normalized_terms(metadata.get("allowed_terms")))
    allowed_terms.update(_tokens(policy.artifact_type))
    for field in policy.required_fields:
        allowed_terms.update(_tokens(field))
    forbidden = _normalized_terms(metadata.get("forbidden_semantic_terms") or metadata.get("semantic_drift_terms"))
    if not forbidden:
        forbidden = list(DEFAULT_FORBIDDEN_SEMANTIC_TERMS)
    text = _artifact_text(artifact).lower()
    diagnostics: list[str] = []
    for term in forbidden:
        normalized = str(term or "").strip().lower()
        if not normalized:
            continue
        term_tokens = set(_tokens(normalized))
        if term_tokens and term_tokens.issubset(allowed_terms):
            continue
        if normalized in text:
            diagnostics.append(f"semantic_drift_detected: forbidden_term={normalized}")
        if len(diagnostics) >= 6:
            break
    return diagnostics


def _machine_artifact_view(
    *,
    original_artifact: Any,
    parsed_artifact: Any,
    declared_type: str,
    policy: ArtifactPolicy,
    forced_refolded: bool,
    base_diagnostics: list[str],
) -> dict[str, Any]:
    normalized = deepcopy(parsed_artifact)
    diagnostics = list(base_diagnostics)
    canonical_type, type_refolded, type_ok = _canonical_artifact_type(declared_type, normalized, policy, diagnostics)
    field_refolded = False
    if isinstance(normalized, dict):
        field_refolded = _apply_field_aliases(normalized, policy.field_aliases, diagnostics)
    missing = _missing_required_fields(normalized, policy.required_fields)
    if missing:
        diagnostics.append("missing_required_fields: " + ", ".join(missing))
    changed = forced_refolded or type_refolded or field_refolded
    if missing or not type_ok:
        status = "malformed"
    elif changed:
        status = "refolded"
    else:
        status = "clean"
    final_eligible = status == "clean"
    if status == "refolded":
        final_eligible = bool(policy.allow_refold_for_final and not policy.final_requires_clean_schema)
    if status == "malformed":
        final_eligible = False
    if policy.final_requires_clean_schema and status != "clean":
        diagnostics.append("final_requires_clean_schema")
        final_eligible = False
    probe_eligible = status == "clean" or (status == "refolded" and bool(policy.allow_refold_for_probe))
    cleanliness = _schema_cleanliness(status=status, parsed_artifact=normalized, missing_count=len(missing), type_ok=type_ok)
    return {
        "artifact_type": canonical_type,
        "artifact": original_artifact,
        "normalized_artifact": normalized if status in {"clean", "refolded"} else None,
        "status": status,
        "schema_cleanliness": cleanliness,
        "probe_eligible": probe_eligible,
        "final_eligible": final_eligible,
        "diagnostics": list(dict.fromkeys(diagnostics)),
        "missing_required_fields": missing,
        "normalization_applied": bool(changed),
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


def _canonical_artifact_type(declared_type: str, artifact: Any, policy: ArtifactPolicy, diagnostics: list[str]) -> tuple[str, bool, bool]:
    declared = str(declared_type or "").strip()
    expected = str(policy.artifact_type or "").strip()
    aliases = {str(k): str(v) for k, v in (policy.artifact_type_aliases or {}).items()}
    inferred = declared or _infer_type(artifact)
    if declared in aliases:
        canonical = aliases[declared]
        diagnostics.append(f"artifact_type_alias_normalized: {declared} -> {canonical}")
        return canonical, True, not expected or canonical == expected
    if expected and declared and declared != expected:
        diagnostics.append(f"artifact_type_mismatch: expected {expected} got {declared}")
        return declared, False, False
    return expected or inferred, False, True


def _apply_field_aliases(value: dict[str, Any], aliases: dict[str, str], diagnostics: list[str]) -> bool:
    changed = False
    for alias, canonical in (aliases or {}).items():
        alias = str(alias or "").strip()
        canonical = str(canonical or "").strip()
        if not alias or not canonical or alias not in value:
            continue
        if canonical not in value:
            value[canonical] = value.pop(alias)
        else:
            value.pop(alias, None)
        diagnostics.append(f"field_alias_normalized: {alias} -> {canonical}")
        changed = True
    return changed


def _missing_required_fields(value: Any, required_fields: list[str]) -> list[str]:
    if not required_fields:
        return []
    if not isinstance(value, dict):
        return list(required_fields)
    return [field for field in required_fields if field not in value]


def _schema_cleanliness(*, status: str, parsed_artifact: Any, missing_count: int, type_ok: bool) -> float:
    if status == "clean":
        return 1.0
    if status == "refolded":
        return 0.85
    base = 0.45 if isinstance(parsed_artifact, dict) else 0.2
    if not type_ok:
        base -= 0.15
    base -= min(0.3, 0.08 * missing_count)
    return bounded_score(base)


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


def _artifact_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)


def _normalized_terms(value: Any) -> list[str]:
    if isinstance(value, str):
        items = re.split(r"[,;\n]+", value)
    elif isinstance(value, (list, tuple, set)):
        items = [str(item) for item in value]
    else:
        items = []
    return [str(item).strip().lower() for item in items if str(item or "").strip()]


def _tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", str(value or "").lower()) if token}


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


__all__ = ["artifact_policy_from_config", "normalize_artifact", "semantic_drift_diagnostics"]
