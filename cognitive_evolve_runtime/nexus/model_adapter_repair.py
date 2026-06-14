"""Schema repair helpers for structured model responses."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

_PATCH_HEADER_RE = re.compile(r"^(?:diff --git a/([^ ]+) b/([^ ]+)|---\s+(.+)|\+\+\+\s+(.+))$")

def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, tuple | set):
        return [str(item) for item in value if item is not None]
    text = str(value).strip()
    return [text] if text else []

def _repair_objective_contract_response(data: dict[str, Any], payload: dict[str, Any], *, project: bool) -> dict[str, Any]:
    """Recover required objective-contract fields from request context.

    Real model backends occasionally return a shallow or wrapped contract and
    omit ``original_user_goal`` / ``normalized_goal``.  The immutable user goal
    is already present in the request payload, so these required fields can be
    restored deterministically before schema validation instead of aborting the
    Nexus run at startup.
    """

    contract_like = data
    for key in ("contract", "objective_contract", "result"):
        nested = data.get(key)
        if isinstance(nested, dict):
            contract_like = {**data, **nested}
            break

    repaired = dict(contract_like)
    user_goal = str(payload.get("user_goal") or "").strip()
    original = str(repaired.get("original_user_goal") or user_goal or repaired.get("normalized_goal") or "user goal").strip()
    normalized = str(repaired.get("normalized_goal") or original or user_goal or "user goal").strip()
    repaired["original_user_goal"] = original or "user goal"
    repaired["normalized_goal"] = " ".join(normalized.split()) or repaired["original_user_goal"]
    if repaired["normalized_goal"] != repaired["original_user_goal"]:
        metadata = dict(repaired.get("metadata") or {})
        metadata.setdefault(
            "goal_normalization_rewrite",
            {
                "original_user_goal_sha256": _text_sha256(repaired["original_user_goal"]),
                "normalized_goal_sha256": _text_sha256(repaired["normalized_goal"]),
                "policy": "original_user_goal_remains_frozen_contract_boundary",
            },
        )
        repaired["metadata"] = metadata

    common_list_fields = {
        "input_constraints",
        "allowed_evidence_sources",
        "disallowed_goal_mutations",
        "expected_output_forms",
        "verification_preferences",
        "success_dimensions",
        "failure_dimensions",
    }
    project_list_fields = {
        "frozen_regions",
        "mutable_regions",
        "contract_files",
        "implementation_files",
        "test_contracts",
        "allowed_patch_scope",
        "unsafe_change_patterns",
    }
    for field_name in common_list_fields | (project_list_fields if project else set()):
        if field_name in repaired:
            repaired[field_name] = _as_string_list(repaired.get(field_name))
    if project:
        repaired = _repair_project_contract_paths(repaired, payload)
    if "uncertainty_policy" in repaired and not isinstance(repaired["uncertainty_policy"], str):
        repaired["uncertainty_policy"] = str(repaired["uncertainty_policy"])
    if "outcome_policy" in repaired and not isinstance(repaired["outcome_policy"], dict):
        repaired.pop("outcome_policy", None)
    repaired = _repair_dynamic_artifact_contract(repaired, user_goal=user_goal)
    repaired = _repair_search_space_plan(repaired)
    return repaired

def _repair_search_space_plan(data: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(data)
    outcome_policy = repaired.get("outcome_policy") if isinstance(repaired.get("outcome_policy"), dict) else {}
    plan = repaired.get("search_space_plan")
    if not isinstance(plan, dict):
        plan = outcome_policy.get("search_space_plan") or outcome_policy.get("search_space")
    if isinstance(plan, dict):
        repaired["search_space_plan"] = dict(plan)
        policy = dict(outcome_policy)
        policy["search_space_plan"] = dict(plan)
        repaired["outcome_policy"] = policy
    return repaired

def _text_sha256(text: str) -> str:
    
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()

def _repair_dynamic_artifact_contract(data: dict[str, Any], *, user_goal: str) -> dict[str, Any]:
    """Ensure model-built objective contracts carry a non-domain DAC surface."""

    from cognitive_evolve_runtime.contracts.objective_contract import _default_dynamic_artifact_contract
    from cognitive_evolve_runtime.nexus.artifact_contract import DynamicArtifactContract

    repaired = dict(data)
    outcome_policy = repaired.get("outcome_policy") if isinstance(repaired.get("outcome_policy"), dict) else {}
    dac_value = repaired.get("dynamic_artifact_contract")
    if not isinstance(dac_value, dict):
        dac_value = outcome_policy.get("dynamic_artifact_contract")
    if isinstance(dac_value, dict):
        dac = DynamicArtifactContract.from_any(dac_value, fallback_objective=str(repaired.get("normalized_goal") or user_goal))
        if dac is not None:
            repaired["dynamic_artifact_contract"] = dac.to_dict()
    else:
        repaired["dynamic_artifact_contract"] = _default_dynamic_artifact_contract(str(repaired.get("normalized_goal") or user_goal))
    policy = dict(outcome_policy)
    policy.setdefault("model_driven", True)
    policy["dynamic_artifact_contract"] = repaired["dynamic_artifact_contract"]
    repaired["outcome_policy"] = policy
    return repaired

def _repair_project_contract_paths(repaired: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Constrain model-proposed project path lists to files in the snapshot.

    Model contracts sometimes narrow ``allowed_patch_scope`` to plausible-looking
    basenames such as ``nexus_proof_progress_hardening.py``.  Those names are
    useful intent signals, but letting them through as literal paths causes every
    later patch to fail with ``source_binding_missing_path``.  Keep valid paths
    and globs, resolve unique basenames, and fall back to the snapshot's real
    Python/test files when the model supplied only nonexistent targets.
    """

    manifest = _manifest_paths_from_snapshot(payload.get("snapshot"))
    if not manifest:
        return repaired
    py_files = [path for path in manifest if path.endswith(".py")]
    test_files = [path for path in py_files if path.startswith("tests/") or Path(path).name.startswith("test_")]
    impl_files = [path for path in py_files if path not in set(test_files)]

    notes: list[str] = []
    for field_name, fallback in {
        "mutable_regions": py_files,
        "implementation_files": impl_files,
        "test_contracts": test_files,
        "allowed_patch_scope": py_files + test_files,
    }.items():
        if field_name not in repaired:
            continue
        fixed, field_notes = _resolve_project_path_list(repaired.get(field_name), manifest, fallback=fallback)
        repaired[field_name] = fixed
        notes.extend(f"{field_name}:{note}" for note in field_notes)
    scope = list(repaired.get("allowed_patch_scope") or [])
    if scope and impl_files and not any(path in impl_files for path in scope):
        repaired["allowed_patch_scope"] = list(dict.fromkeys(impl_files + scope))
        notes.append("allowed_patch_scope:added_snapshot_implementation_files")
    if notes:
        repaired["path_repair_notes"] = notes[:20]
    return repaired

def _manifest_paths_from_snapshot(snapshot: Any) -> list[str]:
    manifest = getattr(snapshot, "file_manifest", None)
    if manifest is None and isinstance(snapshot, dict):
        manifest = snapshot.get("file_manifest") or snapshot.get("files")
    paths: list[str] = []
    for item in manifest or []:
        if isinstance(item, dict):
            path = str(item.get("path") or "").strip()
        else:
            path = str(item or "").strip()
        if path:
            paths.append(path.lstrip("./"))
    return list(dict.fromkeys(paths))

def _resolve_project_path_list(values: Any, manifest: list[str], *, fallback: list[str]) -> tuple[list[str], list[str]]:
    manifest_set = set(manifest)
    resolved: list[str] = []
    notes: list[str] = []
    for raw in _as_string_list(values):
        item = raw.strip().lstrip("./")
        if not item:
            continue
        if any(char in item for char in "*?["):
            resolved.append(item)
            continue
        if item in manifest_set:
            resolved.append(item)
            continue
        match = _unique_manifest_match(item, manifest)
        if match:
            resolved.append(match)
            notes.append(f"resolved {item} -> {match}")
        else:
            notes.append(f"dropped missing {item}")
    if not resolved and fallback:
        resolved = list(fallback)
        notes.append("fallback_to_snapshot_scope")
    return list(dict.fromkeys(resolved)), notes

def _unique_manifest_match(item: str, manifest: list[str]) -> str:
    name = Path(item).name
    basename_matches = [path for path in manifest if Path(path).name == name]
    if len(basename_matches) == 1:
        return basename_matches[0]
    suffix_matches = [path for path in manifest if "/" in item and path.endswith(item)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return ""

def _repair_array_response(data: dict[str, Any], *, target_key: str, aliases: tuple[str, ...]) -> dict[str, Any]:
    """Repair common model wrappers around list-valued schema responses."""

    repaired = dict(data)
    if isinstance(repaired.get(target_key), list):
        return repaired
    for key in aliases:
        value = repaired.get(key)
        if isinstance(value, list):
            repaired[target_key] = value
            return repaired
    for key in ("result", "response", "data"):
        nested = repaired.get(key)
        if isinstance(nested, dict):
            nested_repaired = _repair_array_response(nested, target_key=target_key, aliases=aliases)
            if isinstance(nested_repaired.get(target_key), list):
                return {**repaired, target_key: nested_repaired[target_key]}
    return repaired

def _repair_candidate_items(data: dict[str, Any], *, key: str) -> dict[str, Any]:
    repaired = dict(data)
    items = repaired.get(key)
    if not isinstance(items, list):
        return repaired
    fixed: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            fixed.append(item)
            continue
        candidate = dict(item)
        fallback_text = str(candidate.get("concise_claim") or candidate.get("core_mechanism") or "")
        candidate.setdefault("artifact", fallback_text)
        candidate.setdefault("assumptions", [])
        candidate.setdefault("missing_parts", [])
        candidate.setdefault("uncertainty_notes", [])
        repaired_fields = _repair_structured_candidate_contract_fields(candidate)
        if isinstance(candidate.get("search_space"), dict):
            metadata = candidate.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                candidate["metadata"] = metadata
            metadata.setdefault("search_space", dict(candidate.get("search_space") or {}))
        if repaired_fields:
            metadata = candidate.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                candidate["metadata"] = metadata
            metadata["schema_repair_fields"] = list(dict.fromkeys([*(metadata.get("schema_repair_fields") or []), *repaired_fields]))
            metadata["structured_output_fields"] = {
                "touched_files": candidate.get("touched_files", []),
                "evaluation_dimensions": candidate.get("evaluation_dimensions", []),
                "final_gate": candidate.get("final_gate", {}),
            }
        fixed.append(candidate)
    repaired[key] = fixed
    return repaired

def _repair_structured_candidate_contract_fields(candidate: dict[str, Any]) -> list[str]:
    repaired: list[str] = []
    touched_files = _as_string_list(candidate.get("touched_files"))
    source_bindings = [dict(item) for item in candidate.get("source_bindings", []) or [] if isinstance(item, dict)]
    for binding in source_bindings:
        path = str(binding.get("path") or binding.get("ref") or "").strip()
        if path:
            touched_files.append(path)
    artifact = candidate.get("artifact")
    if isinstance(artifact, dict):
        for key in ("path", "target_path", "file"):
            touched_files.extend(_as_string_list(artifact.get(key)))
        for patch_key in ("patch", "patch_content", "diff", "unified_diff", "content"):
            value = artifact.get(patch_key)
            if isinstance(value, str):
                touched_files.extend(_paths_from_patch_headers(value))
    touched_files = list(dict.fromkeys(path for path in touched_files if path and path != "/dev/null"))
    if not isinstance(candidate.get("touched_files"), list):
        candidate["touched_files"] = touched_files
        repaired.append("touched_files")
    if not source_bindings and touched_files:
        candidate["source_bindings"] = [{"path": path, "kind": "source_file", "source": "model_adapter_schema_repair"} for path in touched_files[:12]]
        repaired.append("source_bindings")
    else:
        candidate["source_bindings"] = source_bindings
    evidence_refs = candidate.get("evidence_refs")
    if not isinstance(evidence_refs, list):
        candidate["evidence_refs"] = []
        repaired.append("evidence_refs")
    else:
        candidate["evidence_refs"] = [dict(item) for item in evidence_refs if isinstance(item, dict)]
    if not isinstance(candidate.get("evaluation_dimensions"), list):
        score_keys = list((candidate.get("multihead_scores") or {}).keys()) if isinstance(candidate.get("multihead_scores"), dict) else []
        candidate["evaluation_dimensions"] = [str(item) for item in score_keys if item]
        repaired.append("evaluation_dimensions")
    else:
        candidate["evaluation_dimensions"] = _as_string_list(candidate.get("evaluation_dimensions"))
    if not isinstance(candidate.get("final_gate"), dict):
        candidate["final_gate"] = {}
        repaired.append("final_gate")
    return repaired

def _paths_from_patch_headers(text: str) -> list[str]:
    out: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        match = _PATCH_HEADER_RE.match(line)
        if not match:
            continue
        for value in match.groups():
            if not value:
                continue
            raw = value.split("\t", 1)[0].strip()
            normalized = _normalize_patch_path(raw)
            if normalized and normalized != "/dev/null":
                out.append(normalized)
    return list(dict.fromkeys(out))

def _normalize_patch_path(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text or text == "/dev/null":
        return text
    parts = [part for part in text.split("/") if part]
    if parts and parts[0] in {"a", "b"}:
        parts = parts[1:]
    if not parts or any(part in {"..", "."} for part in parts):
        return ""
    return "/".join(parts)

__all__ = [
    "_repair_array_response", "_repair_candidate_items", "_repair_objective_contract_response",
]
