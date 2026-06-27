"""Behavior descriptor cells for quality-diversity selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.project_candidate import PatchOperation
from cognitive_evolve_runtime.nexus._serde import coerce_dict
from .fingerprints import normalize_token


@dataclass(frozen=True)
class DescriptorCell:
    artifact_family: str
    mechanism_family: str
    evidence_state: str
    verifier_modality: str
    failure_class: str
    lineage_root: str

    def key(self) -> str:
        return ":".join([self.artifact_family, self.mechanism_family, self.evidence_state, self.verifier_modality, self.failure_class, self.lineage_root])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def descriptor_cell(candidate: CandidateGenome) -> DescriptorCell:
    metadata = coerce_dict(candidate.metadata)
    artifact_family = normalize_token(candidate.artifact_type or "answer") or "answer"
    mechanism_family = _first_non_empty(
        metadata.get("operator"),
        candidate.niche_memberships[0] if candidate.niche_memberships else "",
        candidate.novelty_descriptors[0] if candidate.novelty_descriptors else "",
        candidate.core_mechanism,
        "general",
    )
    evidence_state = _evidence_state(candidate)
    verifier_modality = _verifier_modality(candidate)
    failure_class = _failure_class(candidate)
    lineage_root = normalize_token(candidate.lineage[0] if candidate.lineage else candidate.id)[:24] or "root"
    return DescriptorCell(
        artifact_family=artifact_family,
        mechanism_family=normalize_token(mechanism_family)[:32] or "general",
        evidence_state=evidence_state,
        verifier_modality=verifier_modality,
        failure_class=failure_class,
        lineage_root=lineage_root,
    )


def descriptor_cell_key(candidate: CandidateGenome) -> str:
    return descriptor_cell(candidate).key()


def behavior_descriptor(candidate: CandidateGenome) -> tuple[str, ...]:
    """Grounded behavior descriptor for novelty and QD pressure."""

    metadata = coerce_dict(candidate.metadata)
    patch_result = coerce_dict(getattr(candidate, "patch_application_result", {}) or metadata.get("patch_result"))
    search_space = coerce_dict(getattr(candidate, "search_space", None) or metadata.get("search_space"))
    family = _first_non_empty(
        search_space.get("family_id"),
        search_space.get("plane_id"),
        candidate.niche_memberships[0] if candidate.niche_memberships else "",
        candidate.novelty_descriptors[0] if candidate.novelty_descriptors else "",
        candidate.core_mechanism,
        candidate.artifact_type,
        "general",
    )
    paths = _candidate_paths(candidate, patch_result=patch_result, metadata=metadata)
    applied = coerce_dict(patch_result).get("status") == "applied"
    return tuple(
        item
        for item in (
            normalize_token(family)[:40] or "general",
            _path_bucket(paths.get("source", []), prefix="src"),
            _path_bucket(paths.get("patch", []), prefix="patch"),
            _path_bucket(paths.get("target", []), prefix="target"),
            "applied" if applied else "not_applied",
        )
        if item
    )


def _candidate_paths(candidate: CandidateGenome, *, patch_result: dict[str, Any], metadata: dict[str, Any]) -> dict[str, list[str]]:
    source: list[str] = []
    patch: list[str] = []
    target: list[str] = []
    for binding in candidate.source_bindings:
        if not isinstance(binding, dict):
            continue
        path = _path_token(binding.get("path") or binding.get("file") or binding.get("source_path"))
        if path:
            source.append(path)
    for op in getattr(candidate, "patch_set", []) or []:
        path = _path_token(op.path if isinstance(op, PatchOperation) else coerce_dict(op).get("path"))
        if path:
            patch.append(path)
    for value in getattr(candidate, "touched_files", []) or []:
        path = _path_token(value)
        if path:
            patch.append(path)
    for value in patch_result.get("applied_files") or []:
        path = _path_token(value)
        if path:
            patch.append(path)
    for key in ("target_files", "target_paths", "affected_files"):
        for value in metadata.get(key) or []:
            path = _path_token(value)
            if path:
                target.append(path)
    repair = coerce_dict(metadata.get("repair_required") or metadata.get("repair_seed"))
    for value in repair.get("target_files") or repair.get("paths") or []:
        path = _path_token(value)
        if path:
            target.append(path)
    return {"source": list(dict.fromkeys(source))[:4], "patch": list(dict.fromkeys(patch))[:4], "target": list(dict.fromkeys(target))[:4]}


def _path_bucket(paths: list[str], *, prefix: str) -> str:
    if not paths:
        return f"{prefix}:none"
    return f"{prefix}:" + "+".join(normalize_token(path)[-48:] or "path" for path in paths[:3])


def _path_token(value: Any) -> str:
    text = str(value or "").strip().lstrip("./")
    return text if text and ".." not in text.split("/") else ""


def _evidence_state(candidate: CandidateGenome) -> str:
    if candidate.formal_artifacts:
        return "formal"
    if candidate.verification_trace:
        if any(item.get("passed") is True for item in candidate.verification_trace if isinstance(item, dict)):
            return "verified"
        return "checked"
    if candidate.evidence_delta or candidate.evidence_refs or candidate.source_bindings:
        return "grounded"
    return "proposal"


def _verifier_modality(candidate: CandidateGenome) -> str:
    for item in reversed(candidate.verification_trace or []):
        if not isinstance(item, dict):
            continue
        modality = normalize_token(item.get("modality") or item.get("oracle_kind") or item.get("verifier_kind"))
        if modality:
            return modality[:24]
    return "none"


def _failure_class(candidate: CandidateGenome) -> str:
    metadata = coerce_dict(candidate.metadata)
    for key in ("failure_class", "hard_reject_class", "terminal_reject_reason", "hard_reject_reason"):
        token = normalize_token(metadata.get(key))
        if token:
            return token[:24]
    fc = coerce_dict(metadata.get("failure_classification"))
    for key in ("class", "kind", "reason"):
        token = normalize_token(fc.get(key))
        if token:
            return token[:24]
    if candidate.failure_lessons:
        return "has_failure"
    return "none"


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return "general"


__all__ = ["DescriptorCell", "behavior_descriptor", "descriptor_cell", "descriptor_cell_key"]
