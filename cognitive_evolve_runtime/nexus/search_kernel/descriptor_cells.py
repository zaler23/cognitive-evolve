"""Behavior descriptor cells for quality-diversity selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
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


__all__ = ["DescriptorCell", "descriptor_cell", "descriptor_cell_key"]
