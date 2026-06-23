"""Public archive data contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.core.serialization import coerce_dict, coerce_str_list, stable_hash, utc_now

@dataclass
class FateAssignment:
    candidate_id: str
    fate: str
    archive_targets: list[str] = field(default_factory=list)
    failure_signature: str = ""
    inherited_gene_summary: str = ""
    covered_by: str = ""
    future_reactivation_condition: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FateAssignment":
        return cls(
            candidate_id=str(data.get("candidate_id") or ""),
            fate=CandidateFate.normalize(data.get("fate")),
            archive_targets=coerce_str_list(data.get("archive_targets")),
            failure_signature=str(data.get("failure_signature") or ""),
            inherited_gene_summary=str(data.get("inherited_gene_summary") or ""),
            covered_by=str(data.get("covered_by") or ""),
            future_reactivation_condition=str(data.get("future_reactivation_condition") or ""),
        )

@dataclass
class ArchiveConstraintRecord:
    """Durable constraint distilled from archive failures and repeated lessons."""

    id: str
    kind: str
    rule: str
    target: str = ""
    source_candidate_id: str = ""
    severity: str = "warning"
    created_at: str = field(default_factory=utc_now)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchiveConstraintRecord":
        return cls(
            id=str(data.get("id") or stable_hash(data)[:16]),
            kind=str(data.get("kind") or "verification_constraint"),
            rule=str(data.get("rule") or ""),
            target=str(data.get("target") or ""),
            source_candidate_id=str(data.get("source_candidate_id") or ""),
            severity=str(data.get("severity") or "warning"),
            created_at=str(data.get("created_at") or utc_now()),
            evidence=coerce_dict(data.get("evidence")),
        )

@dataclass
class TerminalCandidateTombstone:
    """Lightweight durable record for a candidate removed from live population."""

    candidate_id: str
    fate: str
    lineage_root: str = ""
    parent_ids: list[str] = field(default_factory=list)
    generation: int = 0
    niche_key: str = ""
    score_summary: dict[str, float] = field(default_factory=dict)
    verification_summary_hash: str = ""
    failure_signature: str = ""
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_candidate(cls, candidate: CandidateGenome, *, fate: str, failure_signature: str = "") -> "TerminalCandidateTombstone":
        from cognitive_evolve_runtime.archives.constraints import verification_failure_signature as _verification_failure_signature
        score_axes = [
            "objective_alignment",
            "answer_likelihood",
            "verifiability",
            "tool_progress",
            "proof_progress",
            "evidence_progress",
            "rarity",
            "novelty",
        ]
        score_summary = {axis: float(candidate.multihead_scores.get(axis, 0.0)) for axis in score_axes if axis in candidate.multihead_scores}
        lineage_root = candidate.lineage[0] if candidate.lineage else candidate.id
        niche = (candidate.niche_memberships[0] if candidate.niche_memberships else "") or candidate.core_mechanism or candidate.concise_claim or candidate.id
        verification_payload = {
            "verification_result": candidate.verification_result,
            "verification_trace": candidate.verification_trace[-5:],
            "failure_lessons": candidate.failure_lessons[:5],
        }
        return cls(
            candidate_id=candidate.id,
            fate=CandidateFate.normalize(fate),
            lineage_root=str(lineage_root),
            parent_ids=list(candidate.parent_ids),
            generation=int(candidate.generation or 0),
            niche_key=str(niche)[:240],
            score_summary=score_summary,
            verification_summary_hash=stable_hash(verification_payload),
            failure_signature=str(failure_signature or _verification_failure_signature(candidate) or "; ".join(candidate.failure_lessons[:3]))[:500],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TerminalCandidateTombstone":
        return cls(
            candidate_id=str(data.get("candidate_id") or ""),
            fate=CandidateFate.normalize(data.get("fate")),
            lineage_root=str(data.get("lineage_root") or ""),
            parent_ids=coerce_str_list(data.get("parent_ids")),
            generation=int(data.get("generation") or 0),
            niche_key=str(data.get("niche_key") or ""),
            score_summary={str(k): float(v) for k, v in coerce_dict(data.get("score_summary")).items() if isinstance(v, (int, float))},
            verification_summary_hash=str(data.get("verification_summary_hash") or ""),
            failure_signature=str(data.get("failure_signature") or ""),
            created_at=str(data.get("created_at") or utc_now()),
        )

__all__ = ["ArchiveConstraintRecord", "FateAssignment", "TerminalCandidateTombstone"]
