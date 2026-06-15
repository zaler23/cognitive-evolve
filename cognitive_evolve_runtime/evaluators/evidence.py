"""Progressive evidence primitives for Nexus evolution.

The evidence kernel is intentionally domain-neutral: adapters may describe
sorting networks, patches, prompts, workflows, or mathematical artifacts, but the
runtime only consumes artifact cleanliness, challenge cases, repair value, and
final-certification state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict, utc_now
from cognitive_evolve_runtime.core.scalars import bounded_score

EVIDENCE_LEVELS = ("L0", "L1", "L2", "L3", "L4")
ARTIFACT_STATUSES = {"clean", "refolded", "malformed", "absent"}


@dataclass(frozen=True)
class ArtifactView:
    artifact_type: str = ""
    artifact: Any = None
    normalized_artifact: Any | None = None
    status: str = "absent"
    schema_cleanliness: float = 0.0
    probe_eligible: bool = False
    final_eligible: bool = False
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ArtifactView | None":
        if not isinstance(data, dict):
            return None
        status = str(data.get("status") or "absent").strip().lower()
        if status not in ARTIFACT_STATUSES:
            status = "absent"
        return cls(
            artifact_type=str(data.get("artifact_type") or ""),
            artifact=data.get("artifact"),
            normalized_artifact=data.get("normalized_artifact"),
            status=status,
            schema_cleanliness=bounded_score(data.get("schema_cleanliness", 0.0)),
            probe_eligible=bool(data.get("probe_eligible")),
            final_eligible=bool(data.get("final_eligible")),
            diagnostics=[str(item) for item in data.get("diagnostics", []) if item],
        )


@dataclass(frozen=True)
class ChallengeCase:
    id: str
    domain_id: str = "general"
    kind: str = "evaluator_failure"
    payload: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    first_seen_round: int = 0
    last_seen_round: int = 0
    kill_count: int = 1
    frontier_kill_count: int = 0
    elite_kill_count: int = 0
    resolved_by_candidate_ids: list[str] = field(default_factory=list)
    region_ids: list[str] = field(default_factory=list)
    lineage_ids: list[str] = field(default_factory=list)
    priority: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ChallengeCase | None":
        if not isinstance(data, dict):
            return None
        return cls(
            id=str(data.get("id") or ""),
            domain_id=str(data.get("domain_id") or "general"),
            kind=str(data.get("kind") or "evaluator_failure"),
            payload=coerce_dict(data.get("payload")),
            summary=str(data.get("summary") or ""),
            first_seen_round=_int(data.get("first_seen_round"), 0),
            last_seen_round=_int(data.get("last_seen_round"), 0),
            kill_count=max(0, _int(data.get("kill_count"), 0)),
            frontier_kill_count=max(0, _int(data.get("frontier_kill_count"), 0)),
            elite_kill_count=max(0, _int(data.get("elite_kill_count"), 0)),
            resolved_by_candidate_ids=[str(item) for item in data.get("resolved_by_candidate_ids", []) if item],
            region_ids=[str(item) for item in data.get("region_ids", []) if item],
            lineage_ids=[str(item) for item in data.get("lineage_ids", []) if item],
            priority=bounded_score(data.get("priority", 0.5)),
            metadata=coerce_dict(data.get("metadata")),
        )


@dataclass(frozen=True)
class EvidenceResult:
    candidate_id: str
    domain_id: str = "general"
    level: str = "L2"
    status: str = "unknown"
    passed: bool = False
    hard_reject: bool = False
    final_eligible: bool = False
    score: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    challenge_cases: list[ChallengeCase] = field(default_factory=list)
    resolved_challenge_ids: list[str] = field(default_factory=list)
    repair_hints: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    artifact_view: ArtifactView | None = None
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["challenge_cases"] = [case.to_dict() for case in self.challenge_cases]
        payload["artifact_view"] = self.artifact_view.to_dict() if self.artifact_view is not None else None
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EvidenceResult | None":
        if not isinstance(data, dict):
            return None
        level = str(data.get("level") or "L2").strip().upper()
        if level not in EVIDENCE_LEVELS:
            level = "L2"
        return cls(
            candidate_id=str(data.get("candidate_id") or ""),
            domain_id=str(data.get("domain_id") or "general"),
            level=level,
            status=str(data.get("status") or "unknown"),
            passed=bool(data.get("passed")),
            hard_reject=bool(data.get("hard_reject")),
            final_eligible=bool(data.get("final_eligible")),
            score=bounded_score(data.get("score", 0.0)),
            metrics=coerce_dict(data.get("metrics")),
            challenge_cases=[case for case in (ChallengeCase.from_dict(item) for item in data.get("challenge_cases", [])) if case is not None],
            resolved_challenge_ids=[str(item) for item in data.get("resolved_challenge_ids", []) if item],
            repair_hints=[str(item) for item in data.get("repair_hints", []) if item],
            diagnostics=[str(item) for item in data.get("diagnostics", []) if item],
            artifact_view=ArtifactView.from_dict(data.get("artifact_view")),
            created_at=str(data.get("created_at") or utc_now()),
        )


def apply_evidence_result(candidate: Any, result: EvidenceResult) -> None:
    """Write domain-neutral evidence back to a CandidateGenome-like object."""

    metadata = candidate.metadata if isinstance(getattr(candidate, "metadata", None), dict) else {}
    metadata["progressive_evidence"] = result.to_dict()
    metadata["challenge_failures"] = [case.to_dict() for case in result.challenge_cases]
    metadata["terminal_failure"] = bool(result.hard_reject)
    metadata["repair_value"] = repair_value_from_evidence(result)
    candidate.metadata = metadata

    scores = candidate.multihead_scores if isinstance(getattr(candidate, "multihead_scores", None), dict) else {}
    scores["frontier_score"] = result.score
    scores["challenge_pass_rate"] = bounded_score(result.metrics.get("challenge_pass_rate", 1.0 if result.passed else 0.0))
    scores["schema_cleanliness"] = result.artifact_view.schema_cleanliness if result.artifact_view is not None else bounded_score(result.metrics.get("schema_cleanliness", 0.0))
    scores["evaluator_score"] = bounded_score(result.metrics.get("score", result.score))
    scores["final_verification"] = 1.0 if result.level == "L4" and result.passed and result.final_eligible else 0.0
    scores["repair_value"] = metadata["repair_value"]
    candidate.multihead_scores = scores


def repair_value_from_evidence(result: EvidenceResult) -> float:
    if result.hard_reject:
        return 0.0
    if result.passed:
        return 0.2 if result.level != "L4" else 0.0
    challenge_value = min(1.0, 0.2 + 0.15 * len(result.challenge_cases) + 0.1 * len(result.repair_hints))
    return bounded_score(max(challenge_value, result.score * 0.6))


def progressive_evidence(candidate: Any) -> EvidenceResult | None:
    metadata = getattr(candidate, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    return EvidenceResult.from_dict(metadata.get("progressive_evidence"))


def progressive_evidence_blocks_final(candidate: Any) -> bool:
    evidence = progressive_evidence(candidate)
    if evidence is None:
        return False
    return evidence.hard_reject or not evidence.final_eligible or not (evidence.level == "L4" and evidence.passed)


def progressive_evidence_blocks_parent(candidate: Any) -> bool:
    evidence = progressive_evidence(candidate)
    if evidence is None:
        return False
    if evidence.hard_reject:
        return True
    if evidence.artifact_view is not None and evidence.artifact_view.status in {"malformed", "absent"} and not has_repair_value(candidate):
        return True
    return False


def has_repair_value(candidate: Any) -> bool:
    metadata = getattr(candidate, "metadata", None)
    if isinstance(metadata, dict):
        try:
            return float(metadata.get("repair_value", 0.0) or 0.0) > 0.0
        except (TypeError, ValueError):
            return False
    return False


def evidence_advisory_features(candidates: list[Any]) -> dict[str, Any]:
    """Build ParentSelector-compatible advisory objects from evidence state."""

    out: dict[str, Any] = {}
    for candidate in candidates:
        evidence = progressive_evidence(candidate)
        if evidence is None:
            continue
        risk = 1.0 if evidence.hard_reject else (0.3 if not evidence.final_eligible and not has_repair_value(candidate) else 0.0)
        out[getattr(candidate, "id", "")] = {
            "rank_prior": bounded_score(evidence.score),
            "plan_value": bounded_score(getattr(candidate, "metadata", {}).get("repair_value", 0.0) if isinstance(getattr(candidate, "metadata", None), dict) else 0.0),
            "diversity": bounded_score(getattr(candidate, "multihead_scores", {}).get("novelty", 0.0) if isinstance(getattr(candidate, "multihead_scores", None), dict) else 0.0),
            "risk": bounded_score(risk),
        }
    return out


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "ARTIFACT_STATUSES",
    "EVIDENCE_LEVELS",
    "ArtifactView",
    "ChallengeCase",
    "EvidenceResult",
    "apply_evidence_result",
    "evidence_advisory_features",
    "has_repair_value",
    "progressive_evidence",
    "progressive_evidence_blocks_final",
    "progressive_evidence_blocks_parent",
    "repair_value_from_evidence",
]
