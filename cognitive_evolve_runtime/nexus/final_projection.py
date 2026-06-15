"""Clean user-facing final projection for Nexus runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.evaluators.evidence import evidence_state, latest_evidence_record
from cognitive_evolve_runtime.evaluators.registry import get_adapter
from cognitive_evolve_runtime.core.scalars import bounded_score


@dataclass(frozen=True)
class FinalProjection:
    status: str
    candidate_id: str = ""
    title: str = ""
    artifact: Any = None
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    blocking_issues: list[str] = field(default_factory=list)
    continuation_plan: list[str] = field(default_factory=list)
    objective_solved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [f"# CognitiveEvolve projection: {self.status}", ""]
        if self.candidate_id:
            lines.append(f"- candidate_id: {self.candidate_id}")
        lines.append(f"- objective_solved: {str(self.objective_solved).lower()}")
        lines.append("")
        if self.title:
            lines.extend([self.title, ""])
        if self.artifact not in (None, ""):
            lines.extend(["## Current artifact", "", _render_artifact(self.artifact), ""])
        if self.evidence_summary:
            lines.extend(["## Evidence summary", ""])
            for key, value in self.evidence_summary.items():
                lines.append(f"- {key}: `{value}`")
            lines.append("")
        if self.blocking_issues:
            lines.extend(["## Blocking issues", ""])
            lines.extend(f"- {item}" for item in self.blocking_issues[:12])
            lines.append("")
        if self.continuation_plan:
            lines.extend(["## Continuation plan", ""])
            lines.extend(f"- {item}" for item in self.continuation_plan[:8])
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def build_final_projection(*, population: CandidatePopulation, synthesis: Any, final_certificate: dict[str, Any] | None = None) -> FinalProjection:
    certificate = dict(final_certificate or getattr(synthesis, "closure_certificate", {}) or {})
    solved = bool(certificate.get("objective_solved") or getattr(synthesis, "objective_solved", False))
    if solved:
        candidate = _candidate_by_id(population.candidates, str(certificate.get("candidate_id") or getattr(synthesis, "best_candidate_id", "")))
        return _projection_for_candidate(candidate, status="solved", objective_solved=True, certificate=certificate)
    best = _best_current_candidate(population.candidates)
    if best is not None:
        return _projection_for_candidate(best, status="best_current", objective_solved=False, certificate=certificate)
    return FinalProjection(
        status="no_candidate",
        title="No displayable candidate was available.",
        blocking_issues=[str(item) for item in certificate.get("blocking_reasons", [])] or ["no_candidate"],
        continuation_plan=["resume evolution with broader seeding or refined contract"],
        objective_solved=False,
    )


def _projection_for_candidate(candidate: CandidateGenome | None, *, status: str, objective_solved: bool, certificate: dict[str, Any]) -> FinalProjection:
    if candidate is None:
        return FinalProjection(status="no_candidate", objective_solved=False, blocking_issues=["candidate_not_found"])
    evidence = latest_evidence_record(candidate)
    state = evidence_state(candidate)
    evidence_summary = evidence.to_dict() if evidence is not None else state
    artifact_state = evidence.metadata.get("artifact_state", {}) if evidence is not None and isinstance(evidence.metadata, dict) else {}
    artifact = artifact_state.get("normalized_artifact") if isinstance(artifact_state, dict) and artifact_state.get("normalized_artifact") is not None else candidate.artifact
    adapter = get_adapter(evidence.source if evidence is not None else None)
    rendered_artifact = adapter.project_artifact_for_user(artifact, evidence=evidence_summary)
    blocking = [str(item) for item in certificate.get("blocking_reasons", []) if item]
    if evidence is not None:
        for challenge in (evidence.metadata.get("challenge_items", []) if isinstance(evidence.metadata, dict) else [])[:8]:
            if isinstance(challenge, dict) and challenge.get("summary"):
                blocking.append(str(challenge.get("summary")))
        continuation = list(evidence.hints[:6]) or ["continue challenge-guided repair"]
    else:
        continuation = ["collect evidence for this candidate"]
    return FinalProjection(
        status=status,
        candidate_id=candidate.id,
        title="Final artifact is certified." if objective_solved else "Best current artifact; not certified as solved.",
        artifact=rendered_artifact,
        evidence_summary={
            "stage": evidence.stage if evidence is not None else "none",
            "source": evidence.source if evidence is not None else "none",
            "score": round(float(evidence.score), 4) if evidence is not None else 0.0,
            "final_blocked": bool(state.get("final_blocked", True)),
        },
        blocking_issues=blocking or (["not_final_certified"] if not objective_solved else []),
        continuation_plan=continuation,
        objective_solved=objective_solved,
    )


def _best_current_candidate(candidates: list[CandidateGenome]) -> CandidateGenome | None:
    eligible = [candidate for candidate in candidates if not _hard_rejected(candidate)]
    if not eligible:
        return candidates[0] if candidates else None
    return max(eligible, key=_best_current_score)


def _best_current_score(candidate: CandidateGenome) -> float:
    scores = candidate.multihead_scores or {}
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    hard_penalty = 1.0 if bool(metadata.get("terminal_failure")) else 0.0
    compactness = 1.0 - min(1.0, len(str(candidate.artifact or "")) / 12000.0)
    continuation = bounded_score(metadata.get("repair_value", 0.0))
    return (
        0.30 * bounded_score(scores.get("frontier_score", 0.0))
        + 0.20 * bounded_score(scores.get("challenge_pass_rate", 0.0))
        + 0.15 * bounded_score(scores.get("evidence_progress", 0.0))
        + 0.10 * bounded_score(scores.get("schema_cleanliness", 0.0))
        + 0.10 * bounded_score(scores.get("novelty", 0.0))
        + 0.10 * compactness
        + 0.05 * continuation
        - hard_penalty
    )


def _candidate_by_id(candidates: list[CandidateGenome], candidate_id: str) -> CandidateGenome | None:
    for candidate in candidates:
        if candidate.id == candidate_id:
            return candidate
    return None


def _hard_rejected(candidate: CandidateGenome) -> bool:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    return bool(metadata.get("terminal_failure"))


def _render_artifact(artifact: Any) -> str:
    if isinstance(artifact, str):
        return artifact
    return "```json\n" + __import__("json").dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True, default=str)[:12000] + "\n```"


__all__ = ["FinalProjection", "build_final_projection"]
