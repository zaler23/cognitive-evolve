"""Clean user-facing final projection for Nexus runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.evaluators.evidence import evidence_state, latest_evidence_record
from cognitive_evolve_runtime.evaluators.registry import get_adapter
from cognitive_evolve_runtime.core.scalars import bounded_score
from cognitive_evolve_runtime.nexus.display_selection import select_displayed_candidate
from cognitive_evolve_runtime.nexus.nextgen import (
    best_current_direction_payload,
    candidate_verification_status,
    select_best_current_direction,
    structurally_blocked,
)
from cognitive_evolve_runtime.verification.types import GradedOutput


@dataclass(frozen=True)
class FinalProjection:
    status: str
    candidate_id: str = ""
    title: str = ""
    artifact_type: str = ""
    artifact: Any = None
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    blocking_issues: list[str] = field(default_factory=list)
    continuation_plan: list[str] = field(default_factory=list)
    advisory_issues: list[str] = field(default_factory=list)
    objective_solved: bool = False
    best_current_direction: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [f"# CognitiveEvolve projection: {self.status}", ""]
        if self.candidate_id:
            lines.append(f"- candidate_id: {self.candidate_id}")
        if self.artifact_type:
            lines.append(f"- artifact_type: {self.artifact_type}")
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
        if self.best_current_direction:
            lines.extend(["## Best current direction", ""])
            for key, value in self.best_current_direction.items():
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
        if self.advisory_issues:
            lines.extend(["## Advisory issues", ""])
            lines.extend(f"- {item}" for item in self.advisory_issues[:12])
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def build_final_projection(*, population: CandidatePopulation, synthesis: Any, graded_output: GradedOutput, final_certificate: dict[str, Any] | None = None, display_context: dict[str, Any] | None = None, contract: Any | None = None) -> FinalProjection:
    certificate = dict(final_certificate or getattr(synthesis, "closure_certificate", {}) or {})
    closure_certificate = getattr(synthesis, "closure_certificate", {}) if isinstance(getattr(synthesis, "closure_certificate", {}), dict) else {}
    if display_context is None and isinstance(certificate.get("display_context"), dict):
        display_context = certificate.get("display_context")
    if display_context is None and isinstance(closure_certificate.get("display_context"), dict):
        display_context = closure_certificate.get("display_context")
    answer_text = str(getattr(synthesis, "final_answer", "") or "").strip()
    candidate = _candidate_by_id(population.candidates, _projection_candidate_id(certificate, synthesis))
    candidate = _unwrap_best_current_carrier(candidate, population.candidates)
    if _projection_candidate_answer_eligible(candidate):
        return _projection_for_candidate(candidate, status="completed", objective_solved=False, certificate=certificate, answer_text=answer_text, contract=contract, graded_output=graded_output)
    if answer_text and candidate is not None:
        certificate.setdefault("blocking_reasons", [])
        if isinstance(certificate["blocking_reasons"], list):
            certificate["blocking_reasons"].append("selected_candidate_structural_or_empty")
    if display_context:
        selection = select_displayed_candidate(
            display_context,
            candidates=population.candidates,
            final_eligible=_projection_candidate_answer_eligible,
        )
        if selection.candidate_id:
            candidate = _candidate_by_id(population.candidates, selection.candidate_id)
            return _projection_for_candidate(candidate, status="completed", objective_solved=False, certificate=certificate, answer_text="", contract=contract, graded_output=graded_output)
        certificate.setdefault("blocking_reasons", [])
        if isinstance(certificate["blocking_reasons"], list) and selection.blocked_reason:
            certificate["blocking_reasons"].append(selection.blocked_reason)
    if not (answer_text and candidate is None):
        best = _best_answer_candidate(population.candidates, contract=contract) or select_best_current_direction(population.candidates, contract=contract)
        if best is not None:
            return _projection_for_candidate(best, status="completed", objective_solved=False, certificate=certificate, answer_text="", contract=contract, graded_output=graded_output)
    if answer_text and candidate is None:
        return _projection_for_candidate(None, status="completed", objective_solved=False, certificate=certificate, answer_text=answer_text, contract=contract, graded_output=graded_output)
    return FinalProjection(
        status="no_candidate",
        title="No displayable candidate was available.",
        blocking_issues=[str(item) for item in certificate.get("blocking_reasons", [])] or ["no_candidate"],
        continuation_plan=["resume evolution with broader seeding or refined contract"],
        objective_solved=False,
    )


def _projection_candidate_id(certificate: dict[str, Any], synthesis: Any) -> str:
    best_current = getattr(synthesis, "best_current_direction", {}) or {}
    if not isinstance(best_current, dict):
        best_current = {}
    for value in (certificate.get("candidate_id"), getattr(synthesis, "best_candidate_id", ""), best_current.get("candidate_id")):
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _projection_for_candidate(candidate: CandidateGenome | None, *, status: str, objective_solved: bool, certificate: dict[str, Any], answer_text: str = "", contract: Any | None = None, graded_output: Any | None = None) -> FinalProjection:
    if candidate is None:
        if str(answer_text or "").strip():
            return FinalProjection(
                status=status,
                title="Final answer",
                artifact_type=str(certificate.get("artifact_type") or "answer"),
                artifact=str(answer_text),
                evidence_summary={
                    "stage": "none",
                    "source": "synthesis",
                    "score": 0.0,
                    "advisory_final_blocked": "unknown",
                    "advisory_artifact_final_eligible": "unknown",
                    "projection_status": "answer",
                    "validation_semantics": "user_owned_after_run",
                    "verification_status": "unverified",
                },
                advisory_issues=["answer_unbound_to_candidate_artifact"],
                objective_solved=objective_solved,
                best_current_direction={
                    "candidate_id": "",
                    "route": "best_current",
                    "mechanism_summary": str(answer_text)[:600],
                    "candidate_main_claim": str(answer_text)[:600],
                    "supporting_claims": [],
                    "intent_alignment_rationale": "unbound synthesis answer; no candidate artifact available",
                    "direct_answer_score": 0.0,
                    "why_best": "unbound synthesis answer; no candidate artifact available",
                    "verification_status": "unverified",
                    "blocked_from_verified_claim_reason": "answer_unbound_to_candidate_artifact",
                },
            )
        return FinalProjection(status="no_candidate", objective_solved=False, blocking_issues=["candidate_not_found"])
    evidence = latest_evidence_record(candidate)
    state = evidence_state(candidate)
    evidence_summary = evidence.to_dict() if evidence is not None else state
    artifact_state = evidence.metadata.get("artifact_state", {}) if evidence is not None and isinstance(evidence.metadata, dict) else {}
    artifact = artifact_state.get("normalized_artifact") if isinstance(artifact_state, dict) and artifact_state.get("normalized_artifact") is not None else candidate.artifact
    artifact_type = _projection_artifact_type(candidate, evidence=evidence, artifact_state=artifact_state, certificate=certificate)
    projected_artifact = str(answer_text) if str(answer_text or "").strip() else _project_artifact_for_projection(artifact, evidence=evidence, evidence_summary=evidence_summary)
    candidate_text = str(candidate.artifact or candidate.concise_claim or candidate.core_mechanism or "").strip()
    answer_candidate_mismatch = bool(str(answer_text or "").strip() and candidate_text and _normalize_projection_text(answer_text) != _normalize_projection_text(candidate_text))
    blocking: list[str] = []
    continuation: list[str] = []
    advisory = _candidate_advisory_issues(candidate)
    if answer_candidate_mismatch:
        advisory.insert(0, "projection_answer_candidate_mismatch")
    best_payload = best_current_direction_payload(candidate, route=_projection_route(candidate), contract=contract, final_certificate=certificate, graded_output=graded_output)
    return FinalProjection(
        status=status,
        candidate_id=candidate.id,
        title="Final answer",
        artifact_type=artifact_type,
        artifact=projected_artifact,
        evidence_summary={
            "stage": evidence.stage if evidence is not None else "none",
            "source": evidence.source if evidence is not None else "none",
            "score": round(float(evidence.score), 4) if evidence is not None else 0.0,
            "advisory_final_blocked": bool(state.get("final_blocked", False)),
            "artifact_type": artifact_type,
            "artifact_cleanliness": artifact_state.get("status") if isinstance(artifact_state, dict) else "",
            "advisory_artifact_final_eligible": artifact_state.get("final_eligible") if isinstance(artifact_state, dict) and "final_eligible" in artifact_state else "unknown",
            "answer_candidate_mismatch": answer_candidate_mismatch,
            "projection_status": "answer",
            "validation_semantics": "user_owned_after_run",
            "verification_status": best_payload.get("verification_status", "unverified"),
        },
        blocking_issues=blocking,
        continuation_plan=continuation,
        advisory_issues=list(dict.fromkeys(advisory))[:12],
        objective_solved=objective_solved,
        best_current_direction=best_payload,
    )


def _best_answer_candidate(candidates: list[CandidateGenome], *, contract: Any | None = None) -> CandidateGenome | None:
    eligible = [candidate for candidate in candidates if _final_answer_candidate_eligible(candidate)]
    if not eligible:
        return None
    return max(eligible, key=lambda candidate: (candidate_verification_status(candidate) == "verified", _answer_candidate_score(candidate, contract=contract)))


def _answer_candidate_score(candidate: CandidateGenome, *, contract: Any | None = None) -> float:
    scores = candidate.multihead_scores or {}
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    hard_penalty = 1.0 if bool(metadata.get("terminal_failure")) else 0.0
    compactness = 1.0 - min(1.0, len(str(candidate.artifact or "")) / 12000.0)
    continuation = bounded_score(metadata.get("repair_value", 0.0))
    intent_score = float(best_current_direction_payload(candidate, contract=contract).get("direct_answer_score") or 0.0)
    return (
        0.25 * intent_score
        + 0.30 * bounded_score(scores.get("frontier_score", 0.0))
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


def _unwrap_best_current_carrier(candidate: CandidateGenome | None, candidates: list[CandidateGenome]) -> CandidateGenome | None:
    """Use an explicit candidate reference inside a status/support artifact.

    Model synthesis can produce a small "best_current_direction" status object
    whose artifact names the real mechanism candidate.  That object is useful
    supporting material, but the user-facing final direction must bind to the
    named mechanism candidate when it exists and is displayable.
    """

    if candidate is None or not isinstance(candidate.artifact, dict):
        return candidate
    best = candidate.artifact.get("best_current_direction")
    if not isinstance(best, dict):
        return candidate
    target_id = str(best.get("candidate_id") or "").strip()
    if not target_id or target_id == candidate.id:
        return candidate
    target = _candidate_by_id(candidates, target_id)
    if target is None or not _projection_candidate_answer_eligible(target):
        return candidate
    metadata = target.metadata if isinstance(target.metadata, dict) else {}
    target.metadata = metadata
    supporting = metadata.setdefault("best_current_direction_carriers", [])
    if isinstance(supporting, list) and candidate.id not in supporting:
        supporting.append(candidate.id)
    return target


def _hard_rejected(candidate: CandidateGenome) -> bool:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    state = evidence_state(candidate)
    return bool(metadata.get("terminal_failure") or state.get("terminal_reject"))


def _fate_excluded_from_answer(candidate: CandidateGenome) -> bool:
    fate = CandidateFate.normalize(getattr(candidate, "current_fate", ""), default="")
    if not fate and isinstance(getattr(candidate, "metadata", None), dict):
        fate = CandidateFate.normalize(candidate.metadata.get("fate"), default="")
    return fate in {CandidateFate.FAILED.value, CandidateFate.CULLED.value}


def _final_answer_candidate_eligible(candidate: CandidateGenome) -> bool:
    if not _projection_candidate_answer_eligible(candidate):
        return False
    if _hard_rejected(candidate) or _fate_excluded_from_answer(candidate):
        return False
    return True

def _projection_candidate_answer_eligible(candidate: CandidateGenome | None) -> bool:
    if candidate is None:
        return False
    return not structurally_blocked(candidate) and bool(str(candidate.artifact or candidate.concise_claim or candidate.core_mechanism).strip())


def _projection_route(candidate: CandidateGenome) -> str:
    return "final" if candidate_verification_status(candidate) == "verified" else "best_current"


def _projection_artifact_type(candidate: CandidateGenome, *, evidence: Any, artifact_state: dict[str, Any], certificate: dict[str, Any]) -> str:
    if isinstance(artifact_state, dict):
        value = str(artifact_state.get("artifact_type") or "").strip()
        if value:
            return value
    if evidence is not None and isinstance(getattr(evidence, "metadata", None), dict):
        policy = evidence.metadata.get("artifact_policy")
        if isinstance(policy, dict):
            value = str(policy.get("artifact_type") or "").strip()
            if value:
                return value
    value = str(getattr(candidate, "artifact_type", "") or "").strip()
    if value:
        return value
    return str(certificate.get("artifact_type") or "").strip()


def _project_artifact_for_projection(artifact: Any, *, evidence: Any, evidence_summary: dict[str, Any]) -> Any:
    if isinstance(artifact, (dict, list)):
        return artifact
    adapter = get_adapter(evidence.source if evidence is not None else None)
    return adapter.project_artifact_for_user(artifact, evidence=evidence_summary)


def _render_artifact(artifact: Any) -> str:
    if isinstance(artifact, str):
        return artifact
    return "```json\n" + __import__("json").dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True, default=str)[:12000] + "\n```"


__all__ = ["FinalProjection", "build_final_projection"]


def _normalize_projection_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip().lower()


def _candidate_advisory_issues(candidate: CandidateGenome) -> list[str]:
    issues: list[str] = []
    result = getattr(candidate, "verification_result", {}) or {}
    if isinstance(result, dict):
        issues.extend(str(item) for item in result.get("diagnostics", []) or [] if item)
        for section in ("final_gate", "proof_progress", "evidence_obligation"):
            payload = result.get(section)
            if isinstance(payload, dict):
                issues.extend(str(item) for item in payload.get("diagnostics", []) or [] if item)
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    for key in ("source_lineage_advisory", "source_binding_manifest", "display_source_binding_advisory"):
        payload = metadata.get(key)
        if isinstance(payload, dict):
            issues.extend(str(item) for item in payload.get("diagnostics", []) or [] if item)
            binding = payload.get("binding_class")
            if binding in {"invented", "unresolved", "no_binding"}:
                issues.append(f"source_binding_{binding}_advisory")
    state = evidence_state(candidate)
    if state.get("final_blocked"):
        issues.append("evidence_final_blocked_advisory")
    return [item for item in dict.fromkeys(issues) if item]
