"""Final evidence certificate for adaptive runs."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.evaluators.evidence import evidence_final_blocked, evidence_state


def build_final_certificate(
    *,
    population: CandidatePopulation,
    synthesis: Any,
    closure_certificate: dict[str, Any],
    evaluator_required: bool,
) -> dict[str, Any]:
    candidate = _selected_candidate(population, synthesis)
    generic_passed = _generic_verifier_passed(candidate)
    evaluator_passed = _external_evaluator_passed(candidate) if evaluator_required else None
    blocking: list[str] = []
    if candidate is None:
        blocking.append("final_candidate_absent")
    if candidate is not None and not generic_passed:
        blocking.append("generic_verifier_not_passed_advisory")
    if evaluator_required and evaluator_passed is not True:
        blocking.append("external_evaluator_not_passed_advisory")
    closure_claimed = bool(closure_certificate.get("objective_solved"))
    objective_solved = bool(closure_claimed and candidate is not None)
    return {
        "version": "adaptive-final-certificate/v1",
        "objective_solved": objective_solved,
        "closure_objective_solved_before_gate": closure_claimed,
        "candidate_id": candidate.id if candidate is not None else "",
        "generic_verifier_passed": generic_passed,
        "external_evaluator_required": bool(evaluator_required),
        "external_evaluator_passed": evaluator_passed,
        "evidence_state": evidence_state(candidate) if candidate is not None else {},
        "robustness_score": _score(candidate, "robustness") if candidate is not None else None,
        "mdl": dict((candidate.metadata or {}).get("mdl") or {}) if candidate is not None and isinstance(candidate.metadata, dict) else {},
        "judge_quorum": {},
        "blocking_reasons": list(dict.fromkeys(blocking)),
    }


def apply_final_certificate_to_closure(closure_certificate: dict[str, Any], final_certificate: dict[str, Any]) -> dict[str, Any]:
    closure = dict(closure_certificate or {})
    closure["final_certificate"] = dict(final_certificate or {})
    if final_certificate and not final_certificate.get("objective_solved"):
        closure["final_certificate_advisory"] = "nonblocking_answer_first"
    return closure


def apply_research_final_gate_directives(final_certificate: dict[str, Any], directives: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Apply research-extension final gate directives without granting solved authority.

    Research extensions may only report or block finalization.  They cannot set
    ``objective_solved`` to true; solved still requires the base certificate
    gates plus the closure certificate.
    """

    certificate = dict(final_certificate or {})
    candidate_id = str(certificate.get("candidate_id") or "")
    reports: list[dict[str, Any]] = []
    blocking: list[str] = []
    for raw in directives or []:
        directive = dict(raw) if isinstance(raw, dict) else {"kind": "unknown_research_final_gate_directive", "enforcement": "blocking"}
        kind = str(directive.get("kind") or "unknown_research_final_gate_directive")
        enforcement = str(directive.get("enforcement") or "report").strip().lower()
        reports.append(_safe_directive_report(directive))
        if directive.get("final_projection_status"):
            certificate["final_projection_status"] = str(directive.get("final_projection_status"))
        if enforcement != "blocking":
            continue
        reason = _research_directive_blocking_reason(kind=kind, directive=directive, candidate_id=candidate_id)
        if reason:
            blocking.append(reason)
    if reports:
        certificate["research_final_gate_directives"] = reports[-50:]
        certificate["research_final_gate_directive_count"] = len(reports)
    if blocking:
        current = [str(item) for item in certificate.get("blocking_reasons", []) if item]
        certificate["blocking_reasons"] = list(dict.fromkeys([*current, *(item + "_advisory" for item in blocking)]))
        certificate["research_final_gate_passed"] = True
        certificate["research_final_gate_advisory"] = "nonblocking_answer_first"
    elif reports:
        certificate["research_final_gate_passed"] = True
    return certificate


def _selected_candidate(population: CandidatePopulation, synthesis: Any) -> CandidateGenome | None:
    by_id = population.by_id()
    candidate_id = str(getattr(synthesis, "best_candidate_id", "") or "")
    return by_id.get(candidate_id) if candidate_id else None


def _generic_verifier_passed(candidate: CandidateGenome | None) -> bool:
    if candidate is None:
        return False
    return bool(str(candidate.artifact or candidate.concise_claim or candidate.core_mechanism).strip())


def _external_evaluator_passed(candidate: CandidateGenome | None) -> bool:
    if candidate is None or not isinstance(candidate.metadata, dict):
        return False
    state = evidence_state(candidate)
    if state:
        return not evidence_final_blocked(candidate) and float(state.get("final_score") or 0.0) > 0.0
    evaluator = candidate.metadata.get("evaluator")
    return isinstance(evaluator, dict) and evaluator.get("status") == "passed" and evaluator.get("passed") is True


def _score(candidate: CandidateGenome, key: str) -> float | None:
    value = (candidate.multihead_scores or {}).get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _research_directive_blocking_reason(*, kind: str, directive: dict[str, Any], candidate_id: str) -> str:
    if kind == "unknown_research_final_gate_directive":
        return "unknown_research_final_gate_directive"
    if kind == "parametric_candidate_not_final":
        directive_candidate = str(directive.get("candidate_id") or "")
        if not directive_candidate or not candidate_id or directive_candidate == candidate_id:
            return "parametric_candidate_not_collapsed"
        return ""
    if kind == "contract_refinement_proposal":
        if directive.get("requires_user_decision") or str(directive.get("final_projection_status") or "") == "needs_user_decision":
            return "contract_refinement_requires_user_decision"
        return ""
    if kind == "bft_quorum_report":
        status = str(directive.get("status") or "").strip().lower()
        if directive.get("block_final") or status in {"blocked", "failed", "disagreement", "quorum_failed"}:
            return "bft_quorum_blocks_final"
        return ""
    if kind == "chaos_replay_required_if_configured":
        status = str(directive.get("status") or "").strip().lower()
        if directive.get("block_final") or status in {"failed", "not_executed_without_explicit_profile", "replay_failed"}:
            return "chaos_replay_not_passed"
        return ""
    if kind == "immune_necropsy_report":
        if directive.get("hard_reject") or directive.get("terminal_reject") or directive.get("revokes_final"):
            return "immune_necropsy_blocks_final"
        return ""
    return "unknown_research_final_gate_directive"


def _safe_directive_report(directive: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in directive.items():
        key_s = str(key)
        if any(token in key_s.lower() for token in ("key", "secret", "token", "password", "prompt")):
            continue
        if isinstance(value, str):
            safe[key_s] = value[:500]
        elif isinstance(value, (int, float, bool)) or value is None:
            safe[key_s] = value
        elif isinstance(value, list):
            safe[key_s] = [str(item)[:200] for item in value[:20]]
        elif isinstance(value, dict):
            safe[key_s] = {str(k): str(v)[:200] for k, v in list(value.items())[:20] if not any(t in str(k).lower() for t in ("key", "secret", "token", "password", "prompt"))}
        else:
            safe[key_s] = str(value)[:500]
    return safe


__all__ = ["apply_final_certificate_to_closure", "apply_research_final_gate_directives", "build_final_certificate"]
