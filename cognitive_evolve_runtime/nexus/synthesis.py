"""Final synthesis for Nexus runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation, candidate_from_dict
from cognitive_evolve_runtime.nexus._serde import coerce_dict
from cognitive_evolve_runtime.nexus.adaptive_signals import mean_percentile, observed_frontier_signal
from cognitive_evolve_runtime.nexus.obligations import HARD_PROOF_FAILURES, requires_proof_progress
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike
from cognitive_evolve_runtime.nexus.fallbacks import record_fallback


@dataclass
class SynthesizedResult:
    status: str
    final_answer: str
    best_candidate_id: str = ""
    reference_candidate_id: str = ""
    reference_note: str = ""
    best_auxiliary_candidate_id: str = ""
    archives_summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failure_analysis: str = ""
    completion_status: str = "completed"
    objective_solved: bool = False
    continuation_available: bool = False
    closure_certificate: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FinalSynthesizer:
    def __init__(self, model: NexusModelLike | None = None) -> None:
        self.model = model
        self._model_fallback_warning = ""

    def synthesize(self, *, population: CandidatePopulation, archives: ArchiveManager, contract: Any | None = None, world: Any | None = None) -> SynthesizedResult:
        self._model_fallback_warning = ""
        model_result = self._model_synthesis(population=population, archives=archives, contract=contract, world=world)
        if model_result is not None:
            return model_result
        best = archives.best_answer_candidate(population.candidates)
        warnings: list[str] = []
        if self._model_fallback_warning:
            warnings.append(self._model_fallback_warning)
        if best is not None and _is_unconverted_seed(best):
            warnings.append("answer_archive_best_was_initial_search_seed_and_was_not_used_as_final")
            best = None
        if best is None:
            candidates = [c for c in population.candidates if archives.is_final_answer_eligible(c) and not _is_unconverted_seed(c)]
            if candidates:
                best = max(candidates, key=_answer_score)
                warnings.append("no_elite_in_answer_archive_used_best_non_auxiliary_candidate")
        auxiliary = _best_auxiliary(archives, population=population)
        if best is None:
            reference = _best_reference_candidate(population.candidates)
            dormant_reference = _best_dormant_reference_candidate(population.candidates)
            if self._model_fallback_warning and dormant_reference is not None and not _has_active_or_elite_candidate(population.candidates):
                return SynthesizedResult(
                    status="best_current_route",
                    final_answer=_best_current_route_answer(dormant_reference),
                    best_candidate_id=dormant_reference.id,
                    reference_candidate_id=dormant_reference.id,
                    reference_note=_reference_note(dormant_reference) + "; non_final_from_dormant_material",
                    best_auxiliary_candidate_id=auxiliary.id if auxiliary else "",
                    archives_summary=archives.summary(),
                    warnings=warnings
                    + [
                        "model_synthesis_failed_local_dormant_reference_used",
                        "best_current_route_not_claimed_optimal_or_solved",
                        "final_answer_not_strict_global_optimum",
                    ],
                    failure_analysis=_best_current_failure_analysis(dormant_reference),
                    objective_solved=False,
                    continuation_available=True,
                )
            if requires_proof_progress(contract, world) and _proof_gate_blocked(population.candidates):
                reference_answer = _reference_candidate_answer(reference) if reference is not None else ""
                return SynthesizedResult(
                    status="route_incomplete",
                    final_answer=_join_reference_answer(
                        (
                            "No verifier-eligible proof candidate emerged. The run produced proof-route material, "
                            "but candidates lacked concrete formal artifacts, obligation-ledger progress, or non-duplicate proof signatures."
                        ),
                        reference_answer,
                    ),
                    best_auxiliary_candidate_id=auxiliary.id if auxiliary else "",
                    reference_candidate_id=reference.id if reference is not None else "",
                    reference_note=_reference_note(reference) if reference is not None else "",
                    archives_summary=archives.summary(),
                    warnings=warnings
                    + ["proof_progress_gate_blocked_final_synthesis"]
                    + (["reference_candidate_displayed_as_non_final_material"] if reference is not None else []),
                    failure_analysis=_proof_failure_analysis(population.candidates),
                )
            best_current = _best_current_route_candidate(population.candidates, contract=contract)
            if best_current is not None:
                return SynthesizedResult(
                    status="best_current_route",
                    final_answer=_best_current_route_answer(best_current),
                    best_candidate_id=best_current.id,
                    reference_candidate_id=best_current.id,
                    reference_note=_reference_note(best_current),
                    best_auxiliary_candidate_id=auxiliary.id if auxiliary else "",
                    archives_summary=archives.summary(),
                    warnings=warnings
                    + [
                        "best_current_route_not_claimed_optimal_or_solved",
                        "final_answer_not_strict_global_optimum",
                    ],
                    failure_analysis=_best_current_failure_analysis(best_current),
                    objective_solved=False,
                    continuation_available=False,
                )
            seed_count = len([c for c in population.candidates if _is_unconverted_seed(c)])
            detail = (
                "No non-seed answer candidate emerged from the offline evolution loop. "
                f"{seed_count} search seeds and archive records were persisted for resume/debugging."
                if seed_count
                else "No admissible answer candidate emerged from the offline evolution loop."
            )
            reference_answer = _reference_candidate_answer(reference) if reference is not None else ""
            return SynthesizedResult(
                status="failure_report",
                final_answer=_join_reference_answer(detail, reference_answer),
                reference_candidate_id=reference.id if reference is not None else "",
                reference_note=_reference_note(reference) if reference is not None else "",
                best_auxiliary_candidate_id=auxiliary.id if auxiliary else "",
                archives_summary=archives.summary(),
                warnings=warnings
                + (["initial_search_seeds_were_not_returned_as_final_answers"] if seed_count else [])
                + (["reference_candidate_displayed_as_non_final_material"] if reference is not None else []),
                failure_analysis="Return stored failure lessons and archive hints for the next run.",
            )
        return SynthesizedResult(
            status="final_synthesis_local_fallback" if self._model_fallback_warning else "synthesized",
            final_answer=str(best.artifact or best.concise_claim or best.core_mechanism),
            best_candidate_id=best.id,
            best_auxiliary_candidate_id=auxiliary.id if auxiliary else "",
            archives_summary=archives.summary(),
            warnings=warnings,
        )

    def _model_synthesis(self, *, population: CandidatePopulation, archives: ArchiveManager, contract: Any | None, world: Any | None) -> SynthesizedResult | None:
        if self.model is None or not hasattr(self.model, "synthesize_result"):
            return None
        try:
            raw = self.model.synthesize_result(population=population.candidates, archives=archives, contract=contract, world=world)
        except Exception as exc:
            record_fallback(stage="final_synthesis", reason=exc.__class__.__name__, detail=str(exc))
            self._model_fallback_warning = f"model_synthesis_local_fallback:{exc.__class__.__name__}"
            return None
        if not isinstance(raw, dict):
            record_fallback(stage="final_synthesis", reason="non_dict_response", detail=type(raw).__name__)
            self._model_fallback_warning = "model_synthesis_local_fallback:non_dict_response"
            return None
        final = str(raw.get("final_answer") or "").strip()
        if not final:
            record_fallback(stage="final_synthesis", reason="empty_final_answer")
            self._model_fallback_warning = "model_synthesis_local_fallback:empty_final_answer"
            return None
        warnings = [str(item) for item in raw.get("warnings", []) if item]
        candidate_by_id = {candidate.id: candidate for candidate in population.candidates}
        requested_best_id = str(raw.get("best_candidate_id") or "").strip()
        requested_reference_id = str(raw.get("reference_candidate_id") or raw.get("candidate_id") or "").strip()
        runtime_best = _runtime_final_candidate(archives, population.candidates)
        best_candidate_id = ""
        reference_candidate: CandidateGenome | None = None

        if requested_best_id:
            candidate = candidate_by_id.get(requested_best_id)
            if candidate is None:
                self._model_fallback_warning = "model_synthesis_local_fallback:unknown_best_candidate_id"
                return None
            if _is_runtime_final_candidate(candidate, archives):
                best_candidate_id = candidate.id
            elif _best_current_route_eligible(candidate):
                reference_candidate = candidate
                warnings.append("model_synthesis_requested_candidate_requires_external_review")
            else:
                self._model_fallback_warning = "model_synthesis_local_fallback:ineligible_best_candidate_id"
                return None
        elif runtime_best is not None:
            best_candidate_id = runtime_best.id
            warnings.append("model_synthesis_without_best_candidate_id_used_verified_runtime_candidate_context")
        elif requested_reference_id and requested_reference_id in candidate_by_id and _best_current_route_eligible(candidate_by_id[requested_reference_id]):
            reference_candidate = candidate_by_id[requested_reference_id]
            warnings.append("model_synthesis_reference_candidate_requires_external_review")
        else:
            reference_candidate = _best_reference_candidate(population.candidates)
            if reference_candidate is None:
                self._model_fallback_warning = "model_synthesis_local_fallback:no_runtime_final_or_reference_candidate"
                return None
            warnings.append("model_synthesis_used_best_reference_candidate_for_external_review")

        if best_candidate_id:
            return SynthesizedResult(
                status=str(raw.get("status") or "model_synthesized"),
                final_answer=final,
                best_candidate_id=best_candidate_id,
                best_auxiliary_candidate_id=str(raw.get("best_auxiliary_candidate_id") or ""),
                archives_summary=archives.summary(),
                warnings=warnings,
                failure_analysis=str(raw.get("failure_analysis") or ""),
            )

        if reference_candidate is not None:
            return SynthesizedResult(
                status="best_current_route",
                final_answer=_join_model_reference_answer(final, reference_candidate),
                best_candidate_id="",
                reference_candidate_id=reference_candidate.id,
                reference_note=_reference_note(reference_candidate),
                best_auxiliary_candidate_id=str(raw.get("best_auxiliary_candidate_id") or ""),
                archives_summary=archives.summary(),
                warnings=warnings
                + [
                    "model_synthesis_not_runtime_final_external_review_required",
                    "best_current_route_not_claimed_optimal_or_solved",
                ],
                failure_analysis=str(raw.get("failure_analysis") or "") or _best_current_failure_analysis(reference_candidate),
                objective_solved=False,
                continuation_available=False,
            )

        self._model_fallback_warning = "model_synthesis_local_fallback:no_runtime_final_or_reference_candidate"
        return None


def synthesize_result(*, population: CandidatePopulation, archives: ArchiveManager, contract: Any | None = None, world: Any | None = None, model: NexusModelLike | None = None) -> SynthesizedResult:
    return FinalSynthesizer(model=model).synthesize(population=population, archives=archives, contract=contract, world=world)


def _is_runtime_final_candidate(candidate: CandidateGenome, archives: ArchiveManager) -> bool:
    return archives.is_final_answer_eligible(candidate) and not _is_unconverted_seed(candidate)


def _runtime_final_candidate(archives: ArchiveManager, candidates: list[CandidateGenome]) -> CandidateGenome | None:
    best = archives.best_answer_candidate(candidates)
    if best is not None and _is_runtime_final_candidate(best, archives):
        return best
    eligible = [candidate for candidate in candidates if _is_runtime_final_candidate(candidate, archives)]
    if not eligible:
        return None
    return max(eligible, key=_answer_score)


def _join_model_reference_answer(model_answer: str, candidate: CandidateGenome) -> str:
    candidate_note = _reference_candidate_answer(candidate)
    return (
        f"{model_answer.strip()}"
        "\n\n---\n\n"
        "Candidate output prepared for external review — correctness is not project-certified. "
        "Human review or an external verifier must judge correctness before treating this as solved. "
        "\n\n"
        f"{candidate_note}"
    )


def _is_unconverted_seed(candidate: CandidateGenome) -> bool:
    return bool(candidate.metadata.get("search_seed_not_final")) and int(candidate.generation or 0) == 0


def _answer_score(candidate: CandidateGenome) -> float:
    scores = candidate.multihead_scores
    if CandidateFate.normalize(candidate.current_fate) == CandidateFate.AUXILIARY.value:
        return -1.0
    axes = [
        float(scores.get("objective_alignment", 0.0) or 0.0),
        float(scores.get("answer_likelihood", 0.0) or 0.0),
        float(scores.get("verifiability", 0.0) or 0.0),
    ]
    return sum(axes) / len(axes)


def _best_current_route_candidate(candidates: list[CandidateGenome], *, contract: Any | None = None) -> CandidateGenome | None:
    if not _contract_allows_best_current_route(contract):
        return None
    viable = [candidate for candidate in candidates if _best_current_route_eligible(candidate)]
    if not viable:
        return None
    return max(viable, key=lambda candidate: _best_current_route_score(candidate, viable))


def _best_reference_candidate(candidates: list[CandidateGenome]) -> CandidateGenome | None:
    """Select useful non-final material for display without relaxing final gates."""

    viable = [candidate for candidate in candidates if _best_current_route_eligible(candidate)]
    if not viable:
        return None
    return max(viable, key=lambda candidate: (_reference_score(candidate, viable), candidate.id))


def _best_dormant_reference_candidate(candidates: list[CandidateGenome]) -> CandidateGenome | None:
    viable = [
        candidate
        for candidate in candidates
        if CandidateFate.normalize(getattr(candidate, "current_fate", "")) == CandidateFate.DORMANT.value
        and _best_current_route_eligible(candidate)
    ]
    if not viable:
        return None
    return max(viable, key=lambda candidate: (_reference_score(candidate, viable), candidate.id))


def _has_active_or_elite_candidate(candidates: list[CandidateGenome]) -> bool:
    return any(
        CandidateFate.normalize(getattr(candidate, "current_fate", ""))
        in {CandidateFate.ACTIVE.value, CandidateFate.ELITE.value, CandidateFate.INCUBATING.value}
        for candidate in candidates
    )


def _best_current_route_eligible(candidate: CandidateGenome) -> bool:
    if _is_unconverted_seed(candidate):
        return False
    fate = CandidateFate.normalize(getattr(candidate, "current_fate", ""))
    if fate in {CandidateFate.CULLED.value, CandidateFate.FAILED.value}:
        return False
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    if metadata.get("hard_reject_reason") or metadata.get("terminal_reject_reason"):
        return False
    if str(metadata.get("dormant_kind") or "") in {"hard_reject", "duplicate"}:
        return False
    result = getattr(candidate, "verification_result", {}) or {}
    if isinstance(result, dict) and result.get("passed") is False:
        diagnostics = {str(item) for item in result.get("diagnostics", []) if item}
        if diagnostics.intersection({"patch_no_effect", "patch_sandbox_failed", "project_offspring_failed_sandbox_verification"}):
            return False
    return bool(str(candidate.artifact or candidate.concise_claim or candidate.core_mechanism).strip())


def _best_current_route_score(candidate: CandidateGenome, context: list[CandidateGenome] | None = None) -> float:
    context = context or [candidate]
    quality_signal = mean_percentile(
        candidate,
        context,
        ["objective_alignment", "answer_likelihood", "verifiability", "core_mechanism_strength"],
    )
    diversity_signal = 1.0 if observed_frontier_signal(candidate, context) else mean_percentile(candidate, context, ["novelty", "rarity"])
    evidence_items = [
        bool(candidate.evidence_refs),
        bool(candidate.source_bindings),
        bool(candidate.formal_artifacts),
        bool(candidate.evidence_delta),
        bool(candidate.obligation_delta),
    ]
    evidence_signal = sum(1 for item in evidence_items if item) / max(1, len(evidence_items))
    latent_signal = max(
        0.0,
        min(
            1.0,
            float(candidate.multihead_scores.get("latent_reproductive_signal", 0.0) or 0.0)
            + float(candidate.multihead_scores.get("latent_expected_utility", 0.0) or 0.0) / 2,
        ),
    )
    uncertainty_items = len(candidate.failure_lessons + candidate.missing_parts)
    uncertainty_penalty = uncertainty_items / max(1, uncertainty_items + len(context))
    return (quality_signal + diversity_signal + evidence_signal + latent_signal) / 4 - uncertainty_penalty


def _reference_score(candidate: CandidateGenome, context: list[CandidateGenome] | None = None) -> float:
    """Score non-final follow-up material by reproducible repair value.

    This is deliberately not a second ranking authority for winners.  It only
    orders reference material after final gates have already declined to claim a
    solved answer, so source-bound and evidence-backed repair candidates are
    displayed ahead of high-scoring but hallucinated proposals.
    """

    diagnostics = _diagnostic_tokens(candidate)
    rank_eligible = _verification_flag(candidate, "rank_eligible")
    final_eligible = _verification_flag(candidate, "final_eligible")
    source_bound = bool(candidate.source_bindings)
    evidence_bound = bool(candidate.evidence_refs or candidate.evidence_delta or candidate.obligation_delta)
    formal_bound = bool(candidate.formal_artifacts or candidate.proof_obligations)
    repair_ready = _repair_ready(candidate, diagnostics)
    value = _best_current_route_score(candidate, context)
    value += float(candidate.multihead_scores.get("latent_reproductive_signal", 0.0) or 0.0) / 2
    evidence_components = [rank_eligible is True, source_bound, evidence_bound, formal_bound, repair_ready, final_eligible is True]
    value += sum(1 for item in evidence_components if item) / max(1, len(evidence_components))
    value -= _reference_diagnostic_penalty(diagnostics)
    return value


def _verification_flag(candidate: CandidateGenome, key: str) -> bool | None:
    result = getattr(candidate, "verification_result", {}) or {}
    if isinstance(result, dict) and isinstance(result.get(key), bool):
        return bool(result.get(key))
    return None


def _repair_ready(candidate: CandidateGenome, diagnostics: set[str]) -> bool:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    if isinstance(metadata.get("repair_required"), dict):
        return True
    if diagnostics.intersection({"unified_patch_failed", "malformed patch", "patch_no_effect", "source_binding_missing_symbol"}):
        return bool(candidate.source_bindings or candidate.evidence_refs)
    return False


def _diagnostic_tokens(candidate: CandidateGenome) -> set[str]:
    tokens: set[str] = set()
    result = getattr(candidate, "verification_result", {}) or {}
    if isinstance(result, dict):
        tokens.update(str(item) for item in result.get("diagnostics", []) or [] if item)
        for section in ("final_gate", "proof_progress", "evidence_obligation"):
            payload = result.get(section)
            if isinstance(payload, dict):
                tokens.update(str(item) for item in payload.get("diagnostics", []) or [] if item)
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    for key in ("hard_reject_reason", "terminal_reject_reason"):
        value = metadata.get(key)
        if value:
            tokens.add(str(value))
    tokens.update(str(item) for item in candidate.failure_lessons if item)
    return tokens


def _reference_diagnostic_penalty(diagnostics: set[str]) -> float:
    joined = " ".join(sorted(diagnostics)).lower()
    hard_tokens = (
        "seed_note_only_patch",
        "documentation_only_patch",
        "runtime_code_change_absent",
        "source_binding_missing_path",
        "source_binding_missing_symbol",
        "patch_target_missing",
    )
    softer_tokens = (
        "patch_no_effect",
        "unified_patch_failed",
        "malformed patch",
        "unexpected end of file",
        "evidence_ref_not_source_relevant",
        "obligation_delta_absent",
        "evidence_ref_absent",
    )
    matched_hard = [token for token in hard_tokens if token in joined]
    matched_soft = [token for token in softer_tokens if token in joined]
    if not matched_hard and not matched_soft:
        return 0.0
    return (len(matched_hard) / max(1, len(hard_tokens))) + (len(matched_soft) / max(1, len(hard_tokens) + len(softer_tokens)))


def _best_current_route_answer(candidate: CandidateGenome) -> str:
    body = str(candidate.artifact or candidate.concise_claim or candidate.core_mechanism)
    lines = [
        "Candidate result / best current route only — not externally validated, not a guaranteed optimum, and not a solved proof:",
        f"Candidate: {candidate.id}",
        body,
    ]
    unresolved = _candidate_blockers(candidate)
    if unresolved:
        lines.append("\nRemaining uncertainty / external-validation gaps:")
        lines.extend(f"- {item}" for item in unresolved)
    return "\n".join(lines)


def _reference_candidate_answer(candidate: CandidateGenome | None) -> str:
    if candidate is None:
        return ""
    body = str(candidate.artifact or candidate.concise_claim or candidate.core_mechanism)
    lines = [
        "Candidate retained for follow-up — correctness has not been externally validated and must not be treated as solved or merge-ready:",
        f"Candidate: {candidate.id}",
        body,
    ]
    blockers = _candidate_blockers(candidate)
    if blockers:
        lines.append("\nOpen validation notes:")
        lines.extend(f"- {item}" for item in blockers)
    return "\n".join(lines)


def _join_reference_answer(primary: str, reference_answer: str) -> str:
    if not reference_answer:
        return primary
    return f"{primary}\n\n---\n\n{reference_answer}"


def _reference_note(candidate: CandidateGenome | None) -> str:
    if candidate is None:
        return ""
    blockers = _candidate_blockers(candidate)
    return "candidate_output_external_validation_required" + (": " + "; ".join(blockers[:4]) if blockers else "")


def _candidate_blockers(candidate: CandidateGenome) -> list[str]:
    result = getattr(candidate, "verification_result", {}) or {}
    items: list[str] = []
    if isinstance(result, dict):
        items.extend(str(item) for item in result.get("diagnostics", []) or [] if item)
        for section in ("final_gate", "proof_progress", "evidence_obligation"):
            payload = result.get(section)
            if isinstance(payload, dict):
                items.extend(str(item) for item in payload.get("diagnostics", []) or [] if item)
        if result.get("passed") is False:
            items.append("verification_result_failed")
        if result.get("final_eligible") is False:
            items.append("external_validation_not_completed")
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    for key in (
        "final_answer_blocked_until_repaired",
        "final_answer_blocked_until_reverified",
        "final_answer_blocked_until_verified",
        "hard_reject_reason",
        "terminal_reject_reason",
    ):
        value = metadata.get(key)
        if value:
            items.append(key if value is True else f"{key}:{value}")
    items.extend(str(item) for item in candidate.missing_parts if item)
    items.extend(str(item) for item in candidate.uncertainty_notes if item)
    items.extend(str(item) for item in candidate.failure_lessons if item)
    return list(dict.fromkeys(items))[:8]


def _best_current_failure_analysis(candidate: CandidateGenome) -> str:
    gaps = list(dict.fromkeys([str(item) for item in candidate.missing_parts + candidate.failure_lessons if item]))[:8]
    if not gaps:
        return "The task type permits a best-current route because no absolute optimum is required; objective_solved remains false."
    return "Best-current route retained with unresolved gaps: " + "; ".join(gaps)


def _contract_allows_best_current_route(contract: Any | None) -> bool:
    """Use the model/contract outcome policy, not a hard-coded task boundary."""

    if isinstance(contract, dict):
        policy = coerce_dict(contract.get("outcome_policy"))
    else:
        policy = coerce_dict(getattr(contract, "outcome_policy", {}))
    if policy.get("accepts_best_current_route") is False:
        return False
    if policy.get("requires_strict_optimum") is True or policy.get("requires_verified_solution") is True:
        return False
    return True


def _best_auxiliary(archives: ArchiveManager, *, population: CandidatePopulation | None = None) -> CandidateGenome | None:
    current_by_id = {candidate.id: candidate for candidate in population.candidates} if population is not None else {}
    candidates = []
    for data in archives.auxiliary_archive.candidates.values():
        archived = candidate_from_dict(data)
        candidate = current_by_id.get(archived.id, archived)
        if CandidateFate.normalize(archives.fates.get(candidate.id, candidate.current_fate)) == CandidateFate.AUXILIARY:
            candidates.append(candidate)
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.multihead_scores.get("auxiliary_value", 0.0))


def _proof_gate_blocked(candidates: list[CandidateGenome]) -> bool:
    for candidate in candidates:
        result = getattr(candidate, "verification_result", {}) or {}
        if not isinstance(result, dict):
            continue
        diagnostics = set(str(item) for item in result.get("diagnostics", []) if item)
        proof = result.get("proof_progress")
        if isinstance(proof, dict):
            diagnostics.update(str(item) for item in proof.get("diagnostics", []) if item)
        if diagnostics.intersection(HARD_PROOF_FAILURES):
            return True
    return False


def _proof_failure_analysis(candidates: list[CandidateGenome]) -> str:
    counts: dict[str, int] = {}
    for candidate in candidates:
        result = getattr(candidate, "verification_result", {}) or {}
        if not isinstance(result, dict):
            continue
        diagnostics = list(result.get("diagnostics", []) or [])
        proof = result.get("proof_progress")
        if isinstance(proof, dict):
            diagnostics.extend(proof.get("diagnostics", []) or [])
        for item in diagnostics:
            if item in HARD_PROOF_FAILURES:
                counts[str(item)] = counts.get(str(item), 0) + 1
    if not counts:
        return "Proof-progress gate blocked synthesis, but no detailed diagnostic survived serialization."
    parts = [f"{key}: {value}" for key, value in sorted(counts.items())]
    return "Proof-progress gate diagnostics: " + "; ".join(parts)


__all__ = ["SynthesizedResult", "FinalSynthesizer", "synthesize_result"]
