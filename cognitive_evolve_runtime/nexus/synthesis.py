"""Answer-first final synthesis for Nexus runs.

This module has exactly one user-facing selection authority: choose the strongest
non-terminal answer candidate and surface its answer material. Verification,
source binding, proof objects, and patch materialization are advisory telemetry;
they must not create a second final/reference/best-current completion path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation, candidate_from_dict
from cognitive_evolve_runtime.nexus.adaptive_signals import mean_percentile, observed_frontier_signal
from cognitive_evolve_runtime.nexus.fallbacks import record_fallback
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike


@dataclass
class SynthesizedResult:
    status: str
    final_answer: str
    best_candidate_id: str = ""
    best_auxiliary_candidate_id: str = ""
    archives_summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failure_analysis: str = ""
    completion_status: str = "completed"
    objective_solved: bool = False
    answer_produced: bool = False
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

        warnings: list[str] = []
        if self._model_fallback_warning:
            warnings.append(self._model_fallback_warning)
        best = _runtime_answer_candidate(archives, population.candidates)
        if best is None:
            best = _best_answer_candidate(population.candidates)
            if best is not None:
                warnings.append("answer_first_used_best_available_candidate")
        auxiliary = _best_auxiliary(archives, population=population)
        if best is None:
            return SynthesizedResult(
                status="failure_report",
                final_answer="No displayable answer candidate emerged from the evolution loop.",
                best_auxiliary_candidate_id=auxiliary.id if auxiliary else "",
                archives_summary=archives.summary(),
                warnings=warnings,
                failure_analysis="No non-terminal candidate carried answer material.",
            )
        return SynthesizedResult(
            status="final_synthesis_local_fallback" if self._model_fallback_warning else "synthesized",
            final_answer=_candidate_answer_text(best),
            best_candidate_id=best.id,
            best_auxiliary_candidate_id=auxiliary.id if auxiliary else "",
            archives_summary=archives.summary(),
            warnings=warnings,
            objective_solved=False,
            answer_produced=True,
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
        requested_candidate_id = str(
            raw.get("best_candidate_id")
            or raw.get("candidate_id")
            or ""
        ).strip()
        selected = _runtime_answer_candidate(archives, population.candidates)
        if requested_candidate_id:
            requested = candidate_by_id.get(requested_candidate_id)
            if requested is None:
                self._model_fallback_warning = "model_synthesis_local_fallback:unknown_candidate_id"
                return None
            if not _answer_candidate_eligible(requested):
                self._model_fallback_warning = "model_synthesis_local_fallback:ineligible_candidate_id"
                return None
            selected = requested
        elif selected is None:
            selected = _best_answer_candidate(population.candidates)
            if selected is not None:
                warnings.append("model_synthesis_without_candidate_id_used_best_answer_candidate")

        selected_id = selected.id if selected is not None else ""
        if selected is not None and _normalize_answer_text(final) != _normalize_answer_text(_candidate_answer_text(selected)):
            warnings.append("model_final_answer_unbound_to_candidate_artifact")
            selected_id = ""
        return SynthesizedResult(
            status=str(raw.get("status") or "model_synthesized"),
            final_answer=final,
            best_candidate_id=selected_id,
            best_auxiliary_candidate_id=str(raw.get("best_auxiliary_candidate_id") or ""),
            archives_summary=archives.summary(),
            warnings=warnings,
            failure_analysis=str(raw.get("failure_analysis") or ""),
            objective_solved=False,
            answer_produced=True,
        )


def synthesize_result(*, population: CandidatePopulation, archives: ArchiveManager, contract: Any | None = None, world: Any | None = None, model: NexusModelLike | None = None) -> SynthesizedResult:
    return FinalSynthesizer(model=model).synthesize(population=population, archives=archives, contract=contract, world=world)


def _runtime_answer_candidate(archives: ArchiveManager, candidates: list[CandidateGenome]) -> CandidateGenome | None:
    best = archives.best_answer_candidate(candidates)
    if best is not None and _answer_candidate_eligible(best):
        return best
    eligible = [candidate for candidate in candidates if archives.is_final_answer_eligible(candidate) and _answer_candidate_eligible(candidate)]
    if eligible:
        return max(eligible, key=lambda candidate: _answer_candidate_score(candidate, eligible))
    return None


def _best_answer_candidate(candidates: list[CandidateGenome]) -> CandidateGenome | None:
    eligible = [candidate for candidate in candidates if _answer_candidate_eligible(candidate)]
    if not eligible:
        return None
    return max(eligible, key=lambda candidate: _answer_candidate_score(candidate, eligible))


def _answer_candidate_eligible(candidate: CandidateGenome) -> bool:
    fate = CandidateFate.normalize(getattr(candidate, "current_fate", ""))
    if fate in {CandidateFate.CULLED.value, CandidateFate.FAILED.value}:
        return False
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    if metadata.get("terminal_failure") or metadata.get("terminal_reject_reason"):
        return False
    return bool(_candidate_answer_text(candidate).strip())


def _candidate_answer_text(candidate: CandidateGenome) -> str:
    return str(candidate.artifact or candidate.concise_claim or candidate.core_mechanism or "")


def _answer_candidate_score(candidate: CandidateGenome, context: list[CandidateGenome] | None = None) -> float:
    context = context or [candidate]
    scores = candidate.multihead_scores
    if CandidateFate.normalize(candidate.current_fate) == CandidateFate.AUXILIARY.value:
        return -1.0
    quality_signal = mean_percentile(
        candidate,
        context,
        ["objective_alignment", "answer_likelihood", "verifiability", "core_mechanism_strength"],
    )
    direct_axes = [
        float(scores.get("objective_alignment", 0.0) or 0.0),
        float(scores.get("answer_likelihood", 0.0) or 0.0),
        float(scores.get("core_mechanism_strength", 0.0) or 0.0),
    ]
    direct_signal = sum(direct_axes) / len(direct_axes)
    diversity_signal = 1.0 if observed_frontier_signal(candidate, context) else mean_percentile(candidate, context, ["novelty", "rarity"])
    answer_signal = bounded_answer_signal(candidate)
    latent_signal = max(
        0.0,
        min(
            1.0,
            float(scores.get("latent_reproductive_signal", 0.0) or 0.0)
            + float(scores.get("latent_expected_utility", 0.0) or 0.0) / 2,
        ),
    )
    return (quality_signal + direct_signal + diversity_signal + answer_signal + latent_signal) / 5


def bounded_answer_signal(candidate: CandidateGenome) -> float:
    """Return an answer-content signal without requiring verifier/source proof."""

    text = _candidate_answer_text(candidate)
    if not text.strip():
        return 0.0
    length_signal = min(1.0, len(text.strip()) / 800.0)
    mechanism_signal = 1.0 if str(candidate.core_mechanism or "").strip() else 0.0
    novelty_signal = max(
        0.0,
        min(
            1.0,
            float(candidate.multihead_scores.get("novelty", 0.0) or 0.0)
            + float(candidate.multihead_scores.get("rarity", 0.0) or 0.0) / 2,
        ),
    )
    return max(length_signal, mechanism_signal * 0.8, novelty_signal)


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


__all__ = ["SynthesizedResult", "FinalSynthesizer", "synthesize_result"]


def _normalize_answer_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip().lower()
