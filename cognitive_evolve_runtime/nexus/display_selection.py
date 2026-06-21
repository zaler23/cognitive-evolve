"""Unified candidate display selection for final user-facing projections."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.evaluators.evidence import evidence_state, latest_evidence_record
from cognitive_evolve_runtime.nexus._serde import coerce_dict, coerce_str_list
from cognitive_evolve_runtime.nexus.source_binding_resolver import annotate_candidate_source_bindings, candidate_source_binding_class
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult


@dataclass(frozen=True)
class DisplayCandidateSource:
    candidate_id: str
    source: str
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DisplayContext:
    ordered_candidates: list[DisplayCandidateSource] = field(default_factory=list)
    ranking_summary: dict[str, Any] = field(default_factory=dict)
    source_required: bool = False
    project_root: str = ""
    fallback_inputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordered_candidates": [item.to_dict() for item in self.ordered_candidates],
            "ranking_summary": dict(self.ranking_summary),
            "source_required": bool(self.source_required),
            "project_root": self.project_root,
            "fallback_inputs": dict(self.fallback_inputs),
        }

    @classmethod
    def from_any(cls, raw: Any) -> "DisplayContext":
        if isinstance(raw, DisplayContext):
            return raw
        data = coerce_dict(raw)
        return cls(
            ordered_candidates=[
                DisplayCandidateSource(
                    candidate_id=str(item.get("candidate_id") or item.get("id") or ""),
                    source=str(item.get("source") or ""),
                    rationale=str(item.get("rationale") or ""),
                )
                for item in data.get("ordered_candidates", [])
                if isinstance(item, dict)
            ],
            ranking_summary=coerce_dict(data.get("ranking_summary")),
            source_required=bool(data.get("source_required", False)),
            project_root=str(data.get("project_root") or ""),
            fallback_inputs=coerce_dict(data.get("fallback_inputs")),
        )


@dataclass(frozen=True)
class DisplaySelection:
    candidate_id: str = ""
    route: str = "failure"
    eligibility_reason: str = ""
    blocked_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EligibilityFn = Callable[[CandidateGenome], bool]


def build_display_context(
    *,
    candidates: list[CandidateGenome],
    ranking: RelativeRankingResult | dict[str, Any] | None = None,
    contract: Any | None = None,
    project_root: str | Path | None = None,
    fallback_inputs: dict[str, Any] | None = None,
) -> DisplayContext:
    ranking_obj = ranking if isinstance(ranking, RelativeRankingResult) else RelativeRankingResult.from_dict(coerce_dict(ranking)) if ranking else RelativeRankingResult()
    ordered: list[DisplayCandidateSource] = []
    seen: set[str] = set()

    def add(candidate_ids: list[str] | str, source: str) -> None:
        for candidate_id in coerce_str_list(candidate_ids):
            if candidate_id and candidate_id not in seen:
                seen.add(candidate_id)
                ordered.append(DisplayCandidateSource(candidate_id=candidate_id, source=source, rationale=f"ranking:{source}"))

    add(ranking_obj.best_final_answer_id, "best_final_answer_id")
    add(ranking_obj.strongest_mechanism_id, "strongest_mechanism_id")
    add(ranking_obj.mutation_worthy_ids, "mutation_worthy_ids")
    add(ranking_obj.edge_value_ids, "edge_value_ids")
    add(ranking_obj.preserve_incomplete_ids, "preserve_incomplete_ids")
    add(ranking_obj.dormant_ids, "dormant_ids")
    fallback = dict(fallback_inputs or {})
    add(str(fallback.get("best_candidate_id") or fallback.get("answer_candidate_id") or ""), "synthesis_answer_candidate_id")
    add(str(fallback.get("reference" + "_candidate_id") or ""), "legacy_answer_candidate_id")
    for candidate in candidates:
        if candidate.id not in seen:
            seen.add(candidate.id)
            ordered.append(DisplayCandidateSource(candidate_id=candidate.id, source="population_order", rationale="population original order"))
    return DisplayContext(
        ordered_candidates=ordered,
        ranking_summary=ranking_obj.to_dict(),
        source_required=_contract_requires_source(contract),
        project_root=str(project_root or ""),
        fallback_inputs=dict(fallback_inputs or {}),
    )


def select_displayed_candidate(
    display_context: DisplayContext | dict[str, Any] | None,
    *,
    candidates: list[CandidateGenome],
    final_eligible: EligibilityFn | None = None,
) -> DisplaySelection:
    """Select one displayed answer candidate through a single final route."""

    context = DisplayContext.from_any(display_context or build_display_context(candidates=candidates))
    by_id = {candidate.id: candidate for candidate in candidates}
    ordered_ids = [item.candidate_id for item in context.ordered_candidates if item.candidate_id in by_id] or [candidate.id for candidate in candidates]
    predicate = final_eligible or _default_answer_eligible
    blockers: list[str] = []
    for candidate_id in ordered_ids:
        candidate = by_id[candidate_id]
        source_block = _source_block_reason(candidate, context)
        if source_block:
            blockers.append(f"{candidate_id}:{source_block}")
            continue
        if predicate(candidate):
            return DisplaySelection(candidate_id=candidate_id, route="final", eligibility_reason="answer_candidate_eligible", blocked_reason="")
        blockers.append(f"{candidate_id}:answer_predicate_false")
    return DisplaySelection(route="failure", blocked_reason="; ".join(blockers[:12]) or "no_displayable_candidate")


def _source_block_reason(candidate: CandidateGenome, context: DisplayContext) -> str:
    if context.project_root:
        try:
            annotate_candidate_source_bindings(candidate, project_root=context.project_root)
        except Exception:
            pass
    binding_class = candidate_source_binding_class(candidate)
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    metadata["display_source_binding_advisory"] = {
        "binding_class": binding_class,
        "source_required": bool(context.source_required),
        "effect": "advisory_only_nonblocking",
    }
    return ""


def _default_answer_eligible(candidate: CandidateGenome) -> bool:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    fate = CandidateFate.normalize(getattr(candidate, "current_fate", ""), default="")
    if fate in {CandidateFate.FAILED.value, CandidateFate.CULLED.value}:
        return False
    if metadata.get("hard_reject_reason") or metadata.get("terminal_reject_reason"):
        return False
    return bool(str(candidate.artifact or candidate.concise_claim or candidate.core_mechanism).strip())


def _contract_requires_source(contract: Any | None) -> bool:
    data = contract.to_dict() if hasattr(contract, "to_dict") else coerce_dict(contract)
    prefs = {str(item).lower() for item in data.get("verification_preferences", []) if item}
    if "source_binding" in prefs or "local_tests" in prefs:
        return True
    dac = coerce_dict(data.get("dynamic_artifact_contract"))
    adapters = coerce_dict(dac.get("adapter_requirements"))
    return bool(adapters.get("requires_source_binding") or adapters.get("requires_patch") or adapters.get("requires_local_tests"))


__all__ = [
    "DisplayCandidateSource",
    "DisplayContext",
    "DisplaySelection",
    "build_display_context",
    "select_displayed_candidate",
]
