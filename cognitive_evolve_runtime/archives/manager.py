"""Archive manager for Nexus candidate populations."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, candidate_from_dict
from cognitive_evolve_runtime.nexus._serde import coerce_dict, coerce_str_list, utc_now
from cognitive_evolve_runtime.nexus.adaptive_signals import in_bottom_band, in_top_band, observed_frontier_signal, score
from cognitive_evolve_runtime.nexus.obligations import HARD_EVIDENCE_FAILURES, HARD_PROOF_FAILURES, candidate_has_obligation_or_evidence_delta
from cognitive_evolve_runtime.nexus.population_vitality import (
    classify_dormant_kind,
    reactivation_condition_for_kind,
    target_active_floor,
)
from cognitive_evolve_runtime.nexus.stage_policy import EligibilityDecision, annotate_stage_eligibility
from .auxiliary import AuxiliaryArchive
from .constraints import (
    candidate_is_verified_dormant_frontier as _candidate_is_verified_dormant_frontier,
    candidate_verification_blocks_final as _candidate_verification_blocks_final,
    constraint_id as _constraint_id,
    constraint_target as _constraint_target,
    verification_diagnostics as _verification_diagnostics,
    verification_failure_signature as _verification_failure_signature,
)
from .dormant import DormantArchive
from .failure import FailureArchive
from .latent_pareto import LatentParetoIntentArchive, _latent_archive_removal_reason
from .quality_diversity import QualityDiversityArchive, candidate_quality
from .registry import ArchiveRegistry
from .types import ArchiveConstraintRecord, FateAssignment, TerminalCandidateTombstone
from .rarity import RarityArchive

ARCHIVE_NAMES = [
    "AnswerArchive",
    "MechanismArchive",
    "NoveltyArchive",
    "LatentParetoIntentArchive",
    "RarityArchive",
    "FailureArchive",
    "AuxiliaryArchive",
    "DormantArchive",
    "ProjectPatchArchive",
]

FINAL_ANSWER_FATES = {CandidateFate.ACTIVE.value, CandidateFate.ELITE.value}
TERMINAL_FAILURE_FATES = {CandidateFate.CULLED.value, CandidateFate.FAILED.value}


@dataclass
class ArchiveManager:
    archive_schema: dict[str, Any] = field(default_factory=lambda: {name: {"enabled": True} for name in ARCHIVE_NAMES})
    answer_archive: dict[str, dict[str, Any]] = field(default_factory=dict)
    mechanism_archive: dict[str, dict[str, Any]] = field(default_factory=dict)
    novelty_archive: dict[str, dict[str, Any]] = field(default_factory=dict)
    latent_pareto_archive: LatentParetoIntentArchive = field(default_factory=LatentParetoIntentArchive)
    project_patch_archive: dict[str, dict[str, Any]] = field(default_factory=dict)
    quality_diversity: QualityDiversityArchive = field(default_factory=QualityDiversityArchive)
    rarity_archive: RarityArchive = field(default_factory=RarityArchive)
    failure_archive: FailureArchive = field(default_factory=FailureArchive)
    auxiliary_archive: AuxiliaryArchive = field(default_factory=AuxiliaryArchive)
    dormant_archive: DormantArchive = field(default_factory=DormantArchive)
    fates: dict[str, str] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    constraint_records: list[dict[str, Any]] = field(default_factory=list)
    terminal_tombstones: dict[str, dict[str, Any]] = field(default_factory=dict)

    def update(
        self,
        candidates_or_assignments: list[CandidateGenome] | list[FateAssignment] | dict[str, str],
        *,
        candidates: list[CandidateGenome] | None = None,
    ) -> list[FateAssignment]:
        """Apply fate assignments and route candidates into archives.

        ``update(population)`` remains current: it reads each
        candidate's current fate.  Newer loop code should prefer
        ``update(assignments, candidates=population)`` so that a model/rater
        assignment is applied exactly once.  This avoids the old double-routing
        pattern where candidates were first archived under stale fates and then
        archived again under assigned fates.
        """

        assignments = self._normalize_assignments(candidates_or_assignments)
        candidate_by_id: dict[str, CandidateGenome] = {}
        for candidate in candidates or []:
            candidate_by_id[candidate.id] = candidate
        if isinstance(candidates_or_assignments, list):
            for item in candidates_or_assignments:
                if isinstance(item, CandidateGenome):
                    candidate_by_id[item.id] = item
        applied: list[FateAssignment] = []
        for assignment in assignments:
            candidate = candidate_by_id.get(assignment.candidate_id)
            if candidate is None:
                continue
            assignment.fate = CandidateFate.normalize(assignment.fate)
            candidate.mark_fate(assignment.fate)
            assignment.inherited_gene_summary = assignment.inherited_gene_summary or candidate.extract_inheritable_gene_summary()
            self.fates[candidate.id] = assignment.fate
            latent_removal_reason = _latent_archive_removal_reason(candidate, assignment.fate)
            if latent_removal_reason and candidate.id in self.latent_pareto_archive.candidates:
                self.latent_pareto_archive.discard(candidate.id, reason=latent_removal_reason)
            self._remove_candidate_from_archives(candidate.id)
            if assignment.fate in TERMINAL_FAILURE_FATES:
                self._record_terminal_tombstone(candidate, assignment)
            self._route_candidate(candidate, assignment)
            self._record_constraints(candidate, assignment)
            applied.append(assignment)
        self.history.append({"at": utc_now(), "assignments": [assignment.to_dict() for assignment in applied]})
        return applied

    def assign_by_policy(
        self,
        candidates: list[CandidateGenome],
        ranking: Any | None = None,
        *,
        current_round: int = 0,
        round_limit: int = 0,
        branch_factor: int = 0,
        eligibility_policy: dict[str, Any] | None = None,
    ) -> list[FateAssignment]:
        best_answer = getattr(ranking, "best_final_answer_id", None)
        auxiliary_ids = set(getattr(ranking, "auxiliary_ids", []) or [])
        dormant_ids = set(getattr(ranking, "dormant_ids", []) or [])
        edge_ids = set(getattr(ranking, "edge_value_ids", []) or [])
        assignments: list[FateAssignment] = []
        decisions: dict[str, EligibilityDecision] = {}
        for candidate in candidates:
            current_fate = CandidateFate.normalize(candidate.current_fate)
            decision = (
                annotate_stage_eligibility(candidate, current_round=current_round, round_limit=round_limit, policy_config=eligibility_policy)
                if (current_round or round_limit)
                else None
            )
            if decision is not None:
                decisions[candidate.id] = decision
            if current_fate in TERMINAL_FAILURE_FATES:
                fate = current_fate
            elif _candidate_verification_blocks_final(candidate):
                fate = CandidateFate.INCUBATING.value if decision is not None and decision.incubating else CandidateFate.DORMANT.value
            elif candidate.id == best_answer or (score(candidate, "answer_likelihood") > 0 and in_top_band(candidate, candidates, "answer_likelihood") and not candidate.metadata.get("search_seed_not_final")):
                fate = CandidateFate.ELITE.value
            elif candidate.id in auxiliary_ids or candidate.multihead_scores.get("auxiliary_value", 0.0) > max(candidate.multihead_scores.get("answer_likelihood", 0.0), candidate.multihead_scores.get("objective_alignment", 0.0)):
                fate = CandidateFate.AUXILIARY.value
            elif candidate.id in dormant_ids:
                fate = CandidateFate.DORMANT.value
            elif candidate.id in edge_ids or observed_frontier_signal(candidate, candidates):
                fate = CandidateFate.DORMANT.value
            elif candidate.failure_lessons or (in_bottom_band(candidate, candidates, "objective_alignment") and not candidate_has_obligation_or_evidence_delta(candidate)):
                fate = CandidateFate.CULLED.value
            else:
                fate = CandidateFate.ACTIVE.value
            assignment = FateAssignment(candidate.id, fate)
            if _candidate_verification_blocks_final(candidate):
                assignment.failure_signature = _verification_failure_signature(candidate)
                assignment.future_reactivation_condition = (
                    (decision.reactivation_condition or "repair_lane_requires_concrete_formal_artifact_obligation_delta_or_verified_evidence")
                    if fate == CandidateFate.INCUBATING.value and decision is not None
                    else (decision.reactivation_condition if decision is not None else "")
                    or "reactivate_only_with_concrete_formal_artifact_and_obligation_delta"
                )
            if fate == CandidateFate.DORMANT.value:
                dormant_kind = classify_dormant_kind(candidate, decision)
                candidate.metadata["dormant_kind"] = dormant_kind
                assignment.future_reactivation_condition = assignment.future_reactivation_condition or reactivation_condition_for_kind(dormant_kind, decision)
            elif fate == CandidateFate.INCUBATING.value:
                candidate.metadata["dormant_kind"] = "repairable"
            else:
                candidate.metadata.pop("dormant_kind", None)
            assignments.append(assignment)
        _apply_stage_adaptive_active_floor(
            candidates,
            assignments,
            decisions,
            current_round=current_round,
            round_limit=round_limit,
            branch_factor=branch_factor,
            eligibility_policy=eligibility_policy,
        )
        return assignments

    def reactivate_dormant(self, candidate_id: str | None = None) -> CandidateGenome | None:
        if candidate_id is not None:
            preview = self.dormant_archive.candidates.get(candidate_id)
            if preview is not None and not self._reactivation_allowed(candidate_from_dict(preview)):
                return None
            candidate = self.dormant_archive.reactivate(candidate_id)
        else:
            candidate = None
            for key, data in list(self.dormant_archive.candidates.items()):
                preview = candidate_from_dict(data)
                if not self._reactivation_allowed(preview):
                    continue
                candidate = self.dormant_archive.reactivate(key)
                break
        if candidate is not None:
            self._remove_candidate_from_archives(candidate.id)
            self.fates[candidate.id] = CandidateFate.ACTIVE.value
        return candidate

    def best_answer_candidate(self, current_candidates: list[CandidateGenome] | None = None) -> CandidateGenome | None:
        current_by_id = {candidate.id: candidate for candidate in current_candidates or []}
        candidates: list[CandidateGenome] = []
        seen: set[str] = set()
        for data in self.answer_archive.values():
            archived = candidate_from_dict(data)
            candidate = current_by_id.get(archived.id, archived)
            if self.is_final_answer_eligible(candidate) and candidate.id not in seen:
                candidates.append(candidate)
                seen.add(candidate.id)
        for data in self.quality_diversity.elites_by_niche.values():
            if not isinstance(data, dict):
                continue
            candidate_data = data.get("candidate")
            if not isinstance(candidate_data, dict):
                continue
            archived = candidate_from_dict(candidate_data)
            candidate = current_by_id.get(archived.id, archived)
            if self.is_final_answer_eligible(candidate) and candidate.id not in seen:
                candidates.append(candidate)
                seen.add(candidate.id)
        if not candidates:
            return None
        return max(candidates, key=candidate_quality)

    def is_final_answer_eligible(self, candidate: CandidateGenome) -> bool:
        """Return whether a candidate may be used as a final answer now.

        Archives are durable indexes, not proofs of current validity.  A
        candidate that was once elite can later fail verification or be culled;
        final-answer selection must honor the newest fate recorded either on
        the candidate object or in the archive fate index.
        """

        indexed_fate = CandidateFate.normalize(self.fates.get(candidate.id, candidate.current_fate))
        object_fate = CandidateFate.normalize(candidate.current_fate)
        if indexed_fate == CandidateFate.INCUBATING.value or object_fate == CandidateFate.INCUBATING.value:
            return False
        if indexed_fate in FINAL_ANSWER_FATES and object_fate in FINAL_ANSWER_FATES:
            return not _candidate_verification_blocks_final(candidate)
        if indexed_fate == CandidateFate.DORMANT.value and object_fate == CandidateFate.DORMANT.value:
            return _candidate_is_verified_dormant_frontier(candidate)
        if indexed_fate != object_fate:
            candidate.metadata["final_answer_fate_mismatch"] = {"archive_fate": indexed_fate, "candidate_fate": object_fate}
        return False

    def summary(self) -> dict[str, Any]:
        latent_summary = self.latent_pareto_archive.summary()
        return {
            "answer_candidates": len(self.answer_archive),
            "mechanism_elites": len(self.mechanism_archive),
            "novelty_candidates": len(self.novelty_archive),
            "latent_pareto_candidates": latent_summary["candidates"],
            "latent_pareto_governance": latent_summary,
            "rarity_candidates": len(self.rarity_archive.candidates),
            "failure_records": len(self.failure_archive.records),
            "auxiliary_candidates": len(self.auxiliary_archive.candidates),
            "dormant_candidates": len(self.dormant_archive.candidates),
            "incubating_candidates": sum(1 for fate in self.fates.values() if CandidateFate.normalize(fate) == CandidateFate.INCUBATING.value),
            "dormant_kinds": _dormant_kind_counts(self.dormant_archive.candidates),
            "project_patch_candidates": len(self.project_patch_archive),
            "constraint_records": len(self.constraint_records),
            "terminal_tombstones": len(self.terminal_tombstones),
            "fates": dict(self.fates),
        }

    def constraints_for_policy(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return newest active constraints for generation/ranking policy."""

        records = [dict(item) for item in self.constraint_records if isinstance(item, dict)]
        return records[-max(0, int(limit)) :]

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_schema": self.archive_schema,
            "answer_archive": self.answer_archive,
            "mechanism_archive": self.mechanism_archive,
            "novelty_archive": self.novelty_archive,
            "latent_pareto_archive": self.latent_pareto_archive.to_dict(),
            "project_patch_archive": self.project_patch_archive,
            "quality_diversity": self.quality_diversity.to_dict(),
            "rarity_archive": self.rarity_archive.to_dict(),
            "failure_archive": self.failure_archive.to_dict(),
            "auxiliary_archive": self.auxiliary_archive.to_dict(),
            "dormant_archive": self.dormant_archive.to_dict(),
            "fates": self.fates,
            "history": self.history,
            "constraint_records": self.constraint_records,
            "terminal_tombstones": self.terminal_tombstones,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchiveManager":
        return cls(
            archive_schema=_archive_schema_from_dict(data.get("archive_schema")),
            answer_archive=dict(data.get("answer_archive") or {}),
            mechanism_archive=dict(data.get("mechanism_archive") or {}),
            novelty_archive=dict(data.get("novelty_archive") or {}),
            latent_pareto_archive=LatentParetoIntentArchive.from_dict(coerce_dict(data.get("latent_pareto_archive"))),
            project_patch_archive=dict(data.get("project_patch_archive") or {}),
            quality_diversity=QualityDiversityArchive.from_dict(coerce_dict(data.get("quality_diversity"))),
            rarity_archive=RarityArchive.from_dict(coerce_dict(data.get("rarity_archive"))),
            failure_archive=FailureArchive.from_dict(coerce_dict(data.get("failure_archive"))),
            auxiliary_archive=AuxiliaryArchive.from_dict(coerce_dict(data.get("auxiliary_archive"))),
            dormant_archive=DormantArchive.from_dict(coerce_dict(data.get("dormant_archive"))),
            fates={str(k): str(v) for k, v in coerce_dict(data.get("fates")).items()},
            history=[dict(item) for item in data.get("history", []) if isinstance(item, dict)],
            constraint_records=[ArchiveConstraintRecord.from_dict(item).to_dict() for item in data.get("constraint_records", []) if isinstance(item, dict)],
            terminal_tombstones={
                str(k): TerminalCandidateTombstone.from_dict(v).to_dict()
                for k, v in coerce_dict(data.get("terminal_tombstones")).items()
                if isinstance(v, dict)
            },
        )

    def _normalize_assignments(self, value: list[CandidateGenome] | list[FateAssignment] | dict[str, str]) -> list[FateAssignment]:
        if isinstance(value, dict):
            return [FateAssignment(str(candidate_id), CandidateFate.normalize(fate)) for candidate_id, fate in value.items()]
        assignments: list[FateAssignment] = []
        for item in value:
            if isinstance(item, FateAssignment):
                item.fate = CandidateFate.normalize(item.fate)
                assignments.append(item)
            elif isinstance(item, CandidateGenome):
                assignments.append(FateAssignment(item.id, item.current_fate))
        return assignments

    def _route_candidate(self, candidate: CandidateGenome, assignment: FateAssignment) -> None:
        ArchiveRegistry(self).route_candidate(candidate, assignment)

    def _record_constraints(self, candidate: CandidateGenome, assignment: FateAssignment) -> None:
        diagnostics = _verification_diagnostics(candidate)
        hard = diagnostics.intersection(HARD_PROOF_FAILURES | HARD_EVIDENCE_FAILURES)
        records: list[ArchiveConstraintRecord] = []
        if hard:
            target = _constraint_target(candidate)
            records.append(
                ArchiveConstraintRecord(
                    id=_constraint_id("verification_constraint", target, sorted(hard), candidate.id),
                    kind="verification_constraint",
                    rule="do_not_rank_or_reactivate_without_named_obligation_delta_and_verified_evidence",
                    target=target,
                    source_candidate_id=candidate.id,
                    severity="error" if assignment.fate in {CandidateFate.DORMANT.value, CandidateFate.CULLED.value, CandidateFate.FAILED.value} else "warning",
                    evidence={"diagnostics": sorted(hard), "fate": assignment.fate},
                )
            )
        has_failure_context = bool(hard or assignment.failure_signature or candidate.failure_lessons)
        if has_failure_context and assignment.fate in {CandidateFate.DORMANT.value, CandidateFate.CULLED.value, CandidateFate.FAILED.value} and not candidate_has_obligation_or_evidence_delta(candidate):
            target = _constraint_target(candidate)
            records.append(
                ArchiveConstraintRecord(
                    id=_constraint_id("lineage_freeze", target, candidate.extract_inheritable_gene_summary(), candidate.id),
                    kind="lineage_freeze",
                    rule="freeze_repeated_proposal_only_lineage_until_new_evidence_delta_exists",
                    target=target,
                    source_candidate_id=candidate.id,
                    severity="warning",
                    evidence={"fate": assignment.fate, "failure_lessons": list(candidate.failure_lessons[:5])},
                )
            )
        for lesson in candidate.failure_lessons[:5]:
            records.append(
                ArchiveConstraintRecord(
                    id=_constraint_id("failure_lesson_constraint", _constraint_target(candidate), lesson),
                    kind="failure_lesson_constraint",
                    rule=str(lesson)[:500],
                    target=_constraint_target(candidate),
                    source_candidate_id=candidate.id,
                    severity="warning",
                    evidence={"fate": assignment.fate},
                )
            )
        if records:
            existing = {str(item.get("id") or "") for item in self.constraint_records}
            for record in records:
                if record.id in existing:
                    continue
                self.constraint_records.append(record.to_dict())
                existing.add(record.id)
            self.constraint_records = self.constraint_records[-200:]

    def _record_terminal_tombstone(self, candidate: CandidateGenome, assignment: FateAssignment) -> None:
        tombstone = TerminalCandidateTombstone.from_candidate(
            candidate,
            fate=assignment.fate,
            failure_signature=assignment.failure_signature,
        )
        self.terminal_tombstones[candidate.id] = tombstone.to_dict()

    def _reactivation_allowed(self, candidate: CandidateGenome) -> bool:
        target = _constraint_target(candidate)
        blocked = [
            item
            for item in self.constraint_records
            if isinstance(item, dict)
            and str(item.get("kind") or "") in {"lineage_freeze", "verification_constraint"}
            and str(item.get("target") or "") == target
        ]
        if not blocked:
            return True
        return candidate_has_obligation_or_evidence_delta(candidate)

    def _remove_candidate_from_archives(self, candidate_id: str) -> None:
        """Remove stale memberships before re-routing a mutable candidate."""

        ArchiveRegistry(self).remove_candidate_from_archives(candidate_id)


def _apply_stage_adaptive_active_floor(
    candidates: list[CandidateGenome],
    assignments: list[FateAssignment],
    decisions: dict[str, EligibilityDecision],
    *,
    current_round: int = 0,
    round_limit: int = 0,
    branch_factor: int = 0,
    eligibility_policy: dict[str, Any] | None = None,
) -> None:
    """Keep early/mid exploration from collapsing to zero Active candidates.

    This is not a final-answer relaxation.  Verification still blocks final
    eligibility.  The floor only prevents ranking/critique from parking every
    promising incomplete route before a repair operator can develop it.
    """

    if not candidates or not assignments or not (current_round or round_limit):
        return
    by_id = {candidate.id: candidate for candidate in candidates}
    active_count = sum(1 for assignment in assignments if CandidateFate.normalize(assignment.fate) == CandidateFate.ACTIVE.value)
    non_terminal_count = sum(1 for assignment in assignments if CandidateFate.normalize(assignment.fate) not in TERMINAL_FAILURE_FATES)
    viable: list[tuple[FateAssignment, CandidateGenome, EligibilityDecision]] = []
    for assignment in assignments:
        fate = CandidateFate.normalize(assignment.fate)
        if fate in TERMINAL_FAILURE_FATES or fate in {CandidateFate.ELITE.value, CandidateFate.AUXILIARY.value}:
            continue
        if fate == CandidateFate.ACTIVE.value:
            continue
        candidate = by_id.get(assignment.candidate_id)
        decision = decisions.get(assignment.candidate_id)
        if candidate is None or decision is None:
            continue
        # Final-answer strictness is enforced by ``is_final_answer_eligible``
        # and synthesis gates.  Archive routing should not let any pre-synthesis
        # search phase collapse to zero Active parents merely because all useful
        # candidates still need contract/evidence/final-gate repair.
        if decision.hard_reject_reason or decision.repair_exhausted:
            continue
        if not (decision.exploration_eligible and decision.parent_eligible):
            continue
        viable.append((assignment, candidate, decision))
    if len(viable) < 2:
        return
    viable_count = len(viable) + active_count
    active_floor_policy = coerce_dict(coerce_dict(eligibility_policy).get("active_floor"))
    floor = target_active_floor(
        non_terminal_count=non_terminal_count,
        viable_count=viable_count,
        branch_factor=branch_factor,
        branch_multiplier=_float_policy(active_floor_policy.get("branch_multiplier"), default=None),
        minimum=_int_policy(active_floor_policy.get("minimum"), default=None),
        enabled=active_floor_policy.get("enabled", True) is not False,
    )
    if active_count >= floor:
        return
    promote_count = min(len(viable), max(0, floor - active_count))
    if promote_count <= 0:
        return
    viable.sort(
        key=lambda item: (
            item[2].strict_rank_eligible,
            item[2].repair_required,
            candidate_quality(item[1]),
            float(item[1].multihead_scores.get("novelty", 0.0) or 0.0),
            float(item[1].multihead_scores.get("rarity", 0.0) or 0.0),
        ),
        reverse=True,
    )
    for assignment, candidate, _decision in viable[:promote_count]:
        assignment.fate = CandidateFate.ACTIVE.value
        assignment.future_reactivation_condition = ""
        candidate.metadata.pop("dormant_kind", None)
        candidate.metadata["active_repair_floor"] = {
            "round": int(current_round or 0),
            "round_limit": int(round_limit or 0),
            "branch_factor": int(branch_factor or 0),
            "target_active_floor": int(floor),
            "previous_active_count": int(active_count),
            "reason": "stage_adaptive_active_floor_prevented_post_critique_collapse",
        }
        if _candidate_verification_blocks_final(candidate):
            candidate.metadata["final_answer_blocked_until_repaired"] = True


def _dormant_kind_counts(archived_candidates: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for data in archived_candidates.values():
        if not isinstance(data, dict):
            continue
        metadata = coerce_dict(data.get("metadata"))
        kind = str(metadata.get("dormant_kind") or "")
        if not kind:
            try:
                kind = classify_dormant_kind(candidate_from_dict(data))
            except Exception:
                kind = "low_priority_reserve"
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _archive_schema_from_dict(value: Any) -> dict[str, Any]:
    schema = {name: {"enabled": True} for name in ARCHIVE_NAMES}
    for name, settings in coerce_dict(value).items():
        schema[str(name)] = coerce_dict(settings) or {"enabled": True}
    return schema


def _float_policy(value: Any, *, default: float | None) -> float | None:
    if isinstance(value, str) and value.strip().lower() in {"", "auto", "adaptive", "model"}:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _int_policy(value: Any, *, default: int | None) -> int | None:
    if isinstance(value, str) and value.strip().lower() in {"", "auto", "adaptive", "model"}:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed


__all__ = [
    "ARCHIVE_NAMES",
    "ArchiveConstraintRecord",
    "ArchiveManager",
    "FateAssignment",
    "LatentParetoIntentArchive",
    "TerminalCandidateTombstone",
]
