"""Search state diagnosis and policy update actions."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationOperator
from cognitive_evolve_runtime.core.scalars import bounded_score
from cognitive_evolve_runtime.nexus._serde import coerce_dict, coerce_str_list, utc_now
from cognitive_evolve_runtime.nexus.adaptive_signals import in_top_band, observed_majority, score
from cognitive_evolve_runtime.nexus.obligations import (
    blocking_obligations_from_history,
    candidate_has_obligation_or_evidence_delta,
    repeated_proof_failure_counts,
    requires_proof_progress,
)
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike
from cognitive_evolve_runtime.archives.quality_diversity import live_reproductive_candidates
from cognitive_evolve_runtime.ranking.lineage_saturation import detect_lineage_saturation

STAGNATION_TYPES = [
    "None",
    "LocalOptimum",
    "AuxiliaryCollapse",
    "SemanticDrift",
    "VerificationBottleneck",
    "KnowledgeBottleneck",
    "DiversityCollapse",
    "PrematureCulling",
    "ToolOverfitting",
    "ProofObjectAbsence",
    "ObligationBottleneck",
    "SemanticLooping",
    "RouteIncomplete",
    "QuotaPaused",
    "Cancelled",
    "ModelSchemaQuotaOrTransport",
]

CONTROL_ACTIONS = [
    "continue",
    "reweight_policy",
    "increase_rarity_budget",
    "reactivate_dormant",
    "cross_archives",
    "quarantine_lineage",
    "strategy_restart",
    "compress_lessons",
    "return_failure_report",
    "core_extraction",
    "rare_inject",
    "scaffold_removal",
    "instantiate_formal_artifact",
    "discharge_obligation",
    "case_split",
    "construct_witness",
    "route_kill",
]


@dataclass
class SearchDiagnosis:
    stagnation_detected: bool = False
    stagnation_type: str = "None"
    over_explored_families: list[str] = field(default_factory=list)
    under_explored_families: list[str] = field(default_factory=list)
    prematurely_culled_genes: list[str] = field(default_factory=list)
    auxiliary_collapse_risk: float = 0.0
    semantic_drift_risk: float = 0.0
    recommended_actions: list[str] = field(default_factory=lambda: ["continue"])
    notes: str = ""
    grounded_information_gain: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.stagnation_type not in STAGNATION_TYPES:
            self.stagnation_type = "None" if not self.stagnation_detected else self.stagnation_type
        self.over_explored_families = coerce_str_list(self.over_explored_families)
        self.under_explored_families = coerce_str_list(self.under_explored_families)
        self.prematurely_culled_genes = coerce_str_list(self.prematurely_culled_genes)
        self.recommended_actions = coerce_str_list(self.recommended_actions) or ["continue"]
        self.auxiliary_collapse_risk = float(self.auxiliary_collapse_risk or 0.0)
        self.semantic_drift_risk = float(self.semantic_drift_risk or 0.0)
        self.grounded_information_gain = dict(self.grounded_information_gain) if isinstance(self.grounded_information_gain, dict) else {}
        self.metadata = coerce_dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchDiagnosis":
        return cls(
            stagnation_detected=bool(data.get("stagnation_detected", False)),
            stagnation_type=str(data.get("stagnation_type") or "None"),
            over_explored_families=coerce_str_list(data.get("over_explored_families")),
            under_explored_families=coerce_str_list(data.get("under_explored_families")),
            prematurely_culled_genes=coerce_str_list(data.get("prematurely_culled_genes")),
            auxiliary_collapse_risk=float(data.get("auxiliary_collapse_risk", 0.0) or 0.0),
            semantic_drift_risk=float(data.get("semantic_drift_risk", 0.0) or 0.0),
            recommended_actions=coerce_str_list(data.get("recommended_actions")),
            notes=str(data.get("notes") or ""),
            grounded_information_gain=dict(data.get("grounded_information_gain") or {}) if isinstance(data.get("grounded_information_gain"), dict) else {},
            metadata=coerce_dict(data.get("metadata")),
            created_at=str(data.get("created_at") or utc_now()),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, default=str)

    @classmethod
    def from_json(cls, text: str) -> "SearchDiagnosis":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("search diagnosis JSON must decode to an object")
        return cls.from_dict(data)


class SearchStateDiagnoser:
    def __init__(self, model: NexusModelLike | None = None) -> None:
        self.model = model

    def diagnose(
        self,
        *,
        population: list[CandidateGenome],
        archives: ArchiveManager,
        history: list[dict[str, Any]] | None = None,
        contract: Any | None = None,
        policy: EvolutionPolicy | None = None,
        tool_feedback: list[dict[str, Any]] | None = None,
    ) -> SearchDiagnosis:
        if self.model is not None and hasattr(self.model, "diagnose_search_state"):
            raw = self.model.diagnose_search_state(population=population, archives=archives, history=history or [], contract=contract, policy=policy)
            if isinstance(raw, SearchDiagnosis):
                return raw
            if isinstance(raw, dict):
                return SearchDiagnosis.from_dict(raw)
        if not population:
            return SearchDiagnosis(stagnation_detected=True, stagnation_type="KnowledgeBottleneck", recommended_actions=["rare_inject", "strategy_restart"], notes="empty population")
        low_gain = _low_grounded_information_gain(history=history, policy=policy, contract=contract)
        if low_gain:
            return SearchDiagnosis(
                stagnation_detected=True,
                stagnation_type="SemanticLooping",
                recommended_actions=["increase_rarity_budget", "rare_inject", "strategy_restart"],
                notes=f"low engine-grounded marginal information gain for {low_gain.get('count')} rounds",
                grounded_information_gain=low_gain,
            )
        live_context = live_reproductive_candidates(population)
        frontier = _frontier_candidates(population)
        diagnosis_context = frontier or live_context or population
        active_count = len([c for c in population if CandidateFate.normalize(c.current_fate) == CandidateFate.ACTIVE.value])
        incubating = [c for c in population if CandidateFate.normalize(c.current_fate) == CandidateFate.INCUBATING.value]
        dormant_count = len([c for c in population if CandidateFate.normalize(c.current_fate) == CandidateFate.DORMANT.value])
        auxiliary = [c for c in population if c.current_fate == CandidateFate.AUXILIARY or c.multihead_scores.get("auxiliary_value", 0.0) > max(c.multihead_scores.get("answer_likelihood", 0.0), c.multihead_scores.get("objective_alignment", 0.0))]
        rare_count = len(archives.rarity_archive.candidates)
        lineage_report = detect_lineage_saturation(diagnosis_context)
        proof_like = requires_proof_progress(contract)
        if proof_like:
            proof_failure_counts = repeated_proof_failure_counts(diagnosis_context)
            if proof_failure_counts:
                total_hard = sum(proof_failure_counts.values())
                absent = proof_failure_counts.get("proof_object_absent", 0)
                ledger = proof_failure_counts.get("ledger_non_progressing", 0)
                duplicate = proof_failure_counts.get("duplicate_formal_signature", 0)
                blocking = blocking_obligations_from_history(history, threshold=3)
                if absent >= max(2, len(diagnosis_context) // 2):
                    return SearchDiagnosis(
                        stagnation_detected=True,
                        stagnation_type="ProofObjectAbsence",
                        over_explored_families=lineage_report.saturated_families,
                        under_explored_families=["equation_set", "construction", "witness", "case_analysis"],
                        recommended_actions=["instantiate_formal_artifact", "discharge_obligation", "case_split", "construct_witness"],
                        notes=f"proof-like objective has many candidates without concrete formal artifacts; active_count={active_count}; incubating_count={len(incubating)}; dormant_count={dormant_count}",
                    )
                if ledger >= max(2, len(diagnosis_context) // 2) or blocking:
                    return SearchDiagnosis(
                        stagnation_detected=True,
                        stagnation_type="ObligationBottleneck",
                        over_explored_families=[item["id"] for item in blocking] or lineage_report.saturated_families,
                        under_explored_families=["obligation_delta", "blocking_gap_discharge"],
                        recommended_actions=["discharge_obligation", "instantiate_formal_artifact", "route_kill", "return_failure_report"],
                        notes=f"proof obligations repeat without verified discharge; active_count={active_count}; incubating_count={len(incubating)}; dormant_count={dormant_count}",
                    )
                if duplicate >= 2 or total_hard >= max(3, len(diagnosis_context)):
                    return SearchDiagnosis(
                        stagnation_detected=True,
                        stagnation_type="SemanticLooping",
                        over_explored_families=lineage_report.saturated_families,
                        under_explored_families=["new_formal_signature", "counterexample", "case_split"],
                        recommended_actions=["case_split", "construct_witness", "route_kill", "increase_rarity_budget"],
                        notes=f"formal/proof progress is looping over duplicate or hard-failed candidates; active_count={active_count}; incubating_count={len(incubating)}; dormant_count={dormant_count}",
                    )
        if active_count == 0 and incubating:
            reasons: dict[str, int] = {}
            for candidate in incubating:
                repair = candidate.metadata.get("repair_required") if isinstance(candidate.metadata, dict) else None
                if isinstance(repair, dict):
                    for blocker in repair.get("blockers", []) or []:
                        reasons[str(blocker)] = reasons.get(str(blocker), 0) + 1
            top_reasons = ", ".join(f"{key}:{value}" for key, value in sorted(reasons.items(), key=lambda item: item[1], reverse=True)[:5])
            return SearchDiagnosis(
                stagnation_detected=True,
                stagnation_type="VerificationBottleneck",
                recommended_actions=["repair", "instantiate_formal_artifact", "discharge_obligation", "tool_ground"],
                notes=f"no Active candidates, but {len(incubating)} Incubating repair candidates remain; repair_reasons={top_reasons or 'unknown'}; dormant_count={dormant_count}",
            )
        if auxiliary and len(auxiliary) >= max(2, len(population) // 2) and not archives.answer_archive:
            return SearchDiagnosis(
                stagnation_detected=True,
                stagnation_type="AuxiliaryCollapse",
                auxiliary_collapse_risk=min(1.0, len(auxiliary) / max(1, len(population))),
                over_explored_families=lineage_report.saturated_families,
                recommended_actions=["core_extraction", "scaffold_removal", "rare_inject"],
                notes="auxiliary candidates dominate without an answer elite",
            )
        if lineage_report.saturated:
            proposal_only_families: list[str] = []
            for family in lineage_report.saturated_families:
                family_candidates = [candidate for candidate in diagnosis_context if (candidate.lineage[0] if candidate.lineage else candidate.id) == family]
                if family_candidates and not any(candidate_has_obligation_or_evidence_delta(candidate) for candidate in family_candidates):
                    proposal_only_families.append(family)
            if proposal_only_families:
                return SearchDiagnosis(
                    stagnation_detected=True,
                    stagnation_type="SemanticLooping",
                    over_explored_families=proposal_only_families,
                    under_explored_families=["evidence_delta", "verified_evidence_ref", "source_grounding"],
                    recommended_actions=["quarantine_lineage", "route_kill", "increase_rarity_budget", "rare_inject"],
                    notes="lineage saturation without new evidence_delta detected; freeze proposal-only family and spend residual budget on rare/source-grounded candidates",
                )
            return SearchDiagnosis(
                stagnation_detected=True,
                stagnation_type="DiversityCollapse",
                over_explored_families=lineage_report.saturated_families,
                under_explored_families=["rarity", "dormant", "crossover"],
                recommended_actions=["quarantine_lineage", "increase_rarity_budget", "reactivate_dormant"],
                notes="lineage saturation detected",
            )
        if rare_count == 0 and policy and policy.rarity_budget > 0:
            return SearchDiagnosis(
                stagnation_detected=True,
                stagnation_type="KnowledgeBottleneck",
                under_explored_families=["edge_knowledge", "rare_recall"],
                recommended_actions=["rare_inject", "increase_rarity_budget"],
                notes="rarity budget exists but no rare archive entries are preserved",
            )
        attempted_frontier = [candidate for candidate in frontier if _has_attempted_verification(candidate)]
        failed_frontier = [candidate for candidate in attempted_frontier if _has_failed_verification(candidate)]
        if attempted_frontier and (
            len(failed_frontier) == len(attempted_frontier)
            or observed_majority(len(failed_frontier), len(attempted_frontier))
            or _high_value_failed_repair_targets(failed_frontier, attempted_frontier)
        ):
            return SearchDiagnosis(
                stagnation_detected=True,
                stagnation_type="VerificationBottleneck",
                recommended_actions=["repair", "tool_ground", "compress_lessons"],
                notes="frontier candidates are blocked by verification failures",
            )
        return SearchDiagnosis(stagnation_detected=False, recommended_actions=["continue"], notes="no generic stagnation detected")


def _frontier_candidates(population: list[CandidateGenome]) -> list[CandidateGenome]:
    # Diagnosis may inspect Dormant failures to explain a stall, but parent
    # selection remains separate and does not broadly sample Dormant candidates.
    allowed = {CandidateFate.ACTIVE.value, CandidateFate.ELITE.value, CandidateFate.INCUBATING.value, CandidateFate.DORMANT.value}
    out: list[CandidateGenome] = []
    for candidate in population:
        if CandidateFate.normalize(candidate.current_fate) not in allowed:
            continue
        if candidate.metadata.get("search_seed_not_final") and int(candidate.generation or 0) == 0:
            continue
        out.append(candidate)
    return out


def _has_attempted_verification(candidate: CandidateGenome) -> bool:
    result = getattr(candidate, "verification_result", {}) or {}
    return bool(candidate.verification_trace or (isinstance(result, dict) and result))


def _has_failed_verification(candidate: CandidateGenome) -> bool:
    result = getattr(candidate, "verification_result", {}) or {}
    if isinstance(result, dict) and result:
        if result.get("passed") is False or result.get("rank_eligible") is False or result.get("final_eligible") is False:
            return True
        if any(str(item).lower() in {"failed", "error"} for item in result.get("diagnostics", []) or []):
            return True
    return any(str(item.get("status") or "").lower() in {"failed", "error"} for item in candidate.verification_trace if isinstance(item, dict))


def _high_value_failed_repair_targets(candidates: list[CandidateGenome], context: list[CandidateGenome]) -> bool:
    for candidate in candidates:
        if (
            (score(candidate, "answer_likelihood") > 0 and in_top_band(candidate, context, "answer_likelihood"))
            or (score(candidate, "objective_alignment") > 0 and in_top_band(candidate, context, "objective_alignment"))
            or (score(candidate, "rarity") > 0 and in_top_band(candidate, context, "rarity"))
        ):
            return True
    return False


def _low_grounded_information_gain(*, history: list[dict[str, Any]] | None, policy: EvolutionPolicy | None, contract: Any | None) -> dict[str, Any]:
    metadata = getattr(policy, "metadata", {}) if policy is not None else {}
    current = metadata.get("engine_grounded_information_gain") if isinstance(metadata, dict) else None
    reports: list[dict[str, Any]] = []
    if isinstance(current, dict):
        reports.append(current)
    for item in reversed(history or []):
        if not isinstance(item, dict):
            continue
        report = item.get("grounded_information_gain")
        if isinstance(report, dict):
            reports.append(report)
        if len(reports) >= _low_gain_patience(contract):
            break
    patience = _low_gain_patience(contract)
    if len(reports) < patience:
        return {}
    recent = reports[:patience]
    if not all(float(report.get("marginal_information_gain") or 0.0) < 0.01 for report in recent):
        return {}
    return {
        "low_gain": True,
        "count": len(recent),
        "threshold": 0.01,
        "recent_gains": [float(report.get("marginal_information_gain") or 0.0) for report in recent],
        "current_signatures": list(current.get("current_signatures", [])) if isinstance(current, dict) else [],
    }


def _low_gain_patience(contract: Any | None) -> int:
    metadata = getattr(contract, "metadata", {}) if contract is not None else {}
    if isinstance(metadata, dict):
        try:
            return max(1, int(metadata.get("low_gain_patience") or 5))
        except (TypeError, ValueError):
            return 5
    return 5


def _observed_rarity_budget_increment(policy: EvolutionPolicy, archives: ArchiveManager | None) -> float:
    """Scale rarity budget pressure from observed rare depth and population size.

    The updater should not add a fixed rarity increment: a run with a shallow
    rare archive and a small population needs a stronger nudge than a broad run
    that already preserved many rare candidates/seeds.
    """

    try:
        current = max(0.0, float(policy.rarity_budget or 0.0))
    except (TypeError, ValueError):
        current = 0.0
    remaining = max(0.0, 1.0 - current)
    if remaining <= 0.0:
        return 0.0
    observation = _rarity_budget_observation(policy=policy, archives=archives)
    denominator = max(1, int(observation["population_size"]) + int(observation["rare_archive_depth"]) + int(observation["rare_seed_depth"]))
    return min(remaining, 1.0 / denominator)


def _rarity_budget_observation(*, policy: EvolutionPolicy, archives: ArchiveManager | None) -> dict[str, int]:
    if archives is not None:
        summary = archives.summary()
        fates = summary.get("fates") if isinstance(summary.get("fates"), dict) else {}
        population_size = max(
            1,
            len(fates),
            int(summary.get("answer_candidates") or 0)
            + int(summary.get("mechanism_elites") or 0)
            + int(summary.get("novelty_candidates") or 0)
            + int(summary.get("failure_records") or 0)
            + int(summary.get("auxiliary_candidates") or 0)
            + int(summary.get("dormant_candidates") or 0)
            + int(summary.get("incubating_candidates") or 0)
            + int(summary.get("project_patch_candidates") or 0),
        )
        rare_archive_depth = max(0, int(summary.get("rarity_candidates") or 0))
        rare_seed_depth = len(getattr(getattr(archives, "rarity_archive", None), "seeds", []) or [])
        return {
            "population_size": population_size,
            "rare_archive_depth": rare_archive_depth,
            "rare_seed_depth": rare_seed_depth,
        }
    policy_surface = len(policy.candidate_niches) + len(policy.mutation_operators) + len(policy.stagnation_actions)
    return {
        "population_size": max(1, policy_surface),
        "rare_archive_depth": 0,
        "rare_seed_depth": 0,
    }


class PolicyUpdater:
    def update(
        self,
        policy: EvolutionPolicy,
        diagnosis: SearchDiagnosis,
        *,
        model: NexusModelLike | None = None,
        archives: ArchiveManager | None = None,
    ) -> EvolutionPolicy:
        updated: EvolutionPolicy | None = None
        if model is not None and hasattr(model, "update_policy"):
            raw = model.update_policy(policy=policy, diagnosis=diagnosis)
            if isinstance(raw, EvolutionPolicy):
                updated = raw
            elif isinstance(raw, dict):
                updated = EvolutionPolicy.from_dict(raw)
        if updated is None:
            updated = EvolutionPolicy.from_dict(policy.to_dict())
        updated.updated_from_diagnoses.append(diagnosis.stagnation_type)
        actions = set(diagnosis.recommended_actions)
        if diagnosis.over_explored_families or diagnosis.under_explored_families or diagnosis.prematurely_culled_genes:
            metadata = dict(updated.metadata)
            pressure = {
                "source": "search_diagnosis",
                "stagnation_type": diagnosis.stagnation_type,
                "over_explored_families": list(dict.fromkeys(diagnosis.over_explored_families)),
                "under_explored_families": list(dict.fromkeys(diagnosis.under_explored_families)),
                "prematurely_culled_genes": list(dict.fromkeys(diagnosis.prematurely_culled_genes)),
                "effect": "selection_scoring_only_final_gate_unchanged",
            }
            metadata["selection_pressure"] = pressure
            eligibility = coerce_dict(metadata.get("eligibility_policy"))
            eligibility["selection_pressure"] = pressure
            metadata["eligibility_policy"] = eligibility
            updated.metadata = metadata
        mandatory_actions = [
            action
            for action in diagnosis.recommended_actions
            if action in {"instantiate_formal_artifact", "discharge_obligation", "case_split", "construct_witness", "route_kill"}
        ]
        if mandatory_actions:
            metadata = dict(updated.metadata)
            previous = coerce_str_list(metadata.get("mandatory_actions"))
            metadata["mandatory_actions"] = list(dict.fromkeys(previous + mandatory_actions))
            metadata["required_evidence_kinds"] = list(
                dict.fromkeys(coerce_str_list(metadata.get("required_evidence_kinds")) + ["formal_artifact", "obligation_delta"])
            )
            metadata["proof_progress_gate"] = "candidate_must_change_named_obligations_with_concrete_formal_objects"
            if diagnosis.over_explored_families:
                metadata["blocked_or_overexplored_obligations"] = list(dict.fromkeys(diagnosis.over_explored_families))
            updated.metadata = metadata
        if "quarantine_lineage" in actions and diagnosis.over_explored_families:
            metadata = dict(updated.metadata)
            previous = coerce_str_list(metadata.get("frozen_lineages"))
            metadata["frozen_lineages"] = list(dict.fromkeys(previous + diagnosis.over_explored_families))
            metadata["lineage_freeze_policy"] = "same_mechanism_requires_new_evidence_delta_before_selection_or_reactivation"
            updated.metadata = metadata
        if archives is not None:
            constraints = archives.constraints_for_policy(limit=20)
            if constraints:
                metadata = dict(updated.metadata)
                metadata["archive_constraints"] = constraints
                updated.metadata = metadata
        honesty_control = coerce_dict(diagnosis.metadata.get("honesty_control"))
        if honesty_control:
            pressure = _honesty_control_policy_pressure(honesty_control)
            if pressure:
                metadata = dict(updated.metadata)
                metadata["honesty_control"] = {
                    "source": "v23_honesty_pi_control",
                    "signal_id": honesty_control.get("signal_id"),
                    "error_vector": coerce_dict(honesty_control.get("error_vector")),
                    "pressure": pressure,
                    "effect": "search_pressure_only_verification_strength_unchanged",
                }
                metadata["frontier_exploration_pressure"] = pressure.get("frontier_exploration_pressure", 0.0)
                eligibility = coerce_dict(metadata.get("eligibility_policy"))
                eligibility["frontier_exploration_pressure"] = pressure
                metadata["eligibility_policy"] = eligibility
                updated.metadata = metadata
        if "increase_rarity_budget" in actions or "rare_inject" in actions:
            rarity_increment = _observed_rarity_budget_increment(updated, archives)
            updated.rarity_budget = min(1.0, updated.rarity_budget + rarity_increment)
            metadata = dict(updated.metadata)
            metadata["rarity_budget_update"] = {
                "source": "self_observed_archive_pressure",
                "increment": rarity_increment,
                **_rarity_budget_observation(policy=updated, archives=archives),
            }
            updated.metadata = metadata
            if MutationOperator.RARE_INJECT not in updated.mutation_operators:
                updated.mutation_operators.append(MutationOperator.RARE_INJECT)
        if "core_extraction" in actions and MutationOperator.CORE_EXTRACTION not in updated.mutation_operators:
            updated.mutation_operators.append(MutationOperator.CORE_EXTRACTION)
        if "scaffold_removal" in actions and MutationOperator.SCAFFOLD_REMOVAL not in updated.mutation_operators:
            updated.mutation_operators.append(MutationOperator.SCAFFOLD_REMOVAL)
        for action, operator in {
            "instantiate_formal_artifact": MutationOperator.INSTANTIATE_FORMAL_ARTIFACT,
            "discharge_obligation": MutationOperator.DISCHARGE_OBLIGATION,
            "case_split": MutationOperator.CASE_SPLIT,
            "construct_witness": MutationOperator.CONSTRUCT_WITNESS,
            "route_kill": MutationOperator.ROUTE_KILL,
        }.items():
            if action in actions and operator not in updated.mutation_operators:
                updated.mutation_operators.append(operator)
        if "reactivate_dormant" in actions and "DormantArchive" not in updated.archive_schema:
            updated.archive_schema["DormantArchive"] = {"enabled": True}
        return updated


def _honesty_control_policy_pressure(signal: dict[str, Any]) -> dict[str, float]:
    raw = coerce_dict(signal.get("pressure"))
    if not raw:
        return {}
    keys = (
        "adversarial_budget_pressure",
        "rarity_budget_pressure",
        "edge_seed_pressure",
        "frontier_exploration_pressure",
        "replay_verifier_pressure",
        "verification_pressure",
    )
    return {key: bounded_score(raw.get(key)) for key in keys if key in raw}


__all__ = ["CONTROL_ACTIONS", "STAGNATION_TYPES", "SearchDiagnosis", "SearchStateDiagnoser", "PolicyUpdater"]
