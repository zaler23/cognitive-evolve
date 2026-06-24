"""Search state diagnosis and policy update actions."""
from __future__ import annotations

import json
import math
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
from cognitive_evolve_runtime.nexus.nextgen import budget_eligible_candidates
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
        self.metadata = coerce_dict(self.metadata)
        raw_type = str(self.stagnation_type or "None")
        retired_engineering_types = {"ProofObjectAbsence", "ObligationBottleneck", "VerificationBottleneck"}
        if raw_type in retired_engineering_types:
            self.metadata.setdefault("raw_retired_stagnation_type", raw_type)
            self.metadata.setdefault("answer_first_repair", "proof/source/verification bottlenecks are advisory only")
            self.stagnation_type = "DiversityCollapse" if self.stagnation_detected else "None"
        if self.stagnation_type not in STAGNATION_TYPES:
            self.stagnation_type = "None" if not self.stagnation_detected else self.stagnation_type
        self.over_explored_families = coerce_str_list(self.over_explored_families)
        self.under_explored_families = coerce_str_list(self.under_explored_families)
        self.prematurely_culled_genes = coerce_str_list(self.prematurely_culled_genes)
        self.recommended_actions = [
            action
            for action in (coerce_str_list(self.recommended_actions) or ["continue"])
            if action not in {"instantiate_formal_artifact", "discharge_obligation", "tool_ground", "return_failure_report"}
        ] or ["continue"]
        self.auxiliary_collapse_risk = float(self.auxiliary_collapse_risk or 0.0)
        self.semantic_drift_risk = float(self.semantic_drift_risk or 0.0)
        self.grounded_information_gain = dict(self.grounded_information_gain) if isinstance(self.grounded_information_gain, dict) else {}

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
            pressure = _open_family_pressure(budget_eligible_candidates(population) or population)
            return SearchDiagnosis(
                stagnation_detected=True,
                stagnation_type="SemanticLooping",
                recommended_actions=["increase_rarity_budget", "rare_inject", "strategy_restart"],
                over_explored_families=pressure["over_explored_families"],
                under_explored_families=pressure["under_explored_families"],
                notes=f"low engine-grounded marginal information gain for {low_gain.get('count')} rounds",
                grounded_information_gain=low_gain,
                metadata={"open_family_pressure": pressure},
            )
        live_context = budget_eligible_candidates(population)
        frontier = _frontier_candidates(population)
        diagnosis_context = frontier or live_context or population
        active_count = len([c for c in population if CandidateFate.normalize(c.current_fate) == CandidateFate.ACTIVE.value])
        incubating = [c for c in population if CandidateFate.normalize(c.current_fate) == CandidateFate.INCUBATING.value]
        dormant_count = len([c for c in population if CandidateFate.normalize(c.current_fate) == CandidateFate.DORMANT.value])
        auxiliary = [c for c in population if c.current_fate == CandidateFate.AUXILIARY or c.multihead_scores.get("auxiliary_value", 0.0) > max(c.multihead_scores.get("answer_likelihood", 0.0), c.multihead_scores.get("objective_alignment", 0.0))]
        rare_count = len(archives.rarity_archive.candidates)
        lineage_report = detect_lineage_saturation(diagnosis_context)
        if active_count == 0 and incubating:
            pressure = _open_family_pressure(diagnosis_context)
            reasons: dict[str, int] = {}
            for candidate in incubating:
                repair = candidate.metadata.get("repair_required") if isinstance(candidate.metadata, dict) else None
                if isinstance(repair, dict):
                    for blocker in repair.get("blockers", []) or []:
                        reasons[str(blocker)] = reasons.get(str(blocker), 0) + 1
            top_reasons = ", ".join(f"{key}:{value}" for key, value in sorted(reasons.items(), key=lambda item: item[1], reverse=True)[:5])
            return SearchDiagnosis(
                stagnation_detected=True,
                stagnation_type="DiversityCollapse",
                over_explored_families=pressure["over_explored_families"],
                under_explored_families=pressure["under_explored_families"],
                recommended_actions=["reactivate_dormant", "rare_inject", "increase_rarity_budget"],
                notes=f"no Active candidates, but {len(incubating)} Incubating answer candidates remain; advisory_reasons={top_reasons or 'unknown'}; dormant_count={dormant_count}",
                metadata={"open_family_pressure": pressure},
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
            pressure = _open_family_pressure(diagnosis_context, saturated_families=lineage_report.saturated_families)
            proposal_only_families: list[str] = []
            for family in lineage_report.saturated_families:
                family_candidates = [candidate for candidate in diagnosis_context if (candidate.lineage[0] if candidate.lineage else candidate.id) == family]
                if family_candidates and not any(candidate_has_obligation_or_evidence_delta(candidate) for candidate in family_candidates):
                    proposal_only_families.append(family)
            if proposal_only_families:
                return SearchDiagnosis(
                    stagnation_detected=True,
                    stagnation_type="SemanticLooping",
                    over_explored_families=list(dict.fromkeys([*proposal_only_families, *pressure["over_explored_families"]])),
                    under_explored_families=pressure["under_explored_families"],
                    recommended_actions=["quarantine_lineage", "route_kill", "increase_rarity_budget", "rare_inject"],
                    notes="lineage saturation detected; spend residual budget on low-sample families from the current open family distribution",
                    metadata={"open_family_pressure": pressure},
                )
            return SearchDiagnosis(
                stagnation_detected=True,
                stagnation_type="DiversityCollapse",
                over_explored_families=list(dict.fromkeys([*lineage_report.saturated_families, *pressure["over_explored_families"]])),
                under_explored_families=pressure["under_explored_families"],
                recommended_actions=["quarantine_lineage", "increase_rarity_budget", "reactivate_dormant"],
                notes="lineage saturation detected",
                metadata={"open_family_pressure": pressure},
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
            pressure = _open_family_pressure(diagnosis_context)
            return SearchDiagnosis(
                stagnation_detected=True,
                stagnation_type="DiversityCollapse",
                over_explored_families=pressure["over_explored_families"],
                under_explored_families=pressure["under_explored_families"],
                recommended_actions=["increase_rarity_budget", "rare_inject", "continue"],
                notes="verification failures are advisory; continue with broader answer exploration",
                metadata={"open_family_pressure": pressure},
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


def _open_family_pressure(candidates: list[CandidateGenome], *, saturated_families: list[str] | None = None) -> dict[str, Any]:
    """Return data-derived family pressure without finite domain classes.

    Family ids come from candidate metadata/content already produced by the run:
    canonical family id, mechanism family id, niche, core mechanism, lineage, or
    candidate id. The thresholds are distribution-derived, so this does not
    encode a fixed "engineering vs mechanism" taxonomy or a Critical-Branching
    special case.
    """

    distribution: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        family = _open_family_id(candidate)
        record = distribution.setdefault(family, {"count": 0, "candidate_ids": [], "terms": set()})
        record["count"] += 1
        record["candidate_ids"].append(candidate.id)
        record["terms"].update(_candidate_pressure_terms(candidate, family))
    if not distribution:
        return {
            "schema": "cogev.open_family_pressure.v1",
            "family_count": 0,
            "over_explored_families": [],
            "under_explored_families": [],
            "family_counts": {},
        }
    counts = [int(record["count"]) for record in distribution.values()]
    mean = sum(counts) / max(1, len(counts))
    variance = sum((count - mean) ** 2 for count in counts) / max(1, len(counts))
    stdev = math.sqrt(variance)
    saturated = {_normalize_family_term(item) for item in saturated_families or [] if str(item or "").strip()}
    over: list[str] = []
    under: list[str] = []
    low_sample_ceiling = max(1, math.floor(mean))
    high_sample_floor = mean + stdev
    for family, record in sorted(distribution.items(), key=lambda item: (int(item[1]["count"]), item[0])):
        count = int(record["count"])
        terms = sorted(str(item) for item in record["terms"] if str(item))
        if count <= low_sample_ceiling:
            under.extend(terms or [family])
        if family in saturated or count > high_sample_floor:
            over.extend(terms or [family])
    return {
        "schema": "cogev.open_family_pressure.v1",
        "family_count": len(distribution),
        "family_counts": {family: int(record["count"]) for family, record in sorted(distribution.items())},
        "mean_family_size": mean,
        "family_size_stdev": stdev,
        "over_explored_families": list(dict.fromkeys(over)),
        "under_explored_families": list(dict.fromkeys(under)),
        "basis": "candidate_metadata_and_content_distribution",
    }


def _open_family_id(candidate: CandidateGenome) -> str:
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    nextgen = coerce_dict(metadata.get("nextgen"))
    for value in (
        nextgen.get("canonical_mechanism_family_id"),
        nextgen.get("mechanism_family_id"),
        metadata.get("canonical_mechanism_family_id"),
        metadata.get("mechanism_family_id"),
        (candidate.niche_memberships[0] if candidate.niche_memberships else ""),
        candidate.core_mechanism,
        candidate.concise_claim,
        (candidate.lineage[0] if candidate.lineage else ""),
        candidate.id,
    ):
        term = _normalize_family_term(value)
        if term:
            return term
    return _normalize_family_term(candidate.id)


def _candidate_pressure_terms(candidate: CandidateGenome, family: str) -> set[str]:
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    nextgen = coerce_dict(metadata.get("nextgen"))
    values: list[Any] = [
        family,
        nextgen.get("canonical_mechanism_family_id"),
        nextgen.get("mechanism_family_id"),
        metadata.get("canonical_mechanism_family_id"),
        metadata.get("mechanism_family_id"),
        candidate.core_mechanism,
        candidate.concise_claim,
        *(candidate.niche_memberships or []),
        *(candidate.edge_knowledge_seeds or []),
        (candidate.lineage[0] if candidate.lineage else ""),
    ]
    return {term for term in (_normalize_family_term(value) for value in values) if term}


def _normalize_family_term(value: Any) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", "_").replace("|", " ").replace(":", " ").split())


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
        _preserve_search_kernel_metadata(policy, updated)
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
            metadata["answer_first_actions"] = {
                "source": "search_diagnosis",
                "actions": list(dict.fromkeys(mandatory_actions)),
                "effect": "exploration_prompt_pressure_only",
            }
            metadata.pop("required_evidence_kinds", None)
            metadata.pop("proof_progress_gate", None)
            updated.metadata = metadata
        if "quarantine_lineage" in actions and diagnosis.over_explored_families:
            metadata = dict(updated.metadata)
            metadata["lineage_pressure_advisory"] = {
                "families": list(dict.fromkeys(diagnosis.over_explored_families)),
                "effect": "selection_scoring_only_not_a_freeze",
            }
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


def _preserve_search_kernel_metadata(previous: EvolutionPolicy, updated: EvolutionPolicy) -> None:
    previous_metadata = coerce_dict(getattr(previous, "metadata", None))
    if not previous_metadata:
        return
    metadata = coerce_dict(getattr(updated, "metadata", None))
    for key in (
        "seed_harvest",
        "seed_coverage",
        "target_perturb_seed_judgment",
        "factor_resurrection_summary",
        "minimal_core_ablation",
        "seed_active_frontier",
        "algorithm_efficiency",
        "model_parallel_efficiency",
        "seed_reservoir_ref",
        "_seed_reservoir_sidecar_payload",
    ):
        if key in previous_metadata and key not in metadata:
            metadata[key] = previous_metadata[key]
    updated.metadata = metadata


__all__ = ["CONTROL_ACTIONS", "STAGNATION_TYPES", "SearchDiagnosis", "SearchStateDiagnoser", "PolicyUpdater"]
