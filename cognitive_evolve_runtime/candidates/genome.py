"""Serializable candidate genomes for Nexus evolution."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, TypedDict

from cognitive_evolve_runtime.nexus._serde import coerce_dict, coerce_str_list, stable_hash, utc_now
from cognitive_evolve_runtime.core.scalars import bounded_score_or_none as _bounded_score_or_none


class CandidateFate(str, Enum):
    ACTIVE = "Active"
    ELITE = "Elite"
    INCUBATING = "Incubating"
    DORMANT = "Dormant"
    AUXILIARY = "Auxiliary"
    CULLED = "Culled"
    FAILED = "Failed"

    @classmethod
    def normalize(cls, value: Any, *, default: str | None = None) -> str:
        if isinstance(value, cls):
            return value.value
        raw = str(value or "").strip()
        if raw in cls.ALL:
            return raw
        return default if default is not None else cls.ACTIVE.value


CandidateFate.ALL = {fate.value for fate in CandidateFate}  # type: ignore[attr-defined]


class CandidateMetadata(TypedDict, total=False):
    seed_type: str
    search_seed_not_final: bool
    exploration_source: str
    score_source: str
    model_seed_error: str
    model_offspring_degraded: bool
    reactivated_in_round: int
    offspring_verification: dict[str, Any]
    target_obligation_ids: list[str]
    required_evidence_kinds: list[str]
    evidence_need: str
    source_grounding_required: bool
    created_in_round: int
    stage_eligibility: dict[str, Any]
    repair_required: dict[str, Any]
    repair_attempts: int
    incubation_started_round: int
    max_incubation_attempts: int
    max_incubation_age: int
    state_transition_reason: str
    reactivation_condition: str
    hard_reject_reason: str
    dormant_kind: str
    active_repair_floor: dict[str, Any]
    final_answer_blocked_until_repaired: bool
    claim_maturity_stage: str
    failure_micro_guidance: list[dict[str, Any]]
    failure_classification: dict[str, Any]
    repair_context: dict[str, Any]
    offspring_repair_lane: dict[str, Any]
    repair_seed: dict[str, Any]
    dormant_repair_reactivation: dict[str, Any]
    failure_archive_reseed: dict[str, Any]
    dormant_recovery_reject: dict[str, Any]
    selection_pressure: dict[str, Any]
    bootstrap_entry_survival: dict[str, Any]
    generation_plan_id: str
    generation_plan_source: str
    generation_plan_round: int
    generation_plan_fate: str


@dataclass(frozen=True)
class CandidateIdentity:
    id: str
    parent_ids: tuple[str, ...]
    generation: int
    lineage: tuple[str, ...]
    contract_hash: str
    created_at: str


@dataclass(frozen=True)
class CandidateState:
    artifact: Any
    artifact_type: str
    concise_claim: str
    core_mechanism: str
    current_fate: str
    multihead_scores: dict[str, float]
    verification_result: dict[str, Any]
    evidence_refs: list[dict[str, Any]]
    source_bindings: list[dict[str, Any]]
    evidence_delta: dict[str, Any]
    metadata: CandidateMetadata


def _candidate_id() -> str:
    return "C" + uuid.uuid4().hex[:12]


@dataclass
class CandidateGenome:
    id: str = field(default_factory=_candidate_id)
    parent_ids: list[str] = field(default_factory=list)
    generation: int = 0
    lineage: list[str] = field(default_factory=list)
    artifact: Any = ""
    artifact_type: str = "answer"
    concise_claim: str = ""
    core_mechanism: str = ""
    assumptions: list[str] = field(default_factory=list)
    missing_parts: list[str] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)
    edge_knowledge_seeds: list[str] = field(default_factory=list)
    inherited_genes: list[str] = field(default_factory=list)
    mutation_history: list[str] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    verification_trace: list[dict[str, Any]] = field(default_factory=list)
    formal_artifacts: list[dict[str, Any]] = field(default_factory=list)
    proof_obligations: list[dict[str, Any]] = field(default_factory=list)
    obligation_delta: dict[str, Any] = field(default_factory=dict)
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)
    source_bindings: list[dict[str, Any]] = field(default_factory=list)
    evidence_delta: dict[str, Any] = field(default_factory=dict)
    verification_result: dict[str, Any] = field(default_factory=dict)
    novelty_descriptors: list[str] = field(default_factory=list)
    niche_memberships: list[str] = field(default_factory=list)
    failure_lessons: list[str] = field(default_factory=list)
    current_fate: str = CandidateFate.ACTIVE.value
    multihead_scores: dict[str, float] = field(default_factory=dict)
    contract_hash: str = ""
    created_at: str = field(default_factory=utc_now)
    metadata: CandidateMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.parent_ids = coerce_str_list(self.parent_ids)
        if not self.lineage:
            self.lineage = list(self.parent_ids) + [self.id]
        else:
            self.lineage = coerce_str_list(self.lineage)
            if self.id not in self.lineage:
                self.lineage.append(self.id)
        self.assumptions = coerce_str_list(self.assumptions)
        self.missing_parts = coerce_str_list(self.missing_parts)
        self.uncertainty_notes = coerce_str_list(self.uncertainty_notes)
        self.edge_knowledge_seeds = coerce_str_list(self.edge_knowledge_seeds)
        self.inherited_genes = coerce_str_list(self.inherited_genes)
        self.mutation_history = coerce_str_list(self.mutation_history)
        self.tool_results = [dict(item) for item in self.tool_results if isinstance(item, dict)]
        self.verification_trace = [dict(item) for item in self.verification_trace if isinstance(item, dict)]
        self.formal_artifacts = [dict(item) for item in self.formal_artifacts if isinstance(item, dict)]
        self.proof_obligations = [dict(item) for item in self.proof_obligations if isinstance(item, dict)]
        self.obligation_delta = coerce_dict(self.obligation_delta)
        self.evidence_refs = [dict(item) for item in self.evidence_refs if isinstance(item, dict)]
        self.source_bindings = [dict(item) for item in self.source_bindings if isinstance(item, dict)]
        self.evidence_delta = coerce_dict(self.evidence_delta)
        self.verification_result = coerce_dict(self.verification_result)
        self.novelty_descriptors = coerce_str_list(self.novelty_descriptors)
        self.niche_memberships = coerce_str_list(self.niche_memberships)
        self.failure_lessons = coerce_str_list(self.failure_lessons)
        self.current_fate = CandidateFate.normalize(self.current_fate)
        self.multihead_scores = _coerce_scores(self.multihead_scores)
        self.metadata = coerce_dict(self.metadata)

    @property
    def identity(self) -> CandidateIdentity:
        return CandidateIdentity(
            id=self.id,
            parent_ids=tuple(self.parent_ids),
            generation=self.generation,
            lineage=tuple(self.lineage),
            contract_hash=self.contract_hash,
            created_at=self.created_at,
        )

    @property
    def state(self) -> CandidateState:
        return CandidateState(
            artifact=self.artifact,
            artifact_type=self.artifact_type,
            concise_claim=self.concise_claim,
            core_mechanism=self.core_mechanism,
            current_fate=self.current_fate,
            multihead_scores=dict(self.multihead_scores),
            verification_result=dict(self.verification_result),
            evidence_refs=list(self.evidence_refs),
            source_bindings=list(self.source_bindings),
            evidence_delta=dict(self.evidence_delta),
            metadata=dict(self.metadata),
        )

    @property
    def genome_hash(self) -> str:
        data = self.to_dict()
        data.pop("created_at", None)
        return stable_hash(data)

    def mark_fate(self, fate: str) -> "CandidateGenome":
        normalized = CandidateFate.normalize(fate, default="")
        if normalized not in CandidateFate.ALL:
            raise ValueError(f"unknown candidate fate: {fate}")
        self.current_fate = normalized
        return self

    def add_tool_feedback(self, feedback: Any) -> None:
        if hasattr(feedback, "to_dict"):
            self.tool_results.append(feedback.to_dict())
        elif isinstance(feedback, dict):
            self.tool_results.append(dict(feedback))
        else:
            self.tool_results.append({"tool_id": "unknown", "status": "unknown", "raw_output_ref": str(feedback)})

    def add_verification_feedback(self, feedback: Any) -> None:
        if hasattr(feedback, "to_dict"):
            self.verification_trace.append(feedback.to_dict())
        elif isinstance(feedback, dict):
            self.verification_trace.append(dict(feedback))
        else:
            self.verification_trace.append({"status": "unknown", "raw_output_ref": str(feedback)})

    def extract_inheritable_gene_summary(self) -> str:
        parts = [self.core_mechanism or self.concise_claim]
        parts.extend(self.edge_knowledge_seeds[:2])
        parts.extend(self.failure_lessons[:2])
        return "; ".join(part for part in parts if part)[:1000]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateGenome":
        return cls(
            id=str(data.get("id") or _candidate_id()),
            parent_ids=coerce_str_list(data.get("parent_ids")),
            generation=int(data.get("generation") or 0),
            lineage=coerce_str_list(data.get("lineage")),
            artifact=data.get("artifact", ""),
            artifact_type=str(data.get("artifact_type") or "answer"),
            concise_claim=str(data.get("concise_claim") or ""),
            core_mechanism=str(data.get("core_mechanism") or ""),
            assumptions=coerce_str_list(data.get("assumptions")),
            missing_parts=coerce_str_list(data.get("missing_parts")),
            uncertainty_notes=coerce_str_list(data.get("uncertainty_notes")),
            edge_knowledge_seeds=coerce_str_list(data.get("edge_knowledge_seeds")),
            inherited_genes=coerce_str_list(data.get("inherited_genes")),
            mutation_history=coerce_str_list(data.get("mutation_history")),
            tool_results=[dict(item) for item in data.get("tool_results", []) if isinstance(item, dict)],
            verification_trace=[dict(item) for item in data.get("verification_trace", []) if isinstance(item, dict)],
            formal_artifacts=[dict(item) for item in data.get("formal_artifacts", []) if isinstance(item, dict)],
            proof_obligations=[dict(item) for item in data.get("proof_obligations", []) if isinstance(item, dict)],
            obligation_delta=coerce_dict(data.get("obligation_delta")),
            evidence_refs=[dict(item) for item in data.get("evidence_refs", []) if isinstance(item, dict)],
            source_bindings=[dict(item) for item in data.get("source_bindings", []) if isinstance(item, dict)],
            evidence_delta=coerce_dict(data.get("evidence_delta")),
            verification_result=coerce_dict(data.get("verification_result")),
            novelty_descriptors=coerce_str_list(data.get("novelty_descriptors")),
            niche_memberships=coerce_str_list(data.get("niche_memberships")),
            failure_lessons=coerce_str_list(data.get("failure_lessons")),
            current_fate=CandidateFate.normalize(data.get("current_fate")),
            multihead_scores=_coerce_scores(data.get("multihead_scores")),
            contract_hash=str(data.get("contract_hash") or ""),
            created_at=str(data.get("created_at") or utc_now()),
            metadata=coerce_dict(data.get("metadata")),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, default=str, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "CandidateGenome":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("candidate genome JSON must decode to an object")
        return cls.from_dict(data)


def _coerce_scores(value: Any) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key, raw_score in coerce_dict(value).items():
        score = _bounded_score_or_none(raw_score)
        if score is not None:
            scores[str(key)] = score
    return scores


def _is_number(value: Any) -> bool:
    return _bounded_score_or_none(value) is not None




def _bounded_score(value: Any) -> float:
    score = _bounded_score_or_none(value)
    return 0.0 if score is None else score


@dataclass
class CandidatePopulation:
    candidates: list[CandidateGenome] = field(default_factory=list)

    @property
    def active(self) -> list[CandidateGenome]:
        return [candidate for candidate in self.candidates if CandidateFate.normalize(candidate.current_fate) == CandidateFate.ACTIVE.value]

    def by_id(self) -> dict[str, CandidateGenome]:
        return {candidate.id: candidate for candidate in self.candidates}

    def integrate(self, offspring: list[CandidateGenome] | CandidateGenome) -> None:
        incoming = [offspring] if isinstance(offspring, CandidateGenome) else list(offspring)
        known = self.by_id()
        for candidate in incoming:
            known[candidate.id] = candidate
        self.candidates = list(known.values())

    def to_dict(self) -> dict[str, Any]:
        return {"candidates": [candidate.to_dict() for candidate in self.candidates]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidatePopulation":
        return cls([candidate_from_dict(item) for item in data.get("candidates", []) if isinstance(item, dict)])


def candidate_from_dict(data: dict[str, Any]) -> CandidateGenome:
    """Decode a serialized genome while preserving project-candidate fields.

    The base ``CandidateGenome`` deliberately does not import
    ``ProjectCandidateGenome`` at module import time, because that subclass also
    imports the base class.  This late import keeps package initialization simple
    while ensuring project patch candidates reload with their full patch genome.
    """

    if isinstance(data, dict) and (data.get("artifact_type") in {"project_patch", "patch"} or data.get("patch_set")):
        from cognitive_evolve_runtime.candidates.project_candidate import ProjectCandidateGenome

        return ProjectCandidateGenome.from_dict(data)
    return CandidateGenome.from_dict(data)


__all__ = [
    "CandidateFate",
    "CandidateIdentity",
    "CandidateMetadata",
    "CandidateState",
    "CandidateGenome",
    "CandidatePopulation",
    "candidate_from_dict",
]
