"""M6 niche and species isolation primitives.

Niches are runtime-local search lanes.  A species assignment is intentionally
scoped by niche so a mechanism that is valid in one lane cannot silently carry
its identity, closure, or scheduler reward into another lane.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus._serde import coerce_dict, coerce_str_list, stable_hash, utc_now


DEFAULT_NICHE_ID = "default"


@dataclass(frozen=True)
class NicheProfile:
    niche_id: str
    objective_scope: str = ""
    resource_budget: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "niche_id", normalize_niche_id(self.niche_id))
        object.__setattr__(self, "objective_scope", str(self.objective_scope or ""))
        object.__setattr__(self, "resource_budget", max(0.0, _finite_float(self.resource_budget, default=1.0)))
        object.__setattr__(self, "metadata", coerce_dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpeciesAssignment:
    candidate_id: str
    niche_id: str
    species_id: str
    signature: str
    assigned_at_utc: str = field(default_factory=utc_now)
    version: str = "m6-species-assignment/v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeciesAssignment":
        return cls(
            candidate_id=str(data.get("candidate_id") or ""),
            niche_id=normalize_niche_id(data.get("niche_id")),
            species_id=str(data.get("species_id") or ""),
            signature=str(data.get("signature") or ""),
            assigned_at_utc=str(data.get("assigned_at_utc") or utc_now()),
            version=str(data.get("version") or "m6-species-assignment/v1"),
        )


class SpeciesIndex:
    """Deterministic, niche-scoped species index.

    The index mutates candidate metadata only when ``isolate=True``.  That
    default is deliberate: species assignment is the point where runtime code
    should collapse a candidate to one primary niche lane.
    """

    def __init__(self, assignments: dict[str, SpeciesAssignment | dict[str, Any]] | None = None) -> None:
        self.assignments: dict[str, SpeciesAssignment] = {}
        self.members_by_species: dict[str, list[str]] = {}
        for candidate_id, assignment in (assignments or {}).items():
            parsed = assignment if isinstance(assignment, SpeciesAssignment) else SpeciesAssignment.from_dict(assignment)
            self.assignments[str(candidate_id)] = parsed
            self.members_by_species.setdefault(parsed.species_id, []).append(parsed.candidate_id)

    def assign(self, candidate: CandidateGenome, niche_id: str | None = None, *, isolate: bool = True) -> SpeciesAssignment:
        niche = resolve_niche_id(candidate, preferred=niche_id)
        signature = species_signature(candidate, niche)
        species_id = species_id_for_signature(niche, signature)
        assignment = SpeciesAssignment(
            candidate_id=candidate.id,
            niche_id=niche,
            species_id=species_id,
            signature=signature,
        )
        previous = self.assignments.get(candidate.id)
        if previous is not None and previous.species_id in self.members_by_species:
            self.members_by_species[previous.species_id] = [item for item in self.members_by_species[previous.species_id] if item != candidate.id]
        self.assignments[candidate.id] = assignment
        members = self.members_by_species.setdefault(species_id, [])
        if candidate.id not in members:
            members.append(candidate.id)
        if isolate:
            isolate_candidate_to_niche(candidate, niche, species_id=species_id, assignment=assignment)
        return assignment

    def assignment_for(self, candidate_or_id: CandidateGenome | str) -> SpeciesAssignment | None:
        candidate_id = candidate_or_id.id if isinstance(candidate_or_id, CandidateGenome) else str(candidate_or_id)
        return self.assignments.get(candidate_id)

    get_assignment = assignment_for
    assign_candidate = assign

    def species_for(self, candidate_or_id: CandidateGenome | str) -> str:
        assignment = self.assignment_for(candidate_or_id)
        return assignment.species_id if assignment is not None else ""

    def members(self, species_id: str) -> tuple[str, ...]:
        return tuple(self.members_by_species.get(str(species_id), ()))

    def to_dict(self) -> dict[str, Any]:
        return {"assignments": {candidate_id: assignment.to_dict() for candidate_id, assignment in self.assignments.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeciesIndex":
        return cls(assignments=coerce_dict(data.get("assignments")))


def normalize_niche_id(value: Any) -> str:
    raw = str(value or "").strip().lower().replace(" ", "_")
    return raw or DEFAULT_NICHE_ID


def resolve_niche_id(candidate: CandidateGenome | dict[str, Any] | None, *, preferred: str | None = None) -> str:
    if preferred:
        return normalize_niche_id(preferred)
    if isinstance(candidate, CandidateGenome):
        metadata = coerce_dict(candidate.metadata)
        for key in ("niche_id", "primary_niche_id", "target_niche_id"):
            if metadata.get(key):
                return normalize_niche_id(metadata[key])
        runtime = coerce_dict(metadata.get("niche_runtime"))
        for key in ("target_niche_id", "niche_id"):
            if runtime.get(key):
                return normalize_niche_id(runtime[key])
        if candidate.niche_memberships:
            return normalize_niche_id(candidate.niche_memberships[0])
        if candidate.core_mechanism:
            return normalize_niche_id(candidate.core_mechanism)
        return DEFAULT_NICHE_ID
    data = coerce_dict(candidate)
    for key in ("niche_id", "primary_niche_id", "target_niche_id"):
        if data.get(key):
            return normalize_niche_id(data[key])
    memberships = coerce_str_list(data.get("niche_memberships"))
    return normalize_niche_id(memberships[0] if memberships else DEFAULT_NICHE_ID)


def species_signature(candidate: CandidateGenome, niche_id: str | None = None) -> str:
    niche = normalize_niche_id(niche_id or resolve_niche_id(candidate))
    metadata = coerce_dict(candidate.metadata)
    search_space = coerce_dict(metadata.get("search_space"))
    explicit = (
        metadata.get("species_id")
        or metadata.get("species_key")
        or metadata.get("species_hint")
        or metadata.get("family_id")
        or search_space.get("family_id")
    )
    if explicit:
        traits = [str(explicit)]
    else:
        traits = [
            candidate.artifact_type,
            candidate.core_mechanism,
            *(candidate.novelty_descriptors[:4]),
            *(candidate.edge_knowledge_seeds[:2]),
        ]
    clean_traits = [str(item).strip().lower() for item in traits if str(item).strip()]
    if not clean_traits:
        clean_traits = [candidate.concise_claim or candidate.id]
    return stable_hash({"niche_id": niche, "traits": clean_traits})


def species_id_for_signature(niche_id: str, signature: str) -> str:
    return f"sp:{normalize_niche_id(niche_id)}:{str(signature)[:12]}"


def assign_species(candidate: CandidateGenome, niche_id: str | None = None, *, index: SpeciesIndex | None = None, isolate: bool = True) -> SpeciesAssignment:
    runtime_index = index or SpeciesIndex()
    return runtime_index.assign(candidate, niche_id=niche_id, isolate=isolate)


def isolate_candidate_to_niche(
    candidate: CandidateGenome,
    niche_id: str,
    *,
    species_id: str | None = None,
    assignment: SpeciesAssignment | None = None,
) -> CandidateGenome:
    niche = normalize_niche_id(niche_id)
    candidate.niche_memberships = [niche]
    metadata = coerce_dict(candidate.metadata)
    metadata["niche_id"] = niche
    metadata["primary_niche_id"] = niche
    if species_id:
        metadata["species_id"] = str(species_id)
    if assignment is not None:
        metadata["species_assignment"] = assignment.to_dict()
    candidate.metadata = metadata
    return candidate


def is_niche_isolated(candidate: CandidateGenome, niche_id: str | None = None) -> bool:
    niche = normalize_niche_id(niche_id or resolve_niche_id(candidate))
    memberships = [normalize_niche_id(item) for item in candidate.niche_memberships if str(item).strip()]
    if len(set(memberships)) > 1:
        return False
    if memberships and memberships[0] != niche:
        return False
    metadata = coerce_dict(candidate.metadata)
    metadata_niche = metadata.get("niche_id") or metadata.get("primary_niche_id")
    if metadata_niche and normalize_niche_id(metadata_niche) != niche:
        return False
    assignment = coerce_dict(metadata.get("species_assignment"))
    if assignment.get("niche_id") and normalize_niche_id(assignment.get("niche_id")) != niche:
        return False
    return bool(memberships or metadata_niche or assignment)


def assert_niche_isolated(candidate: CandidateGenome, niche_id: str | None = None) -> None:
    if not is_niche_isolated(candidate, niche_id=niche_id):
        raise ValueError(f"candidate {candidate.id} is not isolated to niche {normalize_niche_id(niche_id or resolve_niche_id(candidate))}")


def _finite_float(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return default
    return parsed


__all__ = [
    "DEFAULT_NICHE_ID",
    "NicheProfile",
    "SpeciesAssignment",
    "SpeciesIndex",
    "assign_species",
    "assert_niche_isolated",
    "is_niche_isolated",
    "isolate_candidate_to_niche",
    "normalize_niche_id",
    "resolve_niche_id",
    "species_id_for_signature",
    "species_signature",
]
