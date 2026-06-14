from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome


@dataclass
class FailureRecord:
    candidate_id: str
    failure_signature: str
    inherited_gene_summary: str
    covered_by: str = ""
    future_reactivation_condition: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FailureArchive:
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, candidate: CandidateGenome, *, signature: str = "") -> FailureRecord:
        record = FailureRecord(
            candidate_id=candidate.id,
            failure_signature=signature or "; ".join(candidate.failure_lessons) or "culled_without_specific_signature",
            inherited_gene_summary=candidate.extract_inheritable_gene_summary(),
            future_reactivation_condition="use_if_same_failure_reappears_or_gene_complements_elite",
        )
        self.records[candidate.id] = record.to_dict()
        return record

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailureArchive":
        return cls(records=dict(data.get("records") or {}))
