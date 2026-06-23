"""Candidate dossier types for the generic Exploration Fabric."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.core.serialization import coerce_str_list, utc_now
from .advisory import assert_advisory_payload


@dataclass
class CandidateDossier:
    """A domain-neutral expansion record for a candidate.

    Dossiers are advisory artifacts.  They can make a candidate easier to rank,
    critique, mutate, or verify later, but they never certify correctness.
    """

    candidate_id: str
    summary: str = ""
    expanded_content: dict[str, Any] = field(default_factory=dict)
    applicability_bounds: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    validation_paths: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    counterexamples: list[str] = field(default_factory=list)
    differentiators: list[str] = field(default_factory=list)
    effect_hypotheses: list[str] = field(default_factory=list)
    maturity_level: int = 0
    advisory: bool = True
    diagnostics: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    @property
    def field_completeness(self) -> int:
        fields = [
            self.summary,
            self.expanded_content,
            self.applicability_bounds,
            self.assumptions,
            self.risks,
            self.validation_paths,
            self.variants,
            self.counterexamples,
            self.differentiators,
            self.effect_hypotheses,
        ]
        return sum(1 for item in fields if bool(item))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["advisory"] = True
        payload["field_completeness"] = self.field_completeness
        assert_advisory_payload({k: v for k, v in payload.items() if k not in {"diagnostics", "advisory"}})
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateDossier":
        diagnostics = coerce_str_list(data.get("diagnostics")) if isinstance(data, dict) else []
        if isinstance(data, dict) and data.get("advisory") is False:
            diagnostics.append("advisory_false_coerced_to_true")
        payload = dict(data or {})
        payload.pop("field_completeness", None)
        payload["diagnostics"] = diagnostics
        expanded = payload.get("expanded_content")
        payload["expanded_content"] = dict(expanded) if isinstance(expanded, dict) else {}
        payload["advisory"] = True
        for key in (
            "applicability_bounds",
            "assumptions",
            "risks",
            "validation_paths",
            "variants",
            "counterexamples",
            "differentiators",
            "effect_hypotheses",
        ):
            payload[key] = coerce_str_list(payload.get(key))
        payload["candidate_id"] = str(payload.get("candidate_id") or "")
        payload["summary"] = str(payload.get("summary") or "")
        payload["maturity_level"] = int(payload.get("maturity_level") or 0)
        payload["created_at"] = str(payload.get("created_at") or utc_now())
        candidate = cls(**{k: payload[k] for k in cls.__dataclass_fields__ if k in payload})
        candidate.to_dict()  # validates advisory boundary
        return candidate


@dataclass
class DossierIndexEntry:
    """Bounded checkpoint entry for a dossier sidecar file."""

    candidate_id: str
    summary: str = ""
    field_completeness: int = 0
    content_ref: str = ""
    content_sha256: str = ""
    advisory: bool = True
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": str(self.candidate_id or ""),
            "summary": str(self.summary or "")[:1000],
            "field_completeness": max(0, int(self.field_completeness or 0)),
            "content_ref": str(self.content_ref or ""),
            "content_sha256": str(self.content_sha256 or ""),
            "advisory": True,
            "created_at": str(self.created_at or utc_now()),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DossierIndexEntry":
        return cls(
            candidate_id=str(data.get("candidate_id") or ""),
            summary=str(data.get("summary") or ""),
            field_completeness=int(data.get("field_completeness") or 0),
            content_ref=str(data.get("content_ref") or ""),
            content_sha256=str(data.get("content_sha256") or ""),
            advisory=True,
            created_at=str(data.get("created_at") or utc_now()),
        )


__all__ = ["CandidateDossier", "DossierIndexEntry"]
