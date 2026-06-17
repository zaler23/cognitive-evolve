"""Serializable concept effects for the CognitiveEvolve v2 research bus.

The dataclasses are intentionally small, frozen payload types.  They mirror the
runtime's existing tolerant signal style: malformed input never raises from
``from_dict`` and instead returns the class-specific empty value.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar

from cognitive_evolve_runtime.nexus._serde import coerce_dict


T = TypeVar("T")


@dataclass(frozen=True)
class VerificationObligation:
    """A replayable verification obligation; the only effect that can raise strength."""

    id: str = ""
    verifier_fingerprint: str = ""
    must_pass: bool = False
    strength_contribution: int = 0
    replayable: bool = False
    origin: str = ""
    exogeneity_probe: dict[str, Any] = field(default_factory=dict)
    variety_probe: dict[str, Any] = field(default_factory=dict)
    falsification_budget: dict[str, Any] = field(default_factory=dict)
    replay_record: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def empty(cls) -> "VerificationObligation":
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VerificationObligation":
        try:
            raw = coerce_dict(data)
            if not raw:
                return cls.empty()
            return cls(
                id=str(raw.get("id") or ""),
                verifier_fingerprint=str(raw.get("verifier_fingerprint") or ""),
                must_pass=_bool(raw.get("must_pass")),
                strength_contribution=_int(raw.get("strength_contribution"), 0),
                replayable=_bool(raw.get("replayable")),
                origin=str(raw.get("origin") or ""),
                exogeneity_probe=coerce_dict(raw.get("exogeneity_probe")),
                variety_probe=coerce_dict(raw.get("variety_probe")),
                falsification_budget=coerce_dict(raw.get("falsification_budget")),
                replay_record=coerce_dict(raw.get("replay_record")),
            )
        except Exception:
            return cls.empty()


@dataclass(frozen=True)
class ArchiveDirective:
    """Directive to add/rebalance/archive behavior descriptors."""

    kind: str = ""
    descriptor: tuple[Any, ...] = field(default_factory=tuple)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "descriptor": list(self.descriptor), "payload": dict(self.payload)}

    @classmethod
    def empty(cls) -> "ArchiveDirective":
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ArchiveDirective":
        try:
            raw = coerce_dict(data)
            if not raw:
                return cls.empty()
            descriptor = raw.get("descriptor")
            if isinstance(descriptor, (list, tuple)):
                descriptor_tuple = tuple(descriptor)
            elif descriptor is None:
                descriptor_tuple = tuple()
            else:
                descriptor_tuple = (descriptor,)
            return cls(kind=str(raw.get("kind") or ""), descriptor=descriptor_tuple, payload=coerce_dict(raw.get("payload")))
        except Exception:
            return cls.empty()


@dataclass(frozen=True)
class BudgetDirective:
    """Directive to move compute toward higher estimated ROI."""

    target: str = ""
    weight: float = 0.0
    reason: str = ""
    roi_estimate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def empty(cls) -> "BudgetDirective":
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BudgetDirective":
        try:
            raw = coerce_dict(data)
            if not raw:
                return cls.empty()
            return cls(
                target=str(raw.get("target") or ""),
                weight=_float(raw.get("weight"), 0.0),
                reason=str(raw.get("reason") or ""),
                roi_estimate=_float(raw.get("roi_estimate"), 0.0),
            )
        except Exception:
            return cls.empty()


@dataclass(frozen=True)
class ContextTransform:
    """Context view change that protects long-run memory while pruning noise."""

    protect_refs: list[str] = field(default_factory=list)
    drop_refs: list[str] = field(default_factory=list)
    view_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def empty(cls) -> "ContextTransform":
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ContextTransform":
        try:
            raw = coerce_dict(data)
            if not raw:
                return cls.empty()
            return cls(
                protect_refs=_str_list(raw.get("protect_refs")),
                drop_refs=_str_list(raw.get("drop_refs")),
                view_hash=str(raw.get("view_hash") or ""),
            )
        except Exception:
            return cls.empty()


@dataclass(frozen=True)
class CandidateTransform:
    """Candidate-local transformation request."""

    candidate_id: str = ""
    kind: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    preserve_score_within: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def empty(cls) -> "CandidateTransform":
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CandidateTransform":
        try:
            raw = coerce_dict(data)
            if not raw:
                return cls.empty()
            return cls(
                candidate_id=str(raw.get("candidate_id") or ""),
                kind=str(raw.get("kind") or ""),
                payload=coerce_dict(raw.get("payload")),
                preserve_score_within=_float(raw.get("preserve_score_within"), 0.0),
            )
        except Exception:
            return cls.empty()


@dataclass(frozen=True)
class ContractDeltaProposal:
    """Lazy contract change proposal; approval is required before any effect."""

    delta_id: str = ""
    proposed_change: str = ""
    reason: str = ""
    objective_hash_before: str = ""
    objective_hash_after: str = ""
    requires_approval: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def empty(cls) -> "ContractDeltaProposal":
        return cls(requires_approval=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ContractDeltaProposal":
        try:
            raw = coerce_dict(data)
            if not raw:
                return cls.empty()
            return cls(
                delta_id=str(raw.get("delta_id") or ""),
                proposed_change=str(raw.get("proposed_change") or ""),
                reason=str(raw.get("reason") or ""),
                objective_hash_before=str(raw.get("objective_hash_before") or ""),
                objective_hash_after=str(raw.get("objective_hash_after") or ""),
                requires_approval=_bool(raw.get("requires_approval"), default=True),
            )
        except Exception:
            return cls.empty()


def effect_to_dict(item: Any) -> dict[str, Any]:
    """Best-effort effect serialization helper used by pure signal code."""

    if hasattr(item, "to_dict"):
        try:
            data = item.to_dict()
            return dict(data) if isinstance(data, dict) else {"value": data}
        except Exception:
            return {}
    if isinstance(item, dict):
        return dict(item)
    try:
        return dict(item)
    except Exception:
        return {"value": str(item)} if item is not None else {}


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled", "required"}


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]


__all__ = [
    "ArchiveDirective",
    "BudgetDirective",
    "CandidateTransform",
    "ContextTransform",
    "ContractDeltaProposal",
    "VerificationObligation",
    "effect_to_dict",
]
