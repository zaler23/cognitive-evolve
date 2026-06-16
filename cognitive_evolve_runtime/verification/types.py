"""Typed verification spine payloads."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from cognitive_evolve_runtime.nexus._serde import coerce_dict
from .ladder import VerificationStrength


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    score: float = 0.0
    strength: VerificationStrength = VerificationStrength.NONE
    evidence_ref: str = ""
    replayable: bool = False
    diagnostics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["strength"] = self.strength.name
        data["strength_value"] = int(self.strength)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VerificationResult":
        raw = coerce_dict(data)
        return cls(
            passed=bool(raw.get("passed")),
            score=_float(raw.get("score"), 0.0),
            strength=VerificationStrength.from_value(raw.get("strength") or raw.get("strength_value")),
            evidence_ref=str(raw.get("evidence_ref") or ""),
            replayable=bool(raw.get("replayable")),
            diagnostics=[str(item) for item in raw.get("diagnostics", []) if item],
            metadata=coerce_dict(raw.get("metadata")),
        )


class SynthesizedVerifier(Protocol):
    verifier_id: str
    strength: VerificationStrength
    fingerprint: str

    def check(self, candidate: Any) -> VerificationResult: ...


@dataclass(frozen=True)
class VerificationPlan:
    verifier_id: str
    strength: VerificationStrength
    modality: str
    verifier_fingerprint: str
    replayable: bool = False
    diagnostics: list[str] = field(default_factory=list)
    reformulations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["strength"] = self.strength.name
        data["strength_value"] = int(self.strength)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VerificationPlan":
        raw = coerce_dict(data)
        return cls(
            verifier_id=str(raw.get("verifier_id") or "noop-verifier"),
            strength=VerificationStrength.from_value(raw.get("strength") or raw.get("strength_value")),
            modality=str(raw.get("modality") or "none"),
            verifier_fingerprint=str(raw.get("verifier_fingerprint") or raw.get("fingerprint") or ""),
            replayable=bool(raw.get("replayable")),
            diagnostics=[str(item) for item in raw.get("diagnostics", []) if item],
            reformulations=[dict(item) for item in raw.get("reformulations", []) if isinstance(item, dict)],
            metadata=coerce_dict(raw.get("metadata")),
        )


@dataclass(frozen=True)
class VerifiedResult:
    answer: Any
    replayable: bool
    evidence_ref: str = ""
    verifier_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Direction:
    core_insight: str
    key_assumptions: list[str] = field(default_factory=list)
    falsification_test: str = ""
    lineage: list[str] = field(default_factory=list)
    why_non_obvious: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GradedOutput:
    mode: str
    verification_strength: VerificationStrength
    result: VerifiedResult | None = None
    portfolio: list[Direction] = field(default_factory=list)
    ruled_out_map: list[Any] = field(default_factory=list)
    replay_certificate: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.verification_strength = VerificationStrength.from_value(self.verification_strength)
        if self.mode not in {"verified_result", "graded_portfolio"}:
            raise AssertionError("graded output mode must be verified_result or graded_portfolio")
        if self.result is not None:
            assert self.mode == "verified_result"
            assert self.verification_strength >= VerificationStrength.FORMAL
            assert self.result.replayable and self.replay_certificate

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "verification_strength": self.verification_strength.name,
            "verification_strength_value": int(self.verification_strength),
            "result": self.result.to_dict() if self.result is not None else None,
            "portfolio": [item.to_dict() for item in self.portfolio],
            "ruled_out_map": list(self.ruled_out_map),
            "replay_certificate": dict(self.replay_certificate or {}) if self.replay_certificate else None,
        }


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "Direction",
    "GradedOutput",
    "SynthesizedVerifier",
    "VerificationPlan",
    "VerificationResult",
    "VerifiedResult",
]
