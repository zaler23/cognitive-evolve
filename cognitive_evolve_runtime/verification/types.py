"""Typed verification spine payloads."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from cognitive_evolve_runtime.core.serialization import coerce_dict
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
        metadata = coerce_dict(self.metadata)
        for key in (
            "measured_strength",
            "measured_strength_value",
            "honesty_measurements",
            "grounding_regime_id",
            "replay_record",
            "diagnostics_only",
            "legacy",
        ):
            if key in metadata:
                data[key] = metadata.get(key)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VerificationResult":
        raw = coerce_dict(data)
        metadata = coerce_dict(raw.get("metadata"))
        for key in (
            "measured_strength",
            "measured_strength_value",
            "honesty_measurements",
            "grounding_regime_id",
            "replay_record",
            "diagnostics_only",
            "legacy",
        ):
            if key in raw and key not in metadata:
                metadata[key] = raw.get(key)
        return cls(
            passed=bool(raw.get("passed")),
            score=_float(raw.get("score"), 0.0),
            strength=VerificationStrength.from_value(raw.get("strength") or raw.get("strength_value")),
            evidence_ref=str(raw.get("evidence_ref") or ""),
            replayable=bool(raw.get("replayable")),
            diagnostics=[str(item) for item in raw.get("diagnostics", []) if item],
            metadata=metadata,
        )


class SynthesizedVerifier(Protocol):
    verifier_id: str
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

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VerifiedResult | None":
        raw = coerce_dict(data)
        if not raw:
            return None
        return cls(
            answer=raw.get("answer"),
            replayable=bool(raw.get("replayable")),
            evidence_ref=str(raw.get("evidence_ref") or ""),
            verifier_fingerprint=str(raw.get("verifier_fingerprint") or ""),
        )


@dataclass(frozen=True)
class Direction:
    core_insight: str
    key_assumptions: list[str] = field(default_factory=list)
    falsification_test: str = ""
    lineage: list[str] = field(default_factory=list)
    why_non_obvious: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Direction":
        raw = coerce_dict(data)
        return cls(
            core_insight=str(raw.get("core_insight") or raw.get("insight") or ""),
            key_assumptions=[str(item) for item in raw.get("key_assumptions", []) if item],
            falsification_test=str(raw.get("falsification_test") or ""),
            lineage=[str(item) for item in raw.get("lineage", []) if item],
            why_non_obvious=str(raw.get("why_non_obvious") or ""),
        )


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
        if self.mode == "verified_result":
            assert self.result is not None
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

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GradedOutput":
        raw = coerce_dict(data)
        if not raw:
            return cls(mode="graded_portfolio", verification_strength=VerificationStrength.NONE)
        result = VerifiedResult.from_dict(raw.get("result") if isinstance(raw.get("result"), dict) else None)
        portfolio = [
            Direction.from_dict(item)
            for item in raw.get("portfolio", [])
            if isinstance(item, dict)
        ]
        mode = str(raw.get("mode") or ("verified_result" if result is not None else "graded_portfolio"))
        if mode == "verified_result" and result is None:
            mode = "graded_portfolio"
        return cls(
            mode=mode,
            verification_strength=VerificationStrength.from_value(raw.get("verification_strength") or raw.get("verification_strength_value")),
            result=result,
            portfolio=portfolio,
            ruled_out_map=list(raw.get("ruled_out_map") or []),
            replay_certificate=coerce_dict(raw.get("replay_certificate")) or None,
        )


def graded_output_from_dict(data: dict[str, Any] | None) -> GradedOutput:
    return GradedOutput.from_dict(data)


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
    "graded_output_from_dict",
]
