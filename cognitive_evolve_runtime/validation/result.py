"""Canonical verification result type shared by validation, tools, and API."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class VerificationVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    SKIP = "skip"


@dataclass(frozen=True)
class VerificationResult:
    verdict: VerificationVerdict
    source: str
    confidence: float = 0.0
    reason: str = ""
    code: str = ""
    evidence_ids: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == VerificationVerdict.PASS

    def is_blocking(self, *, threshold: float = 0.8) -> bool:
        return self.verdict == VerificationVerdict.FAIL and self.confidence >= threshold

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["verdict"] = self.verdict.value
        payload["passed"] = self.passed
        payload["evidence_ids"] = list(self.evidence_ids)
        return payload


def verification_result_from_mapping(data: dict[str, Any] | None, *, source: str = "unknown") -> VerificationResult:
    raw = dict(data or {})
    verdict = _verdict_from_raw(raw)
    confidence = _confidence(raw, verdict)
    reason = str(raw.get("reason") or raw.get("message") or raw.get("diagnostic") or raw.get("status") or "")
    code = str(raw.get("code") or raw.get("category") or raw.get("status") or "")
    evidence_ids = tuple(str(item) for item in raw.get("evidence_ids", []) if str(item).strip()) if isinstance(raw.get("evidence_ids"), list) else ()
    return VerificationResult(
        verdict=verdict,
        source=str(raw.get("source") or source),
        confidence=confidence,
        reason=reason,
        code=code,
        evidence_ids=evidence_ids,
        details=raw,
    )


def aggregate_verification_results(results: list[VerificationResult], *, source: str = "aggregate") -> VerificationResult:
    if not results:
        return VerificationResult(VerificationVerdict.INCONCLUSIVE, source=source, confidence=0.0, reason="no verification results")
    if any(item.verdict == VerificationVerdict.FAIL for item in results):
        failures = [item for item in results if item.verdict == VerificationVerdict.FAIL]
        confidence = max(item.confidence for item in failures)
        return VerificationResult(
            VerificationVerdict.FAIL,
            source=source,
            confidence=confidence,
            reason="; ".join(item.reason or item.code or item.source for item in failures[:5]),
            code="aggregate_failure",
            evidence_ids=tuple(eid for item in failures for eid in item.evidence_ids),
            details={"results": [item.to_dict() for item in results]},
        )
    if all(item.verdict == VerificationVerdict.PASS for item in results):
        return VerificationResult(
            VerificationVerdict.PASS,
            source=source,
            confidence=min(item.confidence for item in results),
            reason="all verification results passed",
            code="aggregate_pass",
            evidence_ids=tuple(eid for item in results for eid in item.evidence_ids),
            details={"results": [item.to_dict() for item in results]},
        )
    return VerificationResult(
        VerificationVerdict.INCONCLUSIVE,
        source=source,
        confidence=max(item.confidence for item in results),
        reason="verification results are mixed or skipped",
        code="aggregate_inconclusive",
        evidence_ids=tuple(eid for item in results for eid in item.evidence_ids),
        details={"results": [item.to_dict() for item in results]},
    )


def _verdict_from_raw(raw: dict[str, Any]) -> VerificationVerdict:
    explicit = str(raw.get("verdict") or raw.get("outcome") or "").strip().lower()
    if explicit in {item.value for item in VerificationVerdict}:
        return VerificationVerdict(explicit)
    status = str(raw.get("status") or "").strip().lower()
    if raw.get("passed") is True or status in {"pass", "passed", "success", "ok"}:
        return VerificationVerdict.PASS
    if raw.get("passed") is False or status in {"fail", "failed", "error", "rejected"}:
        return VerificationVerdict.FAIL
    if status in {"skip", "skipped", "not_applicable"}:
        return VerificationVerdict.SKIP
    return VerificationVerdict.INCONCLUSIVE


def _confidence(raw: dict[str, Any], verdict: VerificationVerdict) -> float:
    try:
        value = float(raw.get("confidence"))
    except (TypeError, ValueError):
        value = 1.0 if verdict in {VerificationVerdict.PASS, VerificationVerdict.FAIL} else 0.0
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


__all__ = [
    "VerificationResult",
    "VerificationVerdict",
    "aggregate_verification_results",
    "verification_result_from_mapping",
]
