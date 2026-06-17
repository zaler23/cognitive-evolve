"""Honest final grading gate."""
from __future__ import annotations

from typing import Any

from .ladder import VerificationStrength
from .types import Direction, GradedOutput, VerifiedResult


def grade(campaign_state: dict[str, Any]) -> GradedOutput:
    strength = VerificationStrength.from_value(campaign_state.get("verification_strength") or campaign_state.get("verification_strength_value"))
    verified = campaign_state.get("verified_result") if isinstance(campaign_state.get("verified_result"), dict) else None
    certificate = campaign_state.get("replay_certificate") if isinstance(campaign_state.get("replay_certificate"), dict) else None
    if verified and certificate_allows_verified_result(certificate, VerificationStrength.FORMAL):
        strength = VerificationStrength.from_value(certificate.get("measured_strength") or certificate.get("measured_strength_value"))
        result = VerifiedResult(
            answer=verified.get("answer"),
            replayable=bool(verified.get("replayable")),
            evidence_ref=str(verified.get("evidence_ref") or ""),
            verifier_fingerprint=str(verified.get("verifier_fingerprint") or ""),
        )
        return GradedOutput(mode="verified_result", verification_strength=strength, result=result, replay_certificate=certificate)
    portfolio = [
        Direction(
            core_insight=str(item.get("core_insight") or item.get("insight") or "direction"),
            key_assumptions=[str(v) for v in item.get("key_assumptions", []) if v],
            falsification_test=str(item.get("falsification_test") or "provide a stronger verifier or counterexample"),
            lineage=[str(v) for v in item.get("lineage", []) if v],
            why_non_obvious=str(item.get("why_non_obvious") or "kept because verification is below FORMAL"),
        )
        for item in campaign_state.get("portfolio", [])
        if isinstance(item, dict)
    ]
    return GradedOutput(mode="graded_portfolio", verification_strength=strength, portfolio=portfolio, ruled_out_map=list(campaign_state.get("ruled_out_map") or []), replay_certificate=certificate)


def certificate_allows_verified_result(certificate: dict[str, Any] | None, threshold: VerificationStrength | int | str = VerificationStrength.FORMAL) -> bool:
    cert = certificate if isinstance(certificate, dict) else {}
    threshold = VerificationStrength.from_value(threshold)
    measured = VerificationStrength.from_value(cert.get("measured_strength") or cert.get("measured_strength_value"))
    if measured < threshold:
        return False
    if not str(cert.get("frozen_artifact_hash") or ""):
        return False
    if not str(cert.get("verifier_fingerprint") or ""):
        return False
    measurements = cert.get("honesty_measurements")
    if not isinstance(measurements, dict):
        return False
    for key in ("exogeneity_score", "variety_score", "falsification_score", "replay_score"):
        if measurements.get(key) is None:
            return False
    return True


__all__ = ["Direction", "GradedOutput", "VerifiedResult", "certificate_allows_verified_result", "grade"]
