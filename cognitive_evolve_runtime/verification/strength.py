"""Verification strength aggregation from actual verifier results."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict
from .ladder import VerificationStrength
from .types import VerificationResult


def candidate_verification_results(candidate: Any) -> list[VerificationResult]:
    raw_items: list[Any] = []
    trace = getattr(candidate, "verification_trace", []) if candidate is not None else []
    if isinstance(trace, list):
        raw_items.extend(trace)
    direct = getattr(candidate, "verification_result", {}) if candidate is not None else {}
    if isinstance(direct, dict) and direct:
        raw_items.append(direct)
    metadata = getattr(candidate, "metadata", {}) if candidate is not None else {}
    if isinstance(metadata, dict):
        stored = metadata.get("verification_results")
        if isinstance(stored, list):
            raw_items.extend(stored)
    out: list[VerificationResult] = []
    for item in raw_items:
        data = coerce_dict(item.get("verification_result") if isinstance(item, dict) and isinstance(item.get("verification_result"), dict) else item)
        if not data:
            continue
        out.append(VerificationResult.from_dict(data))
    return out


def candidate_verification_strength(candidate: Any) -> VerificationStrength:
    strength = VerificationStrength.NONE
    for result in candidate_verification_results(candidate):
        if result.passed and result.replayable:
            strength = max(strength, result.strength)
    return strength


def strongest_passed_replayable_result(candidate: Any) -> VerificationResult | None:
    best: VerificationResult | None = None
    for result in candidate_verification_results(candidate):
        if not (result.passed and result.replayable):
            continue
        if best is None or result.strength > best.strength or (result.strength == best.strength and result.score > best.score):
            best = result
    return best


__all__ = ["candidate_verification_results", "candidate_verification_strength", "strongest_passed_replayable_result"]
