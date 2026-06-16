"""Honest final grading gate."""
from __future__ import annotations

from typing import Any

from .ladder import VerificationStrength
from .types import Direction, GradedOutput, VerifiedResult


def grade(campaign_state: dict[str, Any]) -> GradedOutput:
    strength = VerificationStrength.from_value(campaign_state.get("verification_strength") or campaign_state.get("verification_strength_value"))
    verified = campaign_state.get("verified_result") if isinstance(campaign_state.get("verified_result"), dict) else None
    certificate = campaign_state.get("replay_certificate") if isinstance(campaign_state.get("replay_certificate"), dict) else None
    if verified and strength >= VerificationStrength.FORMAL:
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


__all__ = ["Direction", "GradedOutput", "VerifiedResult", "grade"]
