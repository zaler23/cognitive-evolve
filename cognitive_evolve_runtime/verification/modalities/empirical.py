"""Empirical verifier modality."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.nexus._serde import stable_hash
from cognitive_evolve_runtime.verification.types import VerificationResult


class EmpiricalVerifier:
    verifier_id = "empirical-verifier"

    def __init__(self, *, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.fingerprint = "verifier-" + stable_hash({"verifier": self.verifier_id, "threshold": threshold})[:16]

    def check(self, candidate: Any) -> VerificationResult:
        metadata = getattr(candidate, "metadata", {}) if candidate is not None else {}
        empirical_score = float(metadata.get("empirical_score", 0.0)) if isinstance(metadata, dict) else 0.0
        passed = empirical_score >= self.threshold
        return VerificationResult(passed, score=empirical_score, evidence_ref="evidence-" + stable_hash({"score": empirical_score, "threshold": self.threshold})[:16], replayable=True, diagnostics=["empirical_score_checked"], metadata={"fingerprint": self.fingerprint, "oracle_kind": "empirical", "diagnostics_only": True})


__all__ = ["EmpiricalVerifier"]
