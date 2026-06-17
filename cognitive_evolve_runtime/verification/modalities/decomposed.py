"""Decomposed claim verifier modality."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.nexus._serde import stable_hash
from cognitive_evolve_runtime.verification.types import VerificationResult


class DecomposedVerifier:
    verifier_id = "decomposed-verifier"

    def __init__(self, *, required_claims: list[str] | None = None) -> None:
        self.required_claims = [str(item) for item in required_claims or [] if item]
        self.fingerprint = "verifier-" + stable_hash({"verifier": self.verifier_id, "claims": self.required_claims})[:16]

    def check(self, candidate: Any) -> VerificationResult:
        text = str(getattr(candidate, "artifact", candidate) or "")
        if not self.required_claims:
            metadata = getattr(candidate, "metadata", {}) if candidate is not None else {}
            self.required_claims = [str(item) for item in (metadata.get("required_claims", []) if isinstance(metadata, dict) else []) if item]
        if not self.required_claims:
            return VerificationResult(False, score=0.0, replayable=False, diagnostics=["no_required_claims"], metadata={"fingerprint": self.fingerprint, "oracle_kind": "decomposed", "diagnostics_only": True})
        passed_claims = [claim for claim in self.required_claims if claim.lower() in text.lower()]
        score = len(passed_claims) / max(1, len(self.required_claims))
        return VerificationResult(score >= 1.0, score=score, evidence_ref="evidence-" + stable_hash({"claims": self.required_claims, "passed": passed_claims})[:16], replayable=True, diagnostics=[f"claims_passed:{len(passed_claims)}/{len(self.required_claims)}"], metadata={"fingerprint": self.fingerprint, "oracle_kind": "decomposed", "diagnostics_only": True})


__all__ = ["DecomposedVerifier"]
