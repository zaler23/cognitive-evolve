"""Adversarial verifier modality."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.nexus._serde import stable_hash
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.types import VerificationResult


class AdversarialVerifier:
    verifier_id = "adversarial-verifier"
    strength = VerificationStrength.ADVERSARIAL

    def __init__(self, *, perspectives: list[str] | None = None) -> None:
        self.perspectives = perspectives or ["skeptic", "domain_reviewer", "counterexample_hunter"]
        self.fingerprint = "verifier-" + stable_hash({"verifier": self.verifier_id, "perspectives": self.perspectives})[:16]

    def check(self, candidate: Any) -> VerificationResult:
        text = str(getattr(candidate, "artifact", candidate) or "")
        flags = []
        if "TODO" in text or "assume" in text.lower():
            flags.append("assumption_or_todo_detected")
        passed = not flags
        return VerificationResult(passed, score=0.6 if passed else 0.2, strength=self.strength, evidence_ref="evidence-" + stable_hash({"text_hash": stable_hash(text), "flags": flags})[:16], replayable=False, diagnostics=flags or ["no_basic_adversarial_flags"], metadata={"fingerprint": self.fingerprint, "position_swap_required_for_certificate": True})


__all__ = ["AdversarialVerifier"]
