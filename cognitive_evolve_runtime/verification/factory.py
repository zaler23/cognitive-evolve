"""Instantiate synthesized verification plan modalities."""
from __future__ import annotations

from typing import Any

from .modalities.adversarial import AdversarialVerifier
from .modalities.decomposed import DecomposedVerifier
from .modalities.empirical import EmpiricalVerifier
from .modalities.executable import ExecutableVerifier
from .modalities.formal import FormalVerifier
from .types import VerificationPlan, SynthesizedVerifier


def verifier_from_plan(plan: VerificationPlan | dict[str, Any] | None) -> SynthesizedVerifier | None:
    plan = plan if isinstance(plan, VerificationPlan) else VerificationPlan.from_dict(plan)
    modality = str(plan.modality or "none").lower()
    metadata = dict(plan.metadata or {})
    verifier: Any | None
    if modality == "executable":
        command = metadata.get("verification_command") if isinstance(metadata.get("verification_command"), list) else None
        verifier = ExecutableVerifier(command=command)
    elif modality == "formal":
        verifier = FormalVerifier(formula=metadata.get("z3_formula"))
    elif modality == "empirical":
        verifier = EmpiricalVerifier(threshold=float(metadata.get("threshold", 0.5) or 0.5))
    elif modality == "decomposed":
        claims = metadata.get("required_claims") if isinstance(metadata.get("required_claims"), list) else None
        verifier = DecomposedVerifier(required_claims=claims)
    elif modality == "adversarial":
        verifier = AdversarialVerifier()
    else:
        return None
    if plan.verifier_fingerprint:
        try:
            verifier.fingerprint = plan.verifier_fingerprint
        except Exception:
            pass
    return verifier


__all__ = ["verifier_from_plan"]
