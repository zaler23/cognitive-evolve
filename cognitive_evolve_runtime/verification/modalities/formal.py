"""Formal verifier modality.

Uses in-process ``z3-solver`` when available.  It intentionally does not invoke
``z3`` CLI because the tool runner allowlist does not include that executable.
"""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.core.serialization import stable_hash
from cognitive_evolve_runtime.verification.types import VerificationResult


class FormalVerifier:
    verifier_id = "formal-verifier"

    def __init__(self, *, formula: Any | None = None) -> None:
        self.formula = formula
        self.fingerprint = "verifier-" + stable_hash({"verifier": self.verifier_id, "formula": str(formula)})[:16]

    def check(self, candidate: Any) -> VerificationResult:
        try:
            import z3  # type: ignore
        except Exception:
            return VerificationResult(False, score=0.0, replayable=False, diagnostics=["z3_solver_python_binding_unavailable"], metadata={"fingerprint": self.fingerprint, "cli_not_attempted": True, "oracle_kind": "formal", "diagnostics_only": True})
        formula = self.formula or _candidate_formula(candidate)
        if formula is None:
            return VerificationResult(False, score=0.0, replayable=False, diagnostics=["no_formal_obligation_declared"], metadata={"fingerprint": self.fingerprint, "oracle_kind": "formal", "diagnostics_only": True})
        solver = z3.Solver()
        if isinstance(formula, bool):
            solver.add(z3.BoolVal(formula))
        else:
            solver.add(formula)
        result = solver.check()
        passed = result == z3.sat
        evidence_ref = "evidence-" + stable_hash({"formula": str(formula), "result": str(result)})[:16]
        return VerificationResult(passed, score=1.0 if passed else 0.0, evidence_ref=evidence_ref, replayable=True, diagnostics=[f"z3_result:{result}"], metadata={"fingerprint": self.fingerprint, "oracle_kind": "formal", "diagnostics_only": True})


def _candidate_formula(candidate: Any) -> Any | None:
    metadata = getattr(candidate, "metadata", {}) if candidate is not None else {}
    if isinstance(metadata, dict):
        return metadata.get("z3_formula")
    return None


__all__ = ["FormalVerifier"]
