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
            return VerificationResult(False, score=0.0, replayable=False, diagnostics=["z3_solver_python_binding_unavailable"], metadata={"fingerprint": self.fingerprint, "cli_not_attempted": True, "oracle_kind": "formal", "diagnostics_only": True, "validation_status": "not_run"})
        formula = self.formula if self.formula is not None else _candidate_formula(candidate)
        if formula is None:
            return VerificationResult(False, score=0.0, replayable=False, diagnostics=["no_formal_obligation_declared"], metadata={"fingerprint": self.fingerprint, "oracle_kind": "formal", "diagnostics_only": True, "validation_status": "not_run"})
        formal_kind = _candidate_formal_kind(candidate)
        solver = z3.Solver()
        expr = z3.BoolVal(formula) if isinstance(formula, bool) else formula
        if formal_kind == "satisfiability":
            solver.add(expr)
            pass_result = z3.sat
        else:
            solver.add(z3.Not(expr))
            pass_result = z3.unsat
            formal_kind = "proof"
        result = solver.check()
        passed = result == pass_result
        evidence_ref = "evidence-" + stable_hash({"formula": str(formula), "result": str(result)})[:16]
        diagnostics = [f"z3_result:{result}", f"formal_kind:{formal_kind}"]
        if str(result).lower() == "unknown":
            diagnostics.append("z3_unknown_or_timeout")
        metadata = {"fingerprint": self.fingerprint, "oracle_kind": "formal", "diagnostics_only": False, "replay_verified": passed, "formal_kind": formal_kind}
        if str(result).lower() == "unknown":
            metadata["validation_status"] = "inconclusive"
        return VerificationResult(passed, score=1.0 if passed else 0.0, evidence_ref=evidence_ref, replayable=True, diagnostics=diagnostics, metadata=metadata)


def _candidate_formula(candidate: Any) -> Any | None:
    metadata = getattr(candidate, "metadata", {}) if candidate is not None else {}
    if isinstance(metadata, dict):
        return metadata.get("z3_formula")
    return None


def _candidate_formal_kind(candidate: Any) -> str:
    metadata = getattr(candidate, "metadata", {}) if candidate is not None else {}
    if isinstance(metadata, dict) and metadata.get("formal_kind") == "satisfiability":
        return "satisfiability"
    obligations = getattr(candidate, "proof_obligations", []) if candidate is not None else []
    for obligation in obligations or []:
        if isinstance(obligation, dict) and obligation.get("formal_kind") == "satisfiability":
            return "satisfiability"
    return "proof"


__all__ = ["FormalVerifier"]
