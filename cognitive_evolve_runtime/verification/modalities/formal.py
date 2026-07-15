"""Formal verifier modality.

Uses the bounded typed DSL and in-process ``z3-solver`` when available.  It
intentionally does not invoke ``z3`` CLI or accept raw SMT-LIB text.
"""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.core.serialization import stable_hash
from cognitive_evolve_runtime.verification.types import VerificationResult
from cognitive_evolve_runtime.verification.z3_dsl import evaluate_z3_dsl


class FormalVerifier:
    verifier_id = "formal-verifier"

    def __init__(self, *, formula: Any | None = None, dsl: Any | None = None) -> None:
        self.dsl = dsl if dsl is not None else formula
        self.fingerprint = "verifier-" + stable_hash({"verifier": self.verifier_id, "dsl": self.dsl})[:16]

    def check(self, candidate: Any) -> VerificationResult:
        dsl = self.dsl if self.dsl is not None else _candidate_dsl(candidate)
        if dsl is None:
            return VerificationResult(
                False,
                score=0.0,
                replayable=False,
                diagnostics=["no_formal_obligation_declared"],
                metadata={
                    "fingerprint": self.fingerprint,
                    "oracle_kind": "formal",
                    "diagnostics_only": True,
                    "validation_status": "not_run",
                },
            )
        formal_kind = _candidate_formal_kind(candidate)
        return evaluate_z3_dsl(
            dsl,
            expected_status="sat" if formal_kind == "satisfiability" else "unsat",
            fingerprint=self.fingerprint,
            formal_kind=formal_kind,
        )


def _candidate_dsl(candidate: Any) -> Any | None:
    metadata = getattr(candidate, "metadata", {}) if candidate is not None else {}
    if isinstance(metadata, dict):
        if "z3_dsl" in metadata:
            return metadata.get("z3_dsl")
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


def apply_formal_evaluation_evidence(candidate: Any, result: VerificationResult, *, round_index: int = 0) -> Any | None:
    """Write non-passing Z3 receipts through the canonical evidence plane."""

    from cognitive_evolve_runtime.evaluators.evidence import EvidenceRecord, apply_evidence_record, evidence_records

    metadata = dict(result.metadata or {})
    status = str(metadata.get("z3_status") or "")
    dsl_sha256 = str(metadata.get("z3_dsl_sha256") or "")
    if not status or result.passed:
        return None
    for existing in evidence_records(candidate):
        if (
            existing.source == "engine_bounded_z3"
            and existing.metadata.get("z3_status") == status
            and existing.metadata.get("z3_dsl_sha256") == dsl_sha256
        ):
            return existing
    hints = {
        "rejected": "replace raw or invalid formal input with bounded z3_dsl/v1",
        "unavailable": "restore the declared z3-solver binding before treating this obligation as passed",
        "timeout": "reduce the bounded constraint system before retrying the engine-owned solver",
        "unknown": "reformulate the bounded constraint system into a decidable supported fragment",
    }
    record = EvidenceRecord(
        candidate_id=str(getattr(candidate, "id", "")),
        source="engine_bounded_z3",
        stage="verification_formal",
        score=0.0,
        confidence=1.0 if status in {"rejected", "unavailable"} else 0.8,
        final_blocked=True,
        parent_blocked=False,
        terminal_reject=False,
        repair_value=0.4,
        continuation_value=0.5,
        diagnostics=list(result.diagnostics),
        hints=[hints.get(status, "repair the formal claim against the recorded bounded Z3 result")],
        metadata={
            "z3_status": status,
            "z3_expected_status": str(metadata.get("z3_expected_status") or ""),
            "z3_dsl_sha256": dsl_sha256,
            "z3_timeout_ms": metadata.get("z3_timeout_ms"),
            "round_index": int(round_index),
            "verification_receipt": result.to_dict(),
        },
    )
    apply_evidence_record(candidate, record)
    return record


__all__ = ["FormalVerifier", "apply_formal_evaluation_evidence"]
