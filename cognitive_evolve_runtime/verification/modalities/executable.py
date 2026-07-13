"""Explicit operator-owned executable verifier modality."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.tools.runner import ToolRunner
from cognitive_evolve_runtime.core.serialization import stable_hash
from cognitive_evolve_runtime.verification.types import VerificationResult


class ExecutableVerifier:
    verifier_id = "executable-verifier"

    def __init__(
        self,
        *,
        command: list[str] | None = None,
        cwd: str | Path | None = None,
        timeout_seconds: float = 10.0,
        trusted_operator_command: bool = False,
    ) -> None:
        self.command = command or []
        self.cwd = Path(cwd) if cwd is not None else None
        self.timeout_seconds = timeout_seconds
        self.trusted_operator_command = bool(trusted_operator_command)
        self.fingerprint = "verifier-" + stable_hash(
            {
                "verifier": self.verifier_id,
                "command": self.command,
                "timeout": timeout_seconds,
                "trusted_operator_command": self.trusted_operator_command,
            }
        )[:16]

    def check(self, candidate: Any) -> VerificationResult:
        if not self.trusted_operator_command:
            return VerificationResult(False, score=0.0, replayable=False, diagnostics=["trusted_operator_command_required"], metadata={"fingerprint": self.fingerprint, "oracle_kind": "executable", "diagnostics_only": True, "validation_status": "not_run"})
        if not self.command:
            return VerificationResult(False, score=0.0, replayable=False, diagnostics=["no_operator_command_declared"], metadata={"fingerprint": self.fingerprint, "oracle_kind": "executable", "diagnostics_only": True, "validation_status": "not_run"})
        if self.cwd is None:
            with tempfile.TemporaryDirectory(prefix="cogev-exec-verifier-") as tmp:
                return self._check_in_cwd(self.command, Path(tmp))
        return self._check_in_cwd(self.command, self.cwd)

    def _check_in_cwd(self, command: list[str], cwd: Path) -> VerificationResult:
        feedback = ToolRunner(timeout_seconds=self.timeout_seconds).run(command, cwd=cwd, timeout_seconds=self.timeout_seconds)
        passed = feedback.status == "passed"
        evidence_ref = "evidence-" + stable_hash({"command": command, "status": feedback.status, "output": feedback.raw_output_ref})[:16]
        return VerificationResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            evidence_ref=evidence_ref,
            replayable=True,
            diagnostics=list(feedback.diagnostics),
            metadata={"tool_feedback": feedback.to_dict() if hasattr(feedback, "to_dict") else feedback.__dict__, "fingerprint": self.fingerprint, "oracle_kind": "executable", "operator_owned_command": True, "diagnostics_only": False, "replay_verified": True},
        )


__all__ = ["ExecutableVerifier"]
