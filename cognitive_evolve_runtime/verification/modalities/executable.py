"""Executable verifier modality."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.tools.runner import ToolRunner
from cognitive_evolve_runtime.nexus._serde import stable_hash
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.types import VerificationResult


class ExecutableVerifier:
    verifier_id = "executable-verifier"
    strength = VerificationStrength.EXECUTABLE

    def __init__(self, *, command: list[str] | None = None, cwd: str | Path | None = None, timeout_seconds: float = 10.0) -> None:
        self.command = command or []
        self.cwd = Path(cwd) if cwd is not None else None
        self.timeout_seconds = timeout_seconds
        self.fingerprint = "verifier-" + stable_hash({"verifier": self.verifier_id, "command": self.command, "timeout": timeout_seconds})[:16]

    def check(self, candidate: Any) -> VerificationResult:
        command = self.command or _candidate_command(candidate)
        if not command:
            return VerificationResult(False, score=0.0, strength=self.strength, replayable=False, diagnostics=["no_executable_command_declared"], metadata={"fingerprint": self.fingerprint})
        cwd = self.cwd or Path(tempfile.mkdtemp(prefix="cogev-exec-verifier-"))
        if getattr(candidate, "artifact", None) and not self.command:
            artifact = getattr(candidate, "artifact")
            if isinstance(artifact, str):
                (cwd / "candidate.py").write_text(artifact, encoding="utf-8")
        feedback = ToolRunner(timeout_seconds=self.timeout_seconds).run(command, cwd=cwd, timeout_seconds=self.timeout_seconds)
        passed = feedback.status == "passed"
        evidence_ref = "evidence-" + stable_hash({"command": command, "status": feedback.status, "output": feedback.raw_output_ref})[:16]
        return VerificationResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            strength=self.strength,
            evidence_ref=evidence_ref,
            replayable=True,
            diagnostics=list(feedback.diagnostics),
            metadata={"tool_feedback": feedback.to_dict() if hasattr(feedback, "to_dict") else feedback.__dict__, "fingerprint": self.fingerprint},
        )


def _candidate_command(candidate: Any) -> list[str]:
    metadata = getattr(candidate, "metadata", {}) if candidate is not None else {}
    if isinstance(metadata, dict) and isinstance(metadata.get("verification_command"), list):
        return [str(item) for item in metadata["verification_command"]]
    return ["python", "candidate.py"] if isinstance(getattr(candidate, "artifact", None), str) else []


__all__ = ["ExecutableVerifier"]
