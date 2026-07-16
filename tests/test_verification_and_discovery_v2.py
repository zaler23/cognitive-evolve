from __future__ import annotations

import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.evaluators.evidence import EvidenceRecord
from cognitive_evolve_runtime.verification.grading import GradedOutput, VerifiedResult
from cognitive_evolve_runtime.verification.factory import verifier_from_plan
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.modalities.decomposed import DecomposedVerifier
from cognitive_evolve_runtime.verification.modalities.executable import ExecutableVerifier
from cognitive_evolve_runtime.verification.modalities.formal import FormalVerifier
from cognitive_evolve_runtime.tools.adapters import LocalToolSuite
from cognitive_evolve_runtime.verification.synthesizer import VerificationSynthesizer


def test_grading_invariant_rejects_low_strength_verified_result() -> None:
    with pytest.raises(AssertionError):
        GradedOutput(mode="verified_result", verification_strength=VerificationStrength.DECOMPOSED, result=VerifiedResult("x", replayable=True), replay_certificate={"x": 1})


def test_synthesizer_never_infers_executable_from_problem_text() -> None:
    synth = VerificationSynthesizer()
    for problem in ("code", "python", "pytest", "function", "program", "script", "algorithm", "execute", "run"):
        plan = synth.synthesize(problem)
        assert plan.modality != "executable"
        assert plan.metadata.get("diagnostics_only") is True
        assert plan.strength is VerificationStrength.NONE

    open_plan = synth.synthesize("What is a good theory of this phenomenon?")
    assert open_plan.modality in {"adversarial", "decomposed"}
    assert open_plan.strength is VerificationStrength.NONE


def test_automatic_factory_rejects_executable_plan_even_with_opt_in_metadata() -> None:
    assert verifier_from_plan(
        {
            "modality": "executable",
            "metadata": {
                "trusted_operator_command": True,
                "verification_command": ["python", "-c", "raise SystemExit(0)"],
            },
        }
    ) is None


def test_executable_verifier_default_does_not_execute_candidate(tmp_path: Path) -> None:
    external_write = tmp_path / "untrusted-artifact-ran"
    command_write = tmp_path / "untrusted-command-ran"
    candidate = CandidateGenome(
        id="C1",
        artifact=f"from pathlib import Path\nPath({str(external_write)!r}).write_text('owned')\n",
    )

    result = ExecutableVerifier().check(candidate)
    command_result = ExecutableVerifier(
        command=["python", "-c", f"from pathlib import Path; Path({str(command_write)!r}).write_text('owned')"],
    ).check(candidate)

    assert result.passed is False
    assert result.replayable is False
    assert "trusted_operator_command_required" in result.diagnostics
    assert "trusted_operator_command_required" in command_result.diagnostics
    assert result.to_dict()["validation_status"] == "not_run"
    assert command_result.to_dict()["validation_status"] == "not_run"
    assert not external_write.exists()
    assert not command_write.exists()


def test_executable_verifier_runs_only_explicit_operator_owned_command() -> None:
    candidate = CandidateGenome(id="C1", artifact="print('ok')\n")
    result = ExecutableVerifier(
        command=["python", "-c", "print('ok')"],
        trusted_operator_command=True,
    ).check(candidate)
    assert result.passed is True
    assert result.replayable is True


def test_project_tool_defaults_do_not_execute_repository_defined_commands(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"scripts":{"test":"touch escaped"}}', encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[tool.mypy]\nplugins=["candidate_plugin"]\n', encoding="utf-8")

    default_ids = {item.tool_id for item in LocalToolSuite().default_specs_for_project(tmp_path)}
    opted_in_ids = {item.tool_id for item in LocalToolSuite().default_specs_for_project(tmp_path, include_tests=True)}

    assert "npm_test" not in default_ids
    assert "mypy" not in default_ids
    assert "npm_test" in opted_in_ids


def test_formal_verifier_does_not_attempt_z3_cli_when_binding_missing_or_runs_in_process() -> None:
    result = FormalVerifier(
        dsl={"version": "z3_dsl/v1", "symbols": [], "constraints": [{"op": "bool", "value": True}]}
    ).check(CandidateGenome(id="C1", artifact="x", metadata={"formal_kind": "satisfiability"}))
    assert result.metadata["cli_not_attempted"] is True
    assert result.metadata["z3_status"] in {"sat", "unavailable"}


def test_formal_verifier_uses_proof_semantics_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Solver:
        def set(self, *, timeout: int) -> None:
            self.timeout = timeout

        def add(self, *expr: object) -> None:
            self.expr = expr[0]

        def check(self) -> str:
            if self.expr is False:
                return "unsat"
            if self.expr == "unknown":
                return "unknown"
            return "sat"

        def reason_unknown(self) -> str:
            return "incomplete"

    fake_z3 = SimpleNamespace(
        sat="sat",
        unsat="unsat",
        BoolVal=lambda value: bool(value),
        Not=lambda value: not value if isinstance(value, bool) else "unknown" if value == "unknown" else ("not", value),
        Solver=_Solver,
    )
    monkeypatch.setitem(sys.modules, "z3", fake_z3)

    true_dsl = {"version": "z3_dsl/v1", "symbols": [], "constraints": [{"op": "bool", "value": True}]}
    false_dsl = {"version": "z3_dsl/v1", "symbols": [], "constraints": [{"op": "bool", "value": False}]}

    assert FormalVerifier(dsl=false_dsl).check(CandidateGenome()).passed is True
    assert FormalVerifier(formula="x > 0").check(CandidateGenome()).metadata["z3_status"] == "rejected"
    assert FormalVerifier().check(CandidateGenome()).to_dict()["validation_status"] == "not_run"
    assert FormalVerifier(dsl=true_dsl).check(CandidateGenome(metadata={"formal_kind": "satisfiability"})).passed is True


def test_decomposed_check_without_declared_claims_is_not_run() -> None:
    result = DecomposedVerifier().check(CandidateGenome(artifact="candidate"))

    assert result.to_dict()["validation_status"] == "not_run"
