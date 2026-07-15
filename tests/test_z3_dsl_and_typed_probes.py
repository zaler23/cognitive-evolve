from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.evaluators.evidence import evidence_records
from cognitive_evolve_runtime.verification.modalities.formal import (
    FormalVerifier,
    apply_formal_evaluation_evidence,
)
from cognitive_evolve_runtime.verification.probe_executor import execute_probes
from cognitive_evolve_runtime.verification.regime import compile_grounding_regime
from cognitive_evolve_runtime.verification.types import VerificationResult
from cognitive_evolve_runtime.verification.z3_dsl import (
    MAX_AST_DEPTH,
    MAX_BITVEC_WIDTH,
    MAX_SYMBOLS,
    Z3_TIMEOUT_MS,
)


def _dsl(*constraints: dict[str, object], symbols: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "version": "z3_dsl/v1",
        "symbols": list(symbols or []),
        "constraints": list(constraints),
    }


def _bool(value: bool) -> dict[str, object]:
    return {"op": "bool", "value": value}


def _nested_not(depth: int) -> dict[str, object]:
    node = _bool(True)
    for _ in range(depth - 1):
        node = {"op": "not", "args": [node]}
    return node


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ("(assert true)", "dsl_must_be_object"),
        (
            _dsl(
                {"op": "eq", "args": [{"op": "symbol", "name": "x"}, {"op": "bitvec", "value": 0, "width": MAX_BITVEC_WIDTH + 1}]},
                symbols=[{"name": "x", "sort": "BitVec", "width": MAX_BITVEC_WIDTH + 1}],
            ),
            "bitvec_width_exceeds_limit",
        ),
        (_dsl(_nested_not(MAX_AST_DEPTH + 1)), "ast_depth_exceeds_limit"),
        (
            _dsl(_bool(True), symbols=[{"name": f"s{index}", "sort": "Bool"} for index in range(MAX_SYMBOLS + 1)]),
            "symbol_count_exceeds_limit",
        ),
        (_dsl({"op": "eval", "args": []}), "unsupported_operator"),
        ({**_dsl(_bool(True)), "timeout_ms": 999_999}, "unsupported_top_level_fields"),
    ],
)
def test_raw_or_out_of_bounds_dsl_is_rejected_before_z3_import(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    reason: str,
) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):  # noqa: ANN001, ANN202
        if name == "z3":
            raise AssertionError("rejected DSL must not reach Z3")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = FormalVerifier(formula=payload).check(CandidateGenome(id="C-rejected"))

    assert result.passed is False
    assert result.replayable is False
    assert result.metadata["z3_status"] == "rejected"
    assert result.metadata["z3_dsl_sha256"]
    assert reason in result.metadata["z3_reason"]
    assert result.to_dict()["validation_status"] == "not_run"


def test_z3_import_failure_is_unavailable_and_writes_existing_evidence_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def missing_z3(name, *args, **kwargs):  # noqa: ANN001, ANN202
        if name == "z3":
            raise ModuleNotFoundError("No module named 'z3'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_z3)
    candidate = CandidateGenome(id="C-unavailable", metadata={"formal_kind": "satisfiability"})

    result = FormalVerifier(dsl=_dsl(_bool(True))).check(candidate)
    record = apply_formal_evaluation_evidence(candidate, result, round_index=3)

    assert result.passed is False
    assert result.metadata["z3_status"] == "unavailable"
    assert result.to_dict()["validation_status"] == "not_run"
    assert record is not None
    assert record.metadata["z3_status"] == "unavailable"
    assert evidence_records(candidate)[-1].source == "engine_bounded_z3"


class _FakeExpr:
    def __init__(self, sort: str) -> None:
        self.sort = sort

    def _bool_result(self, _other: object = None) -> "_FakeExpr":
        return _FakeExpr("Bool")

    def _same_result(self, _other: object = None) -> "_FakeExpr":
        return _FakeExpr(self.sort)

    __eq__ = _bool_result
    __ne__ = _bool_result
    __lt__ = _bool_result
    __le__ = _bool_result
    __gt__ = _bool_result
    __ge__ = _bool_result
    __add__ = _same_result
    __sub__ = _same_result
    __mul__ = _same_result
    __and__ = _same_result
    __or__ = _same_result
    __xor__ = _same_result
    __lshift__ = _same_result
    __rshift__ = _same_result

    def __neg__(self) -> "_FakeExpr":
        return _FakeExpr(self.sort)

    def __invert__(self) -> "_FakeExpr":
        return _FakeExpr(self.sort)


def _fake_z3(status: str, reason: str = "") -> SimpleNamespace:
    observed: dict[str, int] = {}

    class _Solver:
        def set(self, *, timeout: int) -> None:
            observed["timeout"] = timeout

        def add(self, *_constraints: object) -> None:
            return None

        def check(self) -> str:
            return status

        def reason_unknown(self) -> str:
            return reason

    return SimpleNamespace(
        sat="sat",
        unsat="unsat",
        unknown="unknown",
        Solver=_Solver,
        Bool=lambda _name: _FakeExpr("Bool"),
        Int=lambda _name: _FakeExpr("Int"),
        BitVec=lambda _name, width: _FakeExpr(f"BitVec:{width}"),
        BoolVal=lambda _value: _FakeExpr("Bool"),
        IntVal=lambda _value: _FakeExpr("Int"),
        BitVecVal=lambda _value, width: _FakeExpr(f"BitVec:{width}"),
        And=lambda *_args: _FakeExpr("Bool"),
        Or=lambda *_args: _FakeExpr("Bool"),
        Not=lambda _arg: _FakeExpr("Bool"),
        Implies=lambda _left, _right: _FakeExpr("Bool"),
        Xor=lambda *_args: _FakeExpr("Bool"),
        Distinct=lambda *_args: _FakeExpr("Bool"),
        If=lambda _condition, then, _otherwise: _FakeExpr(then.sort),
        observed=observed,
    )


@pytest.mark.parametrize(
    ("solver_status", "solver_reason", "formal_kind", "expected_status", "passed", "validation_status"),
    [
        ("sat", "", "satisfiability", "sat", True, "preliminary_passed"),
        ("unsat", "", "proof", "unsat", True, "preliminary_passed"),
        ("unknown", "timeout", "satisfiability", "timeout", False, "inconclusive"),
    ],
)
def test_z3_statuses_are_typed_and_timeout_is_engine_owned(
    monkeypatch: pytest.MonkeyPatch,
    solver_status: str,
    solver_reason: str,
    formal_kind: str,
    expected_status: str,
    passed: bool,
    validation_status: str,
) -> None:
    fake_z3 = _fake_z3(solver_status, solver_reason)
    monkeypatch.setitem(sys.modules, "z3", fake_z3)
    candidate = CandidateGenome(id="C-status", metadata={"formal_kind": formal_kind})

    result = FormalVerifier(dsl=_dsl(_bool(True))).check(candidate)

    assert result.metadata["z3_status"] == expected_status
    assert result.passed is passed
    assert result.to_dict()["validation_status"] == validation_status
    assert fake_z3.observed["timeout"] == Z3_TIMEOUT_MS
    assert result.metadata["z3_timeout_ms"] == Z3_TIMEOUT_MS


def _typed_case(*, path: str, operator: str, value: object) -> dict[str, object]:
    return {
        "probe_template_id": "artifact_json_relation/v2",
        "args": {"path": path},
        "expected_relation": {"operator": operator, "value": value},
    }


def _probe_candidate(case: dict[str, object]) -> CandidateGenome:
    return CandidateGenome(
        id="C-typed-probe",
        artifact={"metrics": {"score": 8}},
        proof_obligations=[{"id": "typed", "probe_cases": [case]}],
    )


def _probe_regime(candidate: CandidateGenome):  # noqa: ANN202
    return compile_grounding_regime(
        candidate=candidate,
        verifier_fingerprint="vf-typed",
        artifact_hash="artifact-typed",
        oracle_kind="toolrunner",
        override_adversarial_budget=1,
    )


def test_typed_probe_template_runs_only_after_good_bad_calibration() -> None:
    candidate = _probe_candidate(_typed_case(path="/metrics/score", operator="gte", value=8))
    regime = _probe_regime(candidate)

    observed = execute_probes(VerificationResult(passed=False), regime, candidate=candidate)

    assert regime.probes[0].template_id == "artifact_json_relation/v2"
    assert observed["probe_results"][0]["status"] == "survived"
    assert observed["known_good_bad_distinguishable"] is True
    assert [
        (item["calibration_role"], item["status"])
        for item in observed["known_good_bad_probe_results"]
    ] == [("known_good", "survived"), ("known_bad", "counterexample")]


def test_unknown_typed_probe_template_and_execution_fields_are_rejected_before_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = [
        {
            "probe_template_id": "model_shell/v1",
            "args": {},
            "expected_relation": {"operator": "equal", "value": True},
        },
        {
            **_typed_case(path="/metrics/score", operator="equal", value=8),
            "command": ["python", "-c", "print('owned')"],
            "timeout": 999,
        },
    ]

    def unexpected_run(*_args, **_kwargs):  # noqa: ANN202
        raise AssertionError("rejected template must not reach the engine runner")

    monkeypatch.setattr("cognitive_evolve_runtime.verification.probe_executor.ToolRunner.run", unexpected_run)
    candidate = CandidateGenome(
        id="C-rejected-probes",
        artifact={"metrics": {"score": 8}},
        proof_obligations=[{"id": "typed", "probe_cases": cases}],
    )

    observed = execute_probes(VerificationResult(passed=False), _probe_regime(candidate), candidate=candidate)

    assert [item["status"] for item in observed["probe_results"]] == ["unsupported", "unsupported"]
    assert observed["executed_count"] == 0
    assert "unsupported_template" in observed["probe_results"][0]["reason"]
    assert "unsupported_fields" in observed["probe_results"][1]["reason"]


def test_typed_probe_is_unavailable_when_bad_calibration_does_not_find_counterexample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _probe_candidate(_typed_case(path="/metrics/score", operator="equal", value=8))
    regime = _probe_regime(candidate)

    def calibration_survives(probe, _artifact, role):  # noqa: ANN001, ANN202
        return {
            "probe_id": probe.probe_id,
            "status": "survived",
            "calibration_role": role,
            "engine_generated": True,
            "provenance": "engine",
        }

    def unexpected_candidate_execution(*_args, **_kwargs):  # noqa: ANN202
        raise AssertionError("uncalibrated template must not execute on the candidate artifact")

    monkeypatch.setattr(
        "cognitive_evolve_runtime.verification.probe_executor._run_calibration_artifact",
        calibration_survives,
    )
    monkeypatch.setattr(
        "cognitive_evolve_runtime.verification.probe_executor._run_artifact_assertions",
        unexpected_candidate_execution,
    )

    observed = execute_probes(VerificationResult(passed=False), regime, candidate=candidate)

    assert observed["probe_results"][0]["status"] == "unsupported"
    assert observed["probe_results"][0]["reason"] == "template_calibration_failed"
    assert observed["executed_count"] == 0
