from __future__ import annotations

import json

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.evaluators.evidence import evidence_records
from cognitive_evolve_runtime.evaluators.challenge_memory import ChallengeMemory
from cognitive_evolve_runtime.ranking.parent_selection import evaluator_selection_key
from cognitive_evolve_runtime.nexus.search_kernel.branch_allocator import (
    _payload_failed,
    _payload_passed,
    productive_outcomes,
)
from cognitive_evolve_runtime.tools.feedback import ToolFeedback
from cognitive_evolve_runtime.verification.cache import check_with_cache
from cognitive_evolve_runtime.verification.honesty_core import ProbeCase
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.probe_executor import (
    apply_probe_counterexample_evidence,
    execute_probes,
    result_with_probe_observations,
)
from cognitive_evolve_runtime.verification.regime import compile_grounding_regime
from cognitive_evolve_runtime.verification.types import VerificationResult


def _case(assertion_id: str, *, path: str, operator: str, expected: object) -> dict[str, object]:
    return {
        "template": "artifact_assertion/v1",
        "assertion_id": assertion_id,
        "path": path,
        "operator": operator,
        "expected": expected,
    }


def _candidate(*cases: dict[str, object], artifact: object | None = None) -> CandidateGenome:
    return CandidateGenome(
        id="C-probe",
        parent_ids=["P"],
        artifact={"metrics": {"score": 8}, "tags": ["novel", "grounded"]} if artifact is None else artifact,
        proof_obligations=[{"id": "obl", "probe_cases": list(cases)}],
    )


def test_parameterized_probe_executes_fixed_json_assertions_and_keeps_pending_cases() -> None:
    candidate = _candidate(
        _case("survives", path="/metrics/score", operator="gte", expected=8),
        _case("counterexample", path="/tags", operator="contains", expected="missing"),
        _case("pending", path="/metrics/score", operator="lt", expected=10),
    )
    regime = compile_grounding_regime(
        candidate=candidate,
        verifier_fingerprint="vf",
        artifact_hash="artifact",
        oracle_kind="toolrunner",
        override_adversarial_budget=2,
    )

    observed = execute_probes(VerificationResult(passed=False), regime, candidate=candidate)

    assert [item["status"] for item in observed["probe_results"]] == [
        "survived",
        "counterexample",
        "pending_budget",
    ], observed["probe_results"][0].get("reason")
    assert observed["survived_count"] == 1
    assert observed["counterexample_count"] == 1
    assert observed["pending_count"] == 1
    assert observed["probe_survival_ratio"] == 0.5


def test_probe_harness_preserves_python_loader_path(monkeypatch) -> None:
    candidate = _candidate(_case("survives", path="/metrics/score", operator="gte", expected=8))
    regime = compile_grounding_regime(
        candidate=candidate,
        verifier_fingerprint="vf",
        artifact_hash="artifact",
        oracle_kind="toolrunner",
        override_adversarial_budget=1,
    )
    observed_command: list[str] = []
    observed_env: dict[str, str] = {}
    monkeypatch.setenv("LD_LIBRARY_PATH", "/engine/python/lib")

    def _run(_self, command, *, cwd, env=None, timeout_seconds=None):  # noqa: ANN001, ANN202
        del cwd, timeout_seconds
        observed_command.extend(command)
        observed_env.update(env or {})
        return ToolFeedback(
            tool_id="probe-harness",
            status="passed",
            raw_output_ref=json.dumps(
                [
                    {
                        "probe_id": regime.probes[0].probe_id,
                        "assertion_id": "survives",
                        "status": "survived",
                        "path": "/metrics/score",
                        "operator": "gte",
                    }
                ]
            ),
        )

    monkeypatch.setattr("cognitive_evolve_runtime.verification.probe_executor.ToolRunner.run", _run)

    result = execute_probes(VerificationResult(passed=False), regime, candidate=candidate)

    assert observed_command[1] == "-I"
    assert observed_env == {"LD_LIBRARY_PATH": "/engine/python/lib"}
    assert result["probe_results"][0]["status"] == "survived"


def test_model_command_and_expression_are_recorded_unsupported_and_never_executed(monkeypatch) -> None:
    candidate = _candidate(
        {
            **_case("unsafe", path="/metrics/score", operator="equal", expected=8),
            "command": "rm -rf /",
            "python_expression": "__import__('os').system('false')",
        }
    )
    regime = compile_grounding_regime(
        candidate=candidate,
        verifier_fingerprint="vf",
        artifact_hash="artifact",
        oracle_kind="toolrunner",
        override_adversarial_budget=1,
    )

    def _unexpected_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("unsupported model fields must never reach ToolRunner")

    monkeypatch.setattr("cognitive_evolve_runtime.verification.probe_executor.ToolRunner.run", _unexpected_run)
    observed = execute_probes(VerificationResult(passed=False), regime, candidate=candidate)

    assert observed["probe_results"][0]["status"] == "unsupported"
    assert "command" in observed["probe_results"][0]["reason"]
    assert observed["executed_count"] == 0


def test_non_json_and_pending_parameterized_probes_remain_not_run() -> None:
    candidate = _candidate(
        _case("not-json", path="/answer", operator="exists", expected=None),
        artifact="plain prose",
    )
    regime = compile_grounding_regime(
        candidate=candidate,
        verifier_fingerprint="vf",
        artifact_hash="artifact",
        oracle_kind="toolrunner",
        override_adversarial_budget=1,
    )
    raw = VerificationResult(passed=False, metadata={"diagnostics_only": True})

    observed = execute_probes(raw, regime, candidate=candidate)
    result = result_with_probe_observations(raw, observed)

    assert observed["probe_results"][0]["status"] == "non_json"
    assert result.to_dict()["validation_status"] == "not_run"
    assert _payload_passed(result.to_dict()) is False
    assert _payload_failed(result.to_dict()) is False

    json_null = CandidateGenome(
        id="json-null",
        artifact="null",
        proof_obligations=[{"id": "null", "probe_cases": [_case("null", path="", operator="equal", expected=None)]}],
    )
    null_regime = compile_grounding_regime(
        candidate=json_null,
        verifier_fingerprint="vf",
        artifact_hash="artifact-null",
        oracle_kind="toolrunner",
        override_adversarial_budget=1,
    )
    assert execute_probes(raw, null_regime, candidate=json_null)["probe_results"][0]["status"] == "survived"


def test_unrunnable_probe_does_not_erase_stronger_independent_verification() -> None:
    candidate = _candidate(
        {
            **_case("unsafe", path="/metrics/score", operator="equal", expected=8),
            "command": "ignored",
        }
    )
    regime = compile_grounding_regime(
        candidate=candidate,
        verifier_fingerprint="vf",
        artifact_hash="artifact",
        oracle_kind="formal",
        override_adversarial_budget=1,
    )
    raw = VerificationResult(
        passed=True,
        strength=VerificationStrength.FORMAL,
        replayable=True,
        metadata={"diagnostics_only": False},
    )

    result = result_with_probe_observations(raw, execute_probes(raw, regime, candidate=candidate))

    assert result.to_dict()["validation_status"] == "preliminary_passed"
    assert result.metadata["probe_validation_status"] == "not_run"


def test_probe_survival_is_inconclusive_and_weak_credit_does_not_stack_with_verified_pass() -> None:
    survived = _candidate(_case("survives", path="/metrics/score", operator="gte", expected=8))
    survived.id = "probe-only"
    regime = compile_grounding_regime(
        candidate=survived,
        verifier_fingerprint="vf",
        artifact_hash="artifact",
        oracle_kind="toolrunner",
        override_adversarial_budget=1,
    )
    raw = VerificationResult(passed=True, metadata={"diagnostics_only": True})
    observed = execute_probes(raw, regime, candidate=survived)
    survived.verification_trace = [result_with_probe_observations(raw, observed).to_dict()]

    verified = CandidateGenome(
        id="verified",
        parent_ids=["P"],
        artifact={"different": True},
        verification_result=VerificationResult(
            passed=True,
            strength=VerificationStrength.EXECUTABLE,
            replayable=True,
            metadata={"validation_status": "preliminary_passed"},
        ).to_dict(),
    )
    outcomes = {item.candidate_id: item for item in productive_outcomes([survived, verified])}

    assert survived.verification_trace[0]["validation_status"] == "inconclusive"
    assert outcomes["probe-only"].reward <= outcomes["verified"].reward
    assert "parameterized_probe_survived" in outcomes["probe-only"].reason_codes
    assert "verified_or_evaluator_pass_survived" not in outcomes["probe-only"].reason_codes


def test_counterexample_writes_repair_evidence_without_changing_fate_or_evaluator_tier() -> None:
    candidate = _candidate(_case("fails", path="/metrics/score", operator="gt", expected=9))
    regime = compile_grounding_regime(
        candidate=candidate,
        verifier_fingerprint="vf",
        artifact_hash="artifact",
        oracle_kind="toolrunner",
        override_adversarial_budget=1,
    )
    raw = VerificationResult(passed=True, metadata={"diagnostics_only": True})
    result = result_with_probe_observations(raw, execute_probes(raw, regime, candidate=candidate))

    record = apply_probe_counterexample_evidence(candidate, result, round_index=2)

    assert record is not None
    assert record.final_blocked is True
    assert record.parent_blocked is False
    assert record.terminal_reject is False
    assert candidate.current_fate == CandidateFate.ACTIVE.value
    assert "evaluator" not in candidate.metadata
    assert "correctness" not in candidate.multihead_scores
    assert "objective" not in candidate.multihead_scores
    assert evidence_records(candidate)[-1].source == "engine_parameterized_probe"
    assert evaluator_selection_key(candidate)[0] == 1
    memory = ChallengeMemory()
    assert memory.ingest(record, round_index=2)


class _ProbeVerifier:
    verifier_id = "adversarial-verifier"
    fingerprint = "vf-probe"
    plan = {"modality": "adversarial", "adversarial_budget": {"count": 1}}

    def __init__(self) -> None:
        self.calls = 0

    def check(self, candidate: CandidateGenome) -> VerificationResult:
        self.calls += 1
        return VerificationResult(passed=True, metadata={"diagnostics_only": True, "oracle_kind": "adversarial"})


def test_probe_signature_participates_in_verification_cache_identity() -> None:
    cache: dict[str, dict] = {}
    first = _candidate(_case("same", path="/metrics/score", operator="gte", expected=8))
    second = _candidate(_case("same", path="/metrics/score", operator="gte", expected=9))
    verifier = _ProbeVerifier()

    first_result, first_key, first_hit = check_with_cache(first, verifier, cache)
    second_result, second_key, second_hit = check_with_cache(second, verifier, cache)

    assert first_hit is False
    assert second_hit is False
    assert first_key != second_key
    assert first_result.metadata["probe_signature"] != second_result.metadata["probe_signature"]
    assert verifier.calls == 2


def test_probe_case_keeps_legacy_constructor_compatibility() -> None:
    probe = ProbeCase("p", "engine", "engine", False)
    assert probe.template_id == ""
    assert probe.parameters == {}
