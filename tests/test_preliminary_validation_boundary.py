from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cognitive_evolve_runtime.api.jobs import _job_public
from cognitive_evolve_runtime.api.payloads import _completion_payload
from cognitive_evolve_runtime.api.streaming import _stream_done_chunk
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.nexus.adaptive.elite_gate import build_final_certificate
from cognitive_evolve_runtime.nexus.final_projection import build_final_projection
from cognitive_evolve_runtime.nexus.model_adapter_schemas import _stop_decision_schema
from cognitive_evolve_runtime.nexus.nextgen import best_current_direction_payload, candidate_verification_status
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, EvolutionRound
from cognitive_evolve_runtime.nexus.loop.controller import _replay_certificate_for_final_state
from cognitive_evolve_runtime.nexus.state import nexus_runtime_state, nexus_verification_results
from cognitive_evolve_runtime.nexus.stop_decision import StopDecisionEngine
from cognitive_evolve_runtime.nexus.synthesis import SynthesizedResult
from cognitive_evolve_runtime.verification.grading import grade
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.types import GradedOutput, VerificationPlan, VerificationResult, VerifiedResult
from cognitive_evolve_runtime.validation.result import VerificationVerdict, verification_result_from_mapping


def _replay_certificate() -> dict[str, object]:
    return {
        "frozen_artifact_hash": "artifact-1",
        "verifier_fingerprint": "vf",
        "measured_strength": "FORMAL",
        "measured_strength_value": 4,
        "honesty_measurements": {
            "exogeneity_score": 1.0,
            "variety_score": 1.0,
            "falsification_score": 1.0,
            "replay_score": 1.0,
        },
    }


def _legacy_graded_output() -> dict[str, object]:
    return {
        "mode": "verified_result",
        "verification_strength": "FORMAL",
        "verification_strength_value": 4,
        "result": {
            "answer": "candidate answer",
            "replayable": True,
            "evidence_ref": "e1",
            "verifier_fingerprint": "vf",
        },
        "portfolio": [],
        "ruled_out_map": [],
        "replay_certificate": _replay_certificate(),
    }


def _legacy_run() -> dict[str, object]:
    return {
        "mode": "text",
        "evolution": {
            "completion_status": "solved",
            "stop_reason": "objective_solved",
            "progress_events": [{"round": 1}],
            "synthesis": {
                "completion_status": "solved",
                "objective_solved": True,
                "answer_produced": True,
                "final_answer": "candidate answer",
                "best_candidate_id": "C1",
                "graded_output": _legacy_graded_output(),
                "closure_certificate": {
                    "objective_solved": True,
                    "answer_produced": True,
                    "critical_failures": [],
                    "graded_output": _legacy_graded_output(),
                },
            },
        },
        "verification_summaries": [{"passed": True, "source": "local-check"}],
    }


def test_replay_certificate_distinguishes_terminal_readback_from_continuation() -> None:
    terminal = _replay_certificate_for_final_state(
        synthesis=SynthesizedResult(
            status="completed",
            final_answer="answer",
            closure_certificate={"stop_reason": "candidate_ready_for_external_review"},
        ),
        final_certificate={},
        latent_replay_audit={},
    )
    continuation = _replay_certificate_for_final_state(
        synthesis=SynthesizedResult(
            status="completed",
            final_answer="answer",
            closure_certificate={"stop_reason": "max_rounds"},
        ),
        final_certificate={},
        latent_replay_audit={},
    )

    assert terminal["scope"] == "verifier_on_frozen_artifact_only"
    assert terminal["checkpoint_resume_semantics"] == "terminal_checkpoint_reads_existing_result"
    assert terminal["continuation_may_call_model"] is False
    assert "reads persisted run-result.json" in terminal["replay_command"]
    assert continuation["scope"] == "continued_evolution_not_frozen_replay"
    assert continuation["checkpoint_resume_semantics"] == "non_terminal_checkpoint_continues_evolution"
    assert continuation["continuation_may_call_model"] is True
    assert "may call the model" in continuation["replay_command"]


def test_legacy_verified_result_is_read_as_preliminary_only() -> None:
    restored = GradedOutput.from_dict(_legacy_graded_output())
    serialized = restored.to_dict()

    assert restored.mode == "preliminary_result"
    assert serialized["mode"] == "preliminary_result"
    assert serialized["validation_status"] == "preliminary_passed"
    assert "verified_result" not in json.dumps(serialized)

    produced = grade(
        {
            "verified_result": _legacy_graded_output()["result"],
            "replay_certificate": _replay_certificate(),
        }
    ).to_dict()
    assert produced["mode"] == "preliminary_result"


def test_candidate_local_pass_is_preliminary_and_route_stays_best_current() -> None:
    candidate = CandidateGenome(
        id="C1",
        artifact="candidate answer",
        verification_result={"passed": True, "replayable": True},
    )

    payload = best_current_direction_payload(candidate, route="final")

    assert candidate_verification_status(candidate) == "preliminary_passed"
    assert payload["route"] == "best_current"
    assert payload["validation_status"] == "preliminary_passed"
    assert "verification_status" not in payload
    assert "blocked_from_verified_claim_reason" not in payload


def test_graded_preliminary_result_only_applies_to_its_bound_candidate() -> None:
    graded = _legacy_graded_output()
    graded["replay_certificate"]["candidate_id"] = "C1"  # type: ignore[index]

    bound = best_current_direction_payload(CandidateGenome(id="C1", artifact="one"), graded_output=graded)
    other = best_current_direction_payload(CandidateGenome(id="C2", artifact="two"), graded_output=graded)

    assert bound["validation_status"] == "preliminary_passed"
    assert other["validation_status"] == "not_run"


def test_legacy_solved_state_is_downgraded_to_external_review_handoff() -> None:
    result = nexus_verification_results(_legacy_run())

    assert result["objective_solved"] is False
    assert result["completion_status"] == "completed"
    assert result["validation_status"] == "preliminary_passed"
    assert result["preliminary_validation_passed"] is True
    assert result["ready_for_external_review"] is True


def test_unobserved_validation_is_not_run() -> None:
    result = nexus_verification_results(
        {
            "evolution": {
                "completion_status": "completed",
                "synthesis": {"answer_produced": True, "final_answer": "best current"},
            }
        }
    )

    assert result["validation_status"] == "not_run"
    assert result["preliminary_validation_passed"] is None
    assert result["ready_for_external_review"] is False


def test_model_stop_producer_cannot_claim_solved() -> None:
    schema = _stop_decision_schema()
    assert schema["properties"]["solved"] == {"type": "boolean", "const": False}
    assert "objective_solved" not in schema["properties"]["stop_kind"]["enum"]

    class LegacySolvedModel:
        def should_stop(self, **_: object) -> dict[str, object]:
            return {"stop": True, "solved": True, "reason": "objective_solved"}

    reason = StopDecisionEngine().stop_reason_after_round(
        budget=SimpleNamespace(stop_policy="llm_after_minimum", min_rounds_before_stop=1, history=[]),
        completed_round=1,
        diagnosis=SimpleNamespace(stagnation_detected=False, stagnation_type="none", notes=""),
        best_answer_id="C1",
        population=CandidatePopulation([CandidateGenome(id="C1", artifact="candidate")]),
        model=LegacySolvedModel(),
    )
    assert reason == "candidate_ready_for_external_review"


@pytest.mark.parametrize("policy", ["adaptive_until_solved", "llm_after_minimum"])
def test_model_cannot_stop_for_external_review_without_candidate_material(policy: str) -> None:
    class EmptyReviewModel:
        def should_stop(self, **_: object) -> dict[str, object]:
            return {"stop": True, "solved": False, "continuation_needed": False, "confidence": 0.9}

    reason = StopDecisionEngine().stop_reason_after_round(
        budget=SimpleNamespace(stop_policy=policy, min_rounds_before_stop=1, history=[]),
        completed_round=1,
        diagnosis=SimpleNamespace(stagnation_detected=False, stagnation_type="none", notes=""),
        best_answer_id="",
        population=CandidatePopulation(),
        model=EmptyReviewModel(),
    )

    assert reason == "model_stop_unsolved_needs_continuation"


def test_projection_and_synthesis_never_serialize_a_solved_or_verified_claim() -> None:
    candidate = CandidateGenome(
        id="C1",
        artifact="candidate answer",
        verification_result={"passed": True, "replayable": True},
    )
    synthesis = SynthesizedResult(
        status="completed",
        final_answer="candidate answer",
        best_candidate_id="C1",
        objective_solved=True,
        answer_produced=True,
        closure_certificate={"objective_solved": True, "answer_produced": True},
    )
    graded = GradedOutput(
        mode="verified_result",
        verification_strength=VerificationStrength.FORMAL,
        result=VerifiedResult("candidate answer", replayable=True, evidence_ref="e1", verifier_fingerprint="vf"),
        replay_certificate=_replay_certificate(),
    )
    projection = build_final_projection(
        population=CandidatePopulation([candidate]),
        synthesis=synthesis,
        graded_output=graded,
    )
    markdown = projection.to_markdown()

    assert synthesis.to_dict()["objective_solved"] is False
    assert synthesis.to_dict()["closure_certificate"]["objective_solved"] is False
    assert projection.to_dict()["objective_solved"] is False
    assert projection.title == "Best current candidate"
    assert "preliminary_passed" in markdown
    assert "verified_result" not in markdown


def test_projection_preserves_preliminary_blockers_as_review_limitations() -> None:
    candidate = CandidateGenome(id="C1", artifact="candidate answer")
    projection = build_final_projection(
        population=CandidatePopulation([candidate]),
        synthesis=SynthesizedResult(status="completed", final_answer="candidate answer", best_candidate_id="C1"),
        graded_output=GradedOutput(mode="graded_portfolio", verification_strength=VerificationStrength.NONE),
        final_certificate={
            "candidate_id": "C1",
            "blocking_reasons": ["preliminary_evaluator_failed_advisory", "critical_external_input_missing"],
        },
    )

    assert "preliminary_evaluator_failed_advisory" in projection.advisory_issues
    assert "critical_external_input_missing" in projection.advisory_issues


def test_api_job_and_stream_surfaces_only_preliminary_and_review_readiness() -> None:
    legacy = _legacy_run()
    completion = _completion_payload(
        request_id="req",
        model="cognitive-evolve-one-shot",
        prompt="task",
        answer="candidate answer",
        nexus_data=legacy,
    )["cognitive_evolve"]
    job = _job_public(
        {
            "id": "job",
            "status": "completed",
            "created": 1,
            "updated": 1,
            "model": "cognitive-evolve-one-shot",
            "answer": "candidate answer",
            "nexus_data": legacy,
        }
    )["cognitive_evolve"]
    stream = json.loads(_stream_done_chunk(request_id="req", model="model", created=1, nexus_data=legacy).decode().split("data: ", 1)[1])["cognitive_evolve"]

    for surface in (completion, job, stream):
        assert surface["objective_solved"] is False
        assert surface["preliminary_validation_passed"] is True
        assert surface["validation_status"] == "preliminary_passed"
        assert surface["ready_for_external_review"] is True
        assert "verification_passed" not in surface


def test_adaptive_final_certificate_is_preliminary_only() -> None:
    candidate = CandidateGenome(id="C1", artifact="candidate answer", verification_result={"passed": True})
    certificate = build_final_certificate(
        population=CandidatePopulation([candidate]),
        synthesis=SynthesizedResult(status="completed", final_answer="candidate answer", best_candidate_id="C1"),
        closure_certificate={"objective_solved": True, "stop_reason": "objective_solved"},
        evaluator_required=False,
    )

    assert certificate["objective_solved"] is False
    assert certificate["ready_for_external_review"] is True
    assert certificate["preliminary_validation_passed"] is True
    assert certificate["validation_semantics"] == "preliminary_only_external_review_required"


def test_missing_check_does_not_become_a_synthetic_preliminary_pass() -> None:
    candidate = CandidateGenome(id="C1", artifact="candidate answer")

    certificate = build_final_certificate(
        population=CandidatePopulation([candidate]),
        synthesis=SynthesizedResult(status="completed", final_answer="candidate answer", best_candidate_id="C1"),
        closure_certificate={"stop_reason": "candidate_ready_for_external_review"},
        evaluator_required=False,
    )

    assert candidate_verification_status(candidate) == "not_run"
    assert certificate["preliminary_validation_passed"] is None
    assert "preliminary_check_not_run_advisory" in certificate["blocking_reasons"]


def test_failed_check_uses_preliminary_failed_vocabulary() -> None:
    candidate = CandidateGenome(id="C1", artifact="candidate answer", verification_result={"passed": False})

    assert candidate_verification_status(candidate) == "preliminary_failed"
    assert VerificationResult(passed=False).to_dict()["validation_status"] == "preliminary_failed"
    assert VerificationResult(passed=False, metadata={"validation_status": "inconclusive"}).to_dict()["validation_status"] == "inconclusive"
    assert candidate_verification_status(CandidateGenome(verification_result={"passed": False, "validation_status": "not_run"})) == "not_run"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"passed": True, "validation_status": "preliminary_passed"}, "preliminary_passed"),
        ({"passed": False, "validation_status": "preliminary_failed"}, "preliminary_failed"),
        ({"passed": False, "validation_status": "not_run"}, "not_run"),
        ({"passed": False, "validation_status": "inconclusive"}, "inconclusive"),
    ],
)
def test_verification_result_roundtrip_preserves_canonical_status(payload: dict[str, object], expected: str) -> None:
    assert VerificationResult.from_dict(payload).to_dict()["validation_status"] == expected


def test_not_run_and_inconclusive_are_not_failures() -> None:
    for status, verdict in (("not_run", VerificationVerdict.SKIP), ("inconclusive", VerificationVerdict.INCONCLUSIVE)):
        raw = {"passed": False, "validation_status": status}
        canonical = verification_result_from_mapping(raw)
        candidate = CandidateGenome(id=status, artifact="answer", verification_result=raw)
        certificate = build_final_certificate(
            population=CandidatePopulation([candidate]),
            synthesis=SynthesizedResult(status="completed", final_answer="answer", best_candidate_id=status),
            closure_certificate={},
            evaluator_required=False,
        )

        assert canonical.verdict is verdict
        assert canonical.passed is None
        assert certificate["preliminary_validation_passed"] is None
        assert "preliminary_check_failed_advisory" not in certificate["blocking_reasons"]


def test_unmeasured_verifier_pass_stays_trace_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import cognitive_evolve_runtime.nexus.loop.round as round_module

    evolution_round = EvolutionRound(model=None, budget=EvolutionBudget(branch_factor=1))
    evolution_round.adaptive.state.verification_plan = VerificationPlan(
        verifier_id="test",
        strength=VerificationStrength.NONE,
        modality="decomposed",
        verifier_fingerprint="test-v1",
    ).to_dict()
    raw_result = VerificationResult(
        passed=True,
        replayable=True,
        metadata={"measured_strength": "NONE", "honesty_measurements": {}, "validation_status": "preliminary_passed"},
    )
    monkeypatch.setattr(round_module, "verifier_from_plan", lambda _plan: object())
    monkeypatch.setattr(round_module, "check_with_cache", lambda *_args: (raw_result, "cache-key", False))
    candidate = CandidateGenome(id="C1", current_fate="Active", artifact="answer")

    results = evolution_round._run_synthesized_verifier([candidate], current_round=1)

    assert results == [raw_result]
    assert candidate.verification_result == {}
    assert candidate.verification_trace[-1]["validation_status"] == "preliminary_passed"


def test_failed_or_interrupted_integrity_dominates_a_passing_summary() -> None:
    interrupted = _legacy_run()
    interrupted["evolution"]["interrupted"] = True  # type: ignore[index]
    interrupted["evolution"]["synthesis"]["status"] = "interrupted_checkpointed"  # type: ignore[index]

    result = nexus_verification_results(interrupted)

    assert result["validation_status"] == "preliminary_failed"
    assert result["preliminary_validation_passed"] is False
    assert result["runtime_integrity_passed"] is False


def test_runtime_state_migrates_legacy_public_vocabulary(tmp_path) -> None:
    legacy = _legacy_run()
    legacy["evolution"]["synthesis"]["best_current_direction"] = {  # type: ignore[index]
        "candidate_id": "C1",
        "route": "final",
        "verification_status": "verified",
        "blocked_from_verified_claim_reason": "",
    }

    state = nexus_runtime_state(task_dir=tmp_path, prompt="task", run_data=legacy)
    serialized = json.dumps(state, sort_keys=True)

    assert '"objective_solved": true' not in serialized
    assert '"mode": "verified_result"' not in serialized
    assert '"verification_status": "verified"' not in serialized
    best = state["nexus_runtime"]["evolution"]["synthesis"]["best_current_direction"]
    public_evolution = state["nexus_runtime"]["evolution"]
    assert public_evolution["completion_status"] == "completed"
    assert public_evolution["stop_reason"] == "candidate_ready_for_external_review"
    assert public_evolution["synthesis"]["completion_status"] == "completed"
    assert best["route"] == "best_current"
    assert best["validation_status"] == "preliminary_passed"
    assert "passed" not in state["verification_results"]
    assert state["nexus_search"]["completion_status_note"] == "legacy_solved_status_downgraded_to_completed"


def test_candidate_serialization_preserves_nested_domain_state_and_hashes_it() -> None:
    candidate = CandidateGenome(
        id="C-legacy",
        verification_result={"passed": True, "status": "verified"},
        metadata={
            "objective_solved": True,
            "verification_status": "verified",
            "parent_verification_summary": {"verification_result": {"passed": True, "status": "verified"}},
        },
    )

    payload = candidate.to_dict()
    sibling = CandidateGenome(
        id="C-legacy",
        verification_result={"passed": True, "status": "verified"},
        metadata={
            "objective_solved": False,
            "verification_status": "verified",
            "parent_verification_summary": {"verification_result": {"passed": True, "status": "verified"}},
        },
    )

    assert payload["verification_result"]["status"] == "verified"
    assert payload["metadata"]["objective_solved"] is True
    assert payload["metadata"]["parent_verification_summary"]["verification_result"]["status"] == "verified"
    assert CandidateGenome.from_dict(payload).verification_result["passed"] is True
    assert candidate.genome_hash != sibling.genome_hash


def test_candidate_serialization_keeps_artifact_opaque_and_status_consistent() -> None:
    artifact = {
        "solved": True,
        "objective_solved": True,
        "verification_result": {"status": "passed", "value": 7},
    }
    candidate = CandidateGenome(
        id="C-opaque",
        artifact=artifact,
        verification_result={"status": "passed", "value": 7},
    )

    payload = candidate.to_dict()

    assert payload["artifact"] == artifact
    assert payload["verification_result"] == {"status": "passed", "value": 7}
    assert CandidateGenome.from_dict(payload).artifact == artifact
