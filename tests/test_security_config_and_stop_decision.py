from __future__ import annotations

from typing import Any

import pytest

from cognitive_evolve_runtime.api.config import get_service_config, service_api_key_is_placeholder_or_weak
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.llm.env import LLMConfigurationError, require_llm_config
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, _closure_certificate, _completion_status_for_budget, _is_solved_stop_reason
from cognitive_evolve_runtime.nexus.stop_decision import StopDecisionEngine
from cognitive_evolve_runtime.nexus.synthesis import SynthesizedResult
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.types import VerificationResult


class UnsolvedStopModel:
    def should_stop(self, **_: Any) -> dict[str, Any]:
        return {"stop": True, "solved": False, "reason": "not solved but stop for checkpoint", "open_gaps": ["gap"]}


class SolvedStopModel:
    def should_stop(self, **_: Any) -> dict[str, Any]:
        return {"stop": True, "solved": True, "reason": "objective_solved"}


class LegacyBoolStopModel:
    def should_stop(self, **_: Any) -> bool:
        return True


def test_model_stop_true_solved_false_is_needs_continuation_not_solved() -> None:
    budget = EvolutionBudget(max_rounds=3, stop_policy="llm_after_minimum", min_rounds_before_stop=1)
    reason = StopDecisionEngine().stop_reason_after_round(
        budget=budget,
        completed_round=1,
        diagnosis=SearchDiagnosis(),
        best_answer_id="C1",
        population=CandidatePopulation([CandidateGenome(id="C1")]),
        model=UnsolvedStopModel(),
    )
    budget.stop_reason = reason
    synthesis = SynthesizedResult(status="needs_continuation", final_answer="not solved")

    assert reason == "model_stop_unsolved_needs_continuation"
    assert _completion_status_for_budget(budget=budget, interrupted=False, synthesis=synthesis) == "needs_continuation"
    assert _is_solved_stop_reason("not solved but verified-looking text") is False
    assert _is_solved_stop_reason("model_stop_after_minimum") is False


def test_legacy_bool_stop_signal_is_continuation_not_completed() -> None:
    budget = EvolutionBudget(max_rounds=3, stop_policy="llm_after_minimum", min_rounds_before_stop=1)
    reason = StopDecisionEngine().stop_reason_after_round(
        budget=budget,
        completed_round=1,
        diagnosis=SearchDiagnosis(),
        best_answer_id="C1",
        population=CandidatePopulation([CandidateGenome(id="C1")]),
        model=LegacyBoolStopModel(),
    )
    budget.stop_reason = reason

    assert reason == "model_stop_needs_measured_verification"
    assert _completion_status_for_budget(
        budget=budget,
        interrupted=False,
        synthesis=SynthesizedResult(status="completed", final_answer="ambiguous legacy stop"),
    ) == "needs_continuation"


def test_model_stop_true_solved_true_uses_exact_solved_reason() -> None:
    budget = EvolutionBudget(max_rounds=3, stop_policy="llm_after_minimum", min_rounds_before_stop=1)
    reason = StopDecisionEngine().stop_reason_after_round(
        budget=budget,
        completed_round=1,
        diagnosis=SearchDiagnosis(),
        best_answer_id="C1",
        population=CandidatePopulation([_candidate_with_measured_formal("C1")]),
        model=SolvedStopModel(),
    )

    assert reason == "objective_solved"
    assert _is_solved_stop_reason(reason) is True


def test_model_stop_solved_requires_measured_strength_gate() -> None:
    budget = EvolutionBudget(max_rounds=3, stop_policy="llm_after_minimum", min_rounds_before_stop=1)
    reason = StopDecisionEngine().stop_reason_after_round(
        budget=budget,
        completed_round=1,
        diagnosis=SearchDiagnosis(),
        best_answer_id="C1",
        population=CandidatePopulation([CandidateGenome(id="C1")]),
        model=SolvedStopModel(),
    )

    assert reason == "model_stop_needs_measured_verification"
    assert _is_solved_stop_reason(reason) is False


def _candidate_with_measured_formal(candidate_id: str) -> CandidateGenome:
    candidate = CandidateGenome(id=candidate_id)
    candidate.verification_trace = [
        VerificationResult(
            True,
            score=1.0,
            strength=VerificationStrength.FORMAL,
            evidence_ref="evidence",
            replayable=True,
            metadata={
                "verifier_fingerprint": "vf",
                "measured_strength": "FORMAL",
                "measured_strength_value": 4,
                "honesty_measurements": {
                    "exogeneity_score": 1.0,
                    "variety_score": 1.0,
                    "falsification_score": 1.0,
                    "replay_score": 1.0,
                },
                "diagnostics_only": False,
                "legacy": False,
            },
        ).to_dict()
    ]
    return candidate


def test_closure_certificate_records_legacy_solved_signal_but_is_not_v2_authority() -> None:
    budget = EvolutionBudget(max_rounds=3)
    budget.stop_reason = "objective_solved"
    solved_synthesis = SynthesizedResult(status="completed", final_answer="answer", best_candidate_id="C1")
    completion_status = _completion_status_for_budget(budget=budget, interrupted=False, synthesis=solved_synthesis)

    certificate = _closure_certificate(
        budget=budget,
        interrupted=False,
        synthesis=solved_synthesis,
        completion_status=completion_status,
    )

    assert completion_status == "solved"
    assert certificate["version"] == "closure_certificate_v1"
    # This remains a legacy closure signal. Public/API solved authority is
    # GradedOutput, covered by test_api_verification_passed_requires_v2_graded_verified_result.
    assert certificate["objective_solved"] is True
    assert certificate["terminal_status"] == "solved"
    assert not certificate["critical_failures"]


def test_closure_certificate_blocks_unsolved_or_interrupted_completion_claims() -> None:
    budget = EvolutionBudget(max_rounds=3)
    budget.stop_reason = "model_stop_unsolved_needs_continuation"
    synthesis = SynthesizedResult(status="needs_continuation", final_answer="not solved")
    synthesis.continuation_available = True
    completion_status = _completion_status_for_budget(budget=budget, interrupted=False, synthesis=synthesis)

    certificate = _closure_certificate(
        budget=budget,
        interrupted=False,
        synthesis=synthesis,
        completion_status=completion_status,
    )

    assert completion_status == "needs_continuation"
    assert certificate["objective_solved"] is False
    assert certificate["continuation_available"] is True
    assert "needs_continuation" in certificate["critical_failures"]


def test_closure_certificate_rejects_adversarial_solved_status_without_solved_reason() -> None:
    budget = EvolutionBudget(max_rounds=3)
    budget.stop_reason = "completed"
    certificate = _closure_certificate(
        budget=budget,
        interrupted=False,
        synthesis=SynthesizedResult(status="completed", final_answer="answer"),
        completion_status="solved",
    )

    assert certificate["objective_solved"] is False
    assert "solved_status_failed_certificate_gate" in certificate["critical_failures"]


def test_closure_certificate_rejects_interrupted_or_failed_solved_claims() -> None:
    budget = EvolutionBudget(max_rounds=3)
    budget.stop_reason = "objective_solved"
    interrupted = _closure_certificate(
        budget=budget,
        interrupted=True,
        synthesis=SynthesizedResult(status="completed", final_answer="answer"),
        completion_status="solved",
    )
    failed = _closure_certificate(
        budget=budget,
        interrupted=False,
        synthesis=SynthesizedResult(status="failure_report", final_answer="failed"),
        completion_status="solved",
    )

    assert interrupted["objective_solved"] is False
    assert "interrupted" in interrupted["critical_failures"]
    assert failed["objective_solved"] is False
    assert "failure_report" in failed["critical_failures"]



class ExternalReviewStopModel:
    def should_stop(self, **_: Any) -> dict[str, Any]:
        return {
            "stop": True,
            "solved": False,
            "reason": "candidate_ready_for_external_review",
            "stop_kind": "candidate_ready_for_external_review",
            "continuation_needed": False,
            "confidence": 0.82,
        }


def test_candidate_ready_for_external_review_stops_without_self_certifying_solution() -> None:
    budget = EvolutionBudget(max_rounds=8, stop_policy="llm_after_minimum", min_rounds_before_stop=1)
    candidate = CandidateGenome(id="C-review", artifact="reviewable candidate", current_fate="ELITE")
    reason = StopDecisionEngine().stop_reason_after_round(
        budget=budget,
        completed_round=1,
        diagnosis=SearchDiagnosis(),
        best_answer_id="C-review",
        population=CandidatePopulation([candidate]),
        model=ExternalReviewStopModel(),
    )
    budget.stop_reason = reason
    synthesis = SynthesizedResult(status="model_synthesized", final_answer="review this", reference_candidate_id="C-review")
    completion_status = _completion_status_for_budget(budget=budget, interrupted=False, synthesis=synthesis)
    certificate = _closure_certificate(
        budget=budget,
        interrupted=False,
        synthesis=synthesis,
        completion_status=completion_status,
    )

    assert reason == "candidate_ready_for_external_review"
    assert completion_status == "best_current_route"
    assert certificate["objective_solved"] is False
    assert certificate["terminal_status"] == "best_current_route"
    assert "best_current_route" in certificate["critical_failures"]


def test_diminishing_returns_checkpoint_stops_as_external_review_boundary() -> None:
    budget = EvolutionBudget(max_rounds=8, stop_policy="adaptive_until_solved", min_rounds_before_stop=1)
    reason = StopDecisionEngine().stop_reason_after_round(
        budget=budget,
        completed_round=3,
        diagnosis=SearchDiagnosis(stagnation_detected=True, stagnation_type="diminishing_returns", notes="low marginal gain"),
        best_answer_id="C-review",
        population=CandidatePopulation([CandidateGenome(id="C-review", artifact="candidate")]),
        model=None,
    )
    budget.stop_reason = reason

    assert reason == "diminishing_returns_checkpoint"
    assert _completion_status_for_budget(
        budget=budget,
        interrupted=False,
        synthesis=SynthesizedResult(status="model_synthesized", final_answer="review candidate"),
    ) == "best_current_route"

def test_public_bind_rejects_default_or_weak_service_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("COGEV_HERMETIC_TEST", "1")
    monkeypatch.setenv("COGEV_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("COGEV_SERVER_HOST", "0.0.0.0")
    monkeypatch.setenv("COGEV_SERVER_REQUIRE_AUTH", "true")
    monkeypatch.setenv("COGEV_SERVER_API_KEY", "ce-local-dev-key-change-me")
    monkeypatch.delenv("COGEV_ALLOW_INSECURE_BIND", raising=False)

    config = get_service_config()
    assert service_api_key_is_placeholder_or_weak(config.api_keys[0]) is True
    with pytest.raises(RuntimeError, match="placeholder or low-entropy"):
        config.enforce_safe_to_serve()


def test_loopback_allows_dev_service_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("COGEV_HERMETIC_TEST", "1")
    monkeypatch.setenv("COGEV_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("COGEV_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("COGEV_SERVER_REQUIRE_AUTH", "true")
    monkeypatch.setenv("COGEV_SERVER_API_KEY", "ce-local-dev-key-change-me")

    config = get_service_config()
    assert config.host == "127.0.0.1"
    assert service_api_key_is_placeholder_or_weak(config.api_keys[0]) is True
    config.enforce_safe_to_serve()


def test_require_llm_config_rejects_placeholder_upstream_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COGEV_HERMETIC_TEST", "1")
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "openai/gpt-x")
    monkeypatch.setenv("COGEV_LLM_API_KEY", "replace-with-your-upstream-model-api-key")

    with pytest.raises(LLMConfigurationError, match="placeholder"):
        require_llm_config()
