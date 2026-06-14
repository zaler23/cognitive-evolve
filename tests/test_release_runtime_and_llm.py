from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognitive_evolve_runtime.artifacts.task_files import ensure_task_skeleton
from cognitive_evolve_runtime.nexus.semantics import ensure_enhanced_task_contract
from cognitive_evolve_runtime.llm import (
    LLMConfigurationError,
    LLMResponseError,
    LLMSession,
    ThrottledLLMGovernor,
    _bounded_prompt_for_provider,
    _completion_with_retry,
    _enforce_budget,
    _extract_json_from_text,
    _is_retryable_provider_error,
    _litellm_provider_kwargs,
    _load_fixture_response,
    _provider_error_category,
    _retry_after_seconds,
    _retry_sleep_seconds,
    _usage_dict,
    current_llm_session,
    llm_json,
    llm_session,
    llm_status_cli,
    reset_llm_events,
    write_llm_runtime_report,
)
from cognitive_evolve_runtime.runtime import runtime_run, runtime_status
from cognitive_evolve_runtime.llm.env import env_float as _env_float, env_int as _env_int
from cognitive_evolve_runtime.llm.telemetry import record_event as record_llm_event
from cognitive_evolve_runtime.nexus.evaluation import runtime_validation_run, native_eval_run


def test_fixture_backed_runtime_smoke_same_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = Path(__file__).parent / "fixtures" / "llm_fixture.json"
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "fixture")
    monkeypatch.setenv("COGEV_LLM_FIXTURE", str(fixture))
    monkeypatch.setenv("COGEV_INTERNAL_ROUND_CAP", "1")
    monkeypatch.setenv("COGEV_EVOLUTION_PROFILE", "balanced")

    task_dir = tmp_path / "task"
    ensure_task_skeleton(task_dir, "general", "architecture audit")
    ensure_enhanced_task_contract(task_dir, "Audit this architecture for no silent degradation", print_summary=False, force=True)

    assert runtime_run(str(task_dir), None, activate_all=True, rounds=1) == 0
    assert runtime_status(str(task_dir)) == 0
    assert runtime_validation_run(str(task_dir)) == 0
    assert native_eval_run(str(task_dir)) == 0

    state = json.loads((task_dir / "runtime-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["single_runtime"]["source_of_truth"] == "NexusRuntime"
    assert state["nexus_evolution"]["actual_rounds"] == 1
    assert (task_dir / "evaluations" / "llm-runtime-report.json").exists()
    assert (task_dir / "nexus-runtime" / "nexus-runtime-self-check.json").exists()


def test_llm_fixture_json_budget_governor_and_reporting_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = Path(__file__).parent / "fixtures" / "llm_fixture.json"
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "fixture")
    monkeypatch.setenv("COGEV_LLM_FIXTURE", str(fixture))
    monkeypatch.setenv("COGEV_LLM_MAX_PROMPT_CHARS", "20")
    monkeypatch.setenv("COGEV_LLM_API_KEY", "real-key")
    monkeypatch.setenv("COGEV_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.delenv("COGEV_LLM_BUDGET_USD", raising=False)
    reset_llm_events()

    response = llm_json("score_candidate", {"candidate": {"id": "A"}}, system="Return JSON", schema_hint={})
    assert response["provider"].startswith("fixture")
    assert current_llm_session().snapshot()

    fixture_response = _load_fixture_response("score_candidate", {}, str(fixture))
    assert isinstance(fixture_response, dict)
    assert _extract_json_from_text('```json\n{"ok": true}\n```') == {"ok": True}
    with pytest.raises(LLMResponseError):
        _extract_json_from_text("not-json")
    with pytest.raises(LLMResponseError):
        _load_fixture_response("missing_request_type", {}, str(fixture))

    sent, meta = _bounded_prompt_for_provider("x" * 100)
    assert meta["truncated"] is True
    assert len(sent) <= meta["max_prompt_chars"]
    assert _litellm_provider_kwargs()["api_base"] == "https://example.test/v1"
    assert _usage_dict({"usage": {"input_tokens": 2, "output_tokens": 3}})["prompt_tokens"] == 2

    session = LLMSession()
    with llm_session(session):
        record_llm_event("manual", {"confidence": 0.5}, {"provider": "fixture", "model": "fixture"}, estimated_cost_usd=0.0)
        task_dir = tmp_path / "report-task"
        (task_dir / "evaluations").mkdir(parents=True)
        write_llm_runtime_report(task_dir)
        report = json.loads((task_dir / "evaluations" / "llm-runtime-report.json").read_text(encoding="utf-8"))
        assert report["event_count"] == 1
        assert report["no_llm_fallback"] is True

    monkeypatch.setenv("COGEV_LLM_BUDGET_USD", "0")
    with pytest.raises(LLMResponseError):
        _enforce_budget(preflight=True)
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.delenv("COGEV_LLM_MODEL", raising=False)
    with pytest.raises(LLMConfigurationError):
        llm_json("score_candidate", {}, system="x", schema_hint={})


def test_llm_retry_error_classification_and_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    class RateLimitError(RuntimeError):
        status_code = 429

    class RetryAfterError(RuntimeError):
        retry_after = 0.01

    assert _env_int("COGEV_BAD_INT", 7) == 7
    assert _env_float("COGEV_BAD_FLOAT", 1.5) == 1.5
    assert _provider_error_category(RateLimitError("rate limit")) == "rate_limit_429"
    assert _is_retryable_provider_error(RuntimeError("temporary network failure")) is True
    assert _retry_after_seconds(RetryAfterError("retry-after: 0.01")) == 0.01
    monkeypatch.setenv("COGEV_LLM_RETRY_JITTER", "0")
    assert _retry_sleep_seconds(RetryAfterError("retry-after: 0.01"), 1) == 0.01

    attempts = {"count": 0}

    def flaky_completion(**kwargs: object) -> dict[str, str]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary network failure")
        return {"ok": "yes"}

    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "2")
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    result, attempt_count = _completion_with_retry(flaky_completion, messages=[{"role": "user", "content": "hi"}], max_tokens=1)
    assert result == {"ok": "yes"}
    assert attempt_count == 2

    gov = ThrottledLLMGovernor()
    with gov.acquire(estimated_tokens=1) as status:
        assert status["retry_after_supported"] is True
    assert llm_status_cli() == 0
