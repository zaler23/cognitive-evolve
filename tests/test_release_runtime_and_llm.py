from __future__ import annotations

import asyncio
import json
import threading
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
    logical_llm_call,
    llm_json,
    llm_session,
    llm_status_cli,
    reset_llm_events,
    write_llm_runtime_report,
)
from cognitive_evolve_runtime.runtime import runtime_run, runtime_status
from cognitive_evolve_runtime.llm.env import env_float as _env_float, env_int as _env_int
from cognitive_evolve_runtime.llm.telemetry import record_event as record_llm_event
from cognitive_evolve_runtime.llm.fanout import run_ordered_fanout
from cognitive_evolve_runtime.llm.mock_provider import MockProviderResponse
from cognitive_evolve_runtime.llm.provider_interface import LLMProviderResult
from cognitive_evolve_runtime.llm import transport as transport_module
from cognitive_evolve_runtime.nexus.evaluation import runtime_validation_run, native_eval_run
from cognitive_evolve_runtime.nexus.runtime import NexusRuntime


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
    assert _usage_dict(
        {
            "usage": {
                "input_tokens": 2,
                "output_tokens": 3,
                "cached_input_tokens": 1,
                "reasoning_output_tokens": 2,
            }
        }
    ) == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
        "cached_prompt_tokens": 1,
        "reasoning_tokens": 2,
    }
    assert _usage_dict(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 80},
                "completion_tokens_details": {"reasoning_tokens": 40},
            }
        }
    ) == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "cached_prompt_tokens": 80,
        "reasoning_tokens": 40,
    }

    session = LLMSession()
    with llm_session(session):
        record_llm_event("manual", {"confidence": 0.5}, {"provider": "fixture", "model": "fixture"}, estimated_cost_usd=0.0)
        task_dir = tmp_path / "report-task"
        (task_dir / "evaluations").mkdir(parents=True)
        write_llm_runtime_report(task_dir)
        report = json.loads((task_dir / "evaluations" / "llm-runtime-report.json").read_text(encoding="utf-8"))
        assert report["event_count"] == 1
        assert report["no_llm_fallback"] is True
        assert report["credential_configured"] is False
        assert report["api_base_configured"] is False
        assert "api_base" not in report
        assert "api_key_configured" not in report
        assert "api_key_placeholder" not in report

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


def test_concurrent_calls_cannot_both_spend_the_same_budget_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "test/model")
    monkeypatch.setenv("COGEV_LLM_API_KEY", "test-key")
    monkeypatch.setenv("COGEV_LLM_BUDGET_USD", "1")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")
    class Provider:
        provider_id = "test"
        calls = 0

        def complete_json(self, **kwargs: object) -> LLMProviderResult:
            self.calls += 1
            return LLMProviderResult(response=MockProviderResponse({"ok": True}), estimated_cost_usd=0.75)

    provider = Provider()
    session = LLMSession()

    def call(index: int) -> dict[str, object]:
        return llm_json("unit_test", {"index": index}, system="Return JSON", schema_hint={}, provider=provider)

    with llm_session(session), pytest.raises(LLMResponseError, match="headroom"):
        run_ordered_fanout([1, 2], call, max_workers=2)

    assert provider.calls == 1
    assert session.total_estimated_cost_usd() == 0.75
    assert session.budget_reservation_usd == 0.0


def test_budgeted_fanout_waits_instead_of_rejecting_ample_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "test/model")
    monkeypatch.setenv("COGEV_LLM_API_KEY", "test-key")
    monkeypatch.setenv("COGEV_LLM_BUDGET_USD", "10")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")
    active = 0
    max_active = 0
    active_lock = threading.Lock()

    class Provider:
        provider_id = "test"
        calls = 0

        def complete_json(self, **kwargs: object) -> LLMProviderResult:
            nonlocal active, max_active
            self.calls += 1
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            try:
                threading.Event().wait(0.01)
                return LLMProviderResult(response=MockProviderResponse({"ok": True}), estimated_cost_usd=0.01)
            finally:
                with active_lock:
                    active -= 1

    provider = Provider()
    session = LLMSession()
    with llm_session(session):
        results = run_ordered_fanout([1, 2], lambda index: llm_json("unit_test", {"index": index}, system="Return JSON", schema_hint={}, provider=provider), max_workers=2)

    assert [item["ok"] for item in results] == [True, True]
    assert provider.calls == 2
    assert max_active == 1
    assert session.total_estimated_cost_usd() == 0.02


def test_budget_reservation_releases_after_provider_error_and_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "test/model")
    monkeypatch.setenv("COGEV_LLM_API_KEY", "test-key")
    monkeypatch.setenv("COGEV_LLM_BUDGET_USD", "1")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")

    class FailingProvider:
        provider_id = "test"

        def complete_json(self, **kwargs: object) -> LLMProviderResult:
            raise RuntimeError("provider failed")

    class CancelledProvider:
        provider_id = "test"

        def complete_json(self, **kwargs: object) -> LLMProviderResult:
            raise asyncio.CancelledError

    class SuccessProvider:
        provider_id = "test"

        def complete_json(self, **kwargs: object) -> LLMProviderResult:
            response = MockProviderResponse({"ok": True})
            response.usage = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
            return LLMProviderResult(response=response, estimated_cost_usd=0.25)

    session = LLMSession()
    with llm_session(session):
        with pytest.raises(LLMResponseError, match="provider call failed"):
            llm_json("unit_test", {}, system="Return JSON", schema_hint={}, provider=FailingProvider())
        assert session.budget_reservation_usd == 0.0

        with pytest.raises(asyncio.CancelledError):
            llm_json("unit_test", {}, system="Return JSON", schema_hint={}, provider=CancelledProvider())
        assert session.budget_reservation_usd == 0.0

        assert llm_json("unit_test", {}, system="Return JSON", schema_hint={}, provider=SuccessProvider())["ok"] is True

    assert session.total_estimated_cost_usd() == 0.25
    assert session.snapshot()[-1]["usage_provenance"] == "provider_reported"
    assert session.budget_reservation_usd == 0.0


def test_crash_after_provider_response_does_not_duplicate_remote_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "test/model")
    monkeypatch.setenv("COGEV_LLM_API_KEY", "test-key")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")
    monkeypatch.delenv("COGEV_LLM_BUDGET_USD", raising=False)

    class Provider:
        provider_id = "test"
        calls = 0

        def complete_json(self, **kwargs: object) -> LLMProviderResult:
            self.calls += 1
            return LLMProviderResult(response=MockProviderResponse({"ok": True}), estimated_cost_usd=0.01)

    provider = Provider()
    original_record_event = transport_module.record_event

    def crash_after_response(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated crash after provider response")

    monkeypatch.setattr(transport_module, "record_event", crash_after_response)
    journal_dir = tmp_path / "llm"
    with llm_session(LLMSession(journal_dir=str(journal_dir))), logical_llm_call("round-1/crash-window"), pytest.raises(RuntimeError, match="simulated crash"):
        llm_json("crash_window", {"same": "request"}, system="Return JSON", schema_hint={}, provider=provider)

    monkeypatch.setattr(transport_module, "record_event", original_record_event)
    with llm_session(LLMSession(journal_dir=str(journal_dir))), logical_llm_call("round-1/crash-window"):
        llm_json("crash_window", {"same": "request"}, system="Return JSON", schema_hint={}, provider=provider)

    assert provider.calls == 1


def test_response_replay_does_not_merge_distinct_logical_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "test/model")
    monkeypatch.setenv("COGEV_LLM_API_KEY", "test-key")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")
    monkeypatch.delenv("COGEV_LLM_BUDGET_USD", raising=False)

    class Provider:
        provider_id = "test"
        calls = 0

        def complete_json(self, **kwargs: object) -> LLMProviderResult:
            self.calls += 1
            return LLMProviderResult(response=MockProviderResponse({"ok": True}), estimated_cost_usd=0.01)

    provider = Provider()
    session = LLMSession(journal_dir=str(tmp_path / "llm"))
    with llm_session(session):
        with logical_llm_call("round-1/slot-1"):
            llm_json("same_request", {"same": "request"}, system="Return JSON", schema_hint={}, provider=provider)
        with logical_llm_call("round-1/slot-2"):
            llm_json("same_request", {"same": "request"}, system="Return JSON", schema_hint={}, provider=provider)

    assert provider.calls == 2


def test_response_replay_counts_physical_usage_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "test/model")
    monkeypatch.setenv("COGEV_LLM_API_KEY", "test-key")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")
    monkeypatch.delenv("COGEV_LLM_BUDGET_USD", raising=False)

    class Provider:
        provider_id = "test"
        calls = 0

        def complete_json(self, **kwargs: object) -> LLMProviderResult:
            self.calls += 1
            return LLMProviderResult(response=MockProviderResponse({"ok": True}), estimated_cost_usd=0.01)

    provider = Provider()
    session = LLMSession(journal_dir=str(tmp_path / "llm"))
    with llm_session(session):
        for _ in range(2):
            with logical_llm_call("round-1/slot-1"):
                llm_json("same_request", {"same": "request"}, system="Return JSON", schema_hint={}, provider=provider)

    assert provider.calls == 1
    assert session.total_estimated_cost_usd() == 0.01
    assert sum(event.get("cache_replayed") is True for event in session.snapshot()) == 1


def test_runtime_response_replay_scope_is_stable_across_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "test/model")
    monkeypatch.setenv("COGEV_LLM_API_KEY", "test-key")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")
    monkeypatch.delenv("COGEV_LLM_BUDGET_USD", raising=False)
    monkeypatch.delenv("COGEV_RUN_ID", raising=False)

    class Provider:
        provider_id = "test"
        calls = 0

        def complete_json(self, **kwargs: object) -> LLMProviderResult:
            self.calls += 1
            return LLMProviderResult(response=MockProviderResponse({"ok": True}), estimated_cost_usd=0.01)

    provider = Provider()
    output_dir = tmp_path / "nexus-runtime"
    run_ids: list[str | None] = []
    for _ in range(2):
        session = LLMSession(journal_dir=str(tmp_path / "journal"))
        with llm_session(session):
            NexusRuntime(output_dir=output_dir)._bind_llm_artifact_scope()
            run_ids.append(session.run_id)
            with logical_llm_call("round-1/plan-1/slot-1"):
                llm_json("same_request", {"same": "request"}, system="Return JSON", schema_hint={}, provider=provider)

    assert provider.calls == 1
    assert run_ids[0] == run_ids[1]
    assert len(list((output_dir / "llm-responses" / "v1").glob("*.json"))) == 1


def test_response_replay_signature_tracks_resolved_sampling_run_and_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "test/model")
    monkeypatch.setenv("COGEV_LLM_API_KEY", "test-key")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")
    monkeypatch.delenv("COGEV_LLM_BUDGET_USD", raising=False)

    class Provider:
        provider_id = "test"
        calls = 0

        def complete_json(self, **kwargs: object) -> LLMProviderResult:
            self.calls += 1
            return LLMProviderResult(response=MockProviderResponse({"ok": True}), estimated_cost_usd=0.01)

    provider = Provider()
    journal_dir = str(tmp_path / "llm")
    monkeypatch.setenv("COGEV_LLM_TEMPERATURE", "0.2")
    with llm_session(LLMSession(run_id="run-a", journal_dir=journal_dir)), logical_llm_call("slot", template_version="v1"):
        llm_json("same_request", {"same": "request"}, system="Return JSON", schema_hint={}, provider=provider)
    monkeypatch.setenv("COGEV_LLM_TEMPERATURE", "0.7")
    with llm_session(LLMSession(run_id="run-a", journal_dir=journal_dir)), logical_llm_call("slot", template_version="v1"):
        llm_json("same_request", {"same": "request"}, system="Return JSON", schema_hint={}, provider=provider)
    with llm_session(LLMSession(run_id="run-b", journal_dir=journal_dir)), logical_llm_call("slot", template_version="v1"):
        llm_json("same_request", {"same": "request"}, system="Return JSON", schema_hint={}, provider=provider)
    with llm_session(LLMSession(run_id="run-b", journal_dir=journal_dir)), logical_llm_call("slot", template_version="v2"):
        llm_json("same_request", {"same": "request"}, system="Return JSON", schema_hint={}, provider=provider)

    assert provider.calls == 4
