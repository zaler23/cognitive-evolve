from __future__ import annotations

import json
import socket
import sys
import types
from types import SimpleNamespace
import urllib.request

import pytest

from cognitive_evolve_runtime.durable.idempotency import llm_idempotency_key
from cognitive_evolve_runtime.llm.call_identity import identity_from_status
from cognitive_evolve_runtime.llm.env import (
    LLMConfigurationError,
    llm_public_status,
    normalize_reasoning_effort,
    require_llm_config,
)
from cognitive_evolve_runtime.llm.http_provider import DirectHTTPProvider, DirectHTTPProviderError, normalize_direct_http_model
from cognitive_evolve_runtime.llm.litellm_provider import LiteLLMProvider
from cognitive_evolve_runtime.llm.model_spec import LLMModelSpec
from cognitive_evolve_runtime.llm.retry import provider_error_category
from cognitive_evolve_runtime.llm.provider_interface import LLMProviderResult
from cognitive_evolve_runtime.llm.request_policy import LLMRequestPolicy
from cognitive_evolve_runtime.llm.session import LLMSession, llm_session
from cognitive_evolve_runtime.llm.transport import _default_provider_for_status, llm_json, max_tokens_for_request


def test_direct_http_provider_posts_openai_compatible_request(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "id": "chatcmpl-test",
                    "model": "example-reasoning-model",
                    "choices": [{"message": {"role": "assistant", "content": '{"ok": true}'}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                }
            ).encode()

    def fake_urlopen(request, timeout):  # noqa: ANN001
        calls.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "headers": dict(request.header_items()),
                "payload": json.loads(request.data.decode("utf-8")),
            }
        )
        return FakeResponse()

    monkeypatch.setenv("COGEV_LLM_API_BASE", "http://localhost:8081/v1/")
    monkeypatch.setenv("COGEV_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = DirectHTTPProvider().complete_json(
        model="openai/example-reasoning-model",
        messages=[{"role": "user", "content": "Return JSON."}],
        temperature=0,
        max_tokens=64,
        reasoning_effort="high",
        response_format={"type": "json_object"},
        timeout=10,
    )

    assert result.response.choices[0].message.content == '{"ok": true}'
    assert result.response.usage["total_tokens"] == 3
    assert result.attempts == 1
    assert calls[0]["url"] == "http://localhost:8081/v1/chat/completions"
    assert calls[0]["payload"]["model"] == "example-reasoning-model"
    assert calls[0]["payload"]["reasoning_effort"] == "high"
    assert calls[0]["payload"]["response_format"] == {"type": "json_object"}
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"


def test_direct_http_defaults_to_endpoint_friendly_timeout_and_passes_seed(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [{"message": {"role": "assistant", "content": '{"ok": true}'}, "finish_reason": "stop"}],
                    "usage": {"total_tokens": 3},
                }
            ).encode()

    def fake_urlopen(request, timeout):  # noqa: ANN001
        calls.append({"timeout": timeout, "payload": json.loads(request.data.decode("utf-8"))})
        return FakeResponse()

    monkeypatch.setenv("COGEV_LLM_API_BASE", "http://localhost:8081/v1")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    DirectHTTPProvider().complete_json(
        model="endpoint-model",
        messages=[{"role": "user", "content": "Return JSON."}],
        max_tokens=64,
        seed=123,
    )

    assert calls[0]["timeout"] == 28.0
    assert calls[0]["payload"]["seed"] == 123


def test_direct_http_normalizes_socket_timeout_without_url(monkeypatch) -> None:
    def fake_urlopen(request, timeout):  # noqa: ANN001
        raise socket.timeout("timed out")

    monkeypatch.setenv("COGEV_LLM_API_BASE", "http://localhost:8081/v1")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(DirectHTTPProviderError) as exc_info:
        DirectHTTPProvider().complete_json(
            model="endpoint-model",
            messages=[{"role": "user", "content": "Return JSON."}],
            max_tokens=64,
        )

    assert "Timeout from direct_http provider" in str(exc_info.value)
    assert "http://localhost:8081" not in str(exc_info.value)
    assert provider_error_category(exc_info.value) == "timeout"


def test_direct_http_model_normalization_is_narrow() -> None:
    assert normalize_direct_http_model("openai/example-reasoning-model") == "example-reasoning-model"
    assert normalize_direct_http_model("vendor/model") == "vendor/model"


def test_direct_http_retries_empty_assistant_content(monkeypatch) -> None:
    attempts = {"count": 0}

    class FakeResponse:
        def __init__(self, content: str):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "id": "chatcmpl-test",
                    "model": "example-reasoning-model",
                    "choices": [{"message": {"role": "assistant", "content": self.content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()

    def fake_urlopen(request, timeout):  # noqa: ANN001
        attempts["count"] += 1
        return FakeResponse("" if attempts["count"] == 1 else '{"ok": true}')

    monkeypatch.setenv("COGEV_LLM_API_BASE", "http://localhost:8081/v1")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("COGEV_LLM_RETRY_JITTER", "0")
    monkeypatch.setenv("COGEV_LLM_RETRY_BASE_SLEEP", "0")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = DirectHTTPProvider().complete_json(
        model="example-reasoning-model",
        messages=[{"role": "user", "content": "Return JSON."}],
        max_tokens=64,
        timeout=10,
    )

    assert attempts["count"] == 2
    assert result.attempts == 2
    assert result.response.choices[0].message.content == '{"ok": true}'


def test_direct_http_falls_back_to_reasoning_content(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "", "reasoning_content": '{"ok": true}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"total_tokens": 3},
                }
            ).encode()

    monkeypatch.setenv("COGEV_LLM_API_BASE", "http://localhost:8081/v1")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: FakeResponse())

    result = DirectHTTPProvider().complete_json(model="example-test-model", messages=[{"role": "user", "content": "Return JSON."}], max_tokens=64)

    assert result.response.choices[0].message.content == '{"ok": true}'


def test_direct_http_truncated_length_is_semantic_error(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [{"message": {"role": "assistant", "content": '{"ok":'}, "finish_reason": "length"}],
                    "usage": {"total_tokens": 3},
                }
            ).encode()

    monkeypatch.setenv("COGEV_LLM_API_BASE", "http://localhost:8081/v1")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: FakeResponse())

    try:
        DirectHTTPProvider().complete_json(model="example-test-model", messages=[{"role": "user", "content": "Return JSON."}], max_tokens=64)
    except DirectHTTPProviderError as exc:
        assert "TRUNCATED" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected truncation error")


def test_transport_selects_direct_http_provider(monkeypatch) -> None:
    assert isinstance(_default_provider_for_status({"provider": "direct_http"}), DirectHTTPProvider)

    def fake_complete_json(self, **kwargs):  # noqa: ANN001
        return LLMProviderResult(
            response=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            ),
            attempts=1,
        )

    monkeypatch.setenv("COGEV_LLM_PROVIDER", "direct_http")
    monkeypatch.setenv("COGEV_LLM_MODEL", "openai/example-reasoning-model")
    monkeypatch.setenv("COGEV_LLM_API_BASE", "http://localhost:8081/v1")
    monkeypatch.setenv("COGEV_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")
    monkeypatch.setattr(DirectHTTPProvider, "complete_json", fake_complete_json)

    response = llm_json("unit_test", {"x": 1}, system="Return JSON", schema_hint={})
    assert response["ok"] is True
    assert response["provider"] == "direct_http"


def test_transport_uses_explicit_request_policy_output_budgets(monkeypatch) -> None:
    seen: list[int] = []

    class Provider:
        provider_id = "unit"

        def complete_json(self, **kwargs):  # noqa: ANN001
            seen.append(kwargs["max_tokens"])
            return LLMProviderResult(
                response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))], usage={}),
                attempts=1,
            )

    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "unit/model")
    monkeypatch.delenv("COGEV_LLM_MAX_TOKENS", raising=False)
    monkeypatch.setenv("COGEV_LLM_LONG_MAX_TOKENS", "20000")
    monkeypatch.setenv("COGEV_LLM_LIGHT_MAX_TOKENS", "3000")

    llm_json("nexus_synthesize_result", {"x": 1}, system="Return JSON", schema_hint={}, provider=Provider(), request_policy=LLMRequestPolicy(long_context=True))
    llm_json("nexus_relative_rank", {"x": 1}, system="Return JSON", schema_hint={}, provider=Provider())

    assert seen == [20000, 3000]


def test_long_output_budget_requires_explicit_request_policy(monkeypatch) -> None:
    monkeypatch.delenv("COGEV_LLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("COGEV_LLM_LONG_MAX_TOKENS", raising=False)
    monkeypatch.delenv("COGEV_LLM_LIGHT_MAX_TOKENS", raising=False)

    assert max_tokens_for_request("nexus_seed_population") == 16384
    assert max_tokens_for_request("anything", LLMRequestPolicy(long_context=True)) == 65536
    assert max_tokens_for_request("anything", LLMRequestPolicy(max_output_tokens=123)) == 123


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_reasoning_effort_accepts_canonical_provider_values(effort: str) -> None:
    assert normalize_reasoning_effort(effort) == effort


def test_reasoning_effort_empty_is_unconfigured_and_invalid_fails_closed(monkeypatch) -> None:
    assert normalize_reasoning_effort("") is None
    with pytest.raises(LLMConfigurationError):
        normalize_reasoning_effort("ultra")
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "example-reasoning-model")
    monkeypatch.setenv("COGEV_LLM_REASONING_EFFORT", "turbo")

    with pytest.raises(LLMConfigurationError, match="COGEV_LLM_REASONING_EFFORT"):
        require_llm_config()


def test_model_spec_reasoning_effort_roundtrip_identity_and_idempotency() -> None:
    spec = LLMModelSpec.from_dict(
        {
            "profile_id": "reasoning-high",
            "provider": "direct_http",
            "model": "example-reasoning-model",
            "reasoning_effort": "HIGH",
        }
    )
    assert spec is not None
    assert spec.to_dict()["reasoning_effort"] == "high"
    status = spec.apply_to_status({"reasoning_effort": "medium"})
    identity = identity_from_status(status, request_type="nexus_seed_population")

    assert identity.reasoning_effort == "high"
    assert identity.to_dict()["reasoning_effort"] == "high"
    assert llm_public_status(status)["reasoning_effort"] == "high"
    common = {"provider": "direct_http:example-reasoning-model", "model": "example-reasoning-model", "prompt": {"x": 1}, "schema": {}}
    assert llm_idempotency_key(**common, reasoning_effort="high") != llm_idempotency_key(**common, reasoning_effort="medium")

    with pytest.raises(LLMConfigurationError, match="LLMModelSpec.reasoning_effort"):
        LLMModelSpec(model="example-reasoning-model", reasoning_effort="turbo")


def test_model_spec_override_reaches_transport_identity_journal_and_public_status(tmp_path, monkeypatch) -> None:
    class Provider:
        provider_id = "capture"

        def __init__(self) -> None:
            self.kwargs = {}

        def complete_json(self, **kwargs):  # noqa: ANN001
            self.kwargs = dict(kwargs)
            return LLMProviderResult(
                response=SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                ),
                attempts=1,
            )

    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "example-reasoning-model")
    monkeypatch.setenv("COGEV_LLM_REASONING_EFFORT", "medium")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")
    monkeypatch.delenv("COGEV_LLM_BUDGET_USD", raising=False)
    provider = Provider()
    spec = LLMModelSpec(profile_id="reasoning-high-test", model="example-reasoning-model", reasoning_effort="high")

    with llm_session(LLMSession(journal_dir=str(tmp_path))) as session:
        llm_json("reasoning_effort_test", {"x": 1}, system="Return JSON", schema_hint={}, provider=provider, model_spec=spec)
        event = session.snapshot()[-1]

    rows = [json.loads(line) for line in (tmp_path / "llm-calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert provider.kwargs["model"] == "example-reasoning-model"
    assert provider.kwargs["reasoning_effort"] == "high"
    assert event["llm_call_identity"]["reasoning_effort"] == "high"
    assert {row["reasoning_effort"] for row in rows} == {"high"}
    assert llm_public_status(spec.apply_to_status(require_llm_config()))["reasoning_effort"] == "high"


def test_unconfigured_reasoning_effort_is_not_sent_to_provider(monkeypatch) -> None:
    class Provider:
        provider_id = "capture"

        def complete_json(self, **kwargs):  # noqa: ANN001
            assert "reasoning_effort" not in kwargs
            return LLMProviderResult(
                response=SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                )
            )

    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "ordinary-model")
    monkeypatch.delenv("COGEV_LLM_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("COGEV_LLM_BUDGET_USD", raising=False)
    assert llm_json("ordinary", {}, system="Return JSON", schema_hint={}, provider=Provider())["ok"] is True


def test_litellm_provider_forwards_reasoning_effort(monkeypatch) -> None:
    captured = {}
    fake_litellm = types.ModuleType("litellm")

    def completion(**kwargs):  # noqa: ANN001
        captured.update(kwargs)
        return SimpleNamespace()

    fake_litellm.completion = completion
    fake_litellm.completion_cost = lambda completion_response: 0.0
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")

    LiteLLMProvider().complete_json(model="example-reasoning-model", messages=[], reasoning_effort="high")

    assert captured["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    ("first_content", "first_finish_reason"),
    [('{"ok":', "length"), ("", "stop")],
)
def test_transport_retries_unusable_litellm_response_with_more_output_and_original_messages(
    monkeypatch,
    first_content: str,
    first_finish_reason: str,
) -> None:
    calls: list[dict[str, object]] = []
    fake_litellm = types.ModuleType("litellm")

    def completion(**kwargs):  # noqa: ANN001
        calls.append(dict(kwargs))
        content = first_content if len(calls) == 1 else '{"ok": true}'
        finish_reason = first_finish_reason if len(calls) == 1 else "stop"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    fake_litellm.completion = completion
    fake_litellm.completion_cost = lambda completion_response: 0.0
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "unit/model")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("COGEV_LLM_RETRY_BASE_SLEEP", "0")
    monkeypatch.setenv("COGEV_LLM_RETRY_MAX_TOKENS", "16384")
    monkeypatch.delenv("COGEV_LLM_BUDGET_USD", raising=False)

    response = llm_json(
        "structured-retry",
        {"x": 1},
        system="Return JSON",
        schema_hint={},
        provider=LiteLLMProvider(),
        request_policy=LLMRequestPolicy(structured_prompt=True, max_output_tokens=4096),
    )

    assert response["ok"] is True
    assert [call["max_tokens"] for call in calls] == [4096, 8192]
    assert calls[0]["messages"] == calls[1]["messages"]
    assert len(calls[0]["messages"]) == 2
