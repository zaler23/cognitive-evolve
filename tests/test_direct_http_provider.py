from __future__ import annotations

import json
import socket
from types import SimpleNamespace
import urllib.request

import pytest

from cognitive_evolve_runtime.llm.http_provider import DirectHTTPProvider, DirectHTTPProviderError, normalize_direct_http_model
from cognitive_evolve_runtime.llm.retry import provider_error_category
from cognitive_evolve_runtime.llm.provider_interface import LLMProviderResult
from cognitive_evolve_runtime.llm.request_policy import LLMRequestPolicy
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
        response_format={"type": "json_object"},
        timeout=10,
    )

    assert result.response.choices[0].message.content == '{"ok": true}'
    assert result.response.usage["total_tokens"] == 3
    assert result.attempts == 1
    assert calls[0]["url"] == "http://localhost:8081/v1/chat/completions"
    assert calls[0]["payload"]["model"] == "example-reasoning-model"
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


def test_direct_http_normalizes_socket_timeout_with_url(monkeypatch) -> None:
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

    assert "Timeout from direct_http provider at http://localhost:8081/v1/chat/completions" in str(exc_info.value)
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

    assert max_tokens_for_request("nexus_seed_population") == 4096
    assert max_tokens_for_request("anything", LLMRequestPolicy(long_context=True)) == 32768
    assert max_tokens_for_request("anything", LLMRequestPolicy(max_output_tokens=123)) == 123
