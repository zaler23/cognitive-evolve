from __future__ import annotations

from types import SimpleNamespace

from cognitive_evolve_runtime.llm.provider_interface import LLMProviderResult
from cognitive_evolve_runtime.llm.env import LLMResponseError
from cognitive_evolve_runtime.llm.json_tools import extract_json_from_text
from cognitive_evolve_runtime.llm.transport import llm_json
import pytest


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def test_llm_json_uses_single_five_attempt_default_budget_for_parse_repair(monkeypatch) -> None:
    class Provider:
        provider_id = "test"

        def __init__(self) -> None:
            self.calls = 0

        def complete_json(self, **kwargs):  # noqa: ANN001
            self.calls += 1
            if self.calls < 5:
                return LLMProviderResult(response=_response(""))
            return LLMProviderResult(response=_response('{"ok": true}'))

    provider = Provider()
    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "test/model")
    monkeypatch.delenv("COGEV_LLM_RETRY_ATTEMPTS", raising=False)
    monkeypatch.delenv("COGEV_LLM_JSON_RETRY_ATTEMPTS", raising=False)

    response = llm_json("unit_test", {"x": 1}, system="Return JSON", schema_hint={}, provider=provider)

    assert response["ok"] is True
    assert provider.calls == 5


def test_extract_json_from_text_accepts_embedded_fenced_json() -> None:
    text = "Model note before JSON.\n```json\n{\"ok\": true, \"mode\": \"embedded\"}\n```\ntrailing note"

    assert extract_json_from_text(text) == {"ok": True, "mode": "embedded"}


def test_extract_json_from_text_accepts_prose_wrapped_object() -> None:
    text = "Here is the result: {\"ok\": true, \"value\": 3} and nothing else."

    assert extract_json_from_text(text) == {"ok": True, "value": 3}


def test_extract_json_from_text_rejects_refusal_with_hint() -> None:
    with pytest.raises(LLMResponseError) as exc:
        extract_json_from_text("I cannot fulfill this request.")

    assert "parse_hint=refusal_or_empty" in str(exc.value)


def test_extract_json_from_text_still_rejects_non_object_json() -> None:
    with pytest.raises(LLMResponseError, match="must be a JSON object"):
        extract_json_from_text("```json\n[1, 2, 3]\n```")
