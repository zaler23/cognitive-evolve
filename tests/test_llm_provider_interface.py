from __future__ import annotations

from cognitive_evolve_runtime.llm.provider_interface import LLMProviderResult
from cognitive_evolve_runtime.llm.mock_provider import MockLLMProvider


def test_mock_provider_exposes_provider_interface() -> None:
    provider = MockLLMProvider([{"answer": "ok"}])
    result = provider.complete_json(model="fixture", messages=[], temperature=0, max_tokens=10, response_format={"type": "json_object"}, timeout=1)

    assert isinstance(result, LLMProviderResult)
    assert result.attempts == 1
    assert result.estimated_cost_usd == 0.0
    assert result.response.choices[0].message.content == '{"answer": "ok"}'
    assert provider.calls[0]["model"] == "fixture"
