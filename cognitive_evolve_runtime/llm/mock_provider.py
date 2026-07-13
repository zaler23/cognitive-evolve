"""Deterministic provider for tests and offline adapter checks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .provider_interface import LLMProviderResult


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


class MockProviderResponse:
    def __init__(self, payload: dict[str, Any] | str) -> None:
        content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        self.choices = [_Choice(_Message(content))]


class MockLLMProvider:
    provider_id = "mock"

    def __init__(self, responses: list[dict[str, Any] | str] | None = None) -> None:
        self.responses = list(responses or [{"ok": True, "provider": "mock"}])
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> LLMProviderResult:
        self.calls.append(dict(kwargs))
        payload = self.responses.pop(0) if self.responses else {"ok": True, "provider": "mock"}
        return LLMProviderResult(response=MockProviderResponse(payload), attempts=1, estimated_cost_usd=0.0)


__all__ = ["MockLLMProvider", "MockProviderResponse"]
