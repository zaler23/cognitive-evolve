"""Provider boundary for JSON LLM calls.

The runtime depends on this small interface, not on any concrete SDK.  Concrete
providers may wrap LiteLLM, direct vendor SDKs, or deterministic test doubles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class LLMProviderResult:
    """Raw provider response plus transport metadata."""

    response: Any
    attempts: int = 1
    estimated_cost_usd: float | None = None


class LLMProviderInterface(Protocol):
    """Minimal completion interface used by ``llm.transport``."""

    provider_id: str

    def complete_json(self, **kwargs: Any) -> LLMProviderResult:
        """Return a provider response expected to contain JSON text."""


__all__ = ["LLMProviderInterface", "LLMProviderResult"]
