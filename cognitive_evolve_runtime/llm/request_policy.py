"""Provider-call policy chosen by callers, not inferred from business names."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMRequestPolicy:
    max_output_tokens: int | None = None
    timeout_seconds: float | None = None
    retry_attempts: int | None = None
    long_context: bool = False
    max_prompt_chars: int | None = None
    structured_prompt: bool = False
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None


__all__ = ["LLMRequestPolicy"]
