#!/usr/bin/env python3
"""OpenAI-compatible API package for CognitiveEvolve stable runtime.

The package intentionally avoids constructing the FastAPI app on import so CLI
modules and tests do not accidentally load `.env` or mutate LLM provider config.
Import `cognitive_evolve_runtime.api.openai_compat:app` when serving.
"""
from __future__ import annotations

__all__: list[str] = ["JobQueue"]


def __getattr__(name: str) -> object:
    """Lazily expose lightweight API helpers without constructing the app."""

    if name == "JobQueue":
        from .jobs import JobQueue

        return JobQueue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
