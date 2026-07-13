from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from ..nexus.request_context import evolution_profile, internal_round_cap
from .config import evolution_profile_for_model, round_cap_for_model


def _evolution_profile_for_model(model: str) -> str:
    return evolution_profile_for_model(model) or "balanced"



@contextmanager
def _temporary_model_runtime(model: str) -> Iterator[None]:
    # Request-local contexts avoid mutating os.environ under concurrent FastAPI
    # requests while preserving model-specific caps/profile semantics.
    with internal_round_cap(round_cap_for_model(model)), evolution_profile(evolution_profile_for_model(model) or _evolution_profile_for_model(model)):
        yield



@contextmanager
def _temporary_round_cap(model: str) -> Iterator[None]:
    # Convenience wrapper for request-local model runtime settings.
    with _temporary_model_runtime(model):
        yield


__all__ = ['_evolution_profile_for_model', '_temporary_model_runtime', '_temporary_round_cap']
