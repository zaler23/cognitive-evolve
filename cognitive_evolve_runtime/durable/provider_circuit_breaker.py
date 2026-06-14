"""Explicit provider circuit breaker; never falls back to fixtures."""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ProviderCircuitState:
    provider: str
    state: str = "closed"  # closed|open|half_open
    failure_count: int = 0
    opened_at: float | None = None
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["provider_unavailable"] = self.state == "open"
        data["no_fixture_fallback"] = True
        return data


class ProviderCircuitBreaker:
    def __init__(self, *, failure_threshold: int | None = None, reset_seconds: float | None = None) -> None:
        self.failure_threshold = max(1, int(failure_threshold or os.environ.get("COGEV_PROVIDER_CIRCUIT_FAILURES", "3") or 3))
        self.reset_seconds = max(0.1, float(reset_seconds or os.environ.get("COGEV_PROVIDER_CIRCUIT_RESET_SECONDS", "60") or 60))
        self._states: dict[str, ProviderCircuitState] = {}

    def state_for(self, provider: str) -> ProviderCircuitState:
        return self._states.setdefault(str(provider or "unknown"), ProviderCircuitState(provider=str(provider or "unknown")))

    def before_call(self, provider: str) -> ProviderCircuitState:
        state = self.state_for(provider)
        if state.state == "open" and state.opened_at is not None and (time.monotonic() - state.opened_at) >= self.reset_seconds:
            state.state = "half_open"
        if state.state == "open":
            raise ProviderUnavailableError(f"provider_unavailable: circuit open for {provider}; fixture fallback is forbidden")
        return state

    def record_success(self, provider: str) -> ProviderCircuitState:
        state = self.state_for(provider)
        state.state = "closed"
        state.failure_count = 0
        state.opened_at = None
        state.last_error = ""
        return state

    def record_failure(self, provider: str, error: BaseException | str) -> ProviderCircuitState:
        state = self.state_for(provider)
        state.failure_count += 1
        state.last_error = str(error)[:500]
        if state.failure_count >= self.failure_threshold:
            state.state = "open"
            state.opened_at = time.monotonic()
        return state

    def snapshot(self) -> dict[str, Any]:
        return {provider: state.to_dict() for provider, state in sorted(self._states.items())}


class ProviderUnavailableError(RuntimeError):
    pass


_DEFAULT_BREAKER = ProviderCircuitBreaker()


def default_provider_circuit_breaker() -> ProviderCircuitBreaker:
    return _DEFAULT_BREAKER


__all__ = ["ProviderCircuitState", "ProviderCircuitBreaker", "ProviderUnavailableError", "default_provider_circuit_breaker"]
