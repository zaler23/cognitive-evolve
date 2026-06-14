from __future__ import annotations

import json
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Iterator

from .env import LLM_MAX_CONCURRENT_ENV, LLM_RPM_ENV, LLM_TPM_ENV, env_int


class ThrottledLLMGovernor:
    """Process-wide concurrency/RPM/TPM guard for upstream LLM calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._semaphore: threading.BoundedSemaphore | None = None
        self._limit: int | None = None
        self._request_times: deque[float] = deque()
        self._token_events: deque[tuple[float, int]] = deque()

    def _max_concurrent(self) -> int:
        return max(1, env_int(LLM_MAX_CONCURRENT_ENV, 3))

    def _rpm(self) -> int:
        return max(0, env_int(LLM_RPM_ENV, 0))

    def _tpm(self) -> int:
        return max(0, env_int(LLM_TPM_ENV, 0))

    def _semaphore_for_current_env(self) -> threading.BoundedSemaphore:
        limit = self._max_concurrent()
        with self._lock:
            if self._semaphore is None or self._limit != limit:
                self._semaphore = threading.BoundedSemaphore(limit)
                self._limit = limit
            return self._semaphore

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            self._prune_locked(now)
            return {
                "max_concurrent_env": LLM_MAX_CONCURRENT_ENV,
                "max_concurrent": self._max_concurrent(),
                "rpm_env": LLM_RPM_ENV,
                "rpm": self._rpm(),
                "tpm_env": LLM_TPM_ENV,
                "tpm": self._tpm(),
                "open_request_window_count": len(self._request_times),
                "open_token_window_estimate": sum(tokens for _, tokens in self._token_events),
                "retry_after_supported": True,
                "jittered_exponential_backoff": True,
            }

    def _prune_locked(self, now: float) -> None:
        cutoff = now - 60.0
        while self._request_times and self._request_times[0] <= cutoff:
            self._request_times.popleft()
        while self._token_events and self._token_events[0][0] <= cutoff:
            self._token_events.popleft()

    def _wait_for_rate_window(self, estimated_tokens: int) -> None:
        estimated_tokens = max(1, int(estimated_tokens or 1))
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune_locked(now)
                waits: list[float] = []
                rpm = self._rpm()
                if rpm and len(self._request_times) >= rpm:
                    waits.append(max(0.05, 60.0 - (now - self._request_times[0])))
                tpm = self._tpm()
                used_tokens = sum(tokens for _, tokens in self._token_events)
                if tpm and used_tokens + estimated_tokens > tpm and self._token_events:
                    waits.append(max(0.05, 60.0 - (now - self._token_events[0][0])))
                if not waits:
                    self._request_times.append(now)
                    if tpm:
                        self._token_events.append((now, estimated_tokens))
                    return
                sleep_for = min(waits)
            time.sleep(min(sleep_for, 5.0))

    @contextmanager
    def acquire(self, *, estimated_tokens: int = 1) -> Iterator[dict[str, Any]]:
        semaphore = self._semaphore_for_current_env()
        semaphore.acquire()
        try:
            self._wait_for_rate_window(estimated_tokens)
            yield self.status()
        finally:
            semaphore.release()


def estimate_request_tokens(kwargs: dict[str, Any]) -> int:
    messages = kwargs.get("messages") if isinstance(kwargs.get("messages"), list) else []
    text = json.dumps(messages, ensure_ascii=False, default=str)
    prompt_estimate = max(1, len(text) // 4)
    try:
        max_tokens = int(kwargs.get("max_tokens") or 0)
    except (TypeError, ValueError):
        max_tokens = 0
    return max(1, prompt_estimate + max(0, max_tokens))


_LLM_GOVERNOR = ThrottledLLMGovernor()


def llm_governor() -> ThrottledLLMGovernor:
    return _LLM_GOVERNOR


def llm_governor_status() -> dict[str, Any]:
    return _LLM_GOVERNOR.status()
