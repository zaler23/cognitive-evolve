"""Concurrency wrapper for local verification tasks."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable

VERIFY_CONCURRENCY_ENV = "COGEV_VERIFY_CONCURRENCY"


@dataclass(frozen=True)
class VerificationExecutorConfig:
    mode: str = "local"
    max_workers: int = 4


def config_from_env() -> VerificationExecutorConfig:
    mode = str(os.environ.get(VERIFY_CONCURRENCY_ENV) or "local").strip().lower()
    if mode in {"serial", "none", "0"}:
        return VerificationExecutorConfig(mode="serial", max_workers=1)
    if mode in {"threaded_llm", "llm"}:
        return VerificationExecutorConfig(mode="threaded_llm", max_workers=1)
    if mode in {"threaded_toolrunner", "toolrunner"}:
        return VerificationExecutorConfig(mode="threaded_toolrunner", max_workers=_workers(default=2, maximum=4))
    return VerificationExecutorConfig(mode="threaded_local", max_workers=_workers(default=4, maximum=8))


class VerificationExecutor:
    def __init__(self, config: VerificationExecutorConfig | None = None) -> None:
        self.config = config or config_from_env()

    def map(self, fn: Callable[[Any], Any], items: Iterable[Any]) -> list[Any]:
        values = list(items)
        if self.config.mode == "serial" or self.config.max_workers <= 1 or len(values) <= 1:
            return [fn(item) for item in values]
        results: list[Any] = [None] * len(values)
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = {pool.submit(fn, item): index for index, item in enumerate(values)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results


def _workers(*, default: int, maximum: int) -> int:
    try:
        raw = int(os.environ.get("COGEV_VERIFY_MAX_WORKERS") or default)
    except (TypeError, ValueError):
        raw = default
    return max(1, min(maximum, raw))


__all__ = ["VERIFY_CONCURRENCY_ENV", "VerificationExecutor", "VerificationExecutorConfig", "config_from_env"]
