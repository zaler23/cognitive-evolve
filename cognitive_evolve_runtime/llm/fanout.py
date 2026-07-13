"""Bounded model-call fan-out helpers.

The LLM governor remains the hard provider-facing concurrency/rate boundary.
This module only decides how many independent model batches the Nexus runtime
is allowed to submit concurrently before each individual call enters the
provider governor.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from typing import Any, Callable, Iterable, TypeVar

from .env import env_int
from .governor import llm_governor

MODEL_FANOUT_CONCURRENCY_ENV = "COGEV_MODEL_FANOUT_CONCURRENCY"

T = TypeVar("T")
R = TypeVar("R")


def model_fanout_workers(limit: int | None = None) -> int:
    """Return the bounded model fan-out worker count for this process.

    Operators can set ``COGEV_MODEL_FANOUT_CONCURRENCY=1`` to force a
    deterministic serial model-call path.  Without an explicit override, the
    value follows ``COGEV_LLM_MAX_CONCURRENT`` through the shared governor so
    runtime fan-out cannot exceed the provider concurrency budget.
    """

    configured = env_int(MODEL_FANOUT_CONCURRENCY_ENV, llm_governor()._max_concurrent())
    workers = max(1, configured)
    if limit is not None:
        workers = min(workers, max(1, int(limit or 1)))
    return workers


def model_fanout_enabled(limit: int | None = None) -> bool:
    return model_fanout_workers(limit) > 1


def run_ordered_fanout(
    items: Iterable[T],
    fn: Callable[[T], R],
    *,
    max_workers: int | None = None,
    thread_name_prefix: str = "cogev-model-fanout",
) -> list[R]:
    """Run independent items with bounded fan-out and input-order results."""

    values = list(items)
    if not values:
        return []
    workers = model_fanout_workers(len(values)) if max_workers is None else min(max(1, int(max_workers or 1)), len(values))
    if workers <= 1 or len(values) <= 1:
        return [fn(item) for item in values]
    results: list[Any] = [None] * len(values)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=thread_name_prefix) as pool:
        future_to_idx = {pool.submit(copy_context().run, fn, item): idx for idx, item in enumerate(values)}
        for fut in as_completed(future_to_idx):
            results[future_to_idx[fut]] = fut.result()
    return results


__all__ = [
    "MODEL_FANOUT_CONCURRENCY_ENV",
    "model_fanout_enabled",
    "model_fanout_workers",
    "run_ordered_fanout",
]
