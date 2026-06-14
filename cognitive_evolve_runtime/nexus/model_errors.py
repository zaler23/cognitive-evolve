"""Model-boundary error classification helpers.

The runtime treats ordinary model/schema drift as recoverable degradation in a
few non-critical stages, but quota and rate-limit failures must become a clean
pause.  Falling back and continuing to call the provider after a 429/RESOURCE
EXHAUSTED response wastes quota and obscures run state.
"""
from __future__ import annotations

from typing import Any


QUOTA_ERROR_TOKENS = (
    "resource_exhausted",
    "resource exhausted",
    "429",
    "quota",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "insufficient_quota",
    "provider quota",
)


def is_quota_error(exc: BaseException | Any) -> bool:
    """Return true when an exception represents provider quota/rate exhaustion."""

    text = " ".join(
        part
        for part in [
            exc.__class__.__name__ if isinstance(exc, BaseException) else type(exc).__name__,
            str(exc),
            repr(getattr(exc, "status_code", "")),
            repr(getattr(exc, "code", "")),
        ]
        if part
    ).lower()
    return any(token in text for token in QUOTA_ERROR_TOKENS)


__all__ = ["QUOTA_ERROR_TOKENS", "is_quota_error"]
