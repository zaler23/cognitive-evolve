"""Stable hashing and idempotency helpers for durable runs."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.nexus._serde import stable_hash, stable_json


def canonical_json(value: Any) -> str:
    return stable_json(value)


def idempotency_key(*parts: Any, prefix: str = "idem") -> str:
    return f"{prefix}-{stable_hash(parts)}"


def llm_idempotency_key(
    *,
    provider: str,
    model: str | None,
    prompt: Any,
    schema: Any,
    temperature: Any = None,
    seed: Any = None,
    contract_version: str | None = None,
) -> str:
    return idempotency_key(
        {
            "provider": provider,
            "model": model or "",
            "prompt": prompt,
            "schema": schema,
            "temperature": temperature,
            "seed": seed,
            "contract_version": contract_version or "unknown",
        },
        prefix="llm",
    )


__all__ = ["canonical_json", "stable_hash", "idempotency_key", "llm_idempotency_key"]
