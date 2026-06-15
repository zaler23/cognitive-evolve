"""Small in-process adapter registry for progressive evidence."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.evaluators.adapters import GenericDomainAdapter

_ADAPTERS: dict[str, Any] = {"general": GenericDomainAdapter()}


def register_adapter(domain_id: str, adapter: Any) -> None:
    key = str(domain_id or "general").strip() or "general"
    _ADAPTERS[key] = adapter


def get_adapter(domain_id: str | None = None) -> Any:
    return _ADAPTERS.get(str(domain_id or "general"), _ADAPTERS["general"])


def registry_snapshot() -> dict[str, str]:
    return {key: type(value).__name__ for key, value in sorted(_ADAPTERS.items())}


__all__ = ["get_adapter", "register_adapter", "registry_snapshot"]
