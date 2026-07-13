"""Open runtime option persistence for Nexus runs.

This is intentionally a small namespaced dict, not a run-manifest platform.
It records effective options so resume can preserve semantics without the
runtime owning every future component field.
"""
from __future__ import annotations

import os
from typing import Any, Mapping

from cognitive_evolve_runtime.core.serialization import json_ready


def resolve_runtime_options(
    *,
    request_options: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
    registered_components: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    env = environment or os.environ
    options = _json_dict(request_options)
    sources = _json_dict(options.pop("_sources", {}))
    _set_default(options, sources, "verification.backend", "project-default", "default")
    _set_default(options, sources, "context.provider", "repository-context", "default")
    _set_default(options, sources, "scheduler.policy", "fabric-default", "default")
    _set_default(options, sources, "seed.family_priority_source", "model_authored_search_space", "default")
    if "COGEV_VERIFY_INCLUDE_TESTS" in env and "verification.include_tests" not in options:
        options["verification.include_tests"] = _env_bool(env.get("COGEV_VERIFY_INCLUDE_TESTS"))
        sources["verification.include_tests"] = "environment:COGEV_VERIFY_INCLUDE_TESTS"
    if registered_components:
        for identity, component in registered_components.items():
            defaults = getattr(component, "runtime_option_defaults", None)
            if callable(defaults):
                for key, value in _json_dict(defaults()).items():
                    _set_default(options, sources, key, value, f"component:{identity}")
    if sources:
        options["_sources"] = sources
    return _json_dict(json_ready(options))


def restore_runtime_options(*, persisted: Mapping[str, Any] | None = None, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    options = _json_dict(persisted)
    sources = _json_dict(options.get("_sources"))
    for key, value in _json_dict(overrides).items():
        if key == "_sources":
            continue
        options[key] = value
        sources[key] = "resume_override"
    if sources:
        options["_sources"] = sources
    return _json_dict(json_ready(options))


def option_bool(options: Mapping[str, Any] | None, key: str, *, default: bool = False) -> bool:
    if not isinstance(options, Mapping) or key not in options:
        return default
    value = options.get(key)
    if isinstance(value, bool):
        return value
    return _env_bool(str(value))


def _set_default(options: dict[str, Any], sources: dict[str, Any], key: str, value: Any, source: str) -> None:
    if key not in options:
        options[key] = value
        sources.setdefault(key, source)


def _json_dict(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _env_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "y"}


__all__ = ["option_bool", "resolve_runtime_options", "restore_runtime_options"]
