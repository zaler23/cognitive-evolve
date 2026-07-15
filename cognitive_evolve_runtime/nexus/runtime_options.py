"""Open runtime option persistence for Nexus runs.

This is intentionally a small namespaced dict, not a run-manifest platform.
It records effective options so resume can preserve semantics without the
runtime owning every future component field.
"""
from __future__ import annotations

import json
import os
from typing import Any, Mapping

from cognitive_evolve_runtime.core.serialization import json_ready

OFFSPRING_PARALLEL_MODES = frozenset({"slot", "single_batch"})
PERSISTENCE_MODES = frozenset({"async_full", "sync_full"})
DEFAULT_SLOT_SAMPLING_PROFILES = {
    "explore": {
        "default": [
            {"temperature": 0.8, "top_p": 0.9, "seed": 1001},
            {"temperature": 0.95, "top_p": 0.98, "seed": 1002},
        ],
        "explore_fresh": [
            {"temperature": 0.9, "top_p": 0.95, "seed": 1101},
            {"temperature": 1.0, "top_p": 0.98, "seed": 1102},
        ],
        "standard_variation": [
            {"temperature": 0.8, "top_p": 0.9, "seed": 1201},
            {"temperature": 0.7, "top_p": 0.85, "seed": 1202},
        ],
        "exploit_deepen": [
            {"temperature": 0.55, "top_p": 0.8, "seed": 1301},
            {"temperature": 0.4, "top_p": 0.7, "seed": 1302},
        ],
    },
    "exit_sweep": {
        "default": [{"temperature": 0.15, "top_p": 0.45, "seed": 2001}],
        "explore_fresh": [{"temperature": 0.2, "top_p": 0.5, "seed": 2101}],
        "standard_variation": [{"temperature": 0.15, "top_p": 0.45, "seed": 2102}],
        "exploit_deepen": [{"temperature": 0.1, "top_p": 0.4, "seed": 2201}],
    },
}


def resolve_runtime_options(
    *,
    request_options: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
    registered_components: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    env = os.environ if environment is None else environment
    options = _json_dict(request_options)
    sources = _json_dict(options.pop("_sources", {}))
    _set_default(options, sources, "verification.backend", "project-default", "default")
    _set_default(options, sources, "context.provider", "repository-context", "default")
    _set_default(options, sources, "scheduler.policy", "fabric-default", "default")
    _set_default(options, sources, "seed.family_priority_source", "model_authored_search_space", "default")
    _set_default(
        options,
        sources,
        "search.offspring_parallel_mode",
        str(env.get("COGEV_OFFSPRING_PARALLEL_MODE") or "slot").strip().lower(),
        "environment:COGEV_OFFSPRING_PARALLEL_MODE" if "COGEV_OFFSPRING_PARALLEL_MODE" in env else "default",
    )
    sampling_profiles = DEFAULT_SLOT_SAMPLING_PROFILES
    sampling_source = "default"
    if "search.slot_sampling_profiles" not in options and "COGEV_SLOT_SAMPLING_PROFILES" in env:
        try:
            sampling_profiles = json.loads(str(env["COGEV_SLOT_SAMPLING_PROFILES"]))
        except json.JSONDecodeError as exc:
            raise ValueError("COGEV_SLOT_SAMPLING_PROFILES must be valid JSON") from exc
        sampling_source = "environment:COGEV_SLOT_SAMPLING_PROFILES"
    _set_default(options, sources, "search.slot_sampling_profiles", sampling_profiles, sampling_source)
    truncation_threshold = float(env.get("COGEV_SINGLE_BATCH_TRUNCATION_RATE_THRESHOLD") or 0.05)
    _set_default(
        options,
        sources,
        "search.single_batch_truncation_rate_threshold",
        truncation_threshold,
        "environment:COGEV_SINGLE_BATCH_TRUNCATION_RATE_THRESHOLD"
        if "COGEV_SINGLE_BATCH_TRUNCATION_RATE_THRESHOLD" in env
        else "default",
    )
    _set_default(
        options,
        sources,
        "persistence.mode",
        str(env.get("COGEV_PERSISTENCE_MODE") or "async_full").strip().lower(),
        "environment:COGEV_PERSISTENCE_MODE" if "COGEV_PERSISTENCE_MODE" in env else "default",
    )
    _validate_choice(options, "search.offspring_parallel_mode", OFFSPRING_PARALLEL_MODES)
    _validate_choice(options, "persistence.mode", PERSISTENCE_MODES)
    _validate_slot_sampling_profiles(options.get("search.slot_sampling_profiles"))
    threshold = float(options.get("search.single_batch_truncation_rate_threshold", 0.05))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("search.single_batch_truncation_rate_threshold must be between 0 and 1")
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


def _validate_choice(options: Mapping[str, Any], key: str, allowed: frozenset[str]) -> None:
    value = str(options.get(key) or "").strip().lower()
    if value not in allowed:
        raise ValueError(f"{key} must be one of: {', '.join(sorted(allowed))}")


def _json_dict(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _validate_slot_sampling_profiles(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"explore", "exit_sweep"}:
        raise ValueError("search.slot_sampling_profiles must define explore and exit_sweep objects")
    for phase, intents in value.items():
        if not isinstance(intents, Mapping) or not intents:
            raise ValueError(f"search.slot_sampling_profiles.{phase} must be a non-empty object")
        for intent, profiles in intents.items():
            if not isinstance(profiles, list) or not profiles:
                raise ValueError(f"search.slot_sampling_profiles.{phase}.{intent} must be a non-empty list")
            for profile in profiles:
                if not isinstance(profile, Mapping):
                    raise ValueError(f"search.slot_sampling_profiles.{phase}.{intent} entries must be objects")
                if profile.get("temperature") is not None and not 0.0 <= float(profile["temperature"]) <= 2.0:
                    raise ValueError(f"search.slot_sampling_profiles.{phase}.{intent} temperature must be between 0 and 2")
                if profile.get("top_p") is not None and not 0.0 < float(profile["top_p"]) <= 1.0:
                    raise ValueError(f"search.slot_sampling_profiles.{phase}.{intent} top_p must be greater than 0 and at most 1")
                if profile.get("seed") is not None and (isinstance(profile["seed"], bool) or not isinstance(profile["seed"], int)):
                    raise ValueError(f"search.slot_sampling_profiles.{phase}.{intent} seed must be an integer")


def _env_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "y"}


__all__ = ["DEFAULT_SLOT_SAMPLING_PROFILES", "option_bool", "resolve_runtime_options", "restore_runtime_options"]
