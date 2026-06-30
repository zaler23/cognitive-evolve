#!/usr/bin/env python3
"""Runtime configuration for the CognitiveEvolve OpenAI-compatible service.

The service has two separate credential domains:

1. Upstream model credentials used by LiteLLM / the configured model provider.
2. Frontend-facing service credentials accepted by /v1/* endpoints.

This prevents Cherry Studio, OpenWebUI, Continue, Roo Code, or any other
OpenAI-compatible client from receiving the upstream model provider key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..configuration import hermetic_test_enabled, load_layered_config
from ..llm.env import looks_like_placeholder_secret

try:  # pragma: no cover - import availability is covered by package metadata/tests
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(dotenv_path: Any = None, *, override: bool = False, **_: Any) -> bool:
        path = Path(dotenv_path or ".env").expanduser()
        if not path.exists():
            return False
        loaded = False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and (override or key not in os.environ):
                os.environ[key] = value
                loaded = True
        return loaded

DEFAULT_MODELS = [
    "cognitive-evolve-one-shot",
    "cognitive-evolve-one-shot-deep",
    "cognitive-evolve-one-shot-ultra",
    "cognitive-evolve-one-shot-exhaustive",
]

PLACEHOLDER_SERVICE_API_KEYS = {
    "ce-local-dev-key-change-me",
    "change-me",
    "replace-with-service-api-key",
    "replace-with-your-service-api-key",
}

MODEL_ROUND_CAPS = {
    # 0 means adaptive: the Nexus budget/profile sets a safety checkpoint, and
    # answer-first completion can return candidate output without self-certification. Operators may override
    # per model with COGEV_MODEL_ROUND_CAP_<MODEL>=N for an explicit hard cap.
    "cognitive-evolve-one-shot": "0",
    "cognitive-evolve-one-shot-deep": "0",
    "cognitive-evolve-one-shot-ultra": "0",
    "cognitive-evolve-one-shot-exhaustive": "0",
}


MODEL_EVOLUTION_PROFILES = {
    "cognitive-evolve-one-shot": "balanced",
    "cognitive-evolve-one-shot-deep": "deep",
    "cognitive-evolve-one-shot-ultra": "ultra",
    "cognitive-evolve-one-shot-exhaustive": "exhaustive",
}


def _runtime_root() -> Path:
    return Path(os.environ.get("COGEV_RUNTIME_ROOT", Path.home() / ".cognitive-evolve")).expanduser()


def load_service_env() -> Path | None:
    """Load .env and optional .cogev/config.yaml defaults once."""
    loaded_env: Path | None = None
    explicit = os.environ.get("COGEV_ENV_FILE", "").strip()
    if explicit:
        loaded_env = Path(explicit).expanduser().resolve()
        load_dotenv(loaded_env, override=False)
        load_layered_config(override=False)
        return loaded_env
    candidates = [
        _runtime_root() / ".env",
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    if hermetic_test_enabled():
        candidates = [candidate for candidate in candidates if candidate in {Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"}]
    for candidate in candidates:
        if candidate.exists():
            loaded_env = candidate
            load_dotenv(candidate, override=False)
            break
    loaded_config = load_layered_config(override=False)
    return loaded_env or loaded_config


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _runtime_relative_path(raw: str, default: Path) -> Path:
    path = Path(raw).expanduser() if raw.strip() else default
    if path.is_absolute():
        return path
    return _runtime_root() / path


def _cors_origins(default_port: int) -> tuple[str, ...]:
    raw = os.environ.get("COGEV_CORS_ALLOW_ORIGINS", "").strip()
    if raw:
        origins = tuple(_split_csv(raw))
    else:
        origins = (
            "http://127.0.0.1",
            "http://localhost",
            f"http://127.0.0.1:{default_port}",
            f"http://localhost:{default_port}",
        )
    return origins or ("http://127.0.0.1", "http://localhost")


def _cors_credentials(origins: tuple[str, ...]) -> bool:
    requested = _bool_env("COGEV_CORS_ALLOW_CREDENTIALS", True)
    if "*" in origins and requested:
        return False
    return requested


@dataclass(frozen=True)
class ServiceConfig:
    host: str
    port: int
    public_base_url: str
    require_auth: bool
    api_keys: tuple[str, ...]
    models: tuple[str, ...]
    default_model: str
    api_task_root: Path
    allow_insecure_public_bind: bool = False
    cors_allow_origins: tuple[str, ...] = field(default_factory=tuple)
    cors_allow_credentials: bool = True
    max_request_bytes: int = 4 * 1024 * 1024
    rate_limit_per_minute: int = 120
    job_ttl_seconds: int = 24 * 60 * 60
    max_tracked_jobs: int = 1000
    service_name: str = "CognitiveEvolve"
    api_version: str = "v1"

    @property
    def masked_api_keys(self) -> list[str]:
        return [mask_secret(key) for key in self.api_keys]

    @property
    def api_key_precedence(self) -> str:
        return "COGEV_SERVER_API_KEY is prepended before COGEV_SERVER_API_KEYS; duplicate values are accepted once."

    @property
    def auth_warning(self) -> str:
        if self.require_auth:
            return ""
        if self.host not in {"127.0.0.1", "localhost", "::1"}:
            return "authentication_disabled_on_non_loopback_host"
        return "authentication_disabled"

    @property
    def public_bind_without_auth(self) -> bool:
        return not self.require_auth and self.host not in {"127.0.0.1", "localhost", "::1"}

    @property
    def public_bind(self) -> bool:
        return self.host not in {"127.0.0.1", "localhost", "::1", "[::1]"}

    def enforce_safe_to_serve(self) -> None:
        if self.public_bind_without_auth and not self.allow_insecure_public_bind:
            raise RuntimeError(
                "Refusing to serve CognitiveEvolve without authentication on a non-loopback host. "
                "Set COGEV_SERVER_REQUIRE_AUTH=true or explicitly set COGEV_ALLOW_INSECURE_BIND=1 for a local trusted network."
            )
        if self.public_bind and self.require_auth and not self.allow_insecure_public_bind:
            if not self.api_keys:
                raise RuntimeError("Refusing to serve CognitiveEvolve on a non-loopback host without COGEV_SERVER_API_KEY(S).")
            weak = [key for key in self.api_keys if service_api_key_is_placeholder_or_weak(key)]
            if weak:
                raise RuntimeError(
                    "Refusing to serve CognitiveEvolve on a non-loopback host with a placeholder or low-entropy service API key. "
                    "Replace COGEV_SERVER_API_KEY(S) with a private high-entropy value before binding publicly."
                )
            if "*" in self.cors_allow_origins:
                raise RuntimeError("Refusing public CognitiveEvolve service with wildcard CORS origins; set explicit trusted origins.")


def mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return value[:2] + "***"
    return value[:4] + "***" + value[-4:]


def service_api_key_is_placeholder_or_weak(value: str | None) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    lowered = text.lower()
    if lowered in PLACEHOLDER_SERVICE_API_KEYS or looks_like_placeholder_secret(text):
        return True
    # Publicly bound service keys should not be short dictionary-like dev keys.
    # This is deliberately not used to reject loopback development keys.
    if len(text) < 24:
        return True
    classes = sum(
        bool(check(text))
        for check in (
            str.islower,
            str.isupper,
            str.isdigit,
        )
    )
    has_symbol = any(not char.isalnum() for char in text)
    return classes + int(has_symbol) < 2


def get_service_config() -> ServiceConfig:
    load_service_env()
    models = tuple(_split_csv(os.environ.get("COGEV_SERVE_MODELS", "")) or DEFAULT_MODELS)
    default_model = os.environ.get("COGEV_DEFAULT_SERVE_MODEL", models[0]).strip() or models[0]
    keys = _split_csv(os.environ.get("COGEV_SERVER_API_KEYS", ""))
    single_key = os.environ.get("COGEV_SERVER_API_KEY", "").strip()
    if single_key and single_key not in keys:
        keys.insert(0, single_key)
    host = os.environ.get("COGEV_SERVER_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = _int_env("COGEV_SERVER_PORT", 8765)
    cors_allow_origins = _cors_origins(port)
    public_base_url = os.environ.get("COGEV_SERVER_PUBLIC_BASE_URL", f"http://{host}:{port}/v1").strip()
    api_task_root = _runtime_relative_path(
        os.environ.get("COGEV_API_TASK_ROOT", ""),
        _runtime_root() / ".cogev" / "api-runs",
    )
    return ServiceConfig(
        host=host,
        port=port,
        public_base_url=public_base_url,
        require_auth=_bool_env("COGEV_SERVER_REQUIRE_AUTH", True),
        api_keys=tuple(keys),
        models=models,
        default_model=default_model,
        api_task_root=api_task_root,
        allow_insecure_public_bind=_bool_env("COGEV_ALLOW_INSECURE_BIND", False),
        cors_allow_origins=cors_allow_origins,
        cors_allow_credentials=_cors_credentials(cors_allow_origins),
        max_request_bytes=max(1024, _int_env("COGEV_API_MAX_REQUEST_BYTES", 4 * 1024 * 1024)),
        rate_limit_per_minute=max(0, _int_env("COGEV_API_RATE_LIMIT_PER_MINUTE", 120)),
        job_ttl_seconds=max(60, _int_env("COGEV_API_JOB_TTL_SECONDS", 24 * 60 * 60)),
        max_tracked_jobs=max(1, _int_env("COGEV_API_MAX_TRACKED_JOBS", 1000)),
    )


def round_cap_for_model(model: str) -> str | None:
    load_service_env()
    explicit = os.environ.get(f"COGEV_MODEL_ROUND_CAP_{model.replace('-', '_').upper()}", "").strip()
    if explicit:
        return explicit
    return MODEL_ROUND_CAPS.get(model)


def evolution_profile_for_model(model: str) -> str | None:
    load_service_env()
    explicit = os.environ.get(f"COGEV_MODEL_EVOLUTION_PROFILE_{model.replace('-', '_').upper()}", "").strip().lower()
    if explicit:
        return explicit
    return MODEL_EVOLUTION_PROFILES.get(model)


__all__ = [
    "DEFAULT_MODELS",
    "MODEL_ROUND_CAPS",
    "MODEL_EVOLUTION_PROFILES",
    "ServiceConfig",
    "get_service_config",
    "load_service_env",
    "mask_secret",
    "service_api_key_is_placeholder_or_weak",
    "round_cap_for_model",
    "evolution_profile_for_model",
]
