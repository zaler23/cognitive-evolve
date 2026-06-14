#!/usr/bin/env python3
"""CLI helpers for serving the CognitiveEvolve OpenAI-compatible API."""
from __future__ import annotations

import json
from typing import Any

from .config import get_service_config, load_service_env, mask_secret


def serve() -> int:
    load_service_env()
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("uvicorn is required to serve CognitiveEvolve API. Install dependencies first.") from exc
    config = get_service_config()
    config.enforce_safe_to_serve()
    uvicorn.run(
        "cognitive_evolve_runtime.api.openai_compat:app",
        host=config.host,
        port=config.port,
        reload=False,
        factory=False,
    )
    return 0


def status() -> dict[str, Any]:
    config = get_service_config()
    return {
        "service": config.service_name,
        "host": config.host,
        "port": config.port,
        "base_url": config.public_base_url,
        "auth_required": config.require_auth,
        "auth_warning": config.auth_warning,
        "allow_insecure_public_bind": config.allow_insecure_public_bind,
        "api_key_precedence": config.api_key_precedence,
        "api_keys": [mask_secret(key) for key in config.api_keys],
        "cors_allow_origins": list(config.cors_allow_origins),
        "cors_allow_credentials": config.cors_allow_credentials,
        "models": list(config.models),
        "task_root": str(config.api_task_root),
        "max_request_bytes": config.max_request_bytes,
        "rate_limit_per_minute": config.rate_limit_per_minute,
        "job_ttl_seconds": config.job_ttl_seconds,
        "max_tracked_jobs": config.max_tracked_jobs,
    }


def status_cli() -> int:
    print(json.dumps(status(), ensure_ascii=False, indent=2))
    return 0


__all__ = ["serve", "status", "status_cli"]
