#!/usr/bin/env python3
"""Frontend-facing API-key validation for the v1 service."""
from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Request, status

from .config import get_service_config


def _bearer_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    prefix = "Bearer "
    if authorization.startswith(prefix):
        return authorization[len(prefix):].strip()
    return authorization.strip()


async def require_service_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    config = get_service_config()
    if not config.require_auth:
        return
    if not config.api_keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="COGEV_SERVER_REQUIRE_AUTH=true but no COGEV_SERVER_API_KEY(S) is configured.",
        )
    supplied = x_api_key or _bearer_token(authorization)
    if _matches_service_api_key(supplied, config.api_keys):
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing CognitiveEvolve service API key.")


def _matches_service_api_key(supplied: str, valid_keys: tuple[str, ...]) -> bool:
    supplied = str(supplied or "")
    matched = False
    for key in valid_keys:
        candidate = str(key or "")
        # Evaluate every key so membership does not short-circuit on the first
        # matching prefix/value.  Length still affects compare_digest internals,
        # but the auth path no longer uses Python's early-exit membership test.
        matched = hmac.compare_digest(supplied, candidate) or matched
    return matched


__all__ = ["require_service_api_key", "_matches_service_api_key"]
