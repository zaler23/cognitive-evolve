"""Request-size and rate-limit guards for the OpenAI-compatible API."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from typing import Any, Awaitable, Callable

from .config import get_service_config

ASGIApp = Callable[[dict[str, Any], Callable[[], Awaitable[dict[str, Any]]], Callable[[dict[str, Any]], Awaitable[None]]], Awaitable[None]]

_RATE_LOCK = threading.Lock()
_RATE_WINDOWS: dict[str, deque[float]] = {}


class APIGuardMiddleware:
    """Small ASGI guard for API resource boundaries.

    FastAPI's request model validation happens after the body has been read, so
    this middleware enforces a bounded body before handing the replayable body
    to route handlers.  It also provides a per service-key/client sliding
    window.  The guard is intentionally local and dependency-free; operators can
    still put a stronger reverse proxy in front of public deployments.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path") or "")
        if not path.startswith("/v1/"):
            await self.app(scope, receive, send)
            return

        config = get_service_config()
        key = _rate_limit_key(scope)
        if config.rate_limit_per_minute and not _allow_request(key, config.rate_limit_per_minute):
            await _json_response(send, 429, {"error": {"message": "CognitiveEvolve API rate limit exceeded.", "type": "rate_limit_exceeded"}})
            return

        if _declared_content_length(scope) > config.max_request_bytes:
            await _json_response(
                send,
                413,
                {"error": {"message": f"Request body exceeds {config.max_request_bytes} bytes.", "type": "request_body_too_large"}},
            )
            return

        method = str(scope.get("method") or "GET").upper()
        if method not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        try:
            buffered = await _buffer_request_body(receive, max_bytes=config.max_request_bytes)
        except RequestBodyTooLarge:
            await _json_response(
                send,
                413,
                {"error": {"message": f"Request body exceeds {config.max_request_bytes} bytes.", "type": "request_body_too_large"}},
            )
            return

        index = 0

        async def replay_receive() -> dict[str, Any]:
            nonlocal index
            if index < len(buffered):
                message = buffered[index]
                index += 1
                return message
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay_receive, send)


class RequestBodyTooLarge(ValueError):
    pass


async def _buffer_request_body(receive: Callable[[], Awaitable[dict[str, Any]]], *, max_bytes: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    total = 0
    while True:
        message = await receive()
        messages.append(dict(message))
        if message.get("type") != "http.request":
            break
        body = message.get("body") or b""
        total += len(body)
        if total > max_bytes:
            raise RequestBodyTooLarge(str(total))
        if not message.get("more_body", False):
            break
    return messages


def _declared_content_length(scope: dict[str, Any]) -> int:
    for key, value in scope.get("headers") or []:
        if bytes(key).lower() == b"content-length":
            try:
                return int(bytes(value).decode("ascii"))
            except (TypeError, ValueError, UnicodeDecodeError):
                return 0
    return 0


def _rate_limit_key(scope: dict[str, Any]) -> str:
    auth = ""
    api_key = ""
    for key, value in scope.get("headers") or []:
        lowered = bytes(key).lower()
        if lowered == b"authorization":
            auth = bytes(value).decode("utf-8", errors="ignore")
        elif lowered == b"x-api-key":
            api_key = bytes(value).decode("utf-8", errors="ignore")
    if api_key:
        return "key:" + api_key[-16:]
    if auth:
        return "auth:" + auth[-16:]
    client = scope.get("client") or ("unknown", 0)
    return "client:" + str(client[0])


def _allow_request(key: str, limit: int) -> bool:
    now = time.monotonic()
    cutoff = now - 60.0
    with _RATE_LOCK:
        bucket = _RATE_WINDOWS.setdefault(key, deque())
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True


async def _json_response(send: Callable[[dict[str, Any]], Awaitable[None]], status_code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode("ascii"))],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


__all__ = ["APIGuardMiddleware", "RequestBodyTooLarge"]
