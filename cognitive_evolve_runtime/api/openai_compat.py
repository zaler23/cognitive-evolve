#!/usr/bin/env python3
"""OpenAI-compatible API server for CognitiveEvolve.

The public contract intentionally mirrors the subset used by AI frontends:

- GET /v1/models
- POST /v1/chat/completions

Every completion request is mapped to the canonical EngineOrchestrator pipeline.  The
frontend sends normal chat messages; CognitiveEvolve treats the conversation
as a single seed request, performs internal semantic reconstruction/evolution,
and returns one final answer.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .config import get_service_config, load_service_env
from .executor import QueueFullError, get_job_executor, shutdown_api_executors
from .guards import APIGuardMiddleware
from .security import require_service_api_key
from .jobs import _cancel_job_future, _get_job, _job_public, _now, _pop_job_future, _register_job_future, _set_job, _status_from_nexus_data, _task_dir_for_request
from .models import ChatCompletionRequest, ChatMessage
from .payloads import _completion_payload
from .prompting import build_one_shot_prompt


def _run_engine(prompt: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
    """Lazy API runner wrapper kept patchable for tests."""
    from .engine_runner import _run_engine as run_engine

    return run_engine(prompt, **kwargs)


def _stream_engine_chunks(prompt: str, **kwargs: Any) -> Iterator[bytes]:
    from .streaming import _stream_engine_chunks as stream_engine_chunks

    return stream_engine_chunks(prompt, **kwargs)


def _testclient_compat_patch_enabled() -> bool:
    explicit = os.environ.get("COGEV_ENABLE_TESTCLIENT_COMPAT_PATCH", "").strip().lower()
    return explicit in {"1", "true", "yes", "on"} or "PYTEST_CURRENT_TEST" in os.environ


def _patch_httpx_testclient_compat() -> None:
    """Support Starlette TestClient versions that still pass app= to httpx.

    httpx 0.28 removed the ``app`` keyword from ``Client.__init__`` while some
    FastAPI/Starlette test clients still pass it. This patch is intentionally
    limited to pytest or explicit local smoke runs so production imports do not
    monkey-patch httpx globally.
    """
    if not _testclient_compat_patch_enabled():
        return
    try:
        import inspect
        import httpx
    except Exception:  # pragma: no cover
        return
    init = httpx.Client.__init__
    if "app" in inspect.signature(init).parameters or getattr(init, "_cogev_accepts_app_kw", False):
        return

    def _patched_client_init(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        kwargs.pop("app", None)
        init(self, *args, **kwargs)

    _patched_client_init._cogev_accepts_app_kw = True  # type: ignore[attr-defined]
    httpx.Client.__init__ = _patched_client_init  # type: ignore[assignment]


_patch_httpx_testclient_compat()


def create_app() -> FastAPI:
    load_service_env()
    config = get_service_config()
    config.enforce_safe_to_serve()
    app = FastAPI(title="CognitiveEvolve OpenAI-Compatible API", version="2.0.0", lifespan=_lifespan)
    app.add_middleware(APIGuardMiddleware)
    _install_cors(app, config)
    _install_health_and_model_routes(app)
    _install_job_routes(app)
    _install_chat_routes(app)
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI):
    del app
    try:
        yield
    finally:
        shutdown_api_executors(wait=False, cancel_futures=True)


def _install_cors(app: FastAPI, config: Any) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_allow_origins),
        allow_credentials=config.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _install_health_and_model_routes(app: FastAPI) -> None:
    @app.get("/health")
    async def health() -> dict[str, Any]:
        config = get_service_config()
        from ..llm import llm_status

        status = llm_status()
        upstream_ready = bool(status.get("configured")) and not bool(status.get("api_key_placeholder"))
        return {
            "status": "ok" if upstream_ready else "missing_or_placeholder_upstream_llm_config",
            "service": config.service_name,
            "base_url": config.public_base_url,
            "models": list(config.models),
            "auth_required": config.require_auth,
            "auth_warning": config.auth_warning,
            "configured_service_keys": config.masked_api_keys,
            "llm": {
                "provider": status.get("provider"),
                "model": status.get("model"),
                "configured": status.get("configured"),
                "test_provider_only": status.get("test_provider_only"),
            },
        }

    @app.get("/v1/models", dependencies=[Depends(require_service_api_key)])
    async def models() -> dict[str, Any]:
        config = get_service_config()
        created = _now()
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": created,
                    "owned_by": "cognitive-evolve",
                    "permission": [],
                    "cognitive_evolve": {
                        "runtime_path": "nexus",
                        "completion_semantics": "one request triggers adaptive candidate evolution; safety checkpoints return needs_continuation, not solved",
                        "streaming_semantics": "progress events and safe heartbeats first, final answer chunks after synthesis; not token-by-token model streaming",
                    },
                }
                for model in config.models
            ],
        }


def _install_job_routes(app: FastAPI) -> None:
    @app.post("/v1/cogev/jobs", dependencies=[Depends(require_service_api_key)], response_model=None)
    async def create_job(request: Request) -> JSONResponse:
        raw = await request.json()
        try:
            body = ChatCompletionRequest(**raw)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid job request: {exc}") from exc
        config = get_service_config()
        model = body.model or config.default_model
        if model not in config.models:
            raise HTTPException(status_code=404, detail=f"Unknown CognitiveEvolve model '{model}'. Call /v1/models.")
        prompt = build_one_shot_prompt(body.messages)
        if not prompt:
            raise HTTPException(status_code=400, detail="At least one user message with text content is required.")
        job_id = "job-cogev-" + uuid.uuid4().hex[:24]
        task_dir = _task_dir_for_request(config.api_task_root, job_id)
        _set_job(
            job_id,
            status="queued",
            model=model,
            prompt=prompt,
            task_dir=str(task_dir),
            artifact_root=str(task_dir),
            raw_request=raw,
            error=None,
            cancellation_requested=False,
        )

        def worker() -> None:
            _set_job(job_id, status="running")
            try:
                def cancellation_requested() -> bool:
                    return bool((_get_job(job_id) or {}).get("cancellation_requested"))

                answer, nexus_data = _run_engine(
                    prompt,
                    request_id=job_id,
                    model=model,
                    raw_request=raw,
                    cancellation_callback=cancellation_requested,
                )
                current = _get_job(job_id) or {}
                status = "cancelled" if current.get("cancellation_requested") else _status_from_nexus_data(nexus_data, fallback="completed")
                _set_job(job_id, status=status, answer=answer, nexus_data=nexus_data)
            except InterruptedError as exc:
                _set_job(job_id, status="cancelled", error=str(exc), cancellation_requested=True)
            except Exception as exc:
                _set_job(job_id, status="failed", error=f"CognitiveEvolve pipeline failed: {exc}")

        try:
            future = get_job_executor().submit(worker)
        except QueueFullError as exc:
            rejected = _set_job(job_id, status="rejected", error=str(exc), cancellation_requested=False)
            return JSONResponse(_job_public(rejected, include_answer=False), status_code=503)
        _register_job_future(job_id, future)
        future.add_done_callback(lambda _future: _pop_job_future(job_id))
        return JSONResponse(_job_public(_get_job(job_id) or {"id": job_id}, include_answer=False), status_code=202)

    @app.get("/v1/cogev/jobs/{job_id}", dependencies=[Depends(require_service_api_key)], response_model=None)
    async def get_job(job_id: str) -> JSONResponse:
        job = _get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown CognitiveEvolve job id.")
        return JSONResponse(_job_public(job))

    @app.delete("/v1/cogev/jobs/{job_id}", dependencies=[Depends(require_service_api_key)], response_model=None)
    async def cancel_job(job_id: str) -> JSONResponse:
        job = _get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown CognitiveEvolve job id.")
        if job.get("status") in {"completed", "best_current_route", "failed", "cancelled", "needs_continuation", "route_incomplete", "failed_verification", "interrupted_checkpointed", "paused_quota"}:
            return JSONResponse(_job_public(job))
        _cancel_job_future(job_id)
        updated = _set_job(job_id, cancellation_requested=True, status="cancellation_requested")
        return JSONResponse(_job_public(updated, include_answer=False))


    @app.get("/v1/cogev/jobs/{job_id}/result", dependencies=[Depends(require_service_api_key)], response_model=None)
    async def job_result(job_id: str) -> JSONResponse:
        job = _get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown CognitiveEvolve job id.")
        if job.get("status") not in {"completed", "best_current_route"}:
            return JSONResponse(_job_public(job, include_answer=False), status_code=202)
        payload = _completion_payload(
            request_id=job_id,
            model=str(job.get("model") or "cognitive-evolve-one-shot"),
            prompt=str(job.get("prompt") or ""),
            answer=str(job.get("answer") or ""),
            nexus_data=job.get("nexus_data") if isinstance(job.get("nexus_data"), dict) else {},
        )
        payload["object"] = "cogev.job.result"
        return JSONResponse(payload)

    @app.get("/v1/cogev/jobs/{job_id}/artifacts", dependencies=[Depends(require_service_api_key)], response_model=None)
    async def job_artifacts(job_id: str) -> JSONResponse:
        job = _get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown CognitiveEvolve job id.")
        root = Path(str(job.get("artifact_root") or job.get("task_dir") or ""))
        artifacts: list[dict[str, Any]] = []
        if root.exists():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    artifacts.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size})
        return JSONResponse({"id": job_id, "object": "cogev.job.artifacts", "status": job.get("status"), "artifacts": artifacts})


def _install_chat_routes(app: FastAPI) -> None:
    @app.post("/v1/chat/completions", dependencies=[Depends(require_service_api_key)], response_model=None)
    async def chat_completions(request: Request) -> Response:
        raw = await request.json()
        try:
            body = ChatCompletionRequest(**raw)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid chat completion request: {exc}") from exc
        config = get_service_config()
        model = body.model or config.default_model
        if model not in config.models:
            raise HTTPException(status_code=404, detail=f"Unknown CognitiveEvolve model '{model}'. Call /v1/models.")
        prompt = build_one_shot_prompt(body.messages)
        if not prompt:
            raise HTTPException(status_code=400, detail="At least one user message with text content is required.")
        request_id = "chatcmpl-cogev-" + uuid.uuid4().hex[:24]
        if body.stream:
            return StreamingResponse(
                _stream_engine_chunks(prompt, request_id=request_id, model=model, raw_request=raw),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        try:
            future = get_job_executor().submit(
                _run_engine,
                prompt,
                request_id=request_id,
                model=model,
                raw_request=raw,
            )
            answer, nexus_data = await asyncio.wrap_future(future)
        except QueueFullError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            # Keep the error OpenAI-client friendly while not leaking upstream provider secrets.
            raise HTTPException(status_code=502, detail=f"CognitiveEvolve pipeline failed: {exc}") from exc
        payload = _completion_payload(request_id=request_id, model=model, prompt=prompt, answer=answer, nexus_data=nexus_data)
        return JSONResponse(payload)


app = create_app()


__all__ = ["app", "create_app", "build_one_shot_prompt"]
