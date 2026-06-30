#!/usr/bin/env python3
"""OpenAI-shaped API server for CognitiveEvolve.

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


def _resume_engine(*, output_dir: Path, model: str, max_rounds: int | None = None) -> tuple[str, dict[str, Any]]:
    from ..llm import LLMSession, llm_session
    from ..nexus.runtime import NexusRuntime
    from .profiles import _temporary_model_runtime

    llm_call_dir = output_dir / "llm-calls"
    with llm_session(LLMSession(journal_dir=str(llm_call_dir), call_ledger_path=str(llm_call_dir / "llm-call-ledger.jsonl"))), _temporary_model_runtime(model):
        runtime = NexusRuntime.with_configured_llm(output_dir=output_dir)
        run = runtime.resume_from_checkpoint(max_rounds=max_rounds)
    data = run.to_dict()
    return run.final_answer, data


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
    app = FastAPI(title="CognitiveEvolve OpenAI-Shaped API", version="2.0.0", lifespan=_lifespan)
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
                        "completion_semantics": "one request triggers adaptive candidate evolution; completed means an answer was produced, not externally certified",
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


    @app.post("/v1/cogev/jobs/{job_id}/resume", dependencies=[Depends(require_service_api_key)], response_model=None)
    async def resume_job(job_id: str, request: Request) -> JSONResponse:
        job = _get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown CognitiveEvolve job id.")
        if job.get("status") in {"queued", "running", "cancellation_requested", "resuming"}:
            raise HTTPException(status_code=409, detail="Job is already active.")
        try:
            raw = await request.json()
        except Exception:
            raw = {}
        max_rounds = (raw.get("budget") or raw.get("max_rounds")) if isinstance(raw, dict) else None
        try:
            max_rounds_int = int(max_rounds) if max_rounds is not None else None
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="budget/max_rounds must be an integer")
        config = get_service_config()
        root = _safe_job_root(config.api_task_root, job_id)
        if root is None:
            raise HTTPException(status_code=404, detail="Unknown CognitiveEvolve job id.")
        output_dir = root / "nexus-runtime"
        checkpoint_path = output_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            raise HTTPException(status_code=409, detail="No checkpoint is available for this job.")
        model = str(job.get("model") or config.default_model)
        _set_job(job_id, status="queued", error=None, cancellation_requested=False)

        def worker() -> None:
            _set_job(job_id, status="resuming")
            try:
                answer, nexus_data = _resume_engine(output_dir=output_dir, model=model, max_rounds=max_rounds_int)
                status = _status_from_nexus_data(nexus_data, fallback="completed")
                _set_job(job_id, status=status, answer=answer, nexus_data=nexus_data, artifact_root=str(root), task_dir=str(root))
            except Exception as exc:
                _set_job(job_id, status="failed", error=f"CognitiveEvolve resume failed: {exc}")

        try:
            future = get_job_executor().submit(worker)
        except QueueFullError as exc:
            rejected = _set_job(job_id, status="rejected", error=str(exc), cancellation_requested=False)
            return JSONResponse(_job_public(rejected, include_answer=False), status_code=503)
        _register_job_future(job_id, future)
        future.add_done_callback(lambda _future: _pop_job_future(job_id))
        return JSONResponse(_job_public(_get_job(job_id) or {"id": job_id}, include_answer=False), status_code=202)

    @app.delete("/v1/cogev/jobs/{job_id}", dependencies=[Depends(require_service_api_key)], response_model=None)
    async def cancel_job(job_id: str) -> JSONResponse:
        job = _get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown CognitiveEvolve job id.")
        if job.get("status") in {"completed", "failed", "cancelled", "needs_continuation", "failed_verification", "interrupted_checkpointed", "paused_quota"}:
            return JSONResponse(_job_public(job))
        _cancel_job_future(job_id)
        updated = _set_job(job_id, cancellation_requested=True, status="cancellation_requested")
        return JSONResponse(_job_public(updated, include_answer=False))


    @app.get("/v1/cogev/jobs/{job_id}/result", dependencies=[Depends(require_service_api_key)], response_model=None)
    async def job_result(job_id: str) -> JSONResponse:
        job = _get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown CognitiveEvolve job id.")
        if not _job_has_result_payload(job):
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
        root = _safe_job_artifact_root(job)
        artifacts: list[dict[str, Any]] = []
        if root.exists():
            # codeql[py/path-injection] root is resolved and constrained to the configured API task root.
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    artifacts.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size})
        return JSONResponse({"id": job_id, "object": "cogev.job.artifacts", "status": job.get("status"), "artifacts": artifacts})


def _safe_job_artifact_root(job: dict[str, Any]) -> Path:
    raw_root = str(job.get("artifact_root") or job.get("task_dir") or "").strip()
    if not raw_root:
        raise HTTPException(status_code=404, detail="Job has no artifact root.")
    root = Path(raw_root).expanduser().resolve()
    api_root = get_service_config().api_task_root.expanduser().resolve()
    try:
        root.relative_to(api_root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Job artifact root is outside the configured API task root.") from exc
    return root


def _safe_job_root(api_task_root: Path, job_id: str) -> Path | None:
    normalized = str(job_id or "").strip()
    if not normalized or Path(normalized).name != normalized:
        return None
    base = api_task_root.resolve()
    candidate = (base / normalized).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate


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
            # Keep the error client friendly while not leaking upstream provider secrets.
            raise HTTPException(status_code=502, detail=f"CognitiveEvolve pipeline failed: {exc}") from exc
        payload = _completion_payload(request_id=request_id, model=model, prompt=prompt, answer=answer, nexus_data=nexus_data)
        return JSONResponse(payload)


app = create_app()


__all__ = ["app", "create_app", "build_one_shot_prompt"]
