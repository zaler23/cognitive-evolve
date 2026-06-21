from __future__ import annotations

import inspect
import json
import os
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Iterator

from ..artifacts.store import _write_json
from ..core.redaction import redact
from .config import get_service_config
from .executor import QueueFullError, get_stream_executor
from .payloads import _nexus_actual_rounds, _nexus_answer_produced, _nexus_completion_status, _nexus_objective_solved, _nexus_verification_passed
from .jobs import _now, _task_dir_for_request
from .profiles import _temporary_model_runtime

# Lazy-imported in the worker so importing the API module does not eagerly load
# the full Nexus/LLM stack. Tests may monkeypatch this name with a fake class.
EngineOrchestrator: Any | None = None


def _sse_chunk(chunk: dict[str, Any]) -> bytes:
    return ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode("utf-8")



def _sse_comment(comment: str) -> bytes:
    safe = str(comment).replace("\r", " ").replace("\n", " ")
    return (": " + safe + "\n\n").encode("utf-8")


def _heartbeat_seconds() -> float:
    try:
        return max(1.0, float(os.environ.get("COGEV_STREAM_HEARTBEAT_SECONDS", "5")))
    except ValueError:
        return 5.0


def _heartbeat_think_text() -> str:
    text = os.environ.get("COGEV_STREAM_HEARTBEAT_THINK_TEXT", "wait").strip()
    return text or "wait"


def _stream_queue_size() -> int:
    try:
        return max(8, int(os.environ.get("COGEV_STREAM_EVENT_QUEUE_SIZE", "128")))
    except ValueError:
        return 128


def _stream_max_seconds() -> float:
    try:
        value = float(os.environ.get("COGEV_STREAM_MAX_SECONDS", "0"))
    except ValueError:
        return 0.0
    return max(0.0, value)


def _thinking_delta(text: str) -> dict[str, str]:
    safe = str(text).replace("\r", " ").replace("\n", " ").strip() or "wait"
    display = safe + "\n"
    return {
        # Keep regular answer content empty so clients that do not understand
        # reasoning fields do not append heartbeat text to the final answer.
        "content": "",
        # Common OpenAI-compatible reasoning stream field used by many R1-style
        # frontends.
        "reasoning_content": display,
        # Extra reasoning keys for frontends that label the pane as
        # reasoning/thinking rather than reasoning_content.
        "reasoning": display,
        "thinking": display,
    }


def _stream_heartbeat_chunks(*, request_id: str, model: str, created: int) -> Iterator[bytes]:
    think_text = _heartbeat_think_text()
    heartbeat = {
        "at": _now(),
        "stage": "heartbeat",
        "status": "running",
        "summary": "CognitiveEvolve is still running the long one-shot job.",
        "think_display": think_text,
        "chain_of_thought_exposed": False,
    }
    yield _sse_comment(f"cogev heartbeat: {think_text}")
    yield _sse_chunk({
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": _thinking_delta(think_text), "finish_reason": None}],
        "cognitive_evolve_event": heartbeat,
    })



def _stream_role_chunk(*, request_id: str, model: str, created: int) -> bytes:
    return _sse_chunk({
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    })



def _stream_content_chunks(*, request_id: str, model: str, created: int, content: str) -> Iterator[bytes]:
    chunk_size = max(80, int(os.environ.get("COGEV_STREAM_CHUNK_SIZE", "360")))
    for index in range(0, len(content), chunk_size):
        chunk = content[index:index + chunk_size]
        yield _sse_chunk({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
        })



def _stream_done_chunk(*, request_id: str, model: str, created: int, nexus_data: dict[str, Any] | None = None) -> bytes:
    chunk: dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    if nexus_data:
        chunk["cognitive_evolve"] = {
            "runtime_path": "nexus",
            "actual_rounds": _nexus_actual_rounds(nexus_data),
            "verification_passed": _nexus_verification_passed(nexus_data),
            "objective_solved": _nexus_objective_solved(nexus_data),
            "answer_produced": _nexus_answer_produced(nexus_data),
            "completion_status": _nexus_completion_status(nexus_data),
            "streaming_semantics": "progress events plus final answer chunks; not provider token streaming",
        }
    return _sse_chunk(chunk)



def _stream_chunks(payload: dict[str, Any]) -> Iterator[bytes]:
    request_id = payload["id"]
    model = payload["model"]
    created = payload["created"]
    yield _stream_role_chunk(request_id=request_id, model=model, created=created)
    content = payload["choices"][0]["message"]["content"]
    yield from _stream_content_chunks(request_id=request_id, model=model, created=created, content=content)
    yield _stream_done_chunk(request_id=request_id, model=model, created=created)
    yield b"data: [DONE]\n\n"



def _stream_engine_chunks(prompt: str, *, request_id: str, model: str, raw_request: dict[str, Any]) -> Iterator[bytes]:
    """Stream pipeline progress while the engine is still running.

    This is event-first streaming: clients receive coarse safe progress events
    before the final answer is available, then receive the final answer in normal
    OpenAI chat-completion chunks. It does not expose private reasoning.
    """

    created = _now()
    events: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=_stream_queue_size())
    stop_event = threading.Event()

    def emit_progress(event: dict[str, Any]) -> None:
        _offer_event(events, {"type": "progress", "event": event}, block=False)

    def worker() -> None:
        try:
            from ..llm import LLMSession, llm_session

            engine_cls = EngineOrchestrator
            if engine_cls is None:
                from ..engine.orchestrator import EngineOrchestrator as engine_cls

            config = get_service_config()
            task_dir = _task_dir_for_request(config.api_task_root, request_id)
            _write_json(task_dir / "api-request.json", redact({"request_id": request_id, "model": model, "request": raw_request}))
            llm_call_dir = task_dir / "llm-calls"
            with llm_session(LLMSession(journal_dir=str(llm_call_dir), call_ledger_path=str(llm_call_dir / "llm-call-ledger.jsonl"))), _temporary_model_runtime(model):
                engine = engine_cls()
                run_kwargs: dict[str, Any] = {
                    "context": {
                        "task_dir": str(task_dir),
                        "api_request_id": request_id,
                        "openai_compatible_model": model,
                        "interface": "openai_compatible_api",
                        "raw_request": raw_request,
                        "openai_model_tier": model,
                    },
                    "progress_callback": emit_progress,
                }
                try:
                    if "cancellation_callback" in inspect.signature(engine.run).parameters:
                        run_kwargs["cancellation_callback"] = stop_event.is_set
                except (TypeError, ValueError):
                    run_kwargs["cancellation_callback"] = stop_event.is_set
                result = engine.run(prompt, **run_kwargs)
            data = result.to_dict()
            _write_json(task_dir / "api-response-metadata.json", {"request_id": request_id, "model": model, "nexus_result_path": "nexus-runtime/run-result.json"})
            _offer_event(events, {"type": "final", "answer": result.final_answer, "nexus_data": data}, block=True)
        except InterruptedError as exc:
            _offer_event(events, {"type": "error", "error": f"CognitiveEvolve pipeline cancelled: {exc}"}, block=True)
        except Exception as exc:
            _offer_event(events, {"type": "error", "error": f"CognitiveEvolve pipeline failed: {exc}"}, block=True)
        finally:
            _offer_event(events, None, block=True)

    future: Future[Any] | None = None
    try:
        future = get_stream_executor().submit(worker)
    except QueueFullError as exc:
        yield _stream_role_chunk(request_id=request_id, model=model, created=created)
        yield from _stream_content_chunks(request_id=request_id, model=model, created=created, content=f"CognitiveEvolve stream queue is full: {exc}")
        yield _stream_done_chunk(request_id=request_id, model=model, created=created)
        yield b"data: [DONE]\n\n"
        return
    yield _stream_role_chunk(request_id=request_id, model=model, created=created)

    final_data: dict[str, Any] | None = None
    heartbeat_seconds = _heartbeat_seconds()
    max_seconds = _stream_max_seconds()
    started = time.monotonic()
    try:
        while True:
            elapsed = time.monotonic() - started
            if max_seconds and elapsed >= max_seconds:
                yield from _stream_content_chunks(
                    request_id=request_id,
                    model=model,
                    created=created,
                    content="CognitiveEvolve stream reached COGEV_STREAM_MAX_SECONDS; use /v1/cogev/jobs to continue polling the durable run.",
                )
                break
            wait_seconds = heartbeat_seconds
            if max_seconds:
                wait_seconds = max(0.01, min(heartbeat_seconds, max_seconds - elapsed))
            try:
                item = events.get(timeout=wait_seconds)
            except queue.Empty:
                yield from _stream_heartbeat_chunks(request_id=request_id, model=model, created=created)
                continue
            if item is None:
                break
            if item.get("type") == "progress":
                event = item.get("event", {})
                stage = str(event.get("stage", "progress"))
                yield _sse_comment(f"cogev progress: {stage}")
                yield _sse_chunk({
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                    "cognitive_evolve_event": event,
                })
            elif item.get("type") == "final":
                final_data = item.get("nexus_data") if isinstance(item.get("nexus_data"), dict) else {}
                answer = str(item.get("answer") or "")
                yield from _stream_content_chunks(request_id=request_id, model=model, created=created, content=answer)
            elif item.get("type") == "error":
                message = str(item.get("error") or "CognitiveEvolve pipeline failed.")
                yield from _stream_content_chunks(request_id=request_id, model=model, created=created, content=message)
                break
    finally:
        stop_event.set()
        if future is not None and not future.done():
            future.cancel()
    yield _stream_done_chunk(request_id=request_id, model=model, created=created, nexus_data=final_data)
    yield b"data: [DONE]\n\n"


def _offer_event(events: queue.Queue[dict[str, Any] | None], item: dict[str, Any] | None, *, block: bool) -> None:
    try:
        events.put(item, block=block, timeout=1.0 if block else 0.0)
    except queue.Full:
        if item and item.get("type") == "progress":
            return
        try:
            events.get_nowait()
        except queue.Empty:
            pass
        try:
            events.put(item, block=False)
        except queue.Full:
            pass


__all__ = ['_stream_chunks', '_stream_engine_chunks', '_stream_queue_size']
