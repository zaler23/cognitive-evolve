from __future__ import annotations

from typing import Any

from ..artifacts.store import _write_json
from ..core.redaction import redact
from .config import get_service_config
from .jobs import _task_dir_for_request
from .profiles import _temporary_model_runtime


def _run_engine(
    prompt: str,
    *,
    request_id: str,
    model: str,
    raw_request: dict[str, Any],
    cancellation_callback: Any | None = None,
) -> tuple[str, dict[str, Any]]:
    config = get_service_config()
    task_dir = _task_dir_for_request(config.api_task_root, request_id)
    _write_json(task_dir / "api-request.json", redact({"request_id": request_id, "model": model, "request": raw_request}))
    from ..engine.orchestrator import EngineOrchestrator
    from ..llm import LLMSession, llm_session

    llm_call_dir = task_dir / "llm-calls"
    with llm_session(LLMSession(journal_dir=str(llm_call_dir), call_ledger_path=str(llm_call_dir / "llm-call-ledger.jsonl"))), _temporary_model_runtime(model):
        result = EngineOrchestrator().run(
            prompt,
            context={
                "task_dir": str(task_dir),
                "api_request_id": request_id,
                "openai_compatible_model": model,
                "interface": "openai_compatible_api",
                "raw_request": raw_request,
                "openai_model_tier": model,
            },
            cancellation_callback=cancellation_callback,
        )
    data = result.to_dict()
    _write_json(task_dir / "api-response-metadata.json", {"request_id": request_id, "model": model, "nexus_result_path": "nexus-runtime/run-result.json"})
    return result.final_answer, data


__all__ = ['_run_engine']
