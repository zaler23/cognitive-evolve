"""Nexus public orchestrator.

All CLI/API requests enter the same model-driven Nexus architecture.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from .result import NexusEngineResult
from ..llm.env import LLMConfigurationError
from ..nexus.budgeting import resolve_nexus_round_budget
from ..nexus.budget_factory import evolution_budget_from_round_budget, route_incomplete_round_budget
from ..nexus.model_adapter import StructuredModelAdapter
from ..nexus.protocols import NexusModelLike
from ..nexus.runtime import NexusRuntime
from .pipeline import DEFAULT_PIPELINE, EvolutionPipeline


class EngineOrchestrator:
    """Single public entrypoint for one Nexus evolution run."""

    def __init__(self, pipeline: EvolutionPipeline | None = None, *, model: NexusModelLike | None = None) -> None:
        self.pipeline = pipeline or DEFAULT_PIPELINE
        self.model = model

    def run(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancellation_callback: Callable[[], bool] | None = None,
        timeout_seconds: float | None = None,
    ) -> NexusEngineResult:
        del timeout_seconds  # Nexus budgets are round-based; external timeout wrappers remain allowed.
        if cancellation_callback and cancellation_callback():
            raise InterruptedError("cancelled before nexus runtime start")
        context = dict(context or {})
        context.setdefault("runtime_path", "engine_orchestrator/nexus")
        context.setdefault("pipeline", self.pipeline.to_dict())
        task_dir = Path(str(context["task_dir"])) if context.get("task_dir") else None
        output_dir = task_dir / "nexus-runtime" if task_dir is not None else None
        round_budget = resolve_nexus_round_budget(context)
        if _route_incomplete(context):
            round_budget = route_incomplete_round_budget(round_budget)
        runtime_model = self._resolve_model(context)
        evolution_budget = evolution_budget_from_round_budget(round_budget)
        if progress_callback:
            progress_callback({"type": "pipeline_progress", "stage": "nexus_runtime", "status": "started", "round_budget": round_budget.to_dict()})
        run = NexusRuntime(model=runtime_model, output_dir=output_dir).run_text(
            prompt,
            user_goal=prompt,
            budget=evolution_budget,
            cancellation_callback=cancellation_callback,
            runtime_metadata={"round_budget": round_budget.to_dict(), "model_backed": runtime_model is not None},
        )
        if progress_callback:
            for event in run.pipeline_events:
                progress_callback(event)
        run_dict = run.to_dict()
        result = NexusEngineResult(
            prompt=prompt,
            mode=run.mode,
            contract=dict(run_dict.get("contract") or {}),
            policy=dict(run_dict.get("policy") or {}),
            world=dict(run_dict.get("world") or {}),
            evolution=dict(run_dict.get("evolution") or {}),
            artifacts=dict(run_dict.get("artifacts") or {}),
            pipeline_events=list(run_dict.get("pipeline_events") or []),
            context_protocol=dict(run_dict.get("context_protocol") or {}),
            verification_summaries=list(run_dict.get("verification_summaries") or []),
        )
        result.evolution.setdefault("round_budget", round_budget.to_dict())
        return result

    def _resolve_model(self, context: dict[str, Any]) -> NexusModelLike | None:
        if self.model is not None:
            return self.model
        explicit = context.get("model_adapter") or context.get("nexus_model")
        if explicit is not None:
            return explicit
        if _is_api_request(context) or _truthy(os.environ.get("COGEV_NEXUS_USE_CONFIGURED_LLM")):
            try:
                return StructuredModelAdapter.from_configured_llm()
            except LLMConfigurationError:
                raise
            except Exception as exc:  # keep API failures explicit; no deterministic silent fallback.
                raise RuntimeError(f"failed to configure Nexus LLM adapter: {exc}") from exc
        return None


def _is_api_request(context: dict[str, Any]) -> bool:
    interface = str(context.get("interface") or "").strip().lower()
    return interface == "openai_compatible_api" or bool(context.get("api_request_id") or context.get("openai_compatible_model"))


def _route_incomplete(context: dict[str, Any]) -> bool:
    assessment = context.get("semantic_assessment") if isinstance(context.get("semantic_assessment"), dict) else context
    semantic_control = assessment.get("semantic_control") if isinstance(assessment.get("semantic_control"), dict) else {}
    return assessment.get("task_type") == "route_incomplete" or semantic_control.get("incomplete") is True


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


__all__ = ["EngineOrchestrator"]
