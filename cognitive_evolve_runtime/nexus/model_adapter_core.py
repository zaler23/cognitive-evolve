"""Core transport boundary for the structured Nexus model adapter."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from jsonschema import Draft202012Validator

from cognitive_evolve_runtime.llm.env import LLMConfigurationError, LLMResponseError
from cognitive_evolve_runtime.nexus.prompt_view import build_prompt_view

from .model_adapter_repair import (
    _repair_array_response,
    _repair_candidate_items,
    _repair_objective_contract_response,
)

JsonCaller = Callable[[str, dict[str, Any], dict[str, Any]], dict[str, Any]]


class ModelResponseSchemaError(LLMResponseError):
    """Raised when a configured model response cannot satisfy a Nexus schema."""


def _validate_schema(data: dict[str, Any], schema: dict[str, Any], *, request_type: str) -> None:
    errors = sorted(Draft202012Validator(schema).iter_errors(data), key=lambda err: list(err.path))
    if errors:
        detail = "; ".join(f"/{'/'.join(map(str, err.path))}: {err.message}" for err in errors[:5])
        raise ModelResponseSchemaError(f"{request_type} model response failed schema validation: {detail}")

@dataclass
class StructuredModelAdapterCore:
    """Shared transport, prompt-view, repair, and schema-validation core."""

    caller: JsonCaller | None = None
    system: str = "You are the task-semantics controller for the Nexus offline evolution runtime. Return only JSON matching the supplied schema."
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_configured_llm(cls) -> "StructuredModelAdapterCore":
        """Create an adapter backed by the existing ``llm_json`` transport.

        This method is intentionally explicit.  Merely constructing
        ``NexusRuntime`` never falls back to a real provider or reads API keys.
        """

        def _call(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
            from cognitive_evolve_runtime.llm.transport import llm_json

            return llm_json(request_type, payload, system=cls().system, schema_hint=schema)

        return cls(caller=_call, metadata={"transport": "cogev_llm_json"})

    def _call(self, request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        if self.caller is None:
            raise LLMConfigurationError("StructuredModelAdapter requires an explicit JSON caller or from_configured_llm().")
        controls = self.metadata.get("prompt_context_controls")
        if isinstance(controls, dict) and controls and "_prompt_context_controls" not in payload:
            payload = {**payload, "_prompt_context_controls": dict(controls)}
        prompt_view = build_prompt_view(request_type, payload)
        history = self.metadata.setdefault("prompt_view_history", [])
        if isinstance(history, list):
            history.append(dict(prompt_view.metadata))
            del history[:-20]
        self.metadata["last_prompt_view"] = dict(prompt_view.metadata)
        result = self.caller(request_type, prompt_view.payload, schema)
        if not isinstance(result, dict):
            raise ModelResponseSchemaError(f"{request_type} model response must be a JSON object")
        if request_type in {"nexus_build_objective_contract", "nexus_build_project_objective_contract"}:
            result = _repair_objective_contract_response(result, payload, project=request_type == "nexus_build_project_objective_contract")
        elif request_type == "nexus_seed_population":
            result = _repair_array_response(result, target_key="candidates", aliases=("candidates", "seeds", "genomes", "population", "results"))
            result = _repair_candidate_items(result, key="candidates")
        elif request_type == "nexus_generate_offspring":
            result = _repair_array_response(result, target_key="offspring", aliases=("offspring", "candidates", "children", "genomes", "mutations", "results"))
            result = _repair_candidate_items(result, key="offspring")
        _validate_schema(result, schema, request_type=request_type)
        return result

__all__ = ["JsonCaller", "ModelResponseSchemaError", "StructuredModelAdapterCore", "_validate_schema"]
