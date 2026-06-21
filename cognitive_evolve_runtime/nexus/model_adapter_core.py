"""Core transport boundary for the structured Nexus model adapter."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from jsonschema import Draft202012Validator

from cognitive_evolve_runtime.llm.env import LLMConfigurationError, LLMResponseError
from cognitive_evolve_runtime.llm.model_spec import LLMModelSpec
from cognitive_evolve_runtime.nexus.diagnosis import STAGNATION_TYPES
from cognitive_evolve_runtime.nexus.prompt_audit import maybe_record_prompt_audit
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


def _json_key(value: Any) -> str:
    import json

    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return repr(value)


def _record_schema_repair(metadata: dict[str, Any], event: dict[str, Any]) -> None:
    events = metadata.setdefault("schema_repair_events", [])
    if isinstance(events, list):
        events.append(dict(event))
        del events[:-50]


def _repair_search_diagnosis_response(data: dict[str, Any]) -> dict[str, Any]:
    raw_type = str(data.get("stagnation_type") or "None")
    if raw_type in STAGNATION_TYPES:
        return data
    repaired = dict(data)
    metadata = dict(repaired.get("metadata") or {}) if isinstance(repaired.get("metadata"), dict) else {}
    metadata["raw_stagnation_type"] = raw_type
    notes = str(repaired.get("notes") or "")
    if raw_type and raw_type not in notes:
        notes = (notes + "; " if notes else "") + f"raw_stagnation_type={raw_type}"
    lowered = raw_type.lower()
    if any(token in lowered for token in ("route", "no_parent", "repair", "patch", "source_binding", "docs_only")):
        canonical = "RouteIncomplete"
    elif any(token in lowered for token in ("semantic", "loop", "convergence")):
        canonical = "SemanticLooping"
    elif any(token in lowered for token in ("quota", "schema", "transport", "model")):
        canonical = "ModelSchemaQuotaOrTransport"
    elif bool(repaired.get("stagnation_detected", False)):
        canonical = "RouteIncomplete"
    else:
        canonical = "None"
    repaired["metadata"] = metadata
    repaired["notes"] = notes
    repaired["stagnation_type"] = canonical
    return repaired

@dataclass
class StructuredModelAdapterCore:
    """Shared transport, prompt-view, repair, and schema-validation core."""

    caller: JsonCaller | None = None
    system: str = "You are the task-semantics controller for the Nexus offline evolution runtime. Return only JSON matching the supplied schema."
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_configured_llm(cls, model_spec: LLMModelSpec | None = None) -> "StructuredModelAdapterCore":
        """Create an adapter backed by the existing ``llm_json`` transport.

        This method is intentionally explicit.  Merely constructing
        ``NexusRuntime`` never falls back to a real provider or reads API keys.
        """

        def _call(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
            from cognitive_evolve_runtime.llm.transport import llm_json

            return llm_json(request_type, payload, system=cls().system, schema_hint=schema, model_spec=model_spec)

        metadata = {"transport": "cogev_llm_json"}
        if model_spec is not None:
            metadata["model_spec"] = model_spec.public_summary()
            metadata["model_spec_hash"] = model_spec.spec_hash
        return cls(caller=_call, metadata=metadata)

    @classmethod
    def with_configured_model(cls, model_spec: LLMModelSpec | None = None) -> "StructuredModelAdapterCore":
        return cls.from_configured_llm(model_spec=model_spec)

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
        maybe_record_prompt_audit(request_type, prompt_view, metadata=self.metadata)
        result = self.caller(request_type, prompt_view.payload, schema)
        if not isinstance(result, dict):
            raise ModelResponseSchemaError(f"{request_type} model response must be a JSON object")
        original_result_key = _json_key(result)
        if request_type in {"nexus_build_objective_contract", "nexus_build_project_objective_contract"}:
            result = _repair_objective_contract_response(result, payload, project=request_type == "nexus_build_project_objective_contract")
        elif request_type == "nexus_seed_population":
            result = _repair_array_response(result, target_key="candidates", aliases=("candidates", "seeds", "genomes", "population", "results"))
            result = _repair_candidate_items(result, key="candidates")
        elif request_type == "nexus_generate_offspring":
            result = _repair_array_response(result, target_key="offspring", aliases=("offspring", "candidates", "children", "genomes", "mutations", "results"))
            result = _repair_candidate_items(result, key="offspring")
        elif request_type in {"nexus_search_diagnosis", "nexus_diagnose_search_state"}:
            result = _repair_search_diagnosis_response(result)
        if _json_key(result) != original_result_key:
            _record_schema_repair(self.metadata, {"request_type": request_type, "repair": "schema_repair_applied"})
        try:
            _validate_schema(result, schema, request_type=request_type)
        except ModelResponseSchemaError as exc:
            if request_type != "nexus_seed_population":
                raise
            retry_payload = dict(prompt_view.payload)
            retry_payload["_schema_repair_retry"] = {
                "reason": str(exc),
                "target": "Return a JSON object with a candidates array matching the supplied schema.",
                "max_retries": 1,
            }
            _record_schema_repair(self.metadata, {"request_type": request_type, "repair": "schema_repair_retry", "reason": str(exc)})
            retry_result = self.caller(request_type, retry_payload, schema)
            if not isinstance(retry_result, dict):
                raise ModelResponseSchemaError(f"{request_type} schema repair retry response must be a JSON object") from exc
            retry_result = _repair_array_response(retry_result, target_key="candidates", aliases=("candidates", "seeds", "genomes", "population", "results"))
            retry_result = _repair_candidate_items(retry_result, key="candidates")
            _validate_schema(retry_result, schema, request_type=request_type)
            result = retry_result
        return result

__all__ = ["JsonCaller", "ModelResponseSchemaError", "StructuredModelAdapterCore", "_validate_schema"]
