"""Core transport boundary for the structured Nexus model adapter."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from jsonschema import Draft202012Validator

from cognitive_evolve_runtime.llm.env import LLMConfigurationError, LLMResponseError
from cognitive_evolve_runtime.llm.model_spec import LLMModelSpec
from cognitive_evolve_runtime.llm.request_policy import LLMRequestPolicy
from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationPlan
from cognitive_evolve_runtime.nexus.diagnosis import STAGNATION_TYPES, SearchDiagnosis
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.prompt_audit import maybe_record_prompt_audit
from cognitive_evolve_runtime.nexus.prompt_view import (
    PromptView,
    build_prompt_view,
    candidate_prompt_view,
    is_long_context_request,
    prompt_char_budget_details,
)
from cognitive_evolve_runtime.ranking.relative_rater import _deterministic_rank, relative_rater_schema

from .model_adapter_schemas import (
    _candidate_critiques_schema,
    _candidate_population_schema,
    _context_request_schema,
    _evolution_policy_schema,
    _mutation_plan_schema,
    _objective_contract_schema,
    _offspring_population_schema,
    _pool_preprocess_schema,
    _search_diagnosis_schema,
    _stop_decision_schema,
    _synthesis_schema,
    _task_classification_schema,
    _text_world_model_schema,
)
from .model_adapter_repair import (
    _ingest_bare_offspring_artifact,
    _ingest_bare_seed_artifact,
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
        detail = "; ".join(f"/{'/'.join(map(str, err.path))}: {err.message}" for err in errors)
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


def _request_policy_for_model_call(request_type: str) -> LLMRequestPolicy:
    long_context = is_long_context_request(request_type)
    limit_details = prompt_char_budget_details(long_context=long_context)
    return LLMRequestPolicy(
        long_context=long_context,
        max_prompt_chars=int(limit_details["requested"] or limit_details["effective"] or 0),
        structured_prompt=True,
    )


def _prompt_view_for_model_call(request_type: str, payload: dict[str, Any], schema: dict[str, Any]) -> PromptView:
    limit_details = prompt_char_budget_details(long_context=is_long_context_request(request_type))
    effective_limit = int(limit_details["effective"] or 0)
    prompt_view = build_prompt_view(request_type, payload, max_chars=effective_limit, schema_hint=schema)
    prompt_view.metadata["prompt_limit"] = limit_details
    if int(prompt_view.metadata.get("sent_request_chars") or 0) > effective_limit:
        raise LLMResponseError(
            f"{request_type} structured prompt could not fit effective cap of {effective_limit} characters; "
            "refusing lossy transport excerpt replacement."
        )
    return prompt_view


def _split_exact_candidate_payloads(
    request_type: str,
    payload: dict[str, Any],
    schema: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[CandidateGenome]]:
    """Split model-owned candidate comparisons without clipping a candidate."""

    candidates = [item for item in payload.get("candidates", []) if isinstance(item, CandidateGenome)]
    if not candidates:
        return [payload], []
    batches: list[dict[str, Any]] = []
    effective_limit = int(
        prompt_char_budget_details(long_context=is_long_context_request(request_type))["effective"] or 0
    )
    uncovered = [
        candidate
        for candidate in candidates
        if len(_json_key(candidate_prompt_view(candidate, detail="exact"))) >= effective_limit
    ]
    uncovered_objects = {id(candidate) for candidate in uncovered}
    candidates = [candidate for candidate in candidates if id(candidate) not in uncovered_objects]

    def split(items: list[CandidateGenome]) -> None:
        candidate_payload = {**payload, "candidates": items}
        metadata_snapshots = [(item, deepcopy(item.metadata)) for item in items]
        cap_error: LLMResponseError | None = None
        try:
            _prompt_view_for_model_call(request_type, candidate_payload, schema)
        except LLMResponseError as exc:
            if not str(exc).startswith(f"{request_type} structured prompt could not fit effective cap"):
                raise
            cap_error = exc
        finally:
            for item, metadata in metadata_snapshots:
                item.metadata = metadata
        if cap_error is not None:
            if len(items) == 1:
                uncovered.extend(items)
                return
            middle = len(items) // 2
            split(items[:middle])
            split(items[middle:])
            return
        batches.append(candidate_payload)

    if candidates:
        split(candidates)
    return batches, uncovered


def _merge_relative_rank_results(
    results: list[dict[str, Any]],
    candidates: list[CandidateGenome],
    *,
    playoff: dict[str, Any] | None = None,
    batch_count: int,
    uncovered_count: int,
) -> dict[str, Any]:
    ids = {candidate.id for candidate in candidates}
    by_id = {candidate.id: candidate for candidate in candidates}

    def listed(field: str) -> list[str]:
        selected = {
            str(candidate_id)
            for result in results
            for candidate_id in (result.get(field) if isinstance(result.get(field), list) else [])
            if str(candidate_id) in ids
        }
        return [candidate.id for candidate in candidates if candidate.id in selected]

    def champion(field: str, *, mechanism: bool = False) -> str:
        if playoff is not None:
            selected = str(playoff.get(field) or "")
            return selected if selected in ids else ""
        contender_ids = [
            str(result.get(field) or "")
            for result in results
            if str(result.get(field) or "") in ids
        ]
        contenders = [by_id[candidate_id] for candidate_id in dict.fromkeys(contender_ids)]
        if not contenders:
            return ""
        deterministic = _deterministic_rank(contenders, raw_notes="")
        return deterministic.strongest_mechanism_id if mechanism else deterministic.best_final_answer_id

    def pairs(field: str) -> list[list[str]]:
        out: list[list[str]] = []
        seen: set[tuple[str, str]] = set()
        for result in results:
            values = result.get(field)
            for item in values if isinstance(values, list) else []:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                pair = (str(item[0]), str(item[1]))
                if pair[0] in ids and pair[1] in ids and pair not in seen:
                    seen.add(pair)
                    out.append([pair[0], pair[1]])
        return out

    observations: dict[str, dict[str, Any]] = {}
    for result in [*results, *([playoff] if playoff is not None else [])]:
        raw = result.get("multihead_observations")
        if not isinstance(raw, dict):
            continue
        for candidate_id, axes in raw.items():
            if str(candidate_id) in ids and isinstance(axes, dict):
                observations[str(candidate_id)] = dict(axes)
    for candidate in candidates:
        observations.setdefault(candidate.id, dict(candidate.multihead_scores))

    pairwise_preferences: list[dict[str, Any]] = []
    for result in results:
        raw = result.get("pairwise_preferences")
        if isinstance(raw, list):
            pairwise_preferences.extend(dict(item) for item in raw if isinstance(item, dict))

    notes = [
        str(result.get("raw_notes") or "")
        for result in results
        if str(result.get("raw_notes") or "")
    ]
    notes.append(
        f"exact_candidate_batches={batch_count}; "
        f"deterministic_exact_cap_coverage={uncovered_count}"
    )
    if uncovered_count:
        notes.append("exact_candidate_exceeded_prompt_cap")
    if playoff is not None:
        if str(playoff.get("raw_notes") or ""):
            notes.append(str(playoff.get("raw_notes")))
        notes.append("exact_batch_tournament_playoff")

    return {
        "best_final_answer_id": champion("best_final_answer_id"),
        "strongest_mechanism_id": champion("strongest_mechanism_id", mechanism=True),
        "mutation_worthy_ids": listed("mutation_worthy_ids"),
        "edge_value_ids": listed("edge_value_ids"),
        "auxiliary_ids": listed("auxiliary_ids"),
        "dormant_ids": listed("dormant_ids"),
        "dominated_pairs": pairs("dominated_pairs"),
        "crossover_pairs": pairs("crossover_pairs"),
        "preserve_incomplete_ids": listed("preserve_incomplete_ids"),
        "pairwise_preferences": pairwise_preferences,
        "multihead_observations": observations,
        "raw_notes": "; ".join(notes),
    }


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

            policy = _request_policy_for_model_call(request_type)
            try:
                return llm_json(request_type, payload, system=cls().system, schema_hint=schema, model_spec=model_spec, request_policy=policy)
            except TypeError as exc:
                if "request_policy" not in str(exc):
                    raise
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
        prompt_view = _prompt_view_for_model_call(request_type, payload, schema)
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
            if not isinstance(result.get("candidates"), list):
                result = _ingest_bare_seed_artifact(result, payload)
            result = _repair_candidate_items(result, key="candidates")
        elif request_type == "nexus_generate_offspring":
            result = _repair_array_response(result, target_key="offspring", aliases=("offspring", "candidates", "children", "genomes", "mutations", "results"))
            if not isinstance(result.get("offspring"), list):
                result = _ingest_bare_offspring_artifact(result, payload)
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
            retry_payload = dict(payload)
            retry_payload["_schema_repair_retry"] = {
                "reason": str(exc),
                "target": "Return a JSON object with a candidates array matching the supplied schema.",
                "max_retries": 1,
            }
            _record_schema_repair(self.metadata, {"request_type": request_type, "repair": "schema_repair_retry", "reason": str(exc)})
            retry_prompt_view = _prompt_view_for_model_call(request_type, retry_payload, schema)
            retry_result = self.caller(request_type, retry_prompt_view.payload, schema)
            if not isinstance(retry_result, dict):
                raise ModelResponseSchemaError(f"{request_type} schema repair retry response must be a JSON object") from exc
            retry_result = _repair_array_response(retry_result, target_key="candidates", aliases=("candidates", "seeds", "genomes", "population", "results"))
            if not isinstance(retry_result.get("candidates"), list):
                retry_result = _ingest_bare_seed_artifact(retry_result, payload)
            retry_result = _repair_candidate_items(retry_result, key="candidates")
            _validate_schema(retry_result, schema, request_type=request_type)
            result = retry_result
        return result


class StructuredModelAdapter(StructuredModelAdapterCore):
    """Opt-in structured model adapter with deterministic tests and no implicit API-key use."""

    def build_objective_contract(self, *, user_goal: str, world: Any) -> dict[str, Any]:
        schema = _objective_contract_schema(project=False)
        return self._call("nexus_build_objective_contract", {"user_goal": user_goal, "world": world}, schema)

    def build_text_world_model(self, *, packet: Any) -> dict[str, Any]:
        schema = _text_world_model_schema()
        return self._call("nexus_build_text_world_model", {"packet": packet}, schema)

    def build_project_objective_contract(self, *, user_goal: str, snapshot: Any, world: Any | None = None) -> dict[str, Any]:
        schema = _objective_contract_schema(project=True)
        return self._call("nexus_build_project_objective_contract", {"user_goal": user_goal, "snapshot": snapshot, "world": world}, schema)

    def classify_task(self, *, prompt: str) -> dict[str, Any]:
        schema = _task_classification_schema()
        return self._call("nexus_classify_task", {"prompt": prompt}, schema)

    def build_evolution_policy(self, *, contract: Any, world: Any) -> dict[str, Any]:
        schema = _evolution_policy_schema()
        return self._call("nexus_build_evolution_policy", {"contract": contract, "world": world}, schema)

    def preprocess_candidate_pool(
        self,
        *,
        request_type: str = "nexus_pool_preprocess",
        contract: Any,
        policy: Any,
        coverage_report: dict[str, Any],
        clusters: list[dict[str, Any]],
        representatives: list[dict[str, Any]],
        instructions: dict[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        schema = _pool_preprocess_schema()
        payload = {
            "request_type": request_type,
            "contract": contract,
            "policy": policy,
            "coverage_report": coverage_report,
            "clusters": clusters,
            "representatives": representatives,
            "instructions": dict(instructions or {}),
        }
        if extra:
            payload["extra"] = dict(extra)
        return self._call("nexus_pool_preprocess", payload, schema)

    def seed_population(self, *, contract: Any, world: Any, policy: Any, provided_context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        schema = _candidate_population_schema()
        payload = {"contract": contract, "world": world, "policy": policy}
        policy_metadata = getattr(policy, "metadata", None)
        if not isinstance(policy_metadata, dict) and isinstance(policy, dict):
            policy_metadata = policy.get("metadata")
        if isinstance(policy_metadata, dict):
            requested_count = policy_metadata.get("requested_candidate_count")
            if isinstance(requested_count, int) and requested_count > 0:
                payload["requested_candidate_count"] = requested_count
        if provided_context:
            payload["source_context"] = provided_context
        data = self._call("nexus_seed_population", payload, schema)
        return [dict(item) for item in data.get("candidates", []) if isinstance(item, dict)]

    def relative_rank(self, *, candidates: list[CandidateGenome], contract: Any, policy: Any, archives: Any) -> dict[str, Any]:
        schema = relative_rater_schema()
        payload: dict[str, Any] = {
            "candidates": candidates,
            "contract": contract,
            "policy": policy,
            "archives": archives,
            "comparison_protocol": {
                "mechanism_summary": "Extract no more than three mechanism points per candidate before comparing.",
                "length_signal": "Artifact length and verbosity are not quality signals.",
            },
        }
        controls = self.metadata.get("prompt_context_controls")
        if isinstance(controls, dict) and controls:
            payload["_prompt_context_controls"] = dict(controls)
        batches, uncovered = _split_exact_candidate_payloads("nexus_relative_rank", payload, schema)
        results = [self._call("nexus_relative_rank", batch, schema) for batch in batches]
        if uncovered:
            results.append(
                _deterministic_rank(
                    uncovered,
                    raw_notes="exact_candidate_exceeded_prompt_cap; deterministic_exact_cap_coverage",
                ).to_dict()
            )
        if not results:
            results.append(_deterministic_rank(candidates, raw_notes="no candidates").to_dict())

        finalist_ids = {
            str(result.get(field) or "")
            for result in results
            for field in ("best_final_answer_id", "strongest_mechanism_id")
            if str(result.get(field) or "")
        }
        finalists = [candidate for candidate in candidates if candidate.id in finalist_ids]
        playoff = (
            self.relative_rank(candidates=finalists, contract=contract, policy=policy, archives=archives)
            if len(results) > 1 and 1 < len(finalists) < len(candidates)
            else None
        )
        merged = _merge_relative_rank_results(
            results,
            candidates,
            playoff=playoff,
            batch_count=len(batches),
            uncovered_count=len(uncovered),
        )
        _validate_schema(merged, schema, request_type="nexus_relative_rank")
        self.metadata["last_exact_candidate_batching"] = {
            "request_type": "nexus_relative_rank",
            "candidate_count": len(candidates),
            "model_batch_count": len(batches),
            "deterministic_coverage_count": len(uncovered),
            "tournament_playoff": playoff is not None,
        }
        return merged

    def critique_candidates(self, *, candidates: list[CandidateGenome], round_index: int, contract: Any, policy: Any, archives: Any) -> list[dict[str, Any]]:
        schema = _candidate_critiques_schema()
        payload: dict[str, Any] = {
            "round_index": round_index,
            "candidates": candidates,
            "contract": contract,
            "policy": policy,
            "archives": archives,
        }
        controls = self.metadata.get("prompt_context_controls")
        if isinstance(controls, dict) and controls:
            payload["_prompt_context_controls"] = dict(controls)
        batches, uncovered = _split_exact_candidate_payloads("nexus_critique_candidates", payload, schema)
        critiques: list[dict[str, Any]] = []
        for batch in batches:
            data = self._call("nexus_critique_candidates", batch, schema)
            critiques.extend(dict(item) for item in data.get("critiques", []) if isinstance(item, dict))
        self.metadata["last_exact_candidate_batching"] = {
            "request_type": "nexus_critique_candidates",
            "candidate_count": len(candidates),
            "model_batch_count": len(batches),
            "deterministic_coverage_count": len(uncovered),
            "tournament_playoff": False,
        }
        return critiques

    def diagnose_search_state(self, *, population: list[CandidateGenome], archives: Any, history: list[dict[str, Any]], contract: Any, policy: Any) -> dict[str, Any]:
        schema = _search_diagnosis_schema()
        return self._call(
            "nexus_diagnose_search_state",
            {"population": population, "archives": archives, "history": history, "contract": contract, "policy": policy},
            schema,
        )

    def update_policy(self, *, policy: EvolutionPolicy, diagnosis: SearchDiagnosis) -> dict[str, Any]:
        schema = _evolution_policy_schema()
        return self._call("nexus_update_policy", {"policy": policy, "diagnosis": diagnosis}, schema)

    def request_context(self, *, contract: Any, world: Any, parents: list[CandidateGenome], archives: Any, mutation_instruction: str = "") -> dict[str, Any]:
        schema = _context_request_schema()
        return self._call(
            "nexus_request_context",
            {"contract": contract, "world": world, "parents": parents, "archives": archives, "mutation_instruction": mutation_instruction},
            schema,
        )

    def plan_mutations(self, *, parents: list[CandidateGenome], actions: list[str], archives: Any, diagnosis: SearchDiagnosis, policy: EvolutionPolicy, provided_context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        schema = _mutation_plan_schema()
        payload: dict[str, Any] = {
            "parents": parents,
            "actions": list(actions),
            "archives": archives,
            "diagnosis": diagnosis,
            "policy": policy,
        }
        policy_metadata = getattr(policy, "metadata", None)
        if not isinstance(policy_metadata, dict) and isinstance(policy, dict):
            policy_metadata = policy.get("metadata")
        if isinstance(policy_metadata, dict):
            requested_count = policy_metadata.get("requested_candidate_count")
            if isinstance(requested_count, int) and requested_count > 0:
                payload["requested_candidate_count"] = requested_count
        if provided_context:
            payload["source_context"] = provided_context
        data = self._call(
            "nexus_plan_mutations",
            payload,
            schema,
        )
        return [dict(item) for item in data.get("plans", []) if isinstance(item, dict)]

    def generate_offspring(self, *, plans: list[MutationPlan], parents: list[CandidateGenome], world: Any, contract: Any, policy: EvolutionPolicy, provided_context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        schema = _offspring_population_schema()
        payload: dict[str, Any] = {
            "plans": plans,
            "parents": parents,
            "world": world,
            "contract": contract,
            "policy": policy,
        }
        policy_metadata = getattr(policy, "metadata", None)
        if not isinstance(policy_metadata, dict) and isinstance(policy, dict):
            policy_metadata = policy.get("metadata")
        if isinstance(policy_metadata, dict):
            requested_count = policy_metadata.get("requested_candidate_count")
            if isinstance(requested_count, int) and requested_count > 0:
                payload["requested_candidate_count"] = requested_count
        if provided_context:
            payload["source_context"] = provided_context
        data = self._call(
            "nexus_generate_offspring",
            payload,
            schema,
        )
        return [dict(item) for item in data.get("offspring", []) if isinstance(item, dict)]

    def synthesize_result(self, *, population: list[CandidateGenome], archives: Any, contract: Any, world: Any) -> dict[str, Any]:
        schema = _synthesis_schema()
        return self._call(
            "nexus_synthesize_result",
            {"population": population, "archives": archives, "contract": contract, "world": world},
            schema,
        )

    def should_stop(self, *, budget: Any, diagnosis: Any, best_answer_id: str, population: list[Any]) -> dict[str, Any]:
        schema = _stop_decision_schema()
        return self._call(
            "nexus_should_stop",
            {
                "budget": budget,
                "diagnosis": diagnosis,
                "best_answer_id": best_answer_id,
                "population": population,
                "instruction": (
                    "Return stop=true only for one of these terminal choices: "
                    "candidate_ready_for_external_review or diminishing_returns_checkpoint. "
                    "Always return solved=false: this producer may prepare a best-current candidate for external review but cannot close the objective. "
                    "Use diminishing_returns_checkpoint when more rounds have low expected marginal value. "
                    "Safety checkpoints and local checks are not external correctness verdicts."
                ),
            },
            schema,
        )

__all__ = ["JsonCaller", "ModelResponseSchemaError", "StructuredModelAdapterCore", "StructuredModelAdapter", "_validate_schema"]
