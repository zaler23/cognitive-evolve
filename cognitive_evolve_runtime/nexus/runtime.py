"""Nexus runtime entrypoint."""
from __future__ import annotations

import json
import math
import hashlib
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.contracts.objective_contract import (
    NexusObjectiveContract,
    NexusObjectiveContractBuilder,
    NexusProjectObjectiveContract,
    apply_artifact_policy_to_contract,
)
from cognitive_evolve_runtime.inputs.project_map import ProjectWorldModel
from cognitive_evolve_runtime.inputs.project_snapshot import ProjectSnapshot
from cognitive_evolve_runtime.inputs.text_packet import TextInputPacket, TextWorldModel
from cognitive_evolve_runtime.nexus.context_protocol import ContextOrchestrator, ContextProtocolResult
from cognitive_evolve_runtime.nexus.live_store import LiveNexusStore
from cognitive_evolve_runtime.nexus.budget_factory import evolution_budget_from_params, resume_evolution_budget
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, EvolutionLoopResult, evolve_once, seed_population
from cognitive_evolve_runtime.nexus.model_adapter import StructuredModelAdapter
from cognitive_evolve_runtime.nexus.model_routes import NexusModelRole, NexusModelRoutes, coerce_model_routes
from cognitive_evolve_runtime.llm.model_spec import LLMModelSpec
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy, EvolutionPolicyBuilder
from cognitive_evolve_runtime.verification.synthesizer import VerificationSynthesizer
from cognitive_evolve_runtime.verification.types import VerificationPlan
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike
from cognitive_evolve_runtime.nexus.project_verification import ProjectVerificationSummary
from cognitive_evolve_runtime.nexus.fallbacks import finish_fallback_capture, start_fallback_capture
from cognitive_evolve_runtime.nexus.runtime_services import NexusPersistenceService, NexusProjectVerificationService
from cognitive_evolve_runtime.nexus._shared import positive_int
from cognitive_evolve_runtime.persistence.checkpoint import CheckpointStore


@dataclass
class NexusRunResult:
    mode: str
    contract: dict[str, Any]
    policy: dict[str, Any]
    world: dict[str, Any]
    evolution: dict[str, Any]
    artifacts: dict[str, str] = field(default_factory=dict)
    pipeline_events: list[dict[str, Any]] = field(default_factory=list)
    context_protocol: dict[str, Any] = field(default_factory=dict)
    verification_summaries: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def final_answer(self) -> str:
        return str(self.evolution.get("synthesis", {}).get("final_answer", ""))


class NexusRuntime:
    def __init__(self, *, model: NexusModelLike | None = None, model_routes: NexusModelRoutes | dict[str, Any] | None = None, output_dir: str | Path | None = None) -> None:
        self.model_routes = coerce_model_routes(model=model, model_routes=model_routes)
        self.model = self.model_routes.model_for(NexusModelRole.DEFAULT)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.contract_builder = NexusObjectiveContractBuilder()
        self.policy_builder = EvolutionPolicyBuilder()
        self.context_orchestrator = ContextOrchestrator()
        self.persistence_service = NexusPersistenceService(output_dir=self.output_dir)
        self.project_verification_service = NexusProjectVerificationService(output_dir=self.output_dir)

    def serialize_state(self, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return a fresh Nexus runtime serialization event.

        The returned ``signature`` / ``serialization_signature`` pair identifies
        this serialization event, not a deterministic content hash of the
        runtime state.  Consecutive calls intentionally produce distinct v2
        signatures even when the stable runtime fields are unchanged.  Use a
        separate content-hash API if a future caller needs equality, caching, or
        change-detection semantics.
        """

        state: dict[str, Any] = {
            "runtime_path": "nexus",
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
            "model_type": type(self.model).__name__ if self.model is not None else None,
            "model_routes": self.model_routes.public_summary(),
            "event_nonce": uuid.uuid4().hex,
        }
        if extra is not None:
            state["extra"] = dict(extra)
        encoded = json.dumps(state, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        signature = "NEXUS-RUNTIME-STATE-SIG-v2-" + hashlib.sha256(encoded).hexdigest()
        return {
            "status": "success",
            "runtime_path": "nexus",
            "signature": signature,
            "serialization_signature": signature,
            "signature_semantics": "fresh_serialization_event_not_deterministic_state_hash",
            "state": state,
        }

    @classmethod
    def with_configured_llm(cls, *, output_dir: str | Path | None = None, default_model_spec: LLMModelSpec | None = None, seed_model_spec: LLMModelSpec | None = None) -> "NexusRuntime":
        """Construct an explicit LLM-backed Nexus runtime.

        This is deliberately opt-in so tests and offline runs never silently use
        a real provider or an API key.
        """

        default_model = StructuredModelAdapter.from_configured_llm(model_spec=default_model_spec)
        seed_model = StructuredModelAdapter.from_configured_llm(model_spec=seed_model_spec) if seed_model_spec is not None else None
        return cls(model_routes=NexusModelRoutes(default_model=default_model, seed_model=seed_model), output_dir=output_dir)

    def run_text(
        self,
        text: str,
        *,
        user_goal: str | None = None,
        max_rounds: int = 1,
        min_population_size: int | None = None,
        branch_factor: int = 0,
        budget: EvolutionBudget | None = None,
        stop_policy: str = "llm_after_minimum",
        min_rounds_before_stop: int = 1,
        runtime_metadata: dict[str, Any] | None = None,
        adaptive_config: dict[str, Any] | None = None,
        cancellation_callback: Any | None = None,
    ) -> NexusRunResult:
        fallback_token = start_fallback_capture()
        packet = TextInputPacket.from_text(text)
        world = _build_text_world_model(packet, model=self.model)
        goal = user_goal or packet.raw_text[:500] or "text evolution task"
        artifact_policy_config = _artifact_policy_config_from_adaptive_config(adaptive_config)
        contract = self.contract_builder.build_text_contract(
            user_goal=goal,
            packet=packet,
            world=world,
            model=self.model,
            artifact_policy_config=artifact_policy_config,
        )
        policy = self.policy_builder.build(contract=contract, world=world, model=self.model)
        budget = budget or evolution_budget_from_params(
            max_rounds=max_rounds,
            branch_factor=branch_factor,
            initial_candidate_count=min_population_size or 0,
            stop_policy=stop_policy,
            min_rounds_before_stop=min_rounds_before_stop,
        )
        _resolve_budget_width_from_policy(budget, policy)
        min_population_size = min_population_size if min_population_size is not None else budget.initial_candidate_count
        population = seed_population(contract=contract, world=world, policy=policy, model=self.model_routes.model_for(NexusModelRole.SEED), min_population_size=min_population_size)
        archives = ArchiveManager(policy.archive_schema)
        verification_plan = VerificationSynthesizer(model=self.model).synthesize({"goal": goal, "contract": contract.to_dict()})
        world_payload = _world_to_dict_with_latent_metadata(world, contract)
        observer = self._live_observer(mode="text", contract=contract, world=world_payload, max_rounds=budget.max_rounds, budget=budget.to_dict())
        result = evolve_once(
            population=population,
            archives=archives,
            policy=policy,
            contract=contract,
            world=world,
            budget=budget,
            model=self.model,
            observer=observer,
            cancellation_callback=cancellation_callback,
            adaptive_config=adaptive_config,
            verification_plan=verification_plan,
        )
        run = NexusRunResult(
            mode="text",
            contract=contract.to_dict() | {"contract_hash": contract.contract_hash()},
            policy=result.policy.to_dict(),
            world=world_payload,
            evolution=result.to_dict(),
            pipeline_events=list(result.pipeline_events),
        )
        run.evolution.setdefault("runtime_metadata", {})["model_routes"] = self.model_routes.public_summary()
        if runtime_metadata:
            run.evolution["runtime_metadata"].update(dict(runtime_metadata))
        prompt_metadata = _model_prompt_metadata(self.model)
        if prompt_metadata:
            run.evolution["prompt_view_metadata"] = prompt_metadata
        _sync_runtime_round_metadata(run.evolution, result)
        _attach_fallback_events(run.evolution, finish_fallback_capture(fallback_token))
        run.artifacts = self._persist(run, result, contract=contract, world=world_payload, budget_history=result.budget_history, budget=budget)
        return run

    def run_project(
        self,
        root: str | Path,
        *,
        user_goal: str,
        max_rounds: int = 1,
        include_tests: bool = False,
        min_population_size: int | None = None,
        branch_factor: int = 0,
        budget: EvolutionBudget | None = None,
        stop_policy: str = "llm_after_minimum",
        min_rounds_before_stop: int = 1,
        adaptive_config: dict[str, Any] | None = None,
        cancellation_callback: Any | None = None,
    ) -> NexusRunResult:
        fallback_token = start_fallback_capture()
        snapshot = ProjectSnapshot.from_path(root)
        world = ProjectWorldModel.from_snapshot(snapshot, objective=user_goal)
        artifact_policy_config = _artifact_policy_config_from_adaptive_config(adaptive_config)
        contract = self.contract_builder.build_project_contract(
            user_goal=user_goal,
            snapshot=snapshot,
            world=world,
            model=self.model,
            artifact_policy_config=artifact_policy_config,
        )
        policy = self.policy_builder.build(contract=contract, world=world, model=self.model)
        budget = budget or evolution_budget_from_params(
            max_rounds=max_rounds,
            branch_factor=branch_factor,
            initial_candidate_count=min_population_size or 0,
            stop_policy=stop_policy,
            min_rounds_before_stop=min_rounds_before_stop,
        )
        _resolve_budget_width_from_policy(budget, policy)
        min_population_size = min_population_size if min_population_size is not None else budget.initial_candidate_count
        population = seed_population(contract=contract, world=world, policy=policy, model=self.model_routes.model_for(NexusModelRole.SEED), min_population_size=min_population_size)
        archives = ArchiveManager(policy.archive_schema)
        verification_plan = VerificationSynthesizer(model=self.model).synthesize({"goal": user_goal, "contract": contract.to_dict(), "mode": "project"})
        context_result = self.context_orchestrator.build_for_parents(
            contract=contract,
            snapshot=snapshot,
            world=world,
            parents=population.candidates[:3],
            archives=archives,
            model=self.model,
            mutation_instruction="initial_project_seed_verification",
        )
        verification_summaries = self._verify_project_population(snapshot, population.candidates, include_tests=include_tests)

        def verify_offspring(candidates: list[Any]) -> list[ProjectVerificationSummary]:
            summaries = self._verify_project_population(snapshot, candidates, include_tests=include_tests)
            verification_summaries.extend(summaries)
            return summaries

        project_world_payload = _world_to_dict_with_latent_metadata({"snapshot": snapshot.to_dict(), "project_world_model": world.to_dict()}, contract)
        observer = self._live_observer(mode="project", contract=contract, world=project_world_payload, max_rounds=budget.max_rounds, budget=budget.to_dict())
        result = evolve_once(
            population=population,
            archives=archives,
            policy=policy,
            contract=contract,
            world=world,
            budget=budget,
            model=self.model,
            observer=observer,
            cancellation_callback=cancellation_callback,
            offspring_verifier=verify_offspring,
            adaptive_config=adaptive_config,
            verification_plan=verification_plan,
        )
        run = NexusRunResult(
            mode="project",
            contract=contract.to_dict() | {"contract_hash": contract.contract_hash()},
            policy=result.policy.to_dict(),
            world=project_world_payload,
            evolution=result.to_dict(),
            pipeline_events=list(result.pipeline_events),
            context_protocol=context_result.to_dict(),
            verification_summaries=[summary.to_dict() for summary in verification_summaries],
        )
        run.evolution.setdefault("runtime_metadata", {})["model_routes"] = self.model_routes.public_summary()
        prompt_metadata = _model_prompt_metadata(self.model)
        if prompt_metadata:
            run.evolution["prompt_view_metadata"] = prompt_metadata
        _sync_runtime_round_metadata(run.evolution, result)
        _attach_fallback_events(run.evolution, finish_fallback_capture(fallback_token))
        run.artifacts = self._persist(run, result, contract=contract, world=project_world_payload, budget_history=result.budget_history, budget=budget)
        return run

    def resume_from_checkpoint(self, *, max_rounds: int | None = None) -> NexusRunResult:
        if self.output_dir is None:
            raise ValueError("resume_from_checkpoint requires output_dir")
        checkpoint_path = self.output_dir / "checkpoint.json"
        restored = CheckpointStore(checkpoint_path).restore_state()
        if restored is None:
            raise FileNotFoundError(checkpoint_path)
        checkpoint = restored["checkpoint"]
        mode = str(restored.get("mode") or checkpoint.mode or "text")
        population = restored["population"]
        archives = restored["archives"]
        policy = restored["policy"]
        contract = _contract_from_checkpoint(mode, restored.get("contract") or {})
        restored_artifact_policy_config = _artifact_policy_config_from_adaptive_state(restored.get("adaptive_state") or {})
        if restored_artifact_policy_config:
            previous_contract_hash = contract.contract_hash()
            previous_dynamic_hash = contract.dynamic_artifact_contract_hash()
            apply_artifact_policy_to_contract(contract, restored_artifact_policy_config, source="adaptive_state.resume")
            _rebase_population_contract_hashes(
                population,
                previous_contract_hash=previous_contract_hash,
                current_contract_hash=contract.contract_hash(),
                previous_dynamic_artifact_contract_hash=previous_dynamic_hash,
                current_dynamic_artifact_contract_hash=contract.dynamic_artifact_contract_hash(),
            )
        world = _world_from_checkpoint(mode, restored.get("world") or {})
        verification_summaries: list[dict[str, Any]] = []
        offspring_verifier = None
        if mode == "project":
            snapshot_data = _snapshot_payload_from_world(restored.get("world") or {})
            if snapshot_data:
                snapshot = ProjectSnapshot.from_dict(snapshot_data)

                def verify_offspring(candidates: list[Any]) -> list[ProjectVerificationSummary]:
                    summaries = self._verify_project_population(snapshot, candidates, include_tests=False)
                    verification_summaries.extend(summary.to_dict() for summary in summaries)
                    return summaries

                offspring_verifier = verify_offspring
        budget_data = dict(getattr(checkpoint, "budget", {}) or {})
        adaptive_resume = bool(budget_data.get("adaptive"))
        if max_rounds is not None:
            target_rounds = max(int(max_rounds), int(checkpoint.round or 0) + 1)
        elif adaptive_resume:
            previous_limit = int(budget_data.get("round_safety_limit") or checkpoint.max_rounds or 1)
            safety_window = max(1, previous_limit)
            target_rounds = int(checkpoint.round or 0) + safety_window
        else:
            target_rounds = max(int(checkpoint.max_rounds or checkpoint.round or 1), int(checkpoint.round or 0))
        budget = resume_evolution_budget(
            checkpoint_round=checkpoint.round,
            checkpoint_max_rounds=checkpoint.max_rounds,
            budget_data=budget_data,
            max_rounds=max_rounds,
        )
        budget.max_rounds = target_rounds
        budget.history = list(restored.get("budget_history") or [])
        verification_plan = _verification_plan_from_restored(restored, contract=contract, mode=mode, model=self.model)
        observer = self._live_observer(mode=mode, contract=contract, world=world, max_rounds=target_rounds, budget=budget.to_dict())
        result = evolve_once(
            population=population,
            archives=archives,
            policy=policy,
            contract=contract,
            world=world,
            budget=budget,
            model=self.model,
            observer=observer,
            offspring_verifier=offspring_verifier,
            adaptive_state=restored.get("adaptive_state") or {},
            verification_plan=verification_plan,
            fabric_state=restored.get("fabric") or {},
        )
        world_payload = _world_to_dict_with_latent_metadata(world, contract)
        run = NexusRunResult(
            mode=mode,
            contract=contract.to_dict() | {"contract_hash": contract.contract_hash()},
            policy=result.policy.to_dict(),
            world=world_payload,
            evolution=result.to_dict(),
            pipeline_events=list(result.pipeline_events),
            verification_summaries=verification_summaries,
        )
        prompt_metadata = _model_prompt_metadata(self.model)
        if prompt_metadata:
            run.evolution["prompt_view_metadata"] = prompt_metadata
        run.evolution.setdefault("runtime_metadata", {})["model_routes"] = self.model_routes.public_summary()
        _sync_runtime_round_metadata(run.evolution, result)
        run.artifacts = self._persist(run, result, contract=contract, world=world_payload, budget_history=budget.history, budget=budget)
        return run

    def _live_observer(self, *, mode: str, contract: Any, world: Any, max_rounds: int, budget: dict[str, Any] | None = None) -> Any | None:
        if self.output_dir is None:
            return None
        return LiveNexusStore(self.output_dir, mode=mode, contract=contract, world=world, max_rounds=max_rounds, budget=budget)

    def _verify_project_population(self, snapshot: ProjectSnapshot, candidates: list[Any], *, include_tests: bool = False) -> list[ProjectVerificationSummary]:
        return self.project_verification_service.verify_population(snapshot, candidates, include_tests=include_tests)

    def _persist(self, run: NexusRunResult, result: EvolutionLoopResult, *, contract: Any, world: Any, budget_history: list[dict[str, Any]], budget: EvolutionBudget | None = None) -> dict[str, str]:
        return self.persistence_service.persist(run, result, contract=contract, world=world, budget_history=budget_history, budget=budget)




def _resolve_budget_width_from_policy(budget: EvolutionBudget, policy: EvolutionPolicy) -> None:
    """Fill adaptive candidate width from model/policy when no explicit width was set.

    API/model profiles no longer bake in a candidate count or mutation width.
    A model-authored policy can set metadata/parent-selection width; otherwise
    the fallback derives a small width from the number of policy niches.
    """

    if int(getattr(budget, "branch_factor", 0) or 0) > 0:
        return
    configured = _policy_positive_int(policy, "mutation_branches_per_round") or _policy_positive_int(policy, "branch_factor")
    prefs = getattr(policy, "parent_selection_preferences", {}) if policy is not None else {}
    if configured is None and isinstance(prefs, dict):
        configured = positive_int(prefs.get("mutation_branches_per_round") or prefs.get("branch_factor"))
    if configured is None:
        niche_count = len({str(item).strip().lower() for item in getattr(policy, "candidate_niches", []) if str(item).strip()})
        configured = max(1, int(math.ceil(max(1, niche_count) ** 0.5)))
    budget.branch_factor = configured


def _policy_positive_int(policy: EvolutionPolicy, key: str) -> int | None:
    metadata = getattr(policy, "metadata", {}) if policy is not None else {}
    if not isinstance(metadata, dict):
        return None
    return positive_int(metadata.get(key))


def _build_text_world_model(packet: TextInputPacket, *, model: NexusModelLike | None) -> TextWorldModel:
    """Prefer a model-authored text world model; local extraction is a fallback."""

    if model is not None and hasattr(model, "build_text_world_model"):
        try:
            raw = model.build_text_world_model(packet=packet)
        except Exception:
            raw = None
        if isinstance(raw, TextWorldModel):
            return raw
        if isinstance(raw, dict):
            data = dict(raw)
            data.setdefault("kind", "text")
            data.setdefault("input_packet_id", packet.packet_id)
            return TextWorldModel.from_dict(data)
    return TextWorldModel.from_packet(packet)


def _model_prompt_metadata(model: NexusModelLike | None) -> dict[str, Any]:
    metadata = getattr(model, "metadata", None)
    if not isinstance(metadata, dict):
        return {}
    history = metadata.get("prompt_view_history")
    last = metadata.get("last_prompt_view")
    result: dict[str, Any] = {}
    if isinstance(last, dict):
        result["last"] = dict(last)
    if isinstance(history, list):
        result["history"] = [dict(item) for item in history[-20:] if isinstance(item, dict)]
    return result


def _world_to_dict_with_latent_metadata(world: Any, contract: Any) -> dict[str, Any]:
    if hasattr(world, "to_dict"):
        data = dict(world.to_dict())
    elif isinstance(world, dict):
        data = dict(world)
    else:
        data = {"raw_world": str(world)}
    metadata = dict(data.get("metadata") or {})
    contract_metadata = getattr(contract, "metadata", {}) if contract is not None else {}
    if isinstance(contract_metadata, dict):
        for key in ("latent_problem_state_summary", "latent_problem_state_hash"):
            if key in contract_metadata:
                metadata[key] = contract_metadata[key]
    if metadata:
        data["metadata"] = metadata
    return data



def _attach_fallback_events(evolution: dict[str, Any], events: list[dict[str, str]]) -> None:
    sanitized = [dict(event) for event in events if isinstance(event, dict)]
    evolution["fallback_events"] = sanitized
    evolution["fallback_event_count"] = len(sanitized)

def _sync_runtime_round_metadata(evolution: dict[str, Any], result: EvolutionLoopResult) -> None:
    metadata = dict(evolution.get("runtime_metadata") or {})
    round_budget = dict(metadata.get("round_budget") or evolution.get("round_budget") or {})
    if round_budget:
        round_budget["current_round"] = result.current_round
        round_budget["round_limit"] = result.max_rounds
        round_budget["stop_reason"] = result.stop_reason
        round_budget["completion_status"] = result.completion_status
        metadata["round_budget"] = round_budget
        evolution["runtime_metadata"] = metadata
    evolution["round_budget_runtime"] = {
        "current_round": result.current_round,
        "round_limit": result.max_rounds,
        "stop_reason": result.stop_reason,
        "completion_status": result.completion_status,
    }


def _verification_plan_from_restored(restored: dict[str, Any], *, contract: NexusObjectiveContract, mode: str, model: Any | None) -> VerificationPlan:
    adaptive_state = dict(restored.get("adaptive_state") or {})
    research = dict(adaptive_state.get("research_extensions") or {})
    plan = dict(research.get("verification_plan") or restored.get("verification_plan") or {})
    if plan:
        return VerificationPlan.from_dict(plan)
    return VerificationSynthesizer(model=model).synthesize({"goal": getattr(contract, "normalized_goal", "") or contract.to_dict(), "mode": mode, "resynthesized_from_checkpoint": True})


def _contract_from_checkpoint(mode: str, data: dict[str, Any]) -> NexusObjectiveContract:
    if mode == "project" or any(key in data for key in ["allowed_patch_scope", "implementation_files", "test_contracts"]):
        return NexusProjectObjectiveContract.from_dict(data)
    return NexusObjectiveContract.from_dict(data)


def _artifact_policy_config_from_adaptive_config(adaptive_config: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(adaptive_config or {})
    evidence = data.get("evidence") if isinstance(data.get("evidence"), dict) else {}
    return dict(evidence or {})


def _artifact_policy_config_from_adaptive_state(adaptive_state: dict[str, Any]) -> dict[str, Any]:
    state = dict(adaptive_state or {})
    config = state.get("config") if isinstance(state.get("config"), dict) else {}
    evidence = config.get("evidence") if isinstance(config.get("evidence"), dict) else {}
    return dict(evidence or {})


def _rebase_population_contract_hashes(
    population: Any,
    *,
    previous_contract_hash: str,
    current_contract_hash: str,
    previous_dynamic_artifact_contract_hash: str,
    current_dynamic_artifact_contract_hash: str,
) -> None:
    """Keep resumed candidates aligned with a policy overlay applied in memory.

    Historical checkpoint bytes are not rewritten.  Only restored candidates that
    clearly point at the pre-overlay contract hash are rebased so future
    verifier checks do not turn an approved ArtifactPolicy overlay into a stale
    contract failure for the whole population.
    """

    if not previous_contract_hash or previous_contract_hash == current_contract_hash:
        return
    for candidate in getattr(population, "candidates", []) or []:
        if getattr(candidate, "contract_hash", "") != previous_contract_hash:
            continue
        metadata = dict(getattr(candidate, "metadata", {}) or {})
        metadata["contract_hash_overlay_rebased"] = {
            "previous_contract_hash": previous_contract_hash,
            "current_contract_hash": current_contract_hash,
            "previous_dynamic_artifact_contract_hash": previous_dynamic_artifact_contract_hash,
            "current_dynamic_artifact_contract_hash": current_dynamic_artifact_contract_hash,
            "reason": "adaptive_artifact_policy_overlay_on_resume",
        }
        candidate.metadata = metadata
        candidate.contract_hash = current_contract_hash


def _world_from_checkpoint(mode: str, data: dict[str, Any]) -> Any:
    if mode == "project":
        if "project_world_model" in data or "snapshot" in data:
            return {
                "snapshot": dict(data.get("snapshot") or {}),
                "project_world_model": ProjectWorldModel.from_dict(dict(data.get("project_world_model") or data)).to_dict(),
            }
        return ProjectWorldModel.from_dict(data)
    return TextWorldModel.from_dict(data)


def _snapshot_payload_from_world(data: dict[str, Any]) -> dict[str, Any]:
    snapshot = data.get("snapshot") if isinstance(data.get("snapshot"), dict) else {}
    if snapshot:
        return dict(snapshot)
    if data.get("file_manifest") or data.get("root_path"):
        return dict(data)
    return {}


__all__ = ["NexusRuntime", "NexusRunResult"]
