"""Nexus runtime entrypoint."""
from __future__ import annotations

import json
import math
import hashlib
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import (
    NexusObjectiveContract,
    NexusObjectiveContractBuilder,
    NexusProjectObjectiveContract,
    apply_artifact_policy_to_contract,
)
from cognitive_evolve_runtime.inputs.project_map import ProjectWorldModel
from cognitive_evolve_runtime.inputs.project_snapshot import ProjectSnapshot
from cognitive_evolve_runtime.inputs.text_packet import TextInputPacket, TextWorldModel
from cognitive_evolve_runtime.durable.async_writer import WriteBehindObserver
from cognitive_evolve_runtime.durable.resume_assembly import restore_archives, restore_budget, restore_mode_specific, restore_population
from cognitive_evolve_runtime.llm.call_ledger import ledger_summary
from cognitive_evolve_runtime.llm.session import current_llm_session
from cognitive_evolve_runtime.nexus.context_protocol import ContextOrchestrator
from cognitive_evolve_runtime.nexus.handoff import load_inherited_gene_entries
from cognitive_evolve_runtime.nexus.live_store import LiveNexusStore
from cognitive_evolve_runtime.nexus.budget_factory import evolution_budget_from_params
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, EvolutionLoopResult, evolve_once, seed_population
from cognitive_evolve_runtime.nexus.model_adapter import StructuredModelAdapter
from cognitive_evolve_runtime.nexus.model_routes import NexusModelRole, NexusModelRoutes, coerce_model_routes
from cognitive_evolve_runtime.llm.model_spec import LLMModelSpec
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy, EvolutionPolicyBuilder
from cognitive_evolve_runtime.verification.synthesizer import VerificationSynthesizer
from cognitive_evolve_runtime.verification.types import VerificationPlan
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike
from cognitive_evolve_runtime.nexus.project_verification import ProjectVerificationSummary
from cognitive_evolve_runtime.nexus.fallbacks import capture_fallback_events, record_fallback
from cognitive_evolve_runtime.nexus.runtime_options import option_bool, resolve_runtime_options, restore_runtime_options
from cognitive_evolve_runtime.nexus.runtime_services import NexusPersistenceService, NexusProjectVerificationService
from cognitive_evolve_runtime.nexus._shared import MODEL_BOUNDARY_ERRORS, positive_int
from cognitive_evolve_runtime.nexus.stop_reasons import normalize_external_review_stop_reason
from cognitive_evolve_runtime.persistence.checkpoint import CheckpointStore, contract_payload_for_persistence
from cognitive_evolve_runtime.persistence.transactional_snapshot import read_snapshot_json, snapshot_reader


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
        initial_candidates: list[CandidateGenome | dict[str, Any]] | None = None,
        inherited_handoff_path: str | Path | None = None,
        inherited_candidate_ids: list[str] | None = None,
    ) -> NexusRunResult:
        with capture_fallback_events() as fallback_events:
            self._bind_llm_artifact_scope()
            inherited_gene_entries = load_inherited_gene_entries(inherited_handoff_path, inherited_candidate_ids)
            runtime_options = resolve_runtime_options(request_options={"seed.family_priority_source": "model_authored_search_space"})
            packet = TextInputPacket.from_text(text)
            world = _build_text_world_model(packet, model=self.model)
            goal = user_goal or packet.raw_text or "text evolution task"
            artifact_policy_config = _artifact_policy_config_from_adaptive_config(adaptive_config)
            contract = self.contract_builder.build_text_contract(
                user_goal=packet.raw_text or goal,
                packet=packet,
                world=world,
                model=self.model,
                artifact_policy_config=artifact_policy_config,
            )
            provided_context = _text_provided_context(
                contract,
                initial_candidates=initial_candidates,
                inherited_gene_entries=inherited_gene_entries,
            )
            policy = self.policy_builder.build(contract=contract, world=world, model=self.model)
            _apply_search_mechanics(policy, runtime_options)
            budget = budget or evolution_budget_from_params(
                max_rounds=max_rounds,
                branch_factor=branch_factor,
                initial_candidate_count=min_population_size or 0,
                stop_policy=stop_policy,
                min_rounds_before_stop=min_rounds_before_stop,
            )
            _resolve_budget_width_from_policy(budget, policy)
            min_population_size = min_population_size if min_population_size is not None else budget.initial_candidate_count
            population = seed_population(
                contract=contract,
                world=world,
                policy=policy,
                model=self.model_routes.model_for(NexusModelRole.SEED),
                min_population_size=min_population_size,
                provided_context=provided_context,
                initial_candidates=initial_candidates,
            )
            archives = ArchiveManager(policy.archive_schema)
            world_payload = _world_to_dict_with_latent_metadata(world, contract)
            observer = self._live_observer(mode="text", contract=contract, world=world_payload, max_rounds=budget.max_rounds, budget=budget.to_dict(), runtime_options=runtime_options)
            if observer is not None:
                observer({"phase": "post_seeding", "round": 0, "population": population, "archives": archives, "policy": policy, "progress_event": {"type": "nexus_post_seeding_checkpoint", "round": 0, "phase": "post_seeding", "max_rounds": budget.max_rounds}, "runtime_options": runtime_options})
            verification_plan = VerificationSynthesizer(model=self.model).synthesize({"goal": goal, "contract": contract.to_dict()})
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
                provided_context=provided_context,
            )
            run = NexusRunResult(
                mode="text",
                contract=contract_payload_for_persistence(contract) | {"contract_hash": contract.contract_hash()},
                policy=result.policy.to_dict(),
                world=world_payload,
                evolution=result.to_dict(),
                pipeline_events=list(result.pipeline_events),
            )
            run.evolution["runtime_options"] = runtime_options
            run.evolution.setdefault("runtime_metadata", {})["model_routes"] = self.model_routes.public_summary()
            if runtime_metadata:
                run.evolution["runtime_metadata"].update(dict(runtime_metadata))
            prompt_metadata = _model_prompt_metadata(self.model)
            if prompt_metadata:
                run.evolution["prompt_view_metadata"] = prompt_metadata
            _sync_runtime_round_metadata(run.evolution, result)
            _attach_fallback_events(run.evolution, fallback_events)
            _attach_limit_pressure(run.evolution, observer)
            run.artifacts = self._persist(run, result, contract=contract, world=world_payload, budget_history=result.budget_history, budget=budget, runtime_options=runtime_options)
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
        inherited_handoff_path: str | Path | None = None,
        inherited_candidate_ids: list[str] | None = None,
    ) -> NexusRunResult:
        with capture_fallback_events() as fallback_events:
            self._bind_llm_artifact_scope()
            inherited_gene_entries = load_inherited_gene_entries(inherited_handoff_path, inherited_candidate_ids)
            runtime_options = resolve_runtime_options(request_options={"verification.include_tests": bool(include_tests), "seed.family_priority_source": "model_authored_search_space"})
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
            _enable_project_latent_exploration(contract)
            policy = self.policy_builder.build(contract=contract, world=world, model=self.model)
            _apply_search_mechanics(policy, runtime_options)
            budget = budget or evolution_budget_from_params(
                max_rounds=max_rounds,
                branch_factor=branch_factor,
                initial_candidate_count=min_population_size or 0,
                stop_policy=stop_policy,
                min_rounds_before_stop=min_rounds_before_stop,
            )
            _resolve_budget_width_from_policy(budget, policy)
            min_population_size = min_population_size if min_population_size is not None else budget.initial_candidate_count
            archives = ArchiveManager(policy.archive_schema)
            # Ground source-binding resolution in the real project root (runtime-only;
            # never serialized into the archive payload).
            archives.project_root = snapshot.root_path
            initial_context_result = self.context_orchestrator.build_for_parents(
                contract=contract,
                snapshot=snapshot,
                world=world,
                parents=[],
                archives=archives,
                model=None,
                mutation_instruction="initial_project_seed",
            )
            provided_context = initial_context_result.to_source_context()
            if inherited_gene_entries:
                provided_context["inherited_gene_entries"] = inherited_gene_entries

            def refresh_project_context(parents: list[CandidateGenome], mutation_instruction: str) -> dict[str, Any]:
                refreshed = self.context_orchestrator.build_for_parents(
                    contract=contract,
                    snapshot=snapshot,
                    world=world,
                    parents=parents,
                    archives=archives,
                    model=None,
                    mutation_instruction=mutation_instruction,
                ).to_source_context()
                if inherited_gene_entries:
                    refreshed["inherited_gene_entries"] = inherited_gene_entries
                return refreshed

            population = seed_population(
                contract=contract,
                world=world,
                policy=policy,
                model=self.model_routes.model_for(NexusModelRole.SEED),
                min_population_size=min_population_size,
                provided_context=provided_context,
            )
            project_world_payload = _world_to_dict_with_latent_metadata({"snapshot": snapshot.to_dict(), "project_world_model": world.to_dict()}, contract)
            observer = self._live_observer(mode="project", contract=contract, world=project_world_payload, max_rounds=budget.max_rounds, budget=budget.to_dict(), runtime_options=runtime_options)
            if observer is not None:
                observer({"phase": "post_seeding", "round": 0, "population": population, "archives": archives, "policy": policy, "progress_event": {"type": "nexus_post_seeding_checkpoint", "round": 0, "phase": "post_seeding", "max_rounds": budget.max_rounds}, "runtime_options": runtime_options})
            verification_plan = VerificationSynthesizer(model=self.model).synthesize({"goal": user_goal, "contract": contract.to_dict(), "mode": "project"})
            applied_overlays = _applied_overlays(artifact_policy_config, contract)
            verification_summaries = self._verify_project_population(
                snapshot,
                population.candidates,
                include_tests=include_tests,
                contract=contract,
                applied_overlays=applied_overlays,
            )

            def verify_offspring(candidates: list[Any]) -> list[ProjectVerificationSummary]:
                summaries = self._verify_project_population(
                    snapshot,
                    candidates,
                    include_tests=include_tests,
                    contract=contract,
                    applied_overlays=applied_overlays,
                )
                verification_summaries.extend(summaries)
                return summaries

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
                provided_context=provided_context,
                context_provider=refresh_project_context,
            )
            run = NexusRunResult(
                mode="project",
                contract=contract_payload_for_persistence(contract) | {"contract_hash": contract.contract_hash()},
                policy=result.policy.to_dict(),
                world=project_world_payload,
                evolution=result.to_dict(),
                pipeline_events=list(result.pipeline_events),
                context_protocol=initial_context_result.to_dict(),
                verification_summaries=[summary.to_dict() for summary in verification_summaries],
            )
            run.evolution["runtime_options"] = runtime_options
            run.evolution.setdefault("runtime_metadata", {})["model_routes"] = self.model_routes.public_summary()
            prompt_metadata = _model_prompt_metadata(self.model)
            if prompt_metadata:
                run.evolution["prompt_view_metadata"] = prompt_metadata
            _sync_runtime_round_metadata(run.evolution, result)
            _attach_fallback_events(run.evolution, fallback_events)
            _attach_limit_pressure(run.evolution, observer)
            run.artifacts = self._persist(run, result, contract=contract, world=project_world_payload, budget_history=result.budget_history, budget=budget, runtime_options=runtime_options)
            return run

    def resume_from_checkpoint(self, *, max_rounds: int | None = None) -> NexusRunResult:
        with capture_fallback_events() as fallback_events:
            self._bind_llm_artifact_scope()
            if self.output_dir is None:
                raise ValueError("resume_from_checkpoint requires output_dir")
            with snapshot_reader(self.output_dir) as snapshot_root:
                checkpoint_path = snapshot_root / "checkpoint.json"
                restored = CheckpointStore(checkpoint_path).restore_state()
                if restored is None:
                    raise FileNotFoundError(checkpoint_path)
                checkpoint = restored["checkpoint"]
                mode = restore_mode_specific(restored, checkpoint)
                snapshot: ProjectSnapshot | None = None
                if mode == "project":
                    snapshot_data = _snapshot_payload_from_world(restored.get("world") or {})
                    if snapshot_data:
                        snapshot = ProjectSnapshot.from_dict(snapshot_data)
                        current_root_hash = ProjectSnapshot.from_path(snapshot.root_path).root_hash
                        if current_root_hash != snapshot.root_hash:
                            raise ValueError(
                                "project source drift detected on resume: "
                                f"checkpoint root_hash={snapshot.root_hash}, current root_hash={current_root_hash}"
                            )
                budget_data = dict(getattr(checkpoint, "budget", {}) or {})
                terminal_stop = normalize_external_review_stop_reason(budget_data.get("stop_reason"))
                resume_does_not_extend = max_rounds is None or int(max_rounds) <= int(checkpoint.max_rounds or 0)
                if terminal_stop and resume_does_not_extend:
                    run_result_path = snapshot_root / "run-result.json"
                    if not run_result_path.exists():
                        raise FileNotFoundError(f"terminal checkpoint resume requires persisted run-result.json: {run_result_path}")
                    payload = read_snapshot_json(snapshot_root, "run-result.json")
                    return NexusRunResult(**payload)
            runtime_options = restore_runtime_options(persisted=restored.get("runtime_options") or getattr(checkpoint, "runtime_options", {}), overrides={})
            _restore_legacy_search_mechanics(runtime_options)
            population = restore_population(restored)
            archives = restore_archives(restored)
            policy = restored["policy"]
            _apply_search_mechanics(policy, runtime_options)
            contract = _contract_from_checkpoint(mode, restored.get("contract") or {})
            if mode == "project":
                _enable_project_latent_exploration(contract)
            restored_artifact_policy_config = _artifact_policy_config_from_adaptive_state(restored.get("adaptive_state") or {})
            if restored_artifact_policy_config:
                apply_artifact_policy_to_contract(contract, restored_artifact_policy_config, source="adaptive_state.resume")
            world = _world_from_checkpoint(mode, restored.get("world") or {})
            verification_summaries: list[dict[str, Any]] = []
            offspring_verifier = None
            context_provider = None
            if mode == "project":
                if snapshot is not None:
                    # Re-ground source-binding resolution after resume (runtime-only).
                    archives.project_root = snapshot.root_path

                    def verify_offspring(candidates: list[Any]) -> list[ProjectVerificationSummary]:
                        summaries = self._verify_project_population(snapshot, candidates, include_tests=option_bool(runtime_options, "verification.include_tests", default=False), contract=contract, applied_overlays=_applied_overlays(restored_artifact_policy_config, contract))
                        verification_summaries.extend(summary.to_dict() for summary in summaries)
                        return summaries

                    offspring_verifier = verify_offspring
                    context_world = ProjectWorldModel.from_dict(dict(world.get("project_world_model") or world)) if isinstance(world, dict) else world
                    context_result = self.context_orchestrator.build_for_parents(
                        contract=contract,
                        snapshot=snapshot,
                        world=context_world,
                        parents=population.candidates[:3],
                        archives=archives,
                        model=None,
                        mutation_instruction="resume_project_context",
                    )
                    provided_context = context_result.to_source_context()

                    def refresh_project_context(parents: list[CandidateGenome], mutation_instruction: str) -> dict[str, Any]:
                        return self.context_orchestrator.build_for_parents(
                            contract=contract,
                            snapshot=snapshot,
                            world=context_world,
                            parents=parents,
                            archives=archives,
                            model=None,
                            mutation_instruction=mutation_instruction,
                        ).to_source_context()

                    context_provider = refresh_project_context
                else:
                    provided_context = dict(restored.get("provided_context") or {})
            else:
                restored_context = dict(restored.get("provided_context") or {})
                initial_candidates = [
                    candidate
                    for candidate in population.candidates
                    if isinstance(getattr(candidate, "metadata", None), dict)
                    and candidate.metadata.get("operator_provided_incumbent") is True
                ]
                provided_context = {
                    **restored_context,
                    **_text_provided_context(contract, initial_candidates=initial_candidates),
                }
            budget = restore_budget(restored, checkpoint, max_rounds=max_rounds)
            target_rounds = budget.max_rounds
            seed_harvest = policy.metadata.get("seed_harvest", {}) if isinstance(policy.metadata, dict) else {}
            seed_failure = seed_harvest.get("fatal_model_error") or seed_harvest.get("model_error")
            seed_model = self.model_routes.model_for(NexusModelRole.SEED)
            if int(budget.current_round or 0) == 0 and not population.candidates and seed_failure and seed_model is not None:
                population = seed_population(
                    contract=contract,
                    world=world,
                    policy=policy,
                    model=seed_model,
                    min_population_size=budget.initial_candidate_count or None,
                    provided_context=provided_context,
                )
            verification_plan = _verification_plan_from_restored(restored, contract=contract, mode=mode, model=self.model)
            observer = self._live_observer(mode=mode, contract=contract, world=world, max_rounds=target_rounds, budget=budget.to_dict(), runtime_options=runtime_options)
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
                elo_state=restored.get("elo") or {},
                verification_plan=verification_plan,
                fabric_state=restored.get("fabric") or {},
                provided_context=provided_context,
                context_provider=context_provider,
            )
            world_payload = _world_to_dict_with_latent_metadata(world, contract)
            run = NexusRunResult(
                mode=mode,
                contract=contract_payload_for_persistence(contract) | {"contract_hash": contract.contract_hash()},
                policy=result.policy.to_dict(),
                world=world_payload,
                evolution=result.to_dict(),
                pipeline_events=list(result.pipeline_events),
                verification_summaries=verification_summaries,
            )
            prompt_metadata = _model_prompt_metadata(self.model)
            if prompt_metadata:
                run.evolution["prompt_view_metadata"] = prompt_metadata
            run.evolution["runtime_options"] = runtime_options
            run.evolution.setdefault("runtime_metadata", {})["model_routes"] = self.model_routes.public_summary()
            _sync_runtime_round_metadata(run.evolution, result)
            _attach_fallback_events(run.evolution, fallback_events)
            _attach_limit_pressure(run.evolution, observer)
            run.artifacts = self._persist(run, result, contract=contract, world=world_payload, budget_history=budget.history, budget=budget, runtime_options=runtime_options)
            return run

    def _live_observer(self, *, mode: str, contract: Any, world: Any, max_rounds: int, budget: dict[str, Any] | None = None, runtime_options: dict[str, Any] | None = None) -> Any | None:
        if self.output_dir is None:
            return None
        store = LiveNexusStore(self.output_dir, mode=mode, contract=contract, world=world, max_rounds=max_rounds, budget=budget, runtime_options=runtime_options)
        persistence_mode = str(dict(runtime_options or {}).get("persistence.mode") or os.environ.get("COGEV_PERSISTENCE_MODE") or "async_full").strip().lower()
        if persistence_mode == "async_full":
            return WriteBehindObserver(store)
        if persistence_mode == "sync_full":
            return store
        raise ValueError("COGEV_PERSISTENCE_MODE must be async_full or sync_full")

    def _bind_llm_artifact_scope(self) -> None:
        if self.output_dir is None:
            return
        session = current_llm_session()
        session.response_dir = str(self.output_dir)
        if not session.run_id:
            configured = str(os.environ.get("COGEV_RUN_ID") or "").strip()
            session.run_id = configured or "nexus-" + hashlib.sha256(
                str(self.output_dir.expanduser().resolve()).encode("utf-8")
            ).hexdigest()[:20]

    def _verify_project_population(self, snapshot: ProjectSnapshot, candidates: list[Any], *, include_tests: bool = False, contract: Any | None = None, applied_overlays: dict[str, Any] | None = None) -> list[ProjectVerificationSummary]:
        verification_context = _verification_context(contract=contract, applied_overlays=applied_overlays)
        allowed_patch_scope = [str(item) for item in getattr(contract, "allowed_patch_scope", []) or [] if str(item).strip()]
        return self.project_verification_service.verify_population(snapshot, candidates, include_tests=include_tests, verification_context=verification_context, allowed_patch_scope=allowed_patch_scope)

    def _persist(self, run: NexusRunResult, result: EvolutionLoopResult, *, contract: Any, world: Any, budget_history: list[dict[str, Any]], budget: EvolutionBudget | None = None, runtime_options: dict[str, Any] | None = None) -> dict[str, str]:
        return self.persistence_service.persist(run, result, contract=contract, world=world, budget_history=budget_history, budget=budget, runtime_options=runtime_options)




def _enable_project_theory_advisory_pressure(policy: EvolutionPolicy) -> None:
    metadata = policy.metadata if isinstance(policy.metadata, dict) else {}
    theory = dict(metadata.get("theory") or {})
    producers = dict(theory.get("producers") or {})
    weights = dict(theory.get("weights") or {})
    theory["enabled"] = True
    for name in ("mdl", "boed", "geometry"):
        producers[name] = True
    for name, weight in {"mdl": 0.02, "boed": 0.015, "geometry": 0.015}.items():
        weights.setdefault(name, weight)
    theory["producers"] = producers
    theory["weights"] = weights
    metadata["theory"] = theory
    policy.metadata = metadata


def _enable_project_latent_exploration(contract: NexusObjectiveContract) -> None:
    metadata = contract.metadata if isinstance(contract.metadata, dict) else {}
    metadata["latent_objective_enabled"] = True
    contract.metadata = metadata


def _apply_search_mechanics(policy: EvolutionPolicy, runtime_options: dict[str, Any]) -> None:
    metadata = policy.metadata if isinstance(policy.metadata, dict) else {}
    metadata["offspring_parallel_mode"] = str(runtime_options["search.offspring_parallel_mode"])
    policy.metadata = metadata


def _restore_legacy_search_mechanics(runtime_options: dict[str, Any]) -> None:
    missing_offspring_mode = "search.offspring_parallel_mode" not in runtime_options
    missing_persistence_mode = "persistence.mode" not in runtime_options
    if not (missing_offspring_mode or missing_persistence_mode):
        return
    sources = dict(runtime_options.get("_sources") or {})
    if missing_offspring_mode:
        runtime_options["search.offspring_parallel_mode"] = "single_batch"
        sources["search.offspring_parallel_mode"] = "legacy_checkpoint_default"
    if missing_persistence_mode:
        runtime_options["persistence.mode"] = "sync_full"
        sources["persistence.mode"] = "legacy_checkpoint_default"
    runtime_options["legacy_mechanics_restored"] = True
    runtime_options["_sources"] = sources


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


def _text_provided_context(
    contract: NexusObjectiveContract,
    *,
    initial_candidates: list[CandidateGenome | dict[str, Any]] | None = None,
    inherited_gene_entries: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    frozen_spec = dict(getattr(contract, "frozen_spec", {}) or {})
    candidates: list[dict[str, Any]] = []
    for candidate in initial_candidates or []:
        if isinstance(candidate, CandidateGenome):
            candidates.append(candidate.to_dict())
        elif isinstance(candidate, dict):
            candidates.append(dict(candidate))
        else:
            raise TypeError("initial_candidates must contain CandidateGenome or dict values")
    context = {
        "frozen_spec": frozen_spec,
        "initial_candidates": candidates,
    }
    if inherited_gene_entries:
        context["inherited_gene_entries"] = list(inherited_gene_entries)
    return context


def _build_text_world_model(packet: TextInputPacket, *, model: NexusModelLike | None) -> TextWorldModel:
    """Prefer a model-authored text world model; local extraction is a fallback."""

    if model is not None and hasattr(model, "build_text_world_model"):
        exceptions = getattr(model, "fallback_exceptions", MODEL_BOUNDARY_ERRORS)
        if not isinstance(exceptions, tuple) or not all(isinstance(item, type) and issubclass(item, Exception) for item in exceptions):
            exceptions = MODEL_BOUNDARY_ERRORS
        try:
            raw = model.build_text_world_model(packet=packet)
        except exceptions as exc:
            record_fallback(
                stage="text_world_model",
                reason=exc.__class__.__name__,
                detail=str(exc),
                target=getattr(model, "identity", type(model).__name__),
            )
            raw = None
        if isinstance(raw, TextWorldModel):
            if raw.goal_summary.strip():
                return TextWorldModel.from_dict(
                    {
                        **raw.to_dict(),
                        "kind": "text",
                        "input_packet_id": packet.packet_id,
                    }
                )
            record_fallback(
                stage="text_world_model",
                reason="empty_goal_summary",
                detail="compatible model returned a text world without goal_summary",
                target=getattr(model, "identity", type(model).__name__),
            )
        if isinstance(raw, dict):
            data = dict(raw)
            data["kind"] = "text"
            data["input_packet_id"] = packet.packet_id
            try:
                world = TextWorldModel.from_dict(data)
            except (TypeError, ValueError) as exc:
                record_fallback(
                    stage="text_world_model",
                    reason=exc.__class__.__name__,
                    detail=str(exc),
                    target=getattr(model, "identity", type(model).__name__),
                )
            else:
                if world.goal_summary.strip():
                    return world
                record_fallback(
                    stage="text_world_model",
                    reason="empty_goal_summary",
                    detail="compatible model returned an offspring-shaped or empty text world",
                    target=getattr(model, "identity", type(model).__name__),
                )
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


def _attach_limit_pressure(evolution: dict[str, Any], observer: Any | None) -> None:
    events = current_llm_session().snapshot()
    physical_ids = {
        str(event.get("physical_call_id"))
        for event in events
        if event.get("cache_replayed") is not True and str(event.get("physical_call_id") or "")
    }
    physical_without_id = sum(
        1
        for event in events
        if not event.get("physical_call_id") and event.get("cache_replayed") is not True
    )
    pressure: dict[str, Any] = dict(evolution.get("limit_pressure") or {})
    pressure.update(
        {
            "provider_calls": len(physical_ids) + physical_without_id,
            "model_boundary_errors": sum(1 for event in events if event.get("error_type")),
            "cache_replayed": sum(1 for event in events if event.get("cache_replayed") is True),
        }
    )
    harvest_raw = 0
    harvest_accepted = 0
    for record in evolution.get("budget_history", []) if isinstance(evolution.get("budget_history"), list) else []:
        plan = record.get("generation_plan") if isinstance(record, dict) and isinstance(record.get("generation_plan"), dict) else {}
        harvest = plan.get("offspring_harvest") if isinstance(plan.get("offspring_harvest"), dict) else {}
        accepted = int(harvest.get("accepted_count") or 0)
        rejected = int(harvest.get("rejected_count") or 0)
        harvest_accepted += accepted
        harvest_raw += accepted + rejected
        slot_errors = harvest.get("slot_errors") if isinstance(harvest.get("slot_errors"), list) else []
        pressure["model_boundary_errors"] += len(slot_errors)
    error = evolution.get("error") if isinstance(evolution.get("error"), dict) else {}
    if error and str(error.get("type") or "").endswith(("LLMResponseError", "ModelResponseSchemaError", "LLMConfigurationError")):
        pressure["model_boundary_errors"] += 1
    pressure["harvest_raw"] = harvest_raw
    pressure["harvest_accepted"] = harvest_accepted
    ledger = ledger_summary()
    pressure["max_observed_concurrent_calls"] = int(ledger.get("max_observed_concurrent_calls") or 0)
    if observer is not None and hasattr(observer, "telemetry"):
        pressure.update(observer.telemetry())
    evolution["limit_pressure"] = pressure

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
    legacy_research = dict(adaptive_state.get("research_extensions") or {})
    plan = dict(adaptive_state.get("verification_plan") or legacy_research.get("verification_plan") or restored.get("verification_plan") or {})
    if plan:
        return VerificationPlan.from_dict(plan)
    return VerificationSynthesizer(model=model).synthesize({"goal": getattr(contract, "normalized_goal", "") or contract.to_dict(), "mode": mode, "resynthesized_from_checkpoint": True})


def _contract_from_checkpoint(mode: str, data: dict[str, Any]) -> NexusObjectiveContract:
    if mode == "project" or any(key in data for key in ["allowed_patch_scope", "implementation_files", "test_contracts"]):
        return NexusProjectObjectiveContract.from_dict(data)
    return NexusObjectiveContract.from_dict(data)


def _verification_context(*, contract: Any | None, applied_overlays: dict[str, Any] | None = None) -> dict[str, Any]:
    if contract is None:
        return {}
    context: dict[str, Any] = {}
    if hasattr(contract, "contract_hash"):
        context["verification_contract_hash"] = contract.contract_hash()
    if hasattr(contract, "dynamic_artifact_contract_hash"):
        context["verification_dynamic_artifact_contract_hash"] = contract.dynamic_artifact_contract_hash()
    overlays = dict(applied_overlays or {})
    if overlays:
        context["applied_overlays"] = overlays
    return context


def _applied_overlays(config: dict[str, Any], contract: Any | None) -> dict[str, Any]:
    if not config:
        return {}
    value = contract.dynamic_artifact_contract_hash() if hasattr(contract, "dynamic_artifact_contract_hash") else dict(config)
    return {"artifact-policy": value}


def _artifact_policy_config_from_adaptive_config(adaptive_config: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(adaptive_config or {})
    evidence = data.get("evidence") if isinstance(data.get("evidence"), dict) else {}
    return dict(evidence or {})


def _artifact_policy_config_from_adaptive_state(adaptive_state: dict[str, Any]) -> dict[str, Any]:
    state = dict(adaptive_state or {})
    config = state.get("config") if isinstance(state.get("config"), dict) else {}
    evidence = config.get("evidence") if isinstance(config.get("evidence"), dict) else {}
    return dict(evidence or {})


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
