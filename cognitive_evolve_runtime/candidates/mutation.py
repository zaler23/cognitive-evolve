"""Task-agnostic mutation operators for evolvable genomes."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any

from .genome import CandidateFate, CandidateGenome
from .patch_merge import merge_patch_sets
from .project_candidate import PatchOperation, ProjectCandidateGenome
from cognitive_evolve_runtime.core.serialization import coerce_dict, coerce_str_list, stable_hash
from cognitive_evolve_runtime.core.scalars import bounded_score
from cognitive_evolve_runtime.theory.bandit import OperatorArmStats, suggest_budget_allocation


class MutationOperator:
    DEEPEN = "Deepen"
    REPAIR = "Repair"
    SIMPLIFY = "Simplify"
    SPECIALIZE = "Specialize"
    GENERALIZE = "Generalize"
    INVERT = "Invert"
    TRANSFER = "Transfer"
    RARE_INJECT = "RareInject"
    CROSSOVER = "CrossOver"
    ADVERSARIAL_PATCH = "AdversarialPatch"
    TOOL_GROUND = "ToolGround"
    CORE_EXTRACTION = "CoreExtraction"
    SCAFFOLD_REMOVAL = "ScaffoldRemoval"
    DORMANT_REACTIVATION = "DormantReactivation"
    LINEAGE_RESTART = "LineageRestart"
    INSTANTIATE_FORMAL_ARTIFACT = "InstantiateFormalArtifact"
    DISCHARGE_OBLIGATION = "DischargeObligation"
    CASE_SPLIT = "CaseSplit"
    CONSTRUCT_WITNESS = "ConstructWitness"
    ROUTE_KILL = "RouteKill"

    ALL = [
        DEEPEN,
        REPAIR,
        SIMPLIFY,
        SPECIALIZE,
        GENERALIZE,
        INVERT,
        TRANSFER,
        RARE_INJECT,
        CROSSOVER,
        ADVERSARIAL_PATCH,
        TOOL_GROUND,
        CORE_EXTRACTION,
        SCAFFOLD_REMOVAL,
        DORMANT_REACTIVATION,
        LINEAGE_RESTART,
        INSTANTIATE_FORMAL_ARTIFACT,
        DISCHARGE_OBLIGATION,
        CASE_SPLIT,
        CONSTRUCT_WITNESS,
        ROUTE_KILL,
    ]


class MoveContractError(ValueError):
    """A typed move cannot be executed within its declared input domain."""


SEMANTIC_HINT = "semantic_hint"
TYPED_MOVE_KINDS = frozenset(
    {
        "path_assignment",
        "field_split",
        "constraint_strengthening",
        "patch.add",
        "patch.replace",
        "patch.delete",
        "patch.append",
        "proof.instantiate_object",
        "proof.split_obligation",
        "proof.construct_witness",
    }
)


@dataclass
class MutationPlan:
    operator: str
    parent_ids: list[str] = field(default_factory=list)
    instruction: str = ""
    rarity_seed: str = ""
    expected_gene_effects: list[str] = field(default_factory=list)
    move_kind: str = SEMANTIC_HINT
    input_domain: str = ""
    declared_invariant_refs: list[str] = field(default_factory=list)
    expected_delta: dict[str, Any] = field(default_factory=dict)
    actual_delta: dict[str, Any] = field(default_factory=dict)
    receipt_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MutationPlan":
        return cls(
            operator=str(data.get("operator") or MutationOperator.DEEPEN),
            parent_ids=coerce_str_list(data.get("parent_ids")),
            instruction=str(data.get("instruction") or ""),
            rarity_seed=str(data.get("rarity_seed") or ""),
            expected_gene_effects=coerce_str_list(data.get("expected_gene_effects")),
            move_kind=str(data.get("move_kind") or SEMANTIC_HINT),
            input_domain=str(data.get("input_domain") or ""),
            declared_invariant_refs=coerce_str_list(data.get("declared_invariant_refs")),
            expected_delta=coerce_dict(data.get("expected_delta")),
            actual_delta=coerce_dict(data.get("actual_delta")),
            receipt_refs=coerce_str_list(data.get("receipt_refs")),
            metadata=coerce_dict(data.get("metadata")),
        )


class MutationEngine:
    """Small deterministic mutation engine used by tests and fake-model runs."""

    def mutate(self, parent: CandidateGenome, plan: MutationPlan) -> CandidateGenome:
        try:
            validate_move_plan(parent, plan)
        except MoveContractError as exc:
            plan.metadata["move_rejection"] = {
                "status": "rejected",
                "reason": str(exc),
                "parent_id": parent.id,
            }
            raise
        operator = plan.operator if plan.operator in MutationOperator.ALL else MutationOperator.DEEPEN
        inherited = [parent.extract_inheritable_gene_summary()] + list(parent.inherited_genes)
        artifact = self._mutated_artifact(parent, plan)
        metadata = _inherited_mutation_metadata(parent, plan)
        if plan.operator not in MutationOperator.ALL:
            metadata.setdefault(
                "action_fallback",
                {
                    "raw_action": str(plan.operator or ""),
                    "fallback_operator": MutationOperator.DEEPEN,
                    "reason": "unknown_action",
                },
            )
        base_kwargs = dict(
            parent_ids=[parent.id],
            generation=parent.generation + 1,
            lineage=parent.lineage + [],
            artifact=artifact,
            artifact_type=parent.artifact_type,
            concise_claim=self._concise_claim(parent, operator),
            core_mechanism=self._core_mechanism(parent, plan),
            assumptions=list(parent.assumptions),
            missing_parts=list(parent.missing_parts),
            uncertainty_notes=list(parent.uncertainty_notes),
            edge_knowledge_seeds=self._edge_seeds(parent, plan),
            inherited_genes=[gene for gene in inherited if gene],
            mutation_history=parent.mutation_history + [operator],
            tool_results=[],
            verification_trace=[],
            formal_artifacts=list(parent.formal_artifacts),
            proof_obligations=list(parent.proof_obligations),
            obligation_delta=dict(parent.obligation_delta),
            evidence_refs=list(parent.evidence_refs),
            source_bindings=self._source_bindings(parent, plan),
            evidence_delta=self._evidence_delta(parent, plan),
            verification_result={},
            novelty_descriptors=list(dict.fromkeys(parent.novelty_descriptors + [operator.lower()])),
            niche_memberships=list(parent.niche_memberships),
            failure_lessons=list(parent.failure_lessons),
            contract_hash=parent.contract_hash,
            multihead_scores=self._mutated_scores(parent, operator),
            metadata=metadata,
        )
        if isinstance(parent, ProjectCandidateGenome):
            child: CandidateGenome = ProjectCandidateGenome(
                **base_kwargs,
                patch_set=self._mutated_patch_set(parent, plan),
                touched_files=list(parent.touched_files),
                touched_symbols=list(parent.touched_symbols),
                expected_effects=list(dict.fromkeys(parent.expected_effects + plan.expected_gene_effects + [operator])),
                affected_tests=list(parent.affected_tests),
                risk_notes=list(parent.risk_notes),
                commands_run=list(parent.commands_run),
                mutation_operator=operator,
            )
        else:
            child = CandidateGenome(**base_kwargs)
        _apply_proof_move(parent, child, plan)
        _apply_patch_conflict_obligations(child, plan)
        _record_move_execution(parent, child, plan)
        if operator == MutationOperator.LINEAGE_RESTART:
            apply_strategy_restart(child, parent, reset_inherited_state=True)
        return child

    def _mutated_artifact(self, parent: CandidateGenome, plan: MutationPlan) -> Any:
        operator = plan.operator if plan.operator in MutationOperator.ALL else MutationOperator.DEEPEN
        if operator == MutationOperator.LINEAGE_RESTART:
            failed_route = parent.core_mechanism or parent.concise_claim or str(parent.artifact or parent.id)
            return (
                "Strategy restart seed. "
                f"Explicit negation constraint: do not reuse failed route `{failed_route}`. "
                f"High-temperature restart directive: {plan.instruction or 'search a materially different mechanism'}."
            )
        transformed = _apply_candidate_transform_artifact(parent.artifact, plan)
        if transformed is not None:
            return transformed
        if parent.artifact is not None and not isinstance(parent.artifact, str):
            return copy.deepcopy(parent.artifact)
        text = str(parent.artifact or parent.concise_claim or parent.core_mechanism)
        repair_note = _repair_note(plan)
        if operator == MutationOperator.CORE_EXTRACTION:
            return parent.core_mechanism or parent.concise_claim or text
        if operator == MutationOperator.SCAFFOLD_REMOVAL:
            return _remove_scaffold_terms(text)
        if operator == MutationOperator.RARE_INJECT:
            seed = plan.rarity_seed or (parent.edge_knowledge_seeds[0] if parent.edge_knowledge_seeds else "rare seed")
            return f"{text}\n\nRare-injected search seed: {seed}"
        if operator == MutationOperator.TOOL_GROUND:
            return f"{text}\n\nTool-grounded verification target: {plan.instruction or 'derive a locally checkable fragment'}{repair_note}"
        if operator in {
            MutationOperator.INSTANTIATE_FORMAL_ARTIFACT,
            MutationOperator.DISCHARGE_OBLIGATION,
            MutationOperator.CASE_SPLIT,
            MutationOperator.CONSTRUCT_WITNESS,
        }:
            return (
                f"{operator} mutation of {parent.id}: {text}\n\n"
                f"Required proof-progress directive: {plan.instruction or 'replace this with a concrete formal object and obligation delta'}{repair_note}"
            )
        if operator == MutationOperator.ROUTE_KILL:
            return f"Route kill analysis for parent {parent.id}: state the concrete obstruction or counterexample that invalidates this route.\n\n{text}"
        if operator == MutationOperator.INVERT:
            return f"Inverted route from parent {parent.id}: {text}"
        return f"{operator} mutation of {parent.id}: {text}{repair_note}"

    def _concise_claim(self, parent: CandidateGenome, operator: str) -> str:
        if operator == MutationOperator.CORE_EXTRACTION:
            return parent.core_mechanism or parent.concise_claim
        return f"{operator} descendant of {parent.id}"

    def _core_mechanism(self, parent: CandidateGenome, plan: MutationPlan) -> str:
        if plan.operator == MutationOperator.LINEAGE_RESTART:
            return plan.instruction or "high-temperature alternative mechanism"
        if plan.operator == MutationOperator.SCAFFOLD_REMOVAL:
            return _remove_scaffold_terms(parent.core_mechanism or parent.concise_claim)
        if plan.operator == MutationOperator.CORE_EXTRACTION:
            return parent.core_mechanism or parent.concise_claim
        if plan.instruction:
            return f"{parent.core_mechanism} | {plan.instruction}".strip(" |")
        return parent.core_mechanism

    def _edge_seeds(self, parent: CandidateGenome, plan: MutationPlan) -> list[str]:
        seeds = list(parent.edge_knowledge_seeds)
        if plan.rarity_seed:
            seeds.append(plan.rarity_seed)
        return list(dict.fromkeys(seed for seed in seeds if seed))

    def _source_bindings(self, parent: CandidateGenome, plan: MutationPlan) -> list[dict[str, Any]]:
        bindings = [dict(item) for item in parent.source_bindings if isinstance(item, dict)]
        for point in plan.metadata.get("required_source_integration_points", []) or []:
            if not isinstance(point, dict):
                continue
            binding = dict(point)
            binding.setdefault("required", True)
            binding.setdefault("source", "mutation_plan")
            bindings.append(binding)
        return _dedupe_dicts(bindings, key_fields=("path", "ref", "kind"))

    def _evidence_delta(self, parent: CandidateGenome, plan: MutationPlan) -> dict[str, Any]:
        delta = dict(parent.evidence_delta)
        delta.pop("verified", None)
        if plan.metadata.get("requires_pre_fail_post_pass"):
            planned = list(delta.get("planned", [])) if isinstance(delta.get("planned"), list) else []
            planned.append("pre-fail/post-pass evidence required by mutation plan")
            delta["planned"] = list(dict.fromkeys(str(item) for item in planned if item))
        for key in ("target_obligation_ids", "required_evidence_kinds"):
            values = plan.metadata.get(key)
            if isinstance(values, list) and values:
                existing = list(delta.get(key, [])) if isinstance(delta.get(key), list) else []
                delta[key] = list(dict.fromkeys(existing + [str(item) for item in values if item]))
        repair = plan.metadata.get("repair_required")
        if isinstance(repair, dict):
            for key in ("blockers", "evidence_needed", "acceptance_criteria"):
                values = repair.get(key)
                if isinstance(values, list) and values:
                    existing = list(delta.get(key, [])) if isinstance(delta.get(key), list) else []
                    delta[key] = list(dict.fromkeys(existing + [str(item) for item in values if item]))
        return delta

    def _mutated_scores(self, parent: CandidateGenome, operator: str) -> dict[str, float]:
        scores = dict(parent.multihead_scores)
        if operator == MutationOperator.RARE_INJECT:
            scores["rarity"] = min(1.0, scores.get("rarity", 0.0) + 0.2)
            scores["novelty"] = min(1.0, scores.get("novelty", 0.0) + 0.1)
        elif operator == MutationOperator.TOOL_GROUND:
            scores["verifiability"] = min(1.0, scores.get("verifiability", 0.0) + 0.15)
            scores["tool_progress"] = min(1.0, scores.get("tool_progress", 0.0) + 0.1)
        elif operator in {MutationOperator.CORE_EXTRACTION, MutationOperator.SCAFFOLD_REMOVAL}:
            scores["auxiliary_value"] = max(0.0, scores.get("auxiliary_value", 0.0) - 0.2)
            scores["core_mechanism_strength"] = min(1.0, scores.get("core_mechanism_strength", 0.0) + 0.1)
        elif operator == MutationOperator.REPAIR:
            scores["robustness"] = min(1.0, scores.get("robustness", 0.0) + 0.1)
        elif operator in {
            MutationOperator.INSTANTIATE_FORMAL_ARTIFACT,
            MutationOperator.DISCHARGE_OBLIGATION,
            MutationOperator.CASE_SPLIT,
            MutationOperator.CONSTRUCT_WITNESS,
            MutationOperator.ROUTE_KILL,
        }:
            # The verifier, not the mutation label, grants proof credit.  These
            # scores merely keep the directed attempt in the reproductive pool.
            scores["verifiability"] = min(1.0, scores.get("verifiability", 0.0) + 0.05)
            scores["deferral_risk"] = min(1.0, scores.get("deferral_risk", 0.0) + 0.05)
        return scores

    def _mutated_patch_set(self, parent: ProjectCandidateGenome, plan: MutationPlan) -> list[PatchOperation]:
        patch_set = [PatchOperation.from_dict(op.to_dict()) for op in parent.patch_set]
        if plan.move_kind.startswith("patch."):
            operation = _patch_operation_for_move(plan)
            merge = merge_patch_sets(patch_set, [operation])
            plan.metadata["_move_patch_conflicts"] = [item.to_dict() for item in merge.conflicts]
            return merge.patch_set
        operator = plan.operator
        if operator == MutationOperator.RARE_INJECT:
            seed = plan.rarity_seed or (parent.edge_knowledge_seeds[0] if parent.edge_knowledge_seeds else "rare seed")
            patch_set.append(PatchOperation(path="NEXUS_RARE_SEED.md", operation="append", content=f"\n- {seed}\n"))
        elif operator == MutationOperator.TOOL_GROUND:
            patch_set.append(PatchOperation(path="NEXUS_VERIFICATION_TARGET.md", operation="write", content=plan.instruction or "local verification target"))
        elif operator == MutationOperator.REPAIR and parent.patch_application_result:
            patch_set.append(PatchOperation(path="NEXUS_REPAIR_NOTE.md", operation="append", content="Repair mutation generated after patch/tool feedback.\n"))
        elif operator == MutationOperator.SCAFFOLD_REMOVAL:
            patch_set = [op for op in patch_set if "scaffold" not in op.path.lower() and "router" not in op.path.lower()] or patch_set
        return patch_set


class MutationPlanner:
    def plan_from_actions(self, parents: list[CandidateGenome], actions: list[str], rarity_seeds: list[str] | None = None) -> list[MutationPlan]:
        seeds = list(rarity_seeds or [])
        shadow_bandit = _shadow_action_palette_bandit(parents, actions)
        plans: list[MutationPlan] = []
        for index, parent in enumerate(parents):
            action = actions[index % len(actions)] if actions else MutationOperator.DEEPEN
            operator, unknown_fallback = _mapped_action(action)
            metadata: dict[str, Any] = {"raw_policy_action": str(action or "")} if actions else {}
            if unknown_fallback:
                metadata["action_fallback"] = {
                    "raw_action": str(action or ""),
                    "fallback_operator": MutationOperator.DEEPEN,
                    "reason": "unknown_action",
                }
            if shadow_bandit:
                metadata["shadow_action_palette_bandit"] = shadow_bandit
            plans.append(
                MutationPlan(
                    operator=operator,
                    parent_ids=[parent.id],
                    instruction=f"Apply {operator} according to the current EvolutionPolicy.",
                    rarity_seed=seeds[index % len(seeds)] if seeds and operator == MutationOperator.RARE_INJECT else "",
                    metadata=metadata,
                )
            )
        return plans


def _action_to_operator(action: str) -> str:
    return _mapped_action(action)[0]


def _mapped_action(action: str) -> tuple[str, bool]:
    normalized = str(action or "").strip().lower()
    if normalized in {"continue", "deepen"}:
        return MutationOperator.DEEPEN, False
    if normalized in {"strategy_restart", "lineage_restart", "strategyrestart", "lineagerestart"} or "strategy restart" in normalized or "lineage restart" in normalized:
        return MutationOperator.LINEAGE_RESTART, False
    if "formal" in normalized or "instantiate" in normalized or "equation" in normalized:
        return MutationOperator.INSTANTIATE_FORMAL_ARTIFACT, False
    if "discharge" in normalized or "obligation" in normalized or "ledger" in normalized:
        return MutationOperator.DISCHARGE_OBLIGATION, False
    if "case" in normalized or "split" in normalized:
        return MutationOperator.CASE_SPLIT, False
    if "witness" in normalized or "counterexample" in normalized:
        return MutationOperator.CONSTRUCT_WITNESS, False
    if "route_kill" in normalized or "kill" in normalized or "refute" in normalized:
        return MutationOperator.ROUTE_KILL, False
    if "core" in normalized:
        return MutationOperator.CORE_EXTRACTION, False
    if "rare" in normalized:
        return MutationOperator.RARE_INJECT, False
    if "scaffold" in normalized:
        return MutationOperator.SCAFFOLD_REMOVAL, False
    if "dormant" in normalized or "reactivate" in normalized:
        return MutationOperator.DORMANT_REACTIVATION, False
    if "repair" in normalized:
        return MutationOperator.REPAIR, False
    operator = normalized.title().replace("_", "")
    if operator in MutationOperator.ALL:
        return operator, False
    return MutationOperator.DEEPEN, True


def apply_strategy_restart(
    candidate: CandidateGenome,
    source_parent: CandidateGenome,
    *,
    reset_inherited_state: bool,
) -> CandidateGenome:
    """Detach a restart candidate into a new root without changing evaluator state."""

    failed_route = source_parent.core_mechanism or source_parent.concise_claim or str(source_parent.artifact or source_parent.id)
    metadata = coerce_dict(candidate.metadata)
    source_arm = str(metadata.get("branch_arm_id") or (source_parent.lineage[0] if source_parent.lineage else source_parent.id))
    metadata["strategy_restart"] = {
        "mode": "isolated_new_lineage_root",
        "temperature": "high",
        "source_parent_id": source_parent.id,
        "source_lineage_root": source_parent.lineage[0] if source_parent.lineage else source_parent.id,
        "source_branch_arm_id": source_arm,
        "negated_failed_route": failed_route,
        "explicit_negation_constraint": f"do not reuse failed route: {failed_route}",
        "effect": "reproduction_lineage_only_evaluator_authority_unchanged",
    }
    metadata["search_seed_not_final"] = True
    if metadata.get("branch_slot_id"):
        metadata["branch_arm_id"] = candidate.id
    candidate.metadata = metadata
    candidate.parent_ids = []
    candidate.generation = 0
    candidate.lineage = [candidate.id]
    candidate.current_fate = CandidateFate.ACTIVE.value
    candidate.concise_claim = f"Strategy restart seed negating failed route from {source_parent.id}"
    candidate.uncertainty_notes = list(
        dict.fromkeys([*candidate.uncertainty_notes, f"Explicit negation constraint: do not reuse failed route `{failed_route}`"])
    )
    if reset_inherited_state:
        candidate.inherited_genes = []
        candidate.mutation_history = [MutationOperator.LINEAGE_RESTART]
        candidate.tool_results = []
        candidate.verification_trace = []
        candidate.formal_artifacts = []
        candidate.proof_obligations = []
        candidate.obligation_delta = {}
        candidate.evidence_refs = []
        candidate.source_bindings = []
        candidate.evidence_delta = {}
        candidate.verification_result = {}
        candidate.failure_lessons = []
        if isinstance(candidate, ProjectCandidateGenome):
            candidate.patch_set = []
            candidate.touched_files = []
            candidate.touched_symbols = []
            candidate.expected_effects = [MutationOperator.LINEAGE_RESTART]
            candidate.affected_tests = []
            candidate.risk_notes = []
            candidate.commands_run = []
    return candidate


def _shadow_action_palette_bandit(parents: list[CandidateGenome], actions: list[str]) -> dict[str, Any]:
    if not parents or not actions:
        return {}
    raw: dict[str, dict[str, Any]] = {}
    for index, parent in enumerate(parents):
        action = str(actions[index % len(actions)] or MutationOperator.DEEPEN)
        fate = CandidateFate.normalize(getattr(parent, "current_fate", ""))
        current = raw.setdefault(action, {"pulls": 0, "reward_sum": 0.0, "risk_sum": 0.0, "fates": {}})
        current["pulls"] += 1
        current["reward_sum"] += _shadow_action_reward(parent, fate)
        current["risk_sum"] += 1.0 if fate in {CandidateFate.CULLED.value, CandidateFate.FAILED.value} else 0.0
        current["fates"][fate] = current["fates"].get(fate, 0) + 1
    arms = tuple(
        OperatorArmStats(arm_id=arm_id, pulls=int(data["pulls"]), reward_sum=float(data["reward_sum"]), risk_sum=float(data["risk_sum"]))
        for arm_id, data in sorted(raw.items())
    )
    return {
        "schema": "shadow-action-palette-bandit/v1",
        "advisory_only": True,
        "arm_count": len(arms),
        "arms": [arm.to_dict() | {"fates": dict(raw[arm.arm_id]["fates"])} for arm in arms],
        "allocation": [item.to_dict() for item in suggest_budget_allocation(arms)],
    }


def _shadow_action_reward(parent: CandidateGenome, fate: str) -> float:
    signal = bounded_score(coerce_dict(getattr(parent, "multihead_scores", {})).get("latent_reproductive_signal", 0.0))
    if fate == CandidateFate.ELITE.value:
        return max(signal, 1.0)
    if fate in {CandidateFate.ACTIVE.value, CandidateFate.INCUBATING.value}:
        return max(signal, 0.5)
    if fate == CandidateFate.DORMANT.value:
        return max(signal, 0.25)
    return signal


def validate_move_plan(parent: CandidateGenome, plan: MutationPlan) -> None:
    """Normalize and fail-closed validate an engine-executable move."""

    collapse_delta = _collapse_params_delta(plan)
    kind = _inferred_move_kind(plan, collapse_delta=collapse_delta)
    domain = _candidate_input_domain(parent)
    plan.actual_delta = {}
    plan.receipt_refs = []
    if domain == "text" and kind in TYPED_MOVE_KINDS:
        plan.move_kind = SEMANTIC_HINT
        plan.input_domain = "text"
        plan.metadata["move_qualification"] = {
            "typed": False,
            "criteria": [],
            "reason": "free_text_candidates_are_semantic_hints",
        }
        return
    if kind not in TYPED_MOVE_KINDS:
        plan.move_kind = SEMANTIC_HINT
        plan.input_domain = domain
        plan.metadata["move_qualification"] = {
            "typed": False,
            "criteria": [],
            "reason": "no_engine_executable_move_contract",
        }
        return
    if plan.input_domain and plan.input_domain != domain:
        raise MoveContractError(
            f"move input_domain {plan.input_domain!r} does not match candidate domain {domain!r}"
        )
    plan.move_kind = kind
    plan.input_domain = domain
    if collapse_delta:
        plan.expected_delta = collapse_delta
    if domain == "dict":
        _validate_dict_move(parent.artifact, plan)
    elif domain == "project_patch":
        operation = _patch_operation_for_move(plan)
        merge = merge_patch_sets(list(getattr(parent, "patch_set", [])), [operation])
        if not merge.conflicts and [item.to_dict() for item in merge.patch_set] == [
            item.to_dict() for item in getattr(parent, "patch_set", [])
        ]:
            raise MoveContractError("project patch move produces no patch-set delta")
    elif domain == "proof":
        _validate_proof_move(parent, plan)
    else:
        raise MoveContractError(f"typed move is unsupported for input domain: {domain}")
    _evaluate_move_invariants(parent.artifact, plan)
    criteria = ["input_domain_determined", "state_delta_recomputable", "mapping_recomputable"]
    if plan.declared_invariant_refs:
        criteria.append("declared_invariants_executable")
    plan.metadata["move_qualification"] = {"typed": True, "criteria": criteria}


def recompute_move_delta(
    parent: CandidateGenome,
    child: CandidateGenome,
    plan: MutationPlan,
) -> dict[str, Any]:
    if plan.move_kind in {"path_assignment", "field_split", "constraint_strengthening"}:
        changes: list[dict[str, Any]] = []
        for item in _dict_move_items(plan):
            path = str(item["path"])
            before = _read_dict_path(parent.artifact, path)
            after = _read_dict_path(child.artifact, path)
            changes.append({"path": path, "before": copy.deepcopy(before), "after": copy.deepcopy(after)})
        return {
            "changes": changes,
            "invariants": _evaluate_move_invariants(child.artifact, plan),
        }
    if plan.move_kind.startswith("patch."):
        return {
            "operation": plan.move_kind.removeprefix("patch."),
            "path": str(plan.expected_delta.get("path") or ""),
            "before_count": len(getattr(parent, "patch_set", [])),
            "after_count": len(getattr(child, "patch_set", [])),
            "conflicts": [
                dict(item)
                for item in plan.metadata.get("_move_patch_conflicts", [])
                if isinstance(item, dict)
            ],
            "invariants": _evaluate_move_invariants(child.artifact, plan),
        }
    if plan.move_kind.startswith("proof."):
        parent_artifacts = {stable_hash(item) for item in parent.formal_artifacts}
        child_artifacts = [
            copy.deepcopy(item)
            for item in child.formal_artifacts
            if stable_hash(item) not in parent_artifacts
        ]
        parent_obligations = {_obligation_id(item) for item in parent.proof_obligations}
        child_obligations = [
            copy.deepcopy(item)
            for item in child.proof_obligations
            if _obligation_id(item) not in parent_obligations
        ]
        return {
            "formal_artifacts_added": child_artifacts,
            "obligations_added": child_obligations,
            "obligation_delta": copy.deepcopy(child.obligation_delta),
            "invariants": _evaluate_move_invariants(child.artifact, plan),
        }
    return {}


def _candidate_input_domain(parent: CandidateGenome) -> str:
    if isinstance(parent, ProjectCandidateGenome):
        return "project_patch"
    artifact_type = str(parent.artifact_type or "").lower()
    if parent.proof_obligations or parent.formal_artifacts or "proof" in artifact_type or "formal" in artifact_type:
        return "proof"
    if isinstance(parent.artifact, dict):
        return "dict"
    return "text"


def _inferred_move_kind(plan: MutationPlan, *, collapse_delta: dict[str, Any] | None) -> str:
    declared = str(plan.move_kind or "").strip()
    if declared in TYPED_MOVE_KINDS:
        return declared
    if collapse_delta:
        return "path_assignment"
    operation = str(plan.expected_delta.get("operation") or "").strip().lower()
    inferred = {
        "assign": "path_assignment",
        "assign_many": "path_assignment",
        "split": "field_split",
        "strengthen": "constraint_strengthening",
        "add": "patch.add",
        "replace": "patch.replace",
        "delete": "patch.delete",
        "append": "patch.append",
        "instantiate_object": "proof.instantiate_object",
        "split_obligation": "proof.split_obligation",
        "construct_witness": "proof.construct_witness",
    }.get(operation)
    return inferred or SEMANTIC_HINT


def _collapse_params_delta(plan: MutationPlan) -> dict[str, Any] | None:
    transforms = plan.metadata.get("candidate_transforms", []) if isinstance(plan.metadata, dict) else []
    for transform in transforms or []:
        if not isinstance(transform, dict) or transform.get("kind") != "collapse_params":
            continue
        payload = transform.get("payload") if isinstance(transform.get("payload"), dict) else {}
        assignment = payload.get("assignment") if isinstance(payload.get("assignment"), dict) else {}
        slots = payload.get("parameter_slots") if isinstance(payload.get("parameter_slots"), dict) else {}
        items: list[dict[str, Any]] = []
        for key, value in assignment.items():
            slot = slots.get(key)
            path = slot.get("path") if isinstance(slot, dict) else slot
            if isinstance(path, str) and path:
                items.append({"path": path, "after": copy.deepcopy(value)})
        if items:
            return {"operation": "assign_many", "assignments": items}
    return None


def _dict_move_items(plan: MutationPlan) -> list[dict[str, Any]]:
    if plan.move_kind == "path_assignment":
        assignments = plan.expected_delta.get("assignments")
        if isinstance(assignments, list):
            return [dict(item) for item in assignments if isinstance(item, dict)]
        return [{"path": plan.expected_delta.get("path"), "after": plan.expected_delta.get("after")}]
    if plan.move_kind == "field_split":
        return [{"path": plan.expected_delta.get("path"), "after": plan.expected_delta.get("fields")}]
    if plan.move_kind == "constraint_strengthening":
        path = str(plan.expected_delta.get("path") or "")
        return [{"path": path, "after": [*_read_dict_path_placeholder(plan, path), plan.expected_delta.get("add")]}]
    return []


def _read_dict_path_placeholder(plan: MutationPlan, path: str) -> list[Any]:
    before = plan.metadata.get("_move_constraint_before")
    return list(before) if isinstance(before, list) else []


def _validate_dict_move(artifact: Any, plan: MutationPlan) -> None:
    if not isinstance(artifact, dict):
        raise MoveContractError("dict move requires an object artifact")
    if plan.move_kind == "path_assignment":
        items = _dict_move_items(plan)
        if not items:
            raise MoveContractError("path_assignment requires at least one assignment")
        for item in items:
            path = _required_move_path(item.get("path"))
            before = _read_dict_path(artifact, path)
            after = item.get("after")
            if type(before) is not type(after):
                raise MoveContractError(f"path_assignment changes the type at {path}")
            if before == after:
                raise MoveContractError(f"path_assignment does not change {path}")
        return
    path = _required_move_path(plan.expected_delta.get("path"))
    before = _read_dict_path(artifact, path)
    if plan.move_kind == "field_split":
        fields = plan.expected_delta.get("fields")
        if not isinstance(fields, dict) or not fields:
            raise MoveContractError("field_split requires non-empty fields")
        if before == fields:
            raise MoveContractError(f"field_split does not change {path}")
        return
    if plan.move_kind == "constraint_strengthening":
        addition = plan.expected_delta.get("add")
        if not isinstance(before, list):
            raise MoveContractError("constraint_strengthening target must be an array")
        if addition is None or addition in before:
            raise MoveContractError("constraint_strengthening requires a new constraint")
        plan.metadata["_move_constraint_before"] = copy.deepcopy(before)
        return
    raise MoveContractError(f"unsupported dict move: {plan.move_kind}")


def _apply_dict_move(artifact: dict[str, Any], plan: MutationPlan) -> dict[str, Any]:
    out = copy.deepcopy(artifact)
    for item in _dict_move_items(plan):
        path = str(item["path"])
        after = copy.deepcopy(item["after"])
        if plan.move_kind == "constraint_strengthening":
            after = list(_read_dict_path(out, path)) + [plan.expected_delta.get("add")]
        _write_dict_path(out, path, after)
    return out


def _required_move_path(value: Any) -> str:
    path = str(value or "").strip()
    if not path or any(not part for part in path.split(".")):
        raise MoveContractError("move path must be a non-empty dotted path")
    return path


def _read_dict_path(value: Any, path: str) -> Any:
    current = value
    for part in _required_move_path(path).split("."):
        if not isinstance(current, dict) or part not in current:
            raise MoveContractError(f"move path does not exist: {path}")
        current = current[part]
    return current


def _write_dict_path(value: dict[str, Any], path: str, after: Any) -> None:
    parts = _required_move_path(path).split(".")
    current = value
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = after


def _patch_operation_for_move(plan: MutationPlan) -> PatchOperation:
    operation = plan.move_kind.removeprefix("patch.")
    if operation not in {"add", "replace", "delete", "append"}:
        raise MoveContractError(f"unsupported project patch move: {plan.move_kind}")
    path = str(plan.expected_delta.get("path") or "").strip()
    parsed = PurePosixPath(path)
    if not path or path == "." or parsed.is_absolute() or ".." in parsed.parts or "\\" in path:
        raise MoveContractError(f"unsafe project patch path: {path!r}")
    declared_operation = str(plan.expected_delta.get("operation") or "").strip()
    if declared_operation and declared_operation != operation:
        raise MoveContractError("project patch operation conflicts with move_kind")
    if operation in {"add", "append"}:
        content = plan.expected_delta.get("content")
        if not isinstance(content, str) or not content:
            raise MoveContractError(f"patch.{operation} requires non-empty content")
        return PatchOperation(path=path, operation="write" if operation == "add" else "append", content=content)
    if operation == "replace":
        old_text = plan.expected_delta.get("old_text")
        new_text = plan.expected_delta.get("new_text")
        if not isinstance(old_text, str) or not old_text or not isinstance(new_text, str) or old_text == new_text:
            raise MoveContractError("patch.replace requires distinct old_text and new_text")
        return PatchOperation(path=path, operation="replace", old_text=old_text, new_text=new_text)
    return PatchOperation(path=path, operation="delete")


def _validate_proof_move(parent: CandidateGenome, plan: MutationPlan) -> None:
    delta = plan.expected_delta
    if plan.move_kind == "proof.instantiate_object":
        if not isinstance(delta.get("object"), dict) or not delta["object"]:
            raise MoveContractError("proof.instantiate_object requires an object")
        if stable_hash(delta["object"]) in {stable_hash(item) for item in parent.formal_artifacts}:
            raise MoveContractError("proof.instantiate_object requires a new object")
        return
    obligation_id = str(delta.get("obligation_id") or "")
    known = {_obligation_id(item) for item in parent.proof_obligations}
    if not obligation_id or obligation_id not in known:
        raise MoveContractError("proof move references an unknown obligation")
    if plan.move_kind == "proof.split_obligation":
        children = delta.get("child_obligations")
        if not isinstance(children, list) or len(children) < 2 or any(not _obligation_id(item) for item in children):
            raise MoveContractError("proof.split_obligation requires at least two identified child obligations")
        child_ids = [_obligation_id(item) for item in children]
        if len(set(child_ids)) != len(child_ids) or any(item in known for item in child_ids):
            raise MoveContractError("proof.split_obligation requires distinct new child obligations")
        return
    if plan.move_kind == "proof.construct_witness":
        if not isinstance(delta.get("witness"), dict) or not delta["witness"]:
            raise MoveContractError("proof.construct_witness requires a witness object")
        if stable_hash(delta["witness"]) in {stable_hash(item) for item in parent.formal_artifacts}:
            raise MoveContractError("proof.construct_witness requires a new witness")
        return
    raise MoveContractError(f"unsupported proof move: {plan.move_kind}")


def _apply_proof_move(parent: CandidateGenome, child: CandidateGenome, plan: MutationPlan) -> None:
    if not plan.move_kind.startswith("proof."):
        return
    delta = plan.expected_delta
    if plan.move_kind == "proof.instantiate_object":
        child.formal_artifacts.append(copy.deepcopy(delta["object"]))
        return
    obligation_id = str(delta["obligation_id"])
    if plan.move_kind == "proof.split_obligation":
        children = [copy.deepcopy(item) for item in delta["child_obligations"]]
        child.proof_obligations.extend(children)
        decomposed = list(child.obligation_delta.get("decomposed", [])) + [obligation_id]
        introduced = list(child.obligation_delta.get("introduced", [])) + [_obligation_id(item) for item in children]
        child.obligation_delta["decomposed"] = list(dict.fromkeys(decomposed))
        child.obligation_delta["introduced"] = list(dict.fromkeys(introduced))
        return
    child.formal_artifacts.append(copy.deepcopy(delta["witness"]))
    targeted = list(child.obligation_delta.get("targeted", [])) + [obligation_id]
    child.obligation_delta["targeted"] = list(dict.fromkeys(targeted))


def _apply_patch_conflict_obligations(child: CandidateGenome, plan: MutationPlan) -> None:
    conflicts = [
        dict(item)
        for item in plan.metadata.get("_move_patch_conflicts", [])
        if isinstance(item, dict)
    ]
    if not conflicts:
        return
    introduced = list(child.obligation_delta.get("introduced", []))
    for conflict in conflicts:
        obligation_id = "patch-conflict-" + stable_hash(conflict)[:16]
        child.proof_obligations.append(
            {
                "id": obligation_id,
                "status": "pending",
                "description": f"Resolve incompatible patch operations for {conflict.get('path', '')}",
                "source": "move_contract",
                "conflict": conflict,
            }
        )
        introduced.append(obligation_id)
    child.obligation_delta["introduced"] = list(dict.fromkeys(introduced))


def _record_move_execution(parent: CandidateGenome, child: CandidateGenome, plan: MutationPlan) -> None:
    if plan.move_kind not in TYPED_MOVE_KINDS:
        attach_semantic_move_contract(child, plan)
        return
    actual_delta = recompute_move_delta(parent, child, plan)
    if not actual_delta:
        raise MoveContractError("typed move produced no recomputable delta")
    plan.actual_delta = actual_delta
    receipt_core = {
        "candidate_id": child.id,
        "parent_id": parent.id,
        "move_kind": plan.move_kind,
        "input_domain": plan.input_domain,
        "declared_invariant_refs": list(plan.declared_invariant_refs),
        "expected_delta": copy.deepcopy(plan.expected_delta),
        "actual_delta": copy.deepcopy(actual_delta),
    }
    receipt_id = "move-" + stable_hash(receipt_core)[:20]
    plan.receipt_refs = [receipt_id]
    contract = {
        "move_kind": plan.move_kind,
        "input_domain": plan.input_domain,
        "declared_invariant_refs": list(plan.declared_invariant_refs),
        "expected_delta": copy.deepcopy(plan.expected_delta),
        "actual_delta": copy.deepcopy(actual_delta),
        "receipt_refs": [receipt_id],
    }
    from cognitive_evolve_runtime.evaluators.evidence_authority import stable_artifact_hash

    child.metadata["move_contract"] = contract
    child.metadata["move_receipt"] = {
        "receipt_id": receipt_id,
        **receipt_core,
        "artifact_hash": stable_artifact_hash(child.artifact),
    }
    qualification = child.metadata.get("move_qualification")
    if isinstance(qualification, dict) and qualification.get("typed") is True:
        criteria = list(qualification.get("criteria", []))
        qualification["criteria"] = list(dict.fromkeys([*criteria, "result_receipt_recorded"]))


def attach_semantic_move_contract(candidate: CandidateGenome, plan: MutationPlan) -> None:
    """Attach an auditable non-typed contract without granting a move receipt."""

    if isinstance(candidate.metadata.get("move_receipt"), dict):
        return
    candidate.metadata["move_contract"] = {
        "move_kind": SEMANTIC_HINT,
        "input_domain": plan.input_domain or _candidate_input_domain(candidate),
        "declared_invariant_refs": [],
        "expected_delta": {},
        "actual_delta": {},
        "receipt_refs": [],
    }


def _evaluate_move_invariants(artifact: Any, plan: MutationPlan) -> dict[str, bool]:
    if not plan.declared_invariant_refs:
        return {}
    raw = plan.metadata.get("move_invariants")
    specs = {
        str(item.get("ref") or ""): item
        for item in raw or []
        if isinstance(item, dict) and item.get("ref")
    } if isinstance(raw, list) else {}
    results: dict[str, bool] = {}
    for ref in plan.declared_invariant_refs:
        spec = specs.get(ref)
        if spec is None:
            raise MoveContractError(f"declared invariant has no executable criterion: {ref}")
        observed = _read_dict_path(artifact, str(spec.get("path") or ""))
        operator = str(spec.get("operator") or "")
        expected = spec.get("expected")
        if operator == "equal":
            passed = observed == expected
        elif operator == "not_equal":
            passed = observed != expected
        elif operator == "contains":
            passed = expected in observed
        elif operator == "type_is":
            passed = type(observed).__name__ == str(expected)
        else:
            raise MoveContractError(f"unsupported invariant operator: {operator}")
        if not passed:
            raise MoveContractError(f"declared invariant failed: {ref}")
        results[ref] = True
    return results


def _obligation_id(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("id") or "")
    return str(item or "")


def _remove_scaffold_terms(text: str) -> str:
    stripped = " ".join(str(text or "").split())
    return stripped or "core mechanism needs reconstruction"


def _apply_candidate_transform_artifact(artifact: Any, plan: MutationPlan) -> Any | None:
    if plan.move_kind in {"path_assignment", "field_split", "constraint_strengthening"}:
        return _apply_dict_move(artifact, plan)
    transforms = plan.metadata.get("candidate_transforms", []) if isinstance(plan.metadata, dict) else []
    for transform in transforms or []:
        if not isinstance(transform, dict) or transform.get("kind") != "collapse_params":
            continue
        payload = transform.get("payload") if isinstance(transform.get("payload"), dict) else {}
        assignment = payload.get("assignment") if isinstance(payload.get("assignment"), dict) else {}
        slots = payload.get("parameter_slots") if isinstance(payload.get("parameter_slots"), dict) else {}
        if not assignment or not slots:
            continue
        changed, value = _apply_parameter_slots(artifact, assignment=assignment, slots=slots)
        if changed:
            return value
    return None


def _apply_parameter_slots(artifact: Any, *, assignment: dict[str, Any], slots: dict[str, Any]) -> tuple[bool, Any]:
    if isinstance(artifact, dict):
        import copy
        out = copy.deepcopy(artifact)
        changed = False
        for key, value in assignment.items():
            slot = slots.get(key)
            path = slot.get("path") if isinstance(slot, dict) else slot
            if isinstance(path, str) and path:
                current = out
                parts = [part for part in path.split(".") if part]
                for part in parts[:-1]:
                    if not isinstance(current.get(part), dict):
                        current[part] = {}
                    current = current[part]
                if parts:
                    if current.get(parts[-1]) != value:
                        current[parts[-1]] = value
                        changed = True
        return changed, out
    text = str(artifact or "")
    out = text
    for key, value in assignment.items():
        slot = slots.get(key)
        markers: list[str] = []
        if isinstance(slot, dict):
            marker = slot.get("marker")
            if isinstance(marker, list):
                markers.extend(str(item) for item in marker if item)
            elif marker:
                markers.append(str(marker))
        elif slot:
            markers.append(str(slot))
        for marker in markers:
            out = out.replace(marker, str(value))
    return out != text, out


def _repair_note(plan: MutationPlan) -> str:
    repair = plan.metadata.get("repair_required")
    if not isinstance(repair, dict):
        return ""
    blockers = ", ".join(str(item) for item in repair.get("blockers", []) if item)
    criteria = ", ".join(str(item) for item in repair.get("acceptance_criteria", []) if item)
    if not blockers and not criteria:
        return ""
    return f"\n\nTargeted repair requirement: blockers={blockers or 'unspecified'}; acceptance={criteria or 'emit verifier-readable evidence_delta'}."


def _dedupe_dicts(items: list[dict[str, Any]], *, key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for item in items:
        key = tuple(str(item.get(field) or "") for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _inherited_mutation_metadata(parent: CandidateGenome, plan: MutationPlan) -> dict[str, Any]:
    """Preserve non-final/repair lane constraints across deterministic mutation."""

    parent_metadata = coerce_dict(getattr(parent, "metadata", {}))
    metadata: dict[str, Any] = {
        "mutation_instruction": plan.instruction,
        "mutation_operator": plan.operator if plan.operator in MutationOperator.ALL else MutationOperator.DEEPEN,
    }
    for key in (
        "search_seed_not_final",
        "final_answer_blocked_until_repaired",
        "final_answer_blocked_until_reverified",
        "source_grounding_required",
        "exploration_source",
        "repair_seed",
    ):
        if key in parent_metadata:
            metadata[key] = parent_metadata[key]
    repair_required = plan.metadata.get("repair_required") or parent_metadata.get("repair_required")
    if isinstance(repair_required, dict):
        metadata["repair_required"] = dict(repair_required)
    parent_verification = _parent_verification_summary(parent)
    if parent_verification:
        metadata["parent_verification_summary"] = parent_verification
    metadata.update(plan.metadata)
    coverage_target = plan.metadata.get("coverage_target")
    if isinstance(coverage_target, dict) and (coverage_target.get("axis") or coverage_target.get("family")):
        search_space = coerce_dict(parent_metadata.get("search_space"))
        if coverage_target.get("axis"):
            search_space["seed_axis"] = str(coverage_target["axis"])
        if coverage_target.get("family"):
            search_space["family_id"] = str(coverage_target["family"])
        metadata["search_space"] = search_space
    for transform in plan.metadata.get("candidate_transforms", []) or []:
        if not isinstance(transform, dict) or transform.get("kind") != "collapse_params":
            continue
        payload = transform.get("payload") if isinstance(transform.get("payload"), dict) else {}
        assignment = payload.get("assignment") if isinstance(payload.get("assignment"), dict) else {}
        if assignment:
            metadata["parameter_assignment"] = dict(assignment)
            metadata["parameter_collapsed"] = True
            metadata.pop("parameter_space", None)
            metadata["parameter_space_frozen"] = True
    return metadata


def _parent_verification_summary(parent: CandidateGenome) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if parent.verification_result:
        summary["verification_result"] = dict(parent.verification_result)
    if parent.verification_trace:
        summary["verification_trace_count"] = len(parent.verification_trace)
    if parent.tool_results:
        summary["tool_result_count"] = len(parent.tool_results)
    return summary


__all__ = [
    "apply_strategy_restart",
    "attach_semantic_move_contract",
    "MoveContractError",
    "MutationEngine",
    "MutationOperator",
    "MutationPlan",
    "MutationPlanner",
    "SEMANTIC_HINT",
    "TYPED_MOVE_KINDS",
    "recompute_move_delta",
    "validate_move_plan",
]
