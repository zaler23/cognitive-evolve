"""Task-agnostic mutation operators for evolvable genomes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .genome import CandidateGenome
from .project_candidate import PatchOperation, ProjectCandidateGenome
from cognitive_evolve_runtime.nexus._serde import coerce_dict, coerce_str_list


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


@dataclass
class MutationPlan:
    operator: str
    parent_ids: list[str] = field(default_factory=list)
    instruction: str = ""
    rarity_seed: str = ""
    expected_gene_effects: list[str] = field(default_factory=list)
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
            metadata=coerce_dict(data.get("metadata")),
        )


class MutationEngine:
    """Small deterministic mutation engine used by tests and fake-model runs."""

    def mutate(self, parent: CandidateGenome, plan: MutationPlan) -> CandidateGenome:
        operator = plan.operator if plan.operator in MutationOperator.ALL else MutationOperator.DEEPEN
        inherited = [parent.extract_inheritable_gene_summary()] + list(parent.inherited_genes)
        artifact = self._mutated_artifact(parent, plan)
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
            tool_results=list(parent.tool_results),
            verification_trace=list(parent.verification_trace),
            formal_artifacts=list(parent.formal_artifacts),
            proof_obligations=list(parent.proof_obligations),
            obligation_delta=dict(parent.obligation_delta),
            evidence_refs=list(parent.evidence_refs),
            source_bindings=self._source_bindings(parent, plan),
            evidence_delta=self._evidence_delta(parent, plan),
            verification_result=dict(parent.verification_result),
            novelty_descriptors=list(dict.fromkeys(parent.novelty_descriptors + [operator.lower()])),
            niche_memberships=list(parent.niche_memberships),
            failure_lessons=list(parent.failure_lessons),
            contract_hash=parent.contract_hash,
            multihead_scores=self._mutated_scores(parent, operator),
            metadata=_inherited_mutation_metadata(parent, plan),
        )
        if isinstance(parent, ProjectCandidateGenome):
            return ProjectCandidateGenome(
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
        return CandidateGenome(**base_kwargs)

    def _mutated_artifact(self, parent: CandidateGenome, plan: MutationPlan) -> Any:
        text = str(parent.artifact or parent.concise_claim or parent.core_mechanism)
        operator = plan.operator
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
        plans: list[MutationPlan] = []
        for index, parent in enumerate(parents):
            action = actions[index % len(actions)] if actions else MutationOperator.DEEPEN
            operator = _action_to_operator(action)
            plans.append(
                MutationPlan(
                    operator=operator,
                    parent_ids=[parent.id],
                    instruction=f"Apply {operator} according to the current EvolutionPolicy.",
                    rarity_seed=seeds[index % len(seeds)] if seeds and operator == MutationOperator.RARE_INJECT else "",
                )
            )
        return plans


def _action_to_operator(action: str) -> str:
    normalized = str(action or "").strip().lower()
    if "formal" in normalized or "instantiate" in normalized or "equation" in normalized:
        return MutationOperator.INSTANTIATE_FORMAL_ARTIFACT
    if "discharge" in normalized or "obligation" in normalized or "ledger" in normalized:
        return MutationOperator.DISCHARGE_OBLIGATION
    if "case" in normalized or "split" in normalized:
        return MutationOperator.CASE_SPLIT
    if "witness" in normalized or "counterexample" in normalized:
        return MutationOperator.CONSTRUCT_WITNESS
    if "route_kill" in normalized or "kill" in normalized or "refute" in normalized:
        return MutationOperator.ROUTE_KILL
    if "core" in normalized:
        return MutationOperator.CORE_EXTRACTION
    if "rare" in normalized:
        return MutationOperator.RARE_INJECT
    if "scaffold" in normalized:
        return MutationOperator.SCAFFOLD_REMOVAL
    if "dormant" in normalized or "reactivate" in normalized:
        return MutationOperator.DORMANT_REACTIVATION
    if "repair" in normalized:
        return MutationOperator.REPAIR
    return normalized.title().replace("_", "") if normalized.title().replace("_", "") in MutationOperator.ALL else MutationOperator.DEEPEN


def _remove_scaffold_terms(text: str) -> str:
    stripped = str(text or "")
    for token in ["router", "validator", "framework", "classification layer", "scaffold"]:
        stripped = stripped.replace(token, "").replace(token.title(), "")
    return " ".join(stripped.split()) or "core mechanism needs reconstruction"


def _repair_note(plan: MutationPlan) -> str:
    repair = plan.metadata.get("repair_required")
    if not isinstance(repair, dict):
        return ""
    blockers = ", ".join(str(item) for item in repair.get("blockers", [])[:4] if item)
    criteria = ", ".join(str(item) for item in repair.get("acceptance_criteria", [])[:4] if item)
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
    metadata: dict[str, Any] = {"mutation_instruction": plan.instruction}
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
    metadata.update(plan.metadata)
    return metadata


__all__ = ["MutationOperator", "MutationPlan", "MutationEngine", "MutationPlanner"]
