#!/usr/bin/env python3
"""Task/objective contracts for admissible evolution.

The contract is deliberately deterministic and serializable.  LLMs can generate,
criticize, mutate, and compare candidates, but the contract records which
evidence, schema, and admissibility checks must be satisfied before a candidate
may be treated as final success.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from cognitive_evolve_runtime.nexus.artifact_contract import DynamicArtifactContract, validate_dynamic_artifact_contract
from cognitive_evolve_runtime.nexus._serde import coerce_dict
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike

from ..nexus.task_types import (
    CODE_TASK_TYPES,
    DEFAULT_TASK_TYPE,
    FRONTIER_TASK_TYPES,
    RESEARCH_TASK_TYPES,
    evidence_required_for_task_type,
    normalize_task_type,
    task_type_registry,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Constraint:
    id: str
    description: str
    hard: bool = True
    source: str = "contract_compiler"


@dataclass
class ObjectiveDimension:
    id: str
    weight: float = 1.0
    direction: str = "maximize"
    description: str = ""


@dataclass
class EvidenceContract:
    required: bool
    required_source_types: list[str] = field(default_factory=list)
    claim_policy: str = "claims_that_affect_decision_need_source_test_or_explicit_uncertainty"
    source_grounding_hard: bool = False
    unsupported_claim_policy: str = "label_uncertain_or_remove_from_decisive_path"


@dataclass
class EvaluatorBinding:
    id: str
    kind: str
    hard: bool = True
    description: str = ""


@dataclass
class AbstentionPolicy:
    enabled: bool = True
    no_clear_winner_status: str = "needs_verifier"
    insufficient_evidence_status: str = "insufficient_evidence"
    contradiction_status: str = "contradiction_detected"


@dataclass
class FinalSchema:
    id: str
    required_sections: list[str] = field(default_factory=list)
    required_labels: list[str] = field(default_factory=list)
    hard: bool = False


@dataclass
class TaskContract:
    id: str
    version: str
    task_type: str
    objective: str
    requires_evidence: bool
    objectives: list[ObjectiveDimension]
    constraints: list[Constraint]
    evidence: EvidenceContract
    evaluators: list[EvaluatorBinding]
    abstention: AbstentionPolicy
    final_schema: FinalSchema
    admissibility_rules: list[str]
    task_type_registry: dict[str, Any]
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ObjectiveContractCompiler:
    """Compile an objective contract from route, budget, and evidence plan."""

    def compile(
        self,
        *,
        prompt: str,
        semantic_assessment: dict[str, Any],
        evidence_plan: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> TaskContract:
        del prompt, context  # Inputs are kept for future contract extensions.
        evidence_plan = evidence_plan or {}
        task_type = normalize_task_type(semantic_assessment.get("task_type"))
        objective = str(
            semantic_assessment.get("real_objective")
            or semantic_assessment.get("surface_request")
            or "user request"
        )
        required_source_types = [str(item) for item in evidence_plan.get("required_source_types", [])]
        requires_evidence = bool(evidence_plan.get("required")) or evidence_required_for_task_type(task_type)
        if task_type in RESEARCH_TASK_TYPES and "primary_or_current_external_sources" not in required_source_types:
            required_source_types.append("primary_or_current_external_sources")
        if task_type in CODE_TASK_TYPES:
            for item in ["local_files", "test_runner"]:
                if item not in required_source_types:
                    required_source_types.append(item)

        final_schema = self._final_schema(task_type)
        evaluators = [
            EvaluatorBinding(
                id="final_report_schema",
                kind="local_schema",
                hard=final_schema.hard,
                description="Check that the final answer exposes required epistemic labels/sections for this task type.",
            ),
            EvaluatorBinding(
                id="source_grounding",
                kind="evidence_ledger",
                hard=requires_evidence and task_type in RESEARCH_TASK_TYPES,
                description="Claims on the decisive path must be supported, labeled uncertain, or rejected.",
            ),
            EvaluatorBinding(
                id="source_contradiction",
                kind="evidence_ledger",
                hard=True,
                description="Reject candidates/final answers that contradict bound evidence.",
            ),
            EvaluatorBinding(
                id="admissibility_gate",
                kind="candidate_filter",
                hard=True,
                description="A judge/tournament winner can be rejected before final synthesis.",
            ),
        ]
        if task_type in CODE_TASK_TYPES:
            evaluators.append(
                EvaluatorBinding(
                    id="local_validation",
                    kind="test_or_manifest",
                    hard=False,
                    description="Prefer candidates with local file/test evidence; skipped tests remain an uncertainty, not proof.",
                )
            )

        return TaskContract(
            id=f"task-contract:{task_type}",
            version="objective-contract/v1",
            task_type=task_type,
            objective=objective,
            requires_evidence=requires_evidence,
            objectives=[
                ObjectiveDimension("objective_proximity", 1.0, "maximize", "Closeness to the user objective."),
                ObjectiveDimension("evidence_strength", 1.0, "maximize", "Amount and quality of bound evidence."),
                ObjectiveDimension("mechanism_specificity", 0.8, "maximize", "Specificity of the mechanism, proof path, implementation, or test."),
                ObjectiveDimension("risk_control", 0.8, "maximize", "Safety, reversibility, and clear failure modes."),
                ObjectiveDimension("novelty_or_breakthrough", 0.4, "maximize", "Useful novelty without unsupported closure."),
            ],
            constraints=self._constraints(task_type),
            evidence=EvidenceContract(
                required=requires_evidence,
                required_source_types=required_source_types,
                source_grounding_hard=requires_evidence and task_type in RESEARCH_TASK_TYPES,
            ),
            evaluators=evaluators,
            abstention=AbstentionPolicy(),
            final_schema=final_schema,
            admissibility_rules=[
                "reject_support_only_meta_candidates_for_object_level_objectives",
                "reject_candidates_with_decisive_source_contradictions",
                "do_not_force_single_winner_when_evidence_cannot_decide",
                "label_partial_progress_instead_of_claiming_solved",
            ],
            task_type_registry=task_type_registry(),
        )

    def _constraints(self, task_type: str) -> list[Constraint]:
        constraints = [
            Constraint("no_external_clarification_loop", "Final answer must not ask the user to unblock the current run."),
            Constraint("no_fake_chain_of_thought", "Final answer must not expose or fabricate private chain-of-thought."),
            Constraint("judge_is_selection_pressure_not_truth", "LLM judge preference is not final authority."),
        ]
        if task_type in RESEARCH_TASK_TYPES:
            constraints.extend(
                [
                    Constraint("no_unverified_theorem_claim", "Do not present an unchecked derivation or model conjecture as a theorem."),
                    Constraint("public_sources_over_model_memory", "Prefer bound public sources and local evidence over model memory."),
                ]
            )
        if task_type in FRONTIER_TASK_TYPES:
            constraints.append(
                Constraint(
                    "frontier_result_requires_expert_verification",
                    "Frontier proof/counterexample claims must preserve unresolved gaps unless every key lemma is source-backed.",
                )
            )
        if task_type in CODE_TASK_TYPES:
            constraints.append(Constraint("small_reversible_patch_first", "Prefer minimal reversible changes and local validation evidence."))
        return constraints

    def _final_schema(self, task_type: str) -> FinalSchema:
        if task_type in FRONTIER_TASK_TYPES:
            return FinalSchema(
                id="research_report_schema",
                hard=True,
                required_sections=[
                    "problem_statement",
                    "known_public_facts",
                    "attempted_reconstruction",
                    "toy_examples_or_failed_toy_construction",
                    "lemma_dependency_graph",
                    "verification_plan",
                    "unresolved_gaps",
                    "next_research_steps",
                ],
                required_labels=[
                    "confirmed_fact",
                    "model_conjecture",
                    "unchecked_derivation",
                    "failed_attempt",
                    "needs_expert_verification",
                ],
            )
        if task_type == "research_or_evidence_dependent_plan":
            return FinalSchema(
                id="evidence_dependent_answer_schema",
                hard=False,
                required_sections=["known_public_facts", "verification_plan", "unresolved_gaps"],
                required_labels=["confirmed_fact", "unchecked_derivation"],
            )
        if task_type in CODE_TASK_TYPES:
            return FinalSchema(
                id="technical_change_schema",
                hard=False,
                required_sections=["change_summary", "validation", "risk_or_rollback"],
                required_labels=[],
            )
        return FinalSchema(id="general_answer_schema", hard=False)


class TaskContractValidator:
    """Validate the deterministic structure of a task contract."""

    def validate(self, contract: TaskContract | dict[str, Any]) -> ValidationResult:
        data = contract.to_dict() if isinstance(contract, TaskContract) else dict(contract or {})
        errors: list[str] = []
        warnings: list[str] = []
        task_type = normalize_task_type(data.get("task_type"))
        if task_type != data.get("task_type"):
            errors.append(f"unknown task_type: {data.get('task_type')!r}")
        if not str(data.get("objective") or "").strip():
            errors.append("objective is required")
        evidence = data.get("evidence") if isinstance(data.get("evidence"), dict) else {}
        if data.get("requires_evidence") and not evidence.get("required_source_types"):
            warnings.append("requires_evidence is true but required_source_types is empty")
        if not isinstance(data.get("admissibility_rules"), list) or not data.get("admissibility_rules"):
            errors.append("admissibility_rules must be non-empty")
        return ValidationResult(passed=not errors, errors=errors, warnings=warnings)


def contract_from_dict(data: dict[str, Any] | TaskContract | None) -> TaskContract | None:
    if isinstance(data, TaskContract):
        return data
    if not isinstance(data, dict) or not data:
        return None
    evidence_data = data.get("evidence") if isinstance(data.get("evidence"), dict) else {}
    abstention_data = data.get("abstention") if isinstance(data.get("abstention"), dict) else {}
    schema_data = data.get("final_schema") if isinstance(data.get("final_schema"), dict) else {}
    return TaskContract(
        id=str(data.get("id") or "task-contract:unknown"),
        version=str(data.get("version") or "objective-contract/v1"),
        task_type=normalize_task_type(data.get("task_type")),
        objective=str(data.get("objective") or ""),
        requires_evidence=bool(data.get("requires_evidence")),
        objectives=[
            ObjectiveDimension(**item)
            for item in data.get("objectives", [])
            if isinstance(item, dict) and item.get("id")
        ],
        constraints=[
            Constraint(**item)
            for item in data.get("constraints", [])
            if isinstance(item, dict) and item.get("id")
        ],
        evidence=EvidenceContract(
            required=bool(evidence_data.get("required")),
            required_source_types=[str(item) for item in evidence_data.get("required_source_types", [])],
            claim_policy=str(evidence_data.get("claim_policy") or "claims_that_affect_decision_need_source_test_or_explicit_uncertainty"),
            source_grounding_hard=bool(evidence_data.get("source_grounding_hard")),
            unsupported_claim_policy=str(evidence_data.get("unsupported_claim_policy") or "label_uncertain_or_remove_from_decisive_path"),
        ),
        evaluators=[
            EvaluatorBinding(**item)
            for item in data.get("evaluators", [])
            if isinstance(item, dict) and item.get("id")
        ],
        abstention=AbstentionPolicy(
            enabled=bool(abstention_data.get("enabled", True)),
            no_clear_winner_status=str(abstention_data.get("no_clear_winner_status") or "needs_verifier"),
            insufficient_evidence_status=str(abstention_data.get("insufficient_evidence_status") or "insufficient_evidence"),
            contradiction_status=str(abstention_data.get("contradiction_status") or "contradiction_detected"),
        ),
        final_schema=FinalSchema(
            id=str(schema_data.get("id") or "general_answer_schema"),
            required_sections=[str(item) for item in schema_data.get("required_sections", [])],
            required_labels=[str(item) for item in schema_data.get("required_labels", [])],
            hard=bool(schema_data.get("hard", False)),
        ),
        admissibility_rules=[str(item) for item in data.get("admissibility_rules", [])],
        task_type_registry=dict(data.get("task_type_registry") or task_type_registry()),
        created_at=str(data.get("created_at") or _now()),
    )


__all__ = [
    "Constraint",
    "ObjectiveDimension",
    "EvidenceContract",
    "EvaluatorBinding",
    "AbstentionPolicy",
    "FinalSchema",
    "TaskContract",
    "ValidationResult",
    "ObjectiveContractCompiler",
    "TaskContractValidator",
    "contract_from_dict",
]


# ---- Nexus evaluation-contract helpers ----
from .schemas import ContractItem, EvaluationContract


def objective_contract_from_task(
    objective: str,
    *,
    hard_gates: list[str] | None = None,
    progress_metrics: list[str] | None = None,
) -> EvaluationContract:
    """Create the lightweight evaluation contract used by Nexus meta-evolution.

    This helper lives in the canonical contract module so task contracts and
    evaluation contracts share one source of truth.
    """

    return EvaluationContract(
        objective=str(objective or "user objective"),
        success_conditions=[
            ContractItem(
                "objective_satisfied",
                "A candidate materially satisfies the user's stated objective.",
                hard=True,
            )
        ],
        hard_gates=[
            ContractItem(f"hard_gate_{index}", gate, hard=True)
            for index, gate in enumerate(hard_gates or ["Candidate must bridge directly to the objective."], start=1)
        ],
        progress_metrics=[
            ContractItem(f"progress_{index}", metric)
            for index, metric in enumerate(progress_metrics or ["hard_gate_satisfaction_gain"], start=1)
        ],
    )


def contract_from_any(
    value: EvaluationContract | dict[str, Any] | None,
    *,
    objective: str = "user objective",
) -> EvaluationContract:
    """Coerce arbitrary meta-contract input into an ``EvaluationContract``."""

    if isinstance(value, EvaluationContract):
        return value
    if isinstance(value, dict) and value:
        return EvaluationContract.from_dict(value)
    return objective_contract_from_task(objective)


__all__.extend(["objective_contract_from_task", "contract_from_any", "ContractItem", "EvaluationContract"])


# ---- Nexus offline evolution objective contracts ----
import hashlib
import json


@dataclass
class NexusObjectiveContract:
    """Task-bound objective boundary for the Nexus runtime.

    A model may draft this contract from the input packet, but once persisted the
    platform treats its hash as immutable.  Candidates can reference the hash;
    they cannot silently replace the user's goal with a helper objective.
    """

    original_user_goal: str
    normalized_goal: str
    task_type: str = DEFAULT_TASK_TYPE
    outcome_policy: dict[str, Any] = field(default_factory=lambda: {
        "model_driven": True,
        "accepts_best_current_route": True,
        "requires_strict_optimum": False,
        "requires_verified_solution": False,
        "final_claim_policy": "do_not_claim_absolute_optimality_or_solution_unless_verified",
    })
    dynamic_artifact_contract: dict[str, Any] = field(default_factory=dict)
    input_constraints: list[str] = field(default_factory=list)
    allowed_evidence_sources: list[str] = field(default_factory=lambda: ["input_evidence", "tool_evidence", "model_hypothesis"])
    disallowed_goal_mutations: list[str] = field(default_factory=list)
    expected_output_forms: list[str] = field(default_factory=lambda: ["answer", "patch", "report", "failure_analysis"])
    uncertainty_policy: str = "label_uncertainty_and_keep_search_seeds_separate_from_verified_evidence"
    verification_preferences: list[str] = field(default_factory=list)
    success_dimensions: list[str] = field(default_factory=lambda: ["objective_alignment", "verifiability", "robustness"])
    failure_dimensions: list[str] = field(default_factory=lambda: ["semantic_drift", "unsupported_claim", "auxiliary_substitution"])
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_id: str = "nexus-objective-contract"
    version: str = "nexus/objective-contract/v1"
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("dynamic_artifact_contract"):
            policy = dict(data.get("outcome_policy") or {})
            policy.setdefault("dynamic_artifact_contract", data["dynamic_artifact_contract"])
            data["outcome_policy"] = policy
            data["dynamic_artifact_contract_hash"] = self.dynamic_artifact_contract_hash()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NexusObjectiveContract":
        return cls(
            original_user_goal=str(data.get("original_user_goal") or data.get("normalized_goal") or "user goal"),
            normalized_goal=str(data.get("normalized_goal") or data.get("original_user_goal") or "user goal"),
            task_type=str(data.get("task_type") or DEFAULT_TASK_TYPE).strip() or DEFAULT_TASK_TYPE,
            outcome_policy=dict(data.get("outcome_policy") or {
                "model_driven": True,
                "accepts_best_current_route": True,
                "requires_strict_optimum": False,
                "requires_verified_solution": False,
                "final_claim_policy": "do_not_claim_absolute_optimality_or_solution_unless_verified",
            }),
            dynamic_artifact_contract=_coerce_dynamic_artifact_contract(data),
            input_constraints=[str(item) for item in data.get("input_constraints", [])],
            allowed_evidence_sources=[str(item) for item in data.get("allowed_evidence_sources", [])] or ["input_evidence", "tool_evidence", "model_hypothesis"],
            disallowed_goal_mutations=[str(item) for item in data.get("disallowed_goal_mutations", [])],
            expected_output_forms=[str(item) for item in data.get("expected_output_forms", [])] or ["answer", "patch", "report", "failure_analysis"],
            uncertainty_policy=str(data.get("uncertainty_policy") or "label_uncertainty_and_keep_search_seeds_separate_from_verified_evidence"),
            verification_preferences=[str(item) for item in data.get("verification_preferences", [])],
            success_dimensions=[str(item) for item in data.get("success_dimensions", [])] or ["objective_alignment", "verifiability", "robustness"],
            failure_dimensions=[str(item) for item in data.get("failure_dimensions", [])] or ["semantic_drift", "unsupported_claim", "auxiliary_substitution"],
            metadata=coerce_dict(data.get("metadata")),
            contract_id=str(data.get("contract_id") or "nexus-objective-contract"),
            version=str(data.get("version") or "nexus/objective-contract/v1"),
            created_at=str(data.get("created_at") or _now()),
        )

    def canonical_payload(self) -> dict[str, Any]:
        data = self.to_dict()
        data.pop("created_at", None)
        data.pop("metadata", None)
        return data

    def contract_hash(self) -> str:
        payload = json.dumps(self.canonical_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def dynamic_artifact_contract_hash(self) -> str:
        dac = DynamicArtifactContract.from_any(self.dynamic_artifact_contract, fallback_objective=self.normalized_goal)
        return dac.stable_hash() if dac is not None else ""

    def validate_dynamic_artifact_contract(self) -> ValidationResult:
        dac = DynamicArtifactContract.from_any(self.dynamic_artifact_contract, fallback_objective=self.normalized_goal)
        summary = validate_dynamic_artifact_contract(dac)
        return ValidationResult(summary.valid, errors=list(summary.diagnostics))

    def validate_candidate_contract_hash(self, candidate_hash: str | None) -> ValidationResult:
        expected = self.contract_hash()
        if candidate_hash and candidate_hash == expected:
            return ValidationResult(True)
        return ValidationResult(False, errors=["candidate_contract_hash_does_not_match_objective_contract"])


@dataclass
class NexusProjectObjectiveContract(NexusObjectiveContract):
    frozen_regions: list[str] = field(default_factory=list)
    mutable_regions: list[str] = field(default_factory=list)
    contract_files: list[str] = field(default_factory=list)
    implementation_files: list[str] = field(default_factory=list)
    test_contracts: list[str] = field(default_factory=list)
    allowed_patch_scope: list[str] = field(default_factory=list)
    unsafe_change_patterns: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NexusProjectObjectiveContract":
        base = NexusObjectiveContract.from_dict(data)
        base_data = base.to_dict()
        base_data.pop("dynamic_artifact_contract_hash", None)
        return cls(
            **base_data,
            frozen_regions=[str(item) for item in data.get("frozen_regions", [])],
            mutable_regions=[str(item) for item in data.get("mutable_regions", [])],
            contract_files=[str(item) for item in data.get("contract_files", [])],
            implementation_files=[str(item) for item in data.get("implementation_files", [])],
            test_contracts=[str(item) for item in data.get("test_contracts", [])],
            allowed_patch_scope=[str(item) for item in data.get("allowed_patch_scope", [])],
            unsafe_change_patterns=[str(item) for item in data.get("unsafe_change_patterns", [])],
        )


@dataclass(frozen=True)
class _ArtifactPolicyView:
    machine_readable_required: bool = False
    allow_text_fallback: bool = True
    allow_refold_for_probe: bool = True
    allow_refold_for_final: bool = False
    final_requires_certificate: bool = False
    projection_required: bool = True
    artifact_type: str = ""
    artifact_type_aliases: dict[str, str] = field(default_factory=dict)
    field_aliases: dict[str, str] = field(default_factory=dict)
    required_fields: list[str] = field(default_factory=list)
    final_requires_clean_schema: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_any(cls, data: Any) -> "_ArtifactPolicyView":
        if isinstance(data, cls):
            return data
        if data is not None and not isinstance(data, dict):
            return cls(
                machine_readable_required=_truthy(getattr(data, "machine_readable_required", False)),
                allow_text_fallback=_truthy(getattr(data, "allow_text_fallback", True)),
                allow_refold_for_probe=_truthy(getattr(data, "allow_refold_for_probe", True)),
                allow_refold_for_final=_truthy(getattr(data, "allow_refold_for_final", False)),
                final_requires_certificate=_truthy(getattr(data, "final_requires_certificate", False)),
                projection_required=_truthy(getattr(data, "projection_required", True)),
                artifact_type=str(getattr(data, "artifact_type", "") or ""),
                artifact_type_aliases={str(k): str(v) for k, v in coerce_dict(getattr(data, "artifact_type_aliases", {})).items()},
                field_aliases={str(k): str(v) for k, v in coerce_dict(getattr(data, "field_aliases", {})).items()},
                required_fields=[str(item) for item in getattr(data, "required_fields", []) if str(item).strip()],
                final_requires_clean_schema=_truthy(getattr(data, "final_requires_clean_schema", True)),
                metadata=coerce_dict(getattr(data, "metadata", {})),
            )
        cfg = coerce_dict(data)
        evidence = coerce_dict(cfg.get("evidence"))
        merged = {**cfg, **evidence}
        if "machine_artifact_required" in merged and "machine_readable_required" not in merged:
            merged["machine_readable_required"] = merged.get("machine_artifact_required")
        return cls(
            machine_readable_required=_truthy(merged.get("machine_readable_required")),
            allow_text_fallback=_truthy(merged.get("allow_text_fallback", True)),
            allow_refold_for_probe=_truthy(merged.get("allow_refold_for_probe", True)),
            allow_refold_for_final=_truthy(merged.get("allow_refold_for_final", False)),
            final_requires_certificate=_truthy(merged.get("final_requires_certificate", False)),
            projection_required=_truthy(merged.get("projection_required", True)),
            artifact_type=str(merged.get("artifact_type") or ""),
            artifact_type_aliases={str(k): str(v) for k, v in coerce_dict(merged.get("artifact_type_aliases")).items() if str(k or "").strip() and str(v or "").strip()},
            field_aliases={str(k): str(v) for k, v in coerce_dict(merged.get("field_aliases")).items() if str(k or "").strip() and str(v or "").strip()},
            required_fields=[str(item) for item in merged.get("required_fields", []) if str(item).strip()] if isinstance(merged.get("required_fields"), list) else [],
            final_requires_clean_schema=_truthy(merged.get("final_requires_clean_schema", True)),
            metadata=coerce_dict(merged.get("metadata")),
        )


class NexusObjectiveContractBuilder:
    """Deterministic fallback builder; production callers may delegate to a model."""

    def build_text_contract(
        self,
        *,
        user_goal: str,
        packet: Any,
        world: Any | None = None,
        model: NexusModelLike | None = None,
        artifact_policy_config: dict[str, Any] | None = None,
    ) -> NexusObjectiveContract:
        if model is not None and hasattr(model, "build_objective_contract"):
            raw = model.build_objective_contract(user_goal=user_goal, world=world if world is not None else packet)
            if isinstance(raw, NexusObjectiveContract):
                _attach_latent_objective_state(raw, world if world is not None else packet)
                apply_artifact_policy_to_contract(raw, artifact_policy_config, source="adaptive.evidence")
                return raw
            if isinstance(raw, dict):
                contract = NexusObjectiveContract.from_dict(raw)
                _attach_latent_objective_state(contract, world if world is not None else packet)
                apply_artifact_policy_to_contract(contract, artifact_policy_config, source="adaptive.evidence")
                return contract
        constraints = [str(item) for item in getattr(packet, "constraints", [])]
        contract = NexusObjectiveContract(
            original_user_goal=user_goal,
            normalized_goal=" ".join(str(user_goal).split()) or "user goal",
            dynamic_artifact_contract=_default_dynamic_artifact_contract(user_goal),
            input_constraints=constraints,
            disallowed_goal_mutations=["replace_original_goal_with_router_or_validator", "optimize_auxiliary_scaffold_as_final_answer"],
            verification_preferences=["prefer_input_evidence", "prefer_local_tool_evidence"],
        )
        _attach_latent_objective_state(contract, world if world is not None else packet)
        apply_artifact_policy_to_contract(contract, artifact_policy_config, source="adaptive.evidence")
        return contract

    def build_project_contract(
        self,
        *,
        user_goal: str,
        snapshot: Any,
        world: Any | None = None,
        model: NexusModelLike | None = None,
        artifact_policy_config: dict[str, Any] | None = None,
    ) -> NexusProjectObjectiveContract:
        if model is not None and hasattr(model, "build_project_objective_contract"):
            raw = model.build_project_objective_contract(user_goal=user_goal, snapshot=snapshot, world=world)
            if isinstance(raw, NexusProjectObjectiveContract):
                _attach_latent_objective_state(raw, world if world is not None else snapshot)
                apply_artifact_policy_to_contract(raw, artifact_policy_config, source="adaptive.evidence")
                return raw
            if isinstance(raw, dict):
                contract = NexusProjectObjectiveContract.from_dict(raw)
                _attach_latent_objective_state(contract, world if world is not None else snapshot)
                apply_artifact_policy_to_contract(contract, artifact_policy_config, source="adaptive.evidence")
                return contract
        manifest = [str(item.get("path")) for item in getattr(snapshot, "file_manifest", []) if isinstance(item, dict)]
        tests = [path for path in manifest if "/test" in f"/{path}" or path.startswith("tests/")]
        py_files = [path for path in manifest if path.endswith(".py")]
        contract = NexusProjectObjectiveContract(
            original_user_goal=user_goal,
            normalized_goal=" ".join(str(user_goal).split()) or "project goal",
            dynamic_artifact_contract=_default_dynamic_artifact_contract(user_goal),
            input_constraints=["preserve_existing_public_imports", "verify_with_local_tools_when_available"],
            disallowed_goal_mutations=["replace_project_goal_with_framework_only", "claim_success_without_patch_or_verification_trace"],
            verification_preferences=["compileall", "pytest", "schema_validation"],
            mutable_regions=py_files,
            implementation_files=py_files,
            test_contracts=tests,
            allowed_patch_scope=py_files + tests,
            unsafe_change_patterns=["load_user_home_env_in_tests", "real_provider_fallback_in_tests"],
        )
        _attach_latent_objective_state(contract, world if world is not None else snapshot)
        apply_artifact_policy_to_contract(contract, artifact_policy_config, source="adaptive.evidence")
        return contract


def apply_artifact_policy_to_contract(
    contract: NexusObjectiveContract,
    artifact_policy_config: dict[str, Any] | None,
    *,
    source: str = "adaptive.evidence",
) -> NexusObjectiveContract:
    """Overlay explicit machine-artifact policy onto the model-defined contract.

    The objective contract remains the single Nexus contract authority.  The
    Evidence Control Plane may provide stricter artifact requirements for a
    machine artifact task; those requirements must be compiled into the dynamic
    artifact contract instead of living only in evaluator metadata.
    """

    policy = _ArtifactPolicyView.from_any(artifact_policy_config)
    if not _artifact_policy_requires_contract_overlay(policy):
        return contract
    previous_dac = DynamicArtifactContract.from_any(contract.dynamic_artifact_contract, fallback_objective=contract.normalized_goal)
    diagnostics = artifact_policy_contract_conflicts(policy, contract)
    previous_hash = previous_dac.stable_hash() if previous_dac is not None else ""
    overlay = dynamic_artifact_contract_from_artifact_policy(policy, objective=contract.normalized_goal, base=previous_dac)
    contract.dynamic_artifact_contract = overlay.to_dict()
    outcome_policy = dict(contract.outcome_policy or {})
    outcome_policy["dynamic_artifact_contract"] = contract.dynamic_artifact_contract
    outcome_policy["machine_artifact_policy_bound"] = True
    if policy.final_requires_clean_schema:
        outcome_policy["requires_clean_machine_artifact_for_final"] = True
    contract.outcome_policy = outcome_policy
    metadata = coerce_dict(contract.metadata)
    metadata["artifact_policy_contract_overlay"] = {
        "source": str(source or "adaptive.evidence"),
        "artifact_type": policy.artifact_type,
        "required_fields": list(policy.required_fields),
        "previous_dynamic_artifact_contract_hash": previous_hash,
        "new_dynamic_artifact_contract_hash": overlay.stable_hash(),
        "diagnostics": diagnostics,
    }
    if diagnostics:
        metadata["contract_artifact_policy_conflict_diagnostics"] = diagnostics
    contract.metadata = metadata
    return contract


def dynamic_artifact_contract_from_artifact_policy(
    policy: Any,
    *,
    objective: str,
    base: DynamicArtifactContract | dict[str, Any] | None = None,
) -> DynamicArtifactContract:
    """Compile an ArtifactPolicy-compatible mapping into the Nexus contract."""

    artifact_policy = _ArtifactPolicyView.from_any(policy)
    base_dac = DynamicArtifactContract.from_any(base, fallback_objective=objective)
    artifact_type = artifact_policy.artifact_type.strip() or (
        base_dac.artifact_domain_label if base_dac is not None and base_dac.artifact_domain_label != "model_defined_artifact" else "machine_artifact"
    )
    required_fields = list(dict.fromkeys(str(item) for item in artifact_policy.required_fields if str(item).strip()))
    invalid_outputs = list(dict.fromkeys(
        [
            *((base_dac.invalid_outputs if base_dac is not None else []) or []),
            "empty output",
            "meta commentary only",
            "restating objective without artifact",
            "natural-language wrapper instead of the machine artifact",
            "string-wrapped JSON object",
        ]
        + [f"artifact_type alias: {alias}" for alias in artifact_policy.artifact_type_aliases]
        + [f"field alias: {alias}" for alias in artifact_policy.field_aliases]
    ))
    adapter_requirements = dict(base_dac.adapter_requirements if base_dac is not None else {})
    adapter_requirements.update(artifact_policy.to_dict())
    adapter_requirements["artifact_policy_source"] = "evidence_control_plane"
    final_gate = dict(base_dac.final_gate if base_dac is not None else {})
    final_gate.update(
        {
            "check": "clean machine artifact schema plus evaluator/certificate evidence",
            "artifact_type": artifact_type,
            "requires_clean_schema": bool(artifact_policy.final_requires_clean_schema),
            "allow_refold_for_final": bool(artifact_policy.allow_refold_for_final),
            "requires_certificate": bool(artifact_policy.final_requires_certificate),
        }
    )
    repair_contract = dict(base_dac.repair_contract if base_dac is not None else {})
    repair_contract.update(
        {
            "on_missing_artifact": f"emit a clean machine-readable {artifact_type} artifact",
            "on_alias_or_refolded_artifact": "re-emit the artifact with exact artifact_type and exact required field names",
            "refolded_rule": "refolded artifacts may be probed when policy allows, but are not final eligible unless explicitly allowed",
        }
    )
    evaluation_dimensions = list(base_dac.evaluation_dimensions if base_dac is not None else [])
    for item in (
        {"name": "schema_cleanliness", "measurement": "normalized artifact status and required field presence"},
        {"name": "evaluator_score", "measurement": "task-local evaluator result on the normalized artifact"},
        {"name": "semantic_drift_absence", "measurement": "absence of forbidden runtime/internal vocabulary in the machine artifact"},
    ):
        if not any(str(existing.get("name") or "") == item["name"] for existing in evaluation_dimensions):
            evaluation_dimensions.append(item)
    return DynamicArtifactContract(
        objective=str(objective or (base_dac.objective if base_dac is not None else "") or "user objective"),
        artifact_domain_label=artifact_type,
        required_work_product={
            "artifact_type": artifact_type,
            "required_fields": required_fields,
            "description": f"a clean machine-readable {artifact_type} artifact matching the task-local ArtifactPolicy",
        },
        allowed_artifact_shapes=[
            {
                "name": artifact_type,
                "required_fields": required_fields,
                "machine_readable_required": bool(artifact_policy.machine_readable_required),
                "final_eligible": True,
            }
        ],
        minimum_concrete_delta=dict(base_dac.minimum_concrete_delta if base_dac is not None else {})
        or {"observable_signal": "specific artifact field, parameter, rule, score, or evaluator behavior changes relative to parent"},
        invalid_outputs=invalid_outputs,
        evaluation_dimensions=evaluation_dimensions,
        comparison_method=dict(base_dac.comparison_method if base_dac is not None else {})
        or {"method": "schema validation plus evaluator-backed relative comparison under the frozen artifact policy"},
        final_gate=final_gate,
        repair_contract=repair_contract,
        adapter_requirements=adapter_requirements,
        version=(base_dac.version if base_dac is not None else "dynamic-artifact-contract/v1"),
    )


def artifact_policy_contract_conflicts(policy_or_config: Any, contract: NexusObjectiveContract | dict[str, Any] | None) -> list[str]:
    """Return policy/contract mismatch diagnostics before overlay is applied."""

    policy = _ArtifactPolicyView.from_any(policy_or_config)
    if not _artifact_policy_requires_contract_overlay(policy):
        return []
    fallback = ""
    if contract is not None and not isinstance(contract, dict):
        fallback = str(getattr(contract, "normalized_goal", "") or getattr(contract, "original_user_goal", "") or "")
        source = getattr(contract, "dynamic_artifact_contract", None)
    else:
        data = coerce_dict(contract)
        fallback = str(data.get("normalized_goal") or data.get("original_user_goal") or "")
        source = _coerce_dynamic_artifact_contract(data)
    dac = DynamicArtifactContract.from_any(source, fallback_objective=fallback)
    if dac is None:
        return ["contract_artifact_policy_conflict: dynamic_artifact_contract_absent"]
    diagnostics: list[str] = []
    artifact_type = policy.artifact_type.strip()
    shape_names = {str(item.get("name") or item.get("artifact_type") or "").strip() for item in dac.allowed_artifact_shapes if isinstance(item, dict)}
    if artifact_type:
        labels = {str(dac.artifact_domain_label or "").strip(), *shape_names}
        if artifact_type not in labels:
            diagnostics.append(f"contract_artifact_policy_conflict: artifact_type_missing expected={artifact_type}")
    available_fields: set[str] = set()
    for item in dac.allowed_artifact_shapes:
        if isinstance(item, dict):
            available_fields.update(str(field) for field in item.get("required_fields", []) if str(field).strip())
    missing = [field for field in policy.required_fields if field not in available_fields]
    if missing:
        diagnostics.append("contract_artifact_policy_conflict: required_fields_missing=" + ",".join(missing))
    adapter_requirements = coerce_dict(dac.adapter_requirements)
    if policy.machine_readable_required and not (
        _truthy(adapter_requirements.get("machine_readable_required"))
        or _truthy(adapter_requirements.get("machine_artifact_required"))
        or _truthy(adapter_requirements.get("machine_readable"))
    ):
        diagnostics.append("contract_artifact_policy_conflict: machine_readable_requirement_missing")
    final_gate = coerce_dict(dac.final_gate)
    if policy.final_requires_clean_schema and not (
        _truthy(final_gate.get("requires_clean_schema"))
        or "clean" in json.dumps(final_gate, ensure_ascii=False, sort_keys=True, default=str).lower()
    ):
        diagnostics.append("contract_artifact_policy_conflict: final_clean_schema_gate_missing")
    return diagnostics


def _coerce_dynamic_artifact_contract(data: dict[str, Any]) -> dict[str, Any]:
    fallback_objective = str(data.get("normalized_goal") or data.get("original_user_goal") or "")
    for key in ("dynamic_artifact_contract", "artifact_contract", "model_artifact_contract"):
        value = data.get(key)
        if isinstance(value, dict):
            dac = DynamicArtifactContract.from_any(value, fallback_objective=fallback_objective)
            return dac.to_dict() if dac is not None else dict(value)
    outcome_policy = data.get("outcome_policy") if isinstance(data.get("outcome_policy"), dict) else {}
    for key in ("dynamic_artifact_contract", "artifact_contract", "model_artifact_contract"):
        value = outcome_policy.get(key)
        if isinstance(value, dict):
            dac = DynamicArtifactContract.from_any(value, fallback_objective=fallback_objective)
            return dac.to_dict() if dac is not None else dict(value)
    return {}


def _default_dynamic_artifact_contract(objective: str) -> dict[str, Any]:
    objective_text = " ".join(str(objective or "user objective").split()) or "user objective"
    return DynamicArtifactContract(
        objective=objective_text,
        artifact_domain_label="model_defined_artifact",
        required_work_product={"description": "a concrete objective-appropriate artifact, not only commentary about one"},
        allowed_artifact_shapes=[
            {"name": "model_defined_object", "required_fields": ["content_or_structured_object"]},
            {
                "name": "design_candidate",
                "stage": "exploration_non_final",
                "required_fields": ["mechanism", "evaluation_dimensions", "design_diff", "failure_conditions"],
            },
        ],
        minimum_concrete_delta={"observable_signal": "specific changed material, new object structure, or measurable evidence relative to parent"},
        invalid_outputs=["empty output", "meta commentary only", "restating objective without artifact"],
        evaluation_dimensions=[{"name": "objective_fit", "measurement": "comparison against the frozen user objective and artifact contract"}],
        comparison_method={"method": "relative comparison under the frozen contract"},
        final_gate={"check": "structural artifact presence plus independent comparison/evidence linkage"},
        repair_contract={
            "on_missing_artifact": "produce the required work product with an observable delta",
            "design_candidate_rule": "exploration design candidates may rank and reproduce when structurally complete, but are never final eligible until materialized into the contract's required work product",
        },
        adapter_requirements={},
    ).to_dict()


def _artifact_policy_requires_contract_overlay(policy: _ArtifactPolicyView) -> bool:
    return bool(
        policy.machine_readable_required
        or policy.artifact_type.strip()
        or policy.required_fields
        or policy.artifact_type_aliases
        or policy.field_aliases
        or policy.final_requires_certificate
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "enabled", "required"}
    return bool(value)


def _attach_latent_objective_state(contract: NexusObjectiveContract, world: Any | None) -> None:
    """Best-effort bridge hook; contract building must not fail on M5.1 metadata."""

    try:
        from cognitive_evolve_runtime.outcomes.runtime_bridge import attach_latent_state_if_needed

        attach_latent_state_if_needed(contract, world)
    except Exception:
        metadata = coerce_dict(getattr(contract, "metadata", {}))
        metadata["latent_problem_state_error"] = "latent_state_initialization_failed"
        contract.metadata = metadata


__all__.extend([
    "NexusObjectiveContract",
    "NexusProjectObjectiveContract",
    "NexusObjectiveContractBuilder",
    "apply_artifact_policy_to_contract",
    "artifact_policy_contract_conflicts",
    "dynamic_artifact_contract_from_artifact_policy",
])
