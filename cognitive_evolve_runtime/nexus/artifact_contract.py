"""Model-defined artifact contracts for domain-agnostic evolution.

The deterministic runtime must not decide that a task is "code", "proof",
"article", or "fiction" from a finite taxonomy.  A model may define the task's
artifact semantics per run, while this module enforces only meta-invariants:
contract shape, object-level artifact presence, concrete delta, claim binding,
non-self-certified final gates, and explicit opt-in to external adapters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import hashlib
import json
import re

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus._serde import coerce_dict


CONTRACT_KEYS = (
    "dynamic_artifact_contract",
    "artifact_contract",
    "model_artifact_contract",
)

# These are capability adapter identifiers, not domain labels.  Runtime code may
# branch on capabilities because they bind to deterministic tools/safety checks.
ADAPTER_ALIASES = {
    "patch": {"patch", "code_patch", "project_patch", "diff", "unified_diff"},
    "source": {"source", "source_binding", "source_lineage", "filesystem", "file_materialization"},
    "proof": {"proof", "formal", "formal_artifact", "derivation"},
    "test": {"test", "tests", "pytest", "verification_command", "external_tool"},
}

SELF_CERTIFYING_PATTERNS = re.compile(
    r"\b(model|generator|candidate|llm|assistant)\s+(?:says|judges|decides|claims|believes|confirms)|"
    r"\bconfidence\s*[>=]|\bself[-_ ]?certif|模型认为|候选自证|自我认证|我认为已完成",
    re.IGNORECASE,
)

VACUOUS_DELTA_PATTERNS = re.compile(
    r"^(?:better|improved|good|quality|更好|改善|优化|提升|变好)$|"
    r"\b(?:is better|becomes better|more good|quality improves)\b|"
    r"(?:变得更好|质量提升)",
    re.IGNORECASE,
)

META_ONLY_PATTERNS = re.compile(
    r"^(?:\s*(?:we|i|the candidate|this candidate)?\s*(?:should|would|could|will|need to|needs to|must)\b)|"
    r"\b(?:proposal|plan|recommendation|should add|should create|needs a|need a)\b|"
    r"(?:应该|建议|需要|计划|方案|可以考虑)",
    re.IGNORECASE,
)

OBJECT_MARKER_KEYS = {
    "content",
    "text",
    "body",
    "value",
    "revised_text",
    "draft",
    "fragment",
    "scene",
    "answer",
    "patch",
    "diff",
    "unified_diff",
    "patch_set",
    "code",
    "algorithm",
    "pseudocode",
    "proof",
    "derivation",
    "table",
    "items",
}

DELTA_KEYS = {
    "artifact_delta",
    "design_delta",
    "design_diff",
    "revision_delta",
    "material_delta",
    "delta",
    "changes",
    "differences",
    "changed_focus",
    "relative_to_parent",
    "improvement_claim",
    "new_evidence",
    "new_obligation",
}


@dataclass(frozen=True)
class DynamicArtifactContract:
    objective: str
    artifact_domain_label: str = "model_defined_artifact"
    required_work_product: dict[str, Any] = field(default_factory=dict)
    allowed_artifact_shapes: list[dict[str, Any]] = field(default_factory=list)
    minimum_concrete_delta: dict[str, Any] = field(default_factory=dict)
    invalid_outputs: list[str] = field(default_factory=list)
    evaluation_dimensions: list[dict[str, Any]] = field(default_factory=list)
    comparison_method: dict[str, Any] = field(default_factory=dict)
    final_gate: dict[str, Any] = field(default_factory=dict)
    repair_contract: dict[str, Any] = field(default_factory=dict)
    adapter_requirements: dict[str, Any] = field(default_factory=dict)
    version: str = "dynamic-artifact-contract/v1"

    @classmethod
    def from_any(cls, value: Any, *, fallback_objective: str = "") -> "DynamicArtifactContract | None":
        if isinstance(value, DynamicArtifactContract):
            return value
        if not isinstance(value, dict):
            return None
        data = dict(value)
        objective = str(data.get("objective") or data.get("task_objective") or fallback_objective or "user objective")
        return cls(
            objective=objective,
            artifact_domain_label=str(data.get("artifact_domain_label") or data.get("domain_label") or data.get("label") or "model_defined_artifact"),
            required_work_product=_as_mapping(data.get("required_work_product") or data.get("work_product")),
            allowed_artifact_shapes=_as_mapping_list(data.get("allowed_artifact_shapes") or data.get("artifact_shapes")),
            minimum_concrete_delta=_as_mapping(data.get("minimum_concrete_delta") or data.get("concrete_delta") or data.get("minimum_delta")),
            invalid_outputs=[str(item) for item in _as_list(data.get("invalid_outputs"))],
            evaluation_dimensions=_as_mapping_list(data.get("evaluation_dimensions") or data.get("dimensions")),
            comparison_method=_as_mapping(data.get("comparison_method")),
            final_gate=_as_mapping(data.get("final_gate") or data.get("final_gate_criteria")),
            repair_contract=_as_mapping(data.get("repair_contract")),
            adapter_requirements=_as_mapping(data.get("adapter_requirements") or data.get("adapters")),
            version=str(data.get("version") or "dynamic-artifact-contract/v1"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def stable_hash(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ContractValidationSummary:
    required: bool
    valid: bool = True
    diagnostics: list[str] = field(default_factory=list)
    contract_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactContractSummary:
    candidate_id: str
    required: bool
    contract_valid: bool = True
    rank_eligible: bool = True
    final_eligible: bool = True
    diagnostics: list[str] = field(default_factory=list)
    adapter_requirements: dict[str, Any] = field(default_factory=dict)
    contract_hash: str = ""
    artifact_present: bool = False
    concrete_delta_present: bool = False
    claim_bound: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dynamic_artifact_contract_from(contract: Any | None = None, candidate: CandidateGenome | None = None) -> DynamicArtifactContract | None:
    """Extract a model-defined artifact contract without using finite domains."""

    fallback_objective = ""
    if contract is not None:
        fallback_objective = str(getattr(contract, "original_user_goal", "") or getattr(contract, "normalized_goal", "") or "")
    for source in _contract_sources(contract=contract, candidate=candidate):
        for key in CONTRACT_KEYS:
            if key in source:
                dac = DynamicArtifactContract.from_any(source.get(key), fallback_objective=fallback_objective)
                if dac is not None:
                    return dac
        # Allow callers to pass the contract object itself as a plain mapping.
        if any(key in source for key in ("required_work_product", "allowed_artifact_shapes", "minimum_concrete_delta", "final_gate")):
            dac = DynamicArtifactContract.from_any(source, fallback_objective=fallback_objective)
            if dac is not None:
                return dac
    return None


def validate_dynamic_artifact_contract(dac: DynamicArtifactContract | None) -> ContractValidationSummary:
    if dac is None:
        return ContractValidationSummary(required=False, valid=True)
    diagnostics: list[str] = []
    if not str(dac.objective or "").strip():
        diagnostics.append("contract_objective_absent")
    if not dac.required_work_product:
        diagnostics.append("required_work_product_absent")
    if not dac.allowed_artifact_shapes:
        diagnostics.append("allowed_artifact_shapes_absent")
    if not dac.minimum_concrete_delta:
        diagnostics.append("minimum_concrete_delta_absent")
    elif _mapping_is_vacuous_delta(dac.minimum_concrete_delta):
        diagnostics.append("delta_unmeasurable")
    if not dac.evaluation_dimensions:
        diagnostics.append("evaluation_dimensions_absent")
    if not dac.final_gate:
        diagnostics.append("final_gate_absent")
    elif _final_gate_self_certifies(dac.final_gate):
        diagnostics.append("final_gate_self_certifying")
    if not _invalid_outputs_cover_meta_failures(dac.invalid_outputs):
        diagnostics.append("invalid_outputs_underconstrained")
    return ContractValidationSummary(required=True, valid=not diagnostics, diagnostics=list(dict.fromkeys(diagnostics)), contract_hash=dac.stable_hash())


def evaluate_candidate_against_dynamic_contract(
    candidate: CandidateGenome,
    *,
    contract: Any | None = None,
) -> ArtifactContractSummary:
    dac = dynamic_artifact_contract_from(contract=contract, candidate=candidate)
    validation = validate_dynamic_artifact_contract(dac)
    if dac is None:
        return ArtifactContractSummary(candidate_id=candidate.id, required=False)
    diagnostics = list(validation.diagnostics)
    artifact_present = candidate_has_object_level_artifact(candidate)
    meta_only = candidate_is_meta_commentary_only(candidate)
    concrete_delta = candidate_has_concrete_delta(candidate)
    claim_bound = candidate_claim_bound_to_artifact(candidate)
    design_candidate = _design_candidate_allowed(dac) and _candidate_has_structured_design_candidate(candidate)
    if design_candidate:
        artifact_present = True
        meta_only = False
        concrete_delta = True
        claim_bound = True

    if not artifact_present:
        diagnostics.append("artifact_object_absent")
    if artifact_present and meta_only:
        diagnostics.append("meta_commentary_only")
    if not concrete_delta:
        diagnostics.append("concrete_delta_absent")
    if artifact_present and not claim_bound:
        diagnostics.append("claim_artifact_unbound")
    if _design_candidate_allowed(dac) and _candidate_declares_design_candidate(candidate) and not design_candidate:
        diagnostics.append("design_candidate_incomplete")
    if design_candidate:
        diagnostics.append("design_candidate_non_final")

    final_eligible = not diagnostics and not design_candidate
    rank_eligible = not any(
        item in diagnostics
        for item in ("contract_objective_absent", "final_gate_self_certifying", "artifact_object_absent", "meta_commentary_only", "design_candidate_incomplete")
    )
    return ArtifactContractSummary(
        candidate_id=candidate.id,
        required=True,
        contract_valid=validation.valid,
        rank_eligible=rank_eligible,
        final_eligible=final_eligible,
        diagnostics=list(dict.fromkeys(diagnostics)),
        adapter_requirements=dict(dac.adapter_requirements),
        contract_hash=dac.stable_hash(),
        artifact_present=artifact_present,
        concrete_delta_present=concrete_delta,
        claim_bound=claim_bound,
    )


def contract_requires_adapter(contract: Any | None, adapter: str, *, candidate: CandidateGenome | None = None) -> bool | None:
    """Return True/False for DAC adapter opt-in, or None when no DAC exists."""

    dac = dynamic_artifact_contract_from(contract=contract, candidate=candidate)
    if dac is None:
        return None
    requirements = {str(k).lower(): v for k, v in dict(dac.adapter_requirements or {}).items()}
    aliases = ADAPTER_ALIASES.get(adapter, {adapter})
    for key in aliases | {adapter}:
        if key in requirements:
            return _truthy(requirements[key])
        prefixed = "requires_" + key
        if prefixed in requirements:
            return _truthy(requirements[prefixed])
        needs = "needs_" + key
        if needs in requirements:
            return _truthy(requirements[needs])
    # A generic list form is acceptable because it lists capabilities, not domains.
    for list_key in ("required", "enabled", "adapters", "adapter_ids", "capabilities"):
        raw = requirements.get(list_key)
        if isinstance(raw, (list, tuple, set)):
            lowered = {str(item).strip().lower() for item in raw}
            if aliases.intersection(lowered):
                return True
    return False


def materialization_scope_from_contract(contract: Any | None, *, candidate: CandidateGenome | None = None) -> list[str] | None:
    dac = dynamic_artifact_contract_from(contract=contract, candidate=candidate)
    if dac is None:
        return None
    req = coerce_dict(dac.adapter_requirements)
    raw = req.get("materialization_scope") or req.get("allowed_materialization_scope") or req.get("allowed_paths")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple, set)):
        return [str(item) for item in raw if str(item).strip()]
    return None


def candidate_has_object_level_artifact(candidate: CandidateGenome) -> bool:
    artifact = getattr(candidate, "artifact", None)
    if _value_has_object_content(artifact):
        return True
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    for key in ("artifact_object", "work_product", "required_work_product", "candidate_artifact", "output_artifact"):
        if _value_has_object_content(metadata.get(key)):
            return True
    if getattr(candidate, "patch_set", None) or candidate.formal_artifacts:
        return True
    return False


def candidate_has_concrete_delta(candidate: CandidateGenome) -> bool:
    if coerce_dict(getattr(candidate, "evidence_delta", {})) or coerce_dict(getattr(candidate, "obligation_delta", {})):
        return True
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    for key in DELTA_KEYS:
        if _value_has_object_content(metadata.get(key)):
            return True
    artifact = getattr(candidate, "artifact", None)
    if isinstance(artifact, dict):
        for key in DELTA_KEYS:
            if _value_has_object_content(artifact.get(key)):
                return True
    if candidate.verification_trace or getattr(candidate, "patch_set", None):
        return True
    return False


def _design_candidate_allowed(dac: DynamicArtifactContract | None) -> bool:
    if dac is None:
        return False
    sources: list[Any] = [dac.allowed_artifact_shapes, dac.repair_contract, dac.required_work_product]
    text = _normalize_words(_stringify_artifact(sources))
    return "design_candidate" in text


def _candidate_declares_design_candidate(candidate: CandidateGenome) -> bool:
    artifact = getattr(candidate, "artifact", None)
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    artifact_type = str(getattr(candidate, "artifact_type", "") or "").strip().lower()
    if artifact_type == "design_candidate":
        return True
    for source in (artifact, metadata):
        if isinstance(source, dict):
            kind = str(source.get("kind") or source.get("type") or source.get("artifact_shape") or source.get("shape") or "").strip().lower()
            if kind == "design_candidate":
                return True
    return False


def _candidate_has_structured_design_candidate(candidate: CandidateGenome) -> bool:
    if not _candidate_declares_design_candidate(candidate):
        return False
    artifact = getattr(candidate, "artifact", None)
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    sources = [artifact if isinstance(artifact, dict) else {}, metadata]
    mechanism = bool(str(candidate.core_mechanism or candidate.concise_claim or "").strip()) or any(
        _value_has_object_content(source.get(key))
        for source in sources
        for key in ("mechanism", "core_mechanism", "candidate_mechanism", "design_mechanism")
    )
    evaluation_dimensions = bool(getattr(candidate, "multihead_scores", {})) or any(
        _value_has_object_content(source.get(key))
        for source in sources
        for key in ("evaluation_dimensions", "evaluation_axes", "rubric_dimensions")
    )
    design_diff = any(
        _value_has_object_content(source.get(key))
        for source in sources
        for key in ("design_diff", "design_delta", "relative_to_parent", "differences", "comparison_to_existing_design", "artifact_delta")
    )
    failure_conditions = bool(candidate.missing_parts or candidate.uncertainty_notes or candidate.failure_lessons) or any(
        _value_has_object_content(source.get(key))
        for source in sources
        for key in ("failure_conditions", "failure_modes", "falsification_conditions", "known_failure_cases", "stop_conditions")
    )
    return mechanism and evaluation_dimensions and design_diff and failure_conditions


def candidate_claim_bound_to_artifact(candidate: CandidateGenome) -> bool:
    claim_text = _normalize_words(" ".join([str(candidate.concise_claim or ""), str(candidate.core_mechanism or "")]))
    if not claim_text:
        return True
    artifact_text = _normalize_words(_stringify_artifact(getattr(candidate, "artifact", None)))
    metadata_text = _normalize_words(_stringify_artifact(coerce_dict(getattr(candidate, "metadata", {})).get("artifact_delta")))
    if not artifact_text and not metadata_text:
        return False
    claim_tokens = {tok for tok in claim_text.split() if len(tok) >= 4}
    object_tokens = {tok for tok in (artifact_text + " " + metadata_text).split() if len(tok) >= 4}
    if not claim_tokens:
        return True
    return bool(claim_tokens.intersection(object_tokens)) or bool(candidate.evidence_refs or candidate.source_bindings or candidate.formal_artifacts)


def candidate_is_meta_commentary_only(candidate: CandidateGenome) -> bool:
    artifact_text = _stringify_artifact(getattr(candidate, "artifact", None)).strip()
    if not artifact_text:
        return False
    if isinstance(getattr(candidate, "artifact", None), dict):
        artifact = getattr(candidate, "artifact")
        if any(key in artifact and _value_has_object_content(artifact.get(key)) for key in OBJECT_MARKER_KEYS):
            return False
    # Long concrete text can be the artifact itself.  Only classify short or
    # purely directive text as meta-only.
    if len(artifact_text) >= 400:
        return False
    return bool(META_ONLY_PATTERNS.search(artifact_text)) and not _has_concrete_object_markers(artifact_text)


def _contract_sources(contract: Any | None, candidate: CandidateGenome | None) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    if candidate is not None:
        sources.append(coerce_dict(getattr(candidate, "metadata", {})))
    if contract is not None:
        if isinstance(contract, dict):
            sources.append(contract)
        else:
            sources.append(coerce_dict(getattr(contract, "outcome_policy", {})))
            for attr in ("dynamic_artifact_contract", "artifact_contract"):
                value = getattr(contract, attr, None)
                if value:
                    sources.append({attr: value})
    return sources


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        return {"description": value.strip()}
    return {}


def _as_mapping_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        out: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                out.append(dict(item))
            elif str(item or "").strip():
                out.append({"description": str(item).strip()})
        return out
    if isinstance(value, dict):
        return [dict(value)]
    if isinstance(value, str) and value.strip():
        return [{"description": value.strip()}]
    return []


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "required", "on", "enabled"}
    return bool(value)


def _mapping_is_vacuous_delta(value: dict[str, Any]) -> bool:
    text = _normalize_words(_stringify_artifact(value))
    if not text:
        return True
    return bool(VACUOUS_DELTA_PATTERNS.search(text))


def _final_gate_self_certifies(value: dict[str, Any]) -> bool:
    text = _normalize_words(_stringify_artifact(value))
    if not text:
        return True
    if SELF_CERTIFYING_PATTERNS.search(text):
        return True
    # The final gate needs some independent observable, not just a verdict word.
    independent_markers = {
        "structural",
        "validator",
        "referee",
        "rubric",
        "diff",
        "parser",
        "schema",
        "test",
        "tool",
        "external",
        "check",
        "comparison",
        "artifact",
        "evidence",
        "验证",
        "结构",
        "对比",
        "检查",
        "证据",
    }
    return not any(marker in text for marker in independent_markers)


def _invalid_outputs_cover_meta_failures(values: list[str]) -> bool:
    text = " ".join(values).lower()
    if not text:
        return False
    required = ["empty", "meta", "artifact"]
    return all(item in text for item in required) or ("空" in text and "产物" in text)


def _value_has_object_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        if any(_value_has_object_content(value.get(key)) for key in OBJECT_MARKER_KEYS):
            return True
        return any(_value_has_object_content(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_value_has_object_content(item) for item in value)
    return True


def _stringify_artifact(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def _normalize_words(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _has_concrete_object_markers(text: str) -> bool:
    lowered = text.lower()
    markers = ("```", "def ", "class ", "diff --git", "@@", "scene:", "revised:", "draft:", "version:", "正文", "片段", "改写后")
    return any(marker in lowered for marker in markers)


__all__ = [
    "DynamicArtifactContract",
    "ContractValidationSummary",
    "ArtifactContractSummary",
    "dynamic_artifact_contract_from",
    "validate_dynamic_artifact_contract",
    "evaluate_candidate_against_dynamic_contract",
    "contract_requires_adapter",
    "materialization_scope_from_contract",
    "candidate_has_object_level_artifact",
    "candidate_has_concrete_delta",
    "candidate_claim_bound_to_artifact",
    "candidate_is_meta_commentary_only",
]
