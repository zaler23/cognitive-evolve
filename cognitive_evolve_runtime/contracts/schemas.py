"""Schemas for AI-generated, validator-audited evaluation contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

META_CONTRACT_VERSION = "evaluation-contract/v1.1"

RUN_STATUSES = {
    "continue_search",
    "targeted_resample",
    "branch_restart",
    "stagnation_detected",
    "semantic_drift_detected",
    "contract_underdeveloped",
    "evidence_blocked",
    "ranking_failed",
    "provider_unavailable",
    "budget_exhausted",
    "not_solved",
    "partial_result",
    "completed",
}

MATERIAL_DELTA_TYPES = {
    "new_candidate_family",
    "new_search_axis",
    "new_mutation_operator",
    "new_verifier_result",
    "new_external_evidence",
    "new_computed_evidence",
    "new_failed_assumption",
    "new_counterexample",
    "new_contract_revision",
    "new_failure_memory",
    "frontier_diversity_gain",
    "hard_gate_satisfaction_gain",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ContractItem:
    id: str
    description: str
    verifier: str = "contract_audit"
    hard: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationContract:
    objective: str
    success_conditions: list[ContractItem] = field(default_factory=list)
    hard_gates: list[ContractItem] = field(default_factory=list)
    soft_objectives: list[ContractItem] = field(default_factory=list)
    evidence_requirements: list[ContractItem] = field(default_factory=list)
    tool_requirements: list[ContractItem] = field(default_factory=list)
    disallowed_shortcuts: list[ContractItem] = field(default_factory=list)
    abstention_conditions: list[ContractItem] = field(default_factory=list)
    progress_metrics: list[ContractItem] = field(default_factory=list)
    drift_signals: list[ContractItem] = field(default_factory=list)
    restart_conditions: list[ContractItem] = field(default_factory=list)
    id: str = "evaluation-contract"
    version: str = META_CONTRACT_VERSION
    created_at: str = field(default_factory=utc_now)
    source: str = "ai_contract_synthesizer_with_validator_guardrails"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "created_at": self.created_at,
            "source": self.source,
            "objective": self.objective,
            "success_conditions": [item.to_dict() for item in self.success_conditions],
            "hard_gates": [item.to_dict() for item in self.hard_gates],
            "soft_objectives": [item.to_dict() for item in self.soft_objectives],
            "evidence_requirements": [item.to_dict() for item in self.evidence_requirements],
            "tool_requirements": [item.to_dict() for item in self.tool_requirements],
            "disallowed_shortcuts": [item.to_dict() for item in self.disallowed_shortcuts],
            "abstention_conditions": [item.to_dict() for item in self.abstention_conditions],
            "progress_metrics": [item.to_dict() for item in self.progress_metrics],
            "drift_signals": [item.to_dict() for item in self.drift_signals],
            "restart_conditions": [item.to_dict() for item in self.restart_conditions],
            "runtime_statuses": sorted(RUN_STATUSES),
            "material_delta_types": sorted(MATERIAL_DELTA_TYPES),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | "EvaluationContract") -> "EvaluationContract":
        if isinstance(value, EvaluationContract):
            return value
        data = value if isinstance(value, dict) else {}
        return cls(
            id=str(data.get("id") or "evaluation-contract"),
            version=str(data.get("version") or META_CONTRACT_VERSION),
            created_at=str(data.get("created_at") or utc_now()),
            source=str(data.get("source") or "ai_contract_synthesizer_with_validator_guardrails"),
            objective=str(data.get("objective") or "user objective"),
            success_conditions=_items(data.get("success_conditions")),
            hard_gates=_items(data.get("hard_gates"), default_hard=True),
            soft_objectives=_items(data.get("soft_objectives")),
            evidence_requirements=_items(data.get("evidence_requirements")),
            tool_requirements=_items(data.get("tool_requirements")),
            disallowed_shortcuts=_items(data.get("disallowed_shortcuts"), default_hard=True),
            abstention_conditions=_items(data.get("abstention_conditions"), default_hard=True),
            progress_metrics=_items(data.get("progress_metrics")),
            drift_signals=_items(data.get("drift_signals")),
            restart_conditions=_items(data.get("restart_conditions")),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class ContractValidationReport:
    status: str
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence_policy: dict[str, Any] = field(default_factory=dict)
    verifier_capability_resolution: dict[str, Any] = field(default_factory=dict)
    budget_check: dict[str, Any] = field(default_factory=dict)
    contradiction_check: dict[str, Any] = field(default_factory=dict)
    audit_log: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _items(raw: Any, *, default_hard: bool = False) -> list[ContractItem]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list[ContractItem] = []
    for index, item in enumerate(raw, start=1):
        if isinstance(item, ContractItem):
            out.append(item)
            continue
        if isinstance(item, dict):
            desc = str(item.get("description") or item.get("text") or item.get("name") or item.get("id") or "").strip()
            if not desc:
                continue
            out.append(
                ContractItem(
                    id=str(item.get("id") or f"item_{index}"),
                    description=desc,
                    verifier=str(item.get("verifier") or "contract_audit"),
                    hard=bool(item.get("hard", default_hard)),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
        else:
            desc = str(item).strip()
            if desc:
                out.append(ContractItem(id=f"item_{index}", description=desc, hard=default_hard))
    return out


__all__ = [
    "META_CONTRACT_VERSION",
    "RUN_STATUSES",
    "MATERIAL_DELTA_TYPES",
    "ContractItem",
    "EvaluationContract",
    "ContractValidationReport",
]
