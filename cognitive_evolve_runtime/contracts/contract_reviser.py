"""Contract revision for underdeveloped or blocked contracts."""
from __future__ import annotations

from typing import Any

from .schemas import ContractItem, EvaluationContract, MATERIAL_DELTA_TYPES


class ContractReviser:
    def revise(self, contract: EvaluationContract | dict[str, Any], validation: dict[str, Any] | None = None) -> EvaluationContract:
        c = EvaluationContract.from_dict(contract if isinstance(contract, dict) else contract.to_dict())
        validation = validation or {}
        existing_gate_ids = {item.id for item in c.hard_gates}
        existing_metric_ids = {item.id for item in c.progress_metrics}
        if not c.success_conditions:
            c.success_conditions.append(ContractItem("objective_specific_success", "Define a task-specific success condition before final selection.", hard=True))
        for gate_id, desc, verifier in [
            ("direct_objective_bridge", "Candidate must explicitly bridge its mechanism to the objective.", "contract_gate"),
            ("material_progress_required", "Each continuing round must include material delta, not prettier wording.", "progress_monitor"),
            ("final_authority_requires_verifier_or_abstention", "Final authority requires verifier/evidence support or an explicit partial/blocked status.", "runtime_status_gate"),
        ]:
            if gate_id not in existing_gate_ids:
                c.hard_gates.append(ContractItem(gate_id, desc, verifier=verifier, hard=True))
                existing_gate_ids.add(gate_id)
        for metric in sorted(MATERIAL_DELTA_TYPES):
            if metric not in existing_metric_ids:
                c.progress_metrics.append(ContractItem(metric, metric.replace("_", " "), verifier="progress_monitor"))
                existing_metric_ids.add(metric)
        if validation.get("status") == "evidence_blocked" and not any(item.id == "adapter_required_blocks_support" for item in c.abstention_conditions):
            c.abstention_conditions.append(ContractItem("adapter_required_blocks_support", "Evidence adapter gaps must produce evidence_blocked/partial_result, not supported status.", verifier="evidence_policy", hard=True))
        c.metadata["revision_count"] = int(c.metadata.get("revision_count") or 0) + 1
        c.metadata["last_revision_reason"] = validation.get("status", "manual_or_guardrail_revision")
        return c

    def tighten_overbroad_contract(self, contract: EvaluationContract | dict[str, Any]) -> EvaluationContract:
        return self.revise(contract, {"status": "contract_underdeveloped"})


__all__ = ["ContractReviser"]
