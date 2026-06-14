"""Validation and capability resolution for evaluation contracts."""
from __future__ import annotations

from typing import Any

from .schemas import ContractValidationReport, EvaluationContract


class ContractValidator:
    def validate(
        self,
        contract: EvaluationContract | dict[str, Any],
        *,
        verifier_capabilities: dict[str, bool] | None = None,
        evidence_adapters: dict[str, bool] | None = None,
        budget: dict[str, Any] | None = None,
    ) -> ContractValidationReport:
        c = EvaluationContract.from_dict(contract if isinstance(contract, dict) else contract.to_dict())
        capabilities = verifier_capabilities or {}
        adapters = evidence_adapters or {}
        budget = budget or {}
        errors: list[str] = []
        warnings: list[str] = []
        audit: list[dict[str, Any]] = []

        required_sections = {
            "success_conditions": c.success_conditions,
            "hard_gates": c.hard_gates,
            "progress_metrics": c.progress_metrics,
            "drift_signals": c.drift_signals,
            "restart_conditions": c.restart_conditions,
            "abstention_conditions": c.abstention_conditions,
        }
        for name, values in required_sections.items():
            audit.append({"check": f"nonempty_{name}", "passed": bool(values), "count": len(values)})
            if not values:
                errors.append(f"missing_{name}")
        if not any(item.verifier not in {"contract_audit", "llm_judge"} for item in c.progress_metrics):
            warnings.append("contract_underdeveloped:no_verifiable_progress_metric")
        if not any(item.hard for item in c.hard_gates):
            errors.append("no_hard_gate_marked_hard")

        contradiction_check = self._contradiction_check(c)
        errors.extend(contradiction_check["errors"])
        warnings.extend(contradiction_check["warnings"])

        evidence_policy = self._evidence_policy(c, adapters)
        verifier_resolution = self._resolve_verifiers(c, capabilities)
        if evidence_policy.get("blocked"):
            warnings.append("evidence_blocked:required_adapter_missing")
        if verifier_resolution.get("missing_hard_verifiers"):
            warnings.append("contract_underdeveloped:hard_verifier_capability_unresolved")

        budget_check = self._budget_check(c, budget)
        if budget_check.get("blocked"):
            warnings.append("budget_exhausted:contract_cannot_run_with_current_budget")

        if errors:
            status = "invalid"
        elif evidence_policy.get("blocked"):
            status = "evidence_blocked"
        elif any("contract_underdeveloped" in warning for warning in warnings):
            status = "contract_underdeveloped"
        elif budget_check.get("blocked"):
            status = "budget_exhausted"
        else:
            status = "valid"
        return ContractValidationReport(
            status=status,
            passed=not errors and status == "valid",
            errors=errors,
            warnings=warnings,
            evidence_policy=evidence_policy,
            verifier_capability_resolution=verifier_resolution,
            budget_check=budget_check,
            contradiction_check={"errors": contradiction_check["errors"], "warnings": contradiction_check["warnings"]},
            audit_log=audit,
        )

    def _contradiction_check(self, c: EvaluationContract) -> dict[str, list[str]]:
        success = _words(" ".join(item.description for item in c.success_conditions))
        disallowed = _words(" ".join(item.description for item in c.disallowed_shortcuts))
        overlaps = sorted((success & disallowed) - {"candidate", "objective", "progress", "evidence", "the", "and", "with", "must"})
        errors: list[str] = []
        warnings: list[str] = []
        if any("heuristic final" in item.description.lower() for item in c.success_conditions):
            errors.append("success_condition_allows_heuristic_final_authority")
        if overlaps:
            warnings.append("possible_success_disallowed_overlap:" + ",".join(overlaps[:8]))
        return {"errors": errors, "warnings": warnings}

    def _evidence_policy(self, c: EvaluationContract, adapters: dict[str, bool]) -> dict[str, Any]:
        required = [item for item in c.evidence_requirements if item.hard or item.metadata.get("source_type")]
        missing: list[str] = []
        for item in required:
            source_type = str(item.metadata.get("source_type") or "")
            if source_type and not adapters.get(source_type, False):
                missing.append(source_type)
        return {
            "required": bool(required),
            "required_source_types": sorted(set(item.metadata.get("source_type") for item in required if item.metadata.get("source_type"))),
            "missing_adapters": sorted(set(missing)),
            "blocked": bool(missing),
            "model_hypothesis_can_increase_evidence_score": False,
        }

    def _resolve_verifiers(self, c: EvaluationContract, capabilities: dict[str, bool]) -> dict[str, Any]:
        required = sorted({item.verifier for group in [c.hard_gates, c.tool_requirements, c.evidence_requirements] for item in group if item.hard})
        missing = [name for name in required if capabilities and capabilities.get(name) is False]
        return {
            "required_verifiers": required,
            "missing_hard_verifiers": missing,
            "resolved": not missing,
        }

    def _budget_check(self, c: EvaluationContract, budget: dict[str, Any]) -> dict[str, Any]:
        if not budget:
            return {"checked": False, "blocked": False}
        max_rounds = int(budget.get("max_rounds") or budget.get("rounds") or 0)
        hard_gate_count = len(c.hard_gates)
        blocked = max_rounds == 0 and hard_gate_count > 1 and bool(budget.get("requires_runtime_rounds"))
        return {"checked": True, "blocked": blocked, "max_rounds": max_rounds, "hard_gate_count": hard_gate_count}


def _words(text: str) -> set[str]:
    return {piece.strip(".,;:()[]{}!?").lower() for piece in text.split() if len(piece.strip(".,;:()[]{}!?")) >= 4}


__all__ = ["ContractValidator"]
