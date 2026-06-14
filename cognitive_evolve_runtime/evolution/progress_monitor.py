"""Progress monitor enforcing material contract deltas."""
from __future__ import annotations

from typing import Any

from .novelty_delta import material_delta


class ProgressMonitor:
    def evaluate_round(self, round_record: dict[str, Any], previous_rounds: list[dict[str, Any]] | None = None, *, contract: dict[str, Any] | None = None) -> dict[str, Any]:
        delta = material_delta(previous_rounds or [], round_record, contract=contract)
        status = "continue_search" if delta["material_delta"] else "stagnation_detected"
        return {
            "status": status,
            "material_delta": delta,
            "reason": "material contract delta found" if delta["material_delta"] else "pretty_text_or_repetition_without_contract_delta",
            "allowed_runtime_statuses": ["continue_search", "stagnation_detected", "targeted_resample", "branch_restart"],
        }

    def summarize(self, round_artifacts: list[dict[str, Any]], *, contract: dict[str, Any] | None = None) -> dict[str, Any]:
        reports = []
        previous: list[dict[str, Any]] = []
        for record in round_artifacts:
            report = self.evaluate_round(record, previous, contract=contract)
            reports.append(report)
            previous.append(record)
        return {
            "version": "progress-monitor/v1.1",
            "status": reports[-1]["status"] if reports else "not_started",
            "round_count": len(round_artifacts),
            "reports": reports,
            "material_rounds": sum(1 for item in reports if item["material_delta"]["material_delta"]),
        }


__all__ = ["ProgressMonitor"]
