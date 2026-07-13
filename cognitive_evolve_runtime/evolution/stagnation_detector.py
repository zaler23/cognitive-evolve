"""Stagnation detection over no-delta windows."""
from __future__ import annotations

from typing import Any

from .progress_monitor import ProgressMonitor


class StagnationDetector:
    def detect(self, round_artifacts: list[dict[str, Any]], *, contract: dict[str, Any] | None = None, window: int = 2) -> dict[str, Any]:
        summary = ProgressMonitor().summarize(round_artifacts, contract=contract)
        reports = summary.get("reports", [])[-max(1, int(window)):]
        no_delta = bool(reports) and all(not report.get("material_delta", {}).get("material_delta") for report in reports)
        if no_delta:
            return {
                "status": "stagnation_detected",
                "decision": "targeted_resample",
                "reason": f"last_{len(reports)}_rounds_have_no_material_contract_delta",
                "progress_summary": summary,
            }
        return {
            "status": "continue_search" if reports else "not_started",
            "decision": "continue_search" if reports else "not_started",
            "reason": "recent rounds contain material delta" if reports else "no rounds yet",
            "progress_summary": summary,
        }


__all__ = ["StagnationDetector"]
