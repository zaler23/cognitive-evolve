"""Progressive evaluator wrapper.

This layer turns legacy evaluator output and artifact normalization into a
runtime-level EvidenceResult.  Domain-specific adapters can later override the
L1/L2/L3 details without changing the Nexus loop contract.
"""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.evaluators.artifact_normalizer import artifact_policy_from_config, normalize_artifact
from cognitive_evolve_runtime.evaluators.challenge_bank import challenge_from_diagnostic
from cognitive_evolve_runtime.evaluators.evidence import EvidenceResult
from cognitive_evolve_runtime.evaluators.result import EvaluatorResult
from cognitive_evolve_runtime.evaluators.spec import EvaluatorSpec
from cognitive_evolve_runtime.core.scalars import bounded_score


class ProgressiveEvaluator:
    def evaluate_result(self, candidate: Any, evaluator_result: EvaluatorResult | None, *, spec: EvaluatorSpec | None = None, round_index: int = 0) -> EvidenceResult:
        spec = spec or EvaluatorSpec()
        policy = artifact_policy_from_config(getattr(spec, "progressive", {}) if spec is not None else {})
        artifact_view = normalize_artifact(
            candidate,
            artifact_type=str(policy.get("artifact_type") or getattr(candidate, "artifact_type", "") or ""),
            machine_artifact_required=bool(policy.get("machine_artifact_required")),
        )
        domain_id = str(getattr(spec, "domain_id", "") or artifact_view.artifact_type or "general")
        if evaluator_result is None:
            diagnostics = list(artifact_view.diagnostics)
            passed = artifact_view.probe_eligible and not bool(policy.get("machine_artifact_required"))
            status = "artifact_probe_ready" if passed else f"artifact_{artifact_view.status}"
            metrics = {"schema_cleanliness": artifact_view.schema_cleanliness}
        else:
            diagnostics = list(evaluator_result.diagnostics or []) + list(artifact_view.diagnostics or [])
            passed = bool(evaluator_result.passed)
            status = "passed" if passed else "challenge_failed"
            metrics = dict(evaluator_result.metrics or {})
            metrics.setdefault("schema_cleanliness", artifact_view.schema_cleanliness)
        score = _score_from_metrics(metrics, passed=passed, artifact_score=artifact_view.schema_cleanliness)
        hard_reject = artifact_view.status in {"malformed", "absent"} and bool(policy.get("machine_artifact_required"))
        final_eligible = artifact_view.final_eligible and passed and str(getattr(spec, "level", "L2") or "L2").upper() == "L4"
        level = str(getattr(spec, "level", "L2") or "L2").upper()
        if level not in {"L0", "L1", "L2", "L3", "L4"}:
            level = "L2"
        challenge_cases = []
        if not passed:
            kind = "format_violation" if artifact_view.status in {"malformed", "absent"} else "evaluator_failure"
            for diagnostic in diagnostics[:8] or [status]:
                challenge_cases.append(challenge_from_diagnostic(candidate_id=getattr(candidate, "id", ""), domain_id=domain_id, diagnostic=diagnostic, kind=kind, round_index=round_index))
        repair_hints = _repair_hints(diagnostics, artifact_view.status)
        return EvidenceResult(
            candidate_id=getattr(candidate, "id", ""),
            domain_id=domain_id,
            level=level,
            status="passed" if passed else ("hard_reject" if hard_reject else status),
            passed=passed,
            hard_reject=hard_reject,
            final_eligible=final_eligible,
            score=score,
            metrics=metrics,
            challenge_cases=challenge_cases,
            resolved_challenge_ids=[str(item) for item in metrics.get("resolved_challenge_ids", [])] if isinstance(metrics.get("resolved_challenge_ids"), list) else [],
            repair_hints=repair_hints,
            diagnostics=diagnostics[:20],
            artifact_view=artifact_view,
        )


def _score_from_metrics(metrics: dict[str, Any], *, passed: bool, artifact_score: float) -> float:
    raw_score = metrics.get("frontier_score", metrics.get("score", metrics.get("challenge_pass_rate", 1.0 if passed else 0.0)))
    correctness = 1.0 if bool(metrics.get("correctness", passed)) else 0.0
    return bounded_score(0.60 * bounded_score(raw_score) + 0.25 * correctness + 0.15 * artifact_score)


def _repair_hints(diagnostics: list[str], artifact_status: str) -> list[str]:
    hints: list[str] = []
    if artifact_status in {"refolded", "malformed", "absent"}:
        hints.append("re-emit a clean machine-readable artifact instead of natural-language wrapping")
    for item in diagnostics[:5]:
        text = str(item or "").strip()
        if text:
            hints.append(f"repair diagnostic: {text[:180]}")
    return hints[:8]


__all__ = ["ProgressiveEvaluator"]
