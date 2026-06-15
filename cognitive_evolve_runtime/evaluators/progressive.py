"""Progressive evaluator wrapper for the Evidence Control Plane."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.core.scalars import bounded_score
from cognitive_evolve_runtime.evaluators.artifact_normalizer import artifact_policy_from_config, normalize_artifact
from cognitive_evolve_runtime.evaluators.challenge_memory import challenge_from_diagnostic
from cognitive_evolve_runtime.evaluators.evidence import EvidenceRecord, repair_value_from_record
from cognitive_evolve_runtime.evaluators.result import EvaluatorResult
from cognitive_evolve_runtime.evaluators.spec import EvaluatorSpec


class ProgressiveEvaluator:
    def evaluate_result(self, candidate: Any, evaluator_result: EvaluatorResult | None, *, spec: EvaluatorSpec | None = None, round_index: int = 0) -> EvidenceRecord:
        spec = spec or EvaluatorSpec()
        policy = artifact_policy_from_config(getattr(spec, "progressive", {}) if spec is not None else {})
        artifact_state = normalize_artifact(candidate, artifact_type=policy.artifact_type or getattr(candidate, "artifact_type", "") or "", policy=policy)
        source = str(getattr(spec, "domain_id", "") or artifact_state.get("artifact_type") or "general")
        if evaluator_result is None:
            diagnostics = list(artifact_state.get("diagnostics") or [])
            passed = bool(artifact_state.get("probe_eligible")) and not policy.machine_readable_required
            status = "artifact_probe_ready" if passed else f"artifact_{artifact_state.get('status') or 'unknown'}"
            metrics = {"schema_cleanliness": artifact_state.get("schema_cleanliness", 0.0)}
            cost: dict[str, Any] = {}
        else:
            diagnostics = list(evaluator_result.diagnostics or []) + list(artifact_state.get("diagnostics") or [])
            passed = bool(evaluator_result.passed)
            status = "passed" if passed else "challenge_failed"
            metrics = dict(evaluator_result.metrics or {})
            metrics.setdefault("schema_cleanliness", artifact_state.get("schema_cleanliness", 0.0))
            cost = dict(evaluator_result.cost or {})
        score = _score_from_metrics(metrics, passed=passed, artifact_score=bounded_score(artifact_state.get("schema_cleanliness", 0.0)))
        artifact_status = str(artifact_state.get("status") or "")
        probe_blocked = bool(artifact_status in {"malformed", "absent"} and policy.machine_readable_required)
        terminal_reject = bool(artifact_status == "absent" and policy.machine_readable_required)
        stage = str(getattr(spec, "level", "probe") or "probe")
        final_stage = stage.strip().lower() in {"final", "certificate", "certification", "l4"}
        final_ready = bool(artifact_state.get("final_eligible")) and passed and final_stage
        if policy.final_requires_certificate:
            final_ready = final_ready and bool(metrics.get("certificate_passed") or metrics.get("final_certificate_passed"))
        challenge_items = []
        artifact_blocks_final = artifact_status and artifact_status != "clean"
        if not passed or artifact_blocks_final:
            priority = 0.7 if score >= 0.65 else 0.5
            for diagnostic in diagnostics[:8] or [status]:
                challenge_items.append(challenge_from_diagnostic(candidate_id=getattr(candidate, "id", ""), source=source, diagnostic=diagnostic, round_index=round_index, priority=priority))
        resolved = [str(item) for item in metrics.get("resolved_challenge_ids", [])] if isinstance(metrics.get("resolved_challenge_ids"), list) else []
        targets = []
        metadata = getattr(candidate, "metadata", None)
        if isinstance(metadata, dict) and isinstance(metadata.get("target_challenge_ids"), list):
            targets = [str(item) for item in metadata.get("target_challenge_ids", []) if item]
        hints = _repair_hints(diagnostics, str(artifact_state.get("status") or ""))
        provisional = EvidenceRecord(
            candidate_id=getattr(candidate, "id", ""),
            source=source,
            stage=stage,
            score=score,
            confidence=0.85 if passed else 0.55,
            cost=cost,
            final_blocked=not final_ready,
            parent_blocked=probe_blocked,
            terminal_reject=terminal_reject,
            repair_value=0.0,
            continuation_value=0.0,
            novelty_value=bounded_score(getattr(candidate, "multihead_scores", {}).get("novelty", 0.0) if isinstance(getattr(candidate, "multihead_scores", None), dict) else 0.0),
            target_challenge_ids=targets,
            resolved_challenge_ids=resolved,
            emitted_challenge_ids=[str(item.get("id")) for item in challenge_items if item.get("id")],
            diagnostics=diagnostics[:20],
            hints=hints,
            metadata={
                "status": status,
                "artifact_policy": policy.to_dict(),
                "artifact_state": artifact_state,
                "challenge_items": challenge_items,
                "metrics": metrics,
                "evaluator_score": metrics.get("score", score),
                "challenge_pass_rate": metrics.get("challenge_pass_rate", 1.0 if passed else 0.0),
            },
        )
        repair = repair_value_from_record(provisional)
        return EvidenceRecord(
            **{**provisional.to_dict(), "repair_value": repair, "continuation_value": repair if not terminal_reject else 0.0}
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
