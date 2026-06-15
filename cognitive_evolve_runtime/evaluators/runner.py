"""External evaluator command runner."""
from __future__ import annotations

import json
import shlex
import tempfile
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.evaluators.evidence import apply_evidence_result
from cognitive_evolve_runtime.evaluators.progressive import ProgressiveEvaluator
from cognitive_evolve_runtime.evaluators.result import EvaluatorResult
from cognitive_evolve_runtime.evaluators.spec import EvaluatorSpec
from cognitive_evolve_runtime.tools.runner import ToolRunner


class ExternalEvaluatorRunner:
    def __init__(self, *, runner: ToolRunner | None = None, progressive: ProgressiveEvaluator | None = None) -> None:
        self.runner = runner
        self.progressive = progressive or ProgressiveEvaluator()

    def evaluate_population_if_configured(self, candidates: list[CandidateGenome], *, spec: EvaluatorSpec, round_index: int = 0) -> list[EvaluatorResult]:
        if not spec.enabled:
            return []
        results: list[EvaluatorResult] = []
        for candidate in candidates:
            result = self.evaluate_candidate(candidate, spec=spec)
            apply_evaluator_result(candidate, result, progressive=self.progressive, spec=spec, round_index=round_index)
            results.append(result)
        return results

    def evaluate_candidate(self, candidate: CandidateGenome, *, spec: EvaluatorSpec) -> EvaluatorResult:
        with tempfile.TemporaryDirectory(prefix="cogev-evaluator-") as temp_dir:
            candidate_path = Path(temp_dir) / "candidate.json"
            candidate_path.write_text(json.dumps(candidate.to_dict(), ensure_ascii=False, sort_keys=True, default=str), encoding="utf-8")
            command = _format_command(spec.command, candidate_path)
            runner = self.runner or ToolRunner(timeout_seconds=spec.timeout_seconds)
            feedback = runner.run(command, cwd=spec.cwd_path(), timeout_seconds=spec.timeout_seconds)
            parsed = _parse_evaluator_output(feedback.raw_output_ref)
            passed = _passed_from(parsed, feedback.status)
            status = "passed" if passed else "failed"
            diagnostics = [str(item) for item in parsed.get("diagnostics", []) if item] if isinstance(parsed.get("diagnostics"), list) else []
            if not diagnostics:
                diagnostics = list(feedback.diagnostics[:10])
            metrics = dict(parsed.get("metrics") or {}) if isinstance(parsed.get("metrics"), dict) else {}
            if "correctness" not in metrics:
                metrics["correctness"] = passed
            return EvaluatorResult(
                candidate_id=candidate.id,
                status=status,
                passed=passed,
                metrics=metrics,
                diagnostics=diagnostics[:20],
                cost={"seconds": feedback.cost.get("seconds", 0), "returncode": feedback.cost.get("returncode")},
            )


def apply_evaluator_result(candidate: CandidateGenome, result: EvaluatorResult, *, progressive: ProgressiveEvaluator | None = None, spec: EvaluatorSpec | None = None, round_index: int = 0) -> None:
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    metadata["evaluator"] = result.to_dict()
    candidate.metadata = metadata
    if result.passed:
        candidate.multihead_scores["correctness"] = 1.0
    else:
        candidate.multihead_scores["correctness"] = 0.0
    score = result.metrics.get("score")
    if isinstance(score, (int, float)):
        candidate.multihead_scores["objective_score"] = max(0.0, min(1.0, float(score)))
    elif result.passed:
        candidate.multihead_scores.setdefault("objective_score", 1.0)
    else:
        candidate.multihead_scores.setdefault("objective_score", 0.0)
    runtime_ms = result.metrics.get("runtime_ms")
    try:
        runtime_penalty = min(0.5, max(0.0, float(runtime_ms) / 100000.0)) if runtime_ms is not None else 0.0
    except (TypeError, ValueError):
        runtime_penalty = 0.0
    candidate.multihead_scores["cost_adjusted_fitness"] = max(0.0, float(candidate.multihead_scores.get("objective_score", 0.0) or 0.0) - runtime_penalty)
    evidence = (progressive or ProgressiveEvaluator()).evaluate_result(candidate, result, spec=spec, round_index=round_index)
    apply_evidence_result(candidate, evidence)
    candidate.add_verification_feedback(result.to_feedback())


def _format_command(command: str, candidate_path: Path) -> list[str]:
    text = command.replace("{candidate_path}", str(candidate_path))
    args = shlex.split(text)
    if "{candidate_path}" not in command and str(candidate_path) not in args:
        args.append(str(candidate_path))
    return args


def _parse_evaluator_output(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    for candidate in (raw, raw.splitlines()[-1] if raw.splitlines() else raw):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {"diagnostics": raw.splitlines()[-20:]}


def _passed_from(parsed: dict[str, Any], feedback_status: str) -> bool:
    for key in ("passed", "pass", "correct", "correctness"):
        if key in parsed:
            return bool(parsed.get(key))
    status = str(parsed.get("status") or feedback_status or "").strip().lower()
    return status in {"passed", "pass", "ok", "success"}


__all__ = ["ExternalEvaluatorRunner", "apply_evaluator_result"]
