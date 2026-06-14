"""Project candidate sandboxing and local verification for Nexus."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.candidates.project_candidate import PatchApplicationResult, ProjectCandidateGenome
from cognitive_evolve_runtime.nexus.failure_classifier import classify_recovery_eligibility
from cognitive_evolve_runtime.tools.adapters import LocalToolSuite
from cognitive_evolve_runtime.tools.feedback import ToolFeedback
from cognitive_evolve_runtime.tools.patch_sandbox import PatchSandbox, _looks_like_unified_patch_text


@dataclass
class ProjectVerificationSummary:
    candidate_id: str
    patch_result: dict[str, Any]
    tool_feedback: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        patch_ok = self.patch_result.get("status") == "applied"
        tools_ok = all(item.get("status") == "passed" for item in self.tool_feedback) if self.tool_feedback else patch_ok
        return patch_ok and tools_ok

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "patch_result": self.patch_result, "tool_feedback": self.tool_feedback, "passed": self.passed}


class ProjectCandidateVerifier:
    def __init__(self, *, source_root: str | Path, sandbox_root: str | Path, tool_suite: LocalToolSuite | None = None, include_tests: bool = False, timeout_seconds: float = 60.0) -> None:
        self.source_root = Path(source_root)
        self.sandbox_root = Path(sandbox_root)
        self.tool_suite = tool_suite or LocalToolSuite()
        self.include_tests = include_tests
        self.timeout_seconds = timeout_seconds

    def verify_population(self, candidates: list[CandidateGenome]) -> list[ProjectVerificationSummary]:
        summaries: list[ProjectVerificationSummary] = []
        for candidate in candidates:
            if _is_project_patch_like(candidate):
                summaries.append(self.verify(candidate))
        return summaries

    def verify(self, candidate: CandidateGenome) -> ProjectVerificationSummary:
        sandbox = PatchSandbox(self.source_root, self.sandbox_root)
        patch_result = sandbox.apply(candidate)
        feedback: list[ToolFeedback] = []
        if patch_result.passed:
            specs = self.tool_suite.default_specs_for_project(patch_result.sandbox_path, include_tests=self.include_tests)
            feedback = self.tool_suite.run_specs(specs, cwd=patch_result.sandbox_path, timeout_seconds=self.timeout_seconds)
            for item in feedback:
                candidate.add_tool_feedback(item)
                candidate.add_verification_feedback(item)
            if hasattr(candidate, "commands_run"):
                candidate.commands_run.extend(item.to_dict() for item in feedback)
        else:
            fail_feedback = ToolFeedback(
                tool_id="patch_sandbox",
                status="failed",
                diagnostics=list(patch_result.diagnostics),
                failed_fragments=list(patch_result.failed_files),
                confidence=1.0,
                raw_output_ref=patch_result.raw_output_ref,
            )
            candidate.add_tool_feedback(fail_feedback)
            candidate.add_verification_feedback(fail_feedback)
            feedback = [fail_feedback]
        summary = ProjectVerificationSummary(candidate_id=candidate.id, patch_result=patch_result.to_dict(), tool_feedback=[item.to_dict() for item in feedback])
        setattr(candidate, "patch_application_result", patch_result.to_dict())
        candidate.verification_result = summary.to_dict()
        _update_scores_from_verification(candidate, patch_result, feedback)
        if not summary.passed:
            payload = summary.to_dict()
            payload["source_root"] = str(self.source_root)
            _route_failed_project_candidate(candidate, payload)
        return summary


def _is_project_patch_like(candidate: CandidateGenome) -> bool:
    artifact_type = str(getattr(candidate, "artifact_type", "") or "").lower()
    if isinstance(candidate, ProjectCandidateGenome) or artifact_type in {"project_patch", "patch", "code_patch"}:
        return True
    if getattr(candidate, "patch_set", None):
        return True
    artifact = getattr(candidate, "artifact", None)
    if not isinstance(artifact, dict):
        return False
    if any(key in artifact for key in ("patch", "patch_content", "diff", "unified_diff")):
        return True
    content = artifact.get("content")
    return isinstance(content, str) and _looks_like_unified_patch_text(content)


def _update_scores_from_verification(candidate: CandidateGenome, patch_result: PatchApplicationResult, feedback: list[ToolFeedback]) -> None:
    scores = dict(candidate.multihead_scores)
    if patch_result.passed:
        scores["tool_progress"] = min(1.0, scores.get("tool_progress", 0.0) + 0.15)
        scores["verifiability"] = min(1.0, scores.get("verifiability", 0.0) + 0.1)
    else:
        scores["tool_progress"] = max(0.0, scores.get("tool_progress", 0.0) - 0.1)
        candidate.failure_lessons.append("patch_application_failed")
    if feedback:
        if all(item.status == "passed" for item in feedback):
            scores["tool_progress"] = min(1.0, scores.get("tool_progress", 0.0) + 0.25)
            scores["robustness"] = min(1.0, scores.get("robustness", 0.0) + 0.1)
        else:
            candidate.failure_lessons.extend(item.diagnostics[0] for item in feedback if item.status != "passed" and item.diagnostics)
    candidate.multihead_scores = scores


def _route_failed_project_candidate(candidate: CandidateGenome, payload: dict[str, Any]) -> None:
    """Keep repairable project patch failures in the live repair lane.

    Initial project seeds and model-generated code patches can fail because the
    diff is malformed, stale, or incomplete while still carrying useful
    self-evolution genes.  Treat those like failed offspring: block them from
    final synthesis, but keep them as Incubating repair parents.  Terminal
    failures such as docs-only, seed-note-only, missing targets, and source-free
    narrative claims still become Failed.
    """

    verdict = classify_recovery_eligibility(candidate, payload, project_root=payload.get("source_root"))
    candidate.metadata["failure_classification"] = _verdict_with_candidate_id(verdict.to_dict(), candidate.id)
    if not verdict.repairable:
        candidate.mark_fate(CandidateFate.FAILED.value)
        return
    candidate.mark_fate(CandidateFate.INCUBATING.value)
    candidate.metadata["final_answer_blocked_until_repaired"] = True
    candidate.metadata.setdefault("incubation_started_round", int(candidate.metadata.get("created_in_round") or 0))
    try:
        repair_attempts = int(candidate.metadata.get("repair_attempts") or 0)
    except (TypeError, ValueError):
        repair_attempts = 0
    candidate.metadata["repair_attempts"] = max(0, repair_attempts)
    targets = list(dict.fromkeys(verdict.repair_targets))[:6]
    blockers = list(dict.fromkeys([*verdict.blockers, *verdict.diagnostics]))[:8]
    guidance = [_verdict_guidance_item(item, candidate.id) for item in verdict.failure_guidance]
    if guidance:
        candidate.metadata["failure_micro_guidance"] = _dedupe_guidance(
            [dict(item) for item in candidate.metadata.get("failure_micro_guidance", []) or [] if isinstance(item, dict)]
            + guidance
        )[:5]
    candidate.metadata["repair_required"] = {
        "blockers": blockers,
        "evidence_needed": ["valid_unified_diff_or_patch_set", "source_binding", "post_pass_local_verification"],
        "source_bindings": [{"path": path, "kind": "source_file", "source": "project_verification_repair_lane"} for path in targets],
        "next_actions": [
            "rewrite the patch against exact project-relative source context",
            "preserve the useful core mechanism while fixing sandbox diagnostics",
        ],
        "failure_signature": verdict.failure_signature,
        "source": "project_verification_repair_lane",
    }
    candidate.metadata["bootstrap_entry_survival"] = {
        "reason": "repairable_project_patch_failure_kept_as_incubating_parent",
        "category": verdict.category,
        "final_answer_blocked": True,
    }
    if "project_patch_failed_but_repairable" not in candidate.failure_lessons:
        candidate.failure_lessons.append("project_patch_failed_but_repairable")


def _verdict_with_candidate_id(payload: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    out = dict(payload)
    out["failure_guidance"] = [_verdict_guidance_item(item, candidate_id) for item in payload.get("failure_guidance", []) or [] if isinstance(item, dict)]
    return out


def _verdict_guidance_item(item: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    fixed = dict(item)
    fixed["candidate_id"] = candidate_id
    return fixed


def _dedupe_guidance(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (str(item.get("blocker") or ""), str(item.get("next_action") or ""))
        if not any(key) or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


__all__ = ["ProjectCandidateVerifier", "ProjectVerificationSummary"]
