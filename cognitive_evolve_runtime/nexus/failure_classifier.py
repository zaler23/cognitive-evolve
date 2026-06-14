"""Classify verifier failures before lifecycle/compaction decisions.

The repair lane is intentionally narrower than "anything that failed".  Patch
syntax/context errors and structurally repairable evidence/proof gaps can remain
live as Incubating parents, but docs-only, seed-note, unrelated, missing-path, or
final-claim-without-evidence failures must stay terminal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
import re
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus._serde import coerce_dict


@dataclass(frozen=True)
class FailureVerdict:
    category: str
    repairable: bool
    reason: str
    lifecycle_action: str = ""
    retention: str = ""
    blockers: list[str] = field(default_factory=list)
    repair_targets: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    failure_signature: str = ""
    failure_guidance: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_candidate_failure(candidate: CandidateGenome, payload: dict[str, Any] | None = None) -> FailureVerdict:
    payload = coerce_dict(payload)
    diagnostics = _failure_terms(candidate, payload)
    targets = _repair_targets(candidate, payload)
    text = " ".join(diagnostics + targets).lower()

    terminal_reason = _terminal_reason(text=text, targets=targets, candidate=candidate)
    if terminal_reason:
        blockers = _select_blockers(diagnostics, terminal=True)
        return _verdict(
            category=terminal_reason,
            repairable=False,
            reason=terminal_reason,
            blockers=blockers,
            repair_targets=targets,
            diagnostics=diagnostics,
        )

    category = _repairable_category(text=text, diagnostics=diagnostics, targets=targets)
    if category:
        blockers = _select_blockers(diagnostics, terminal=False)
        return _verdict(
            category=category,
            repairable=True,
            reason="mechanical_or_structural_failure_kept_for_bounded_repair",
            blockers=blockers,
            repair_targets=targets,
            diagnostics=diagnostics,
        )

    blockers = _select_blockers(diagnostics, terminal=True)
    return _verdict(
        category="terminal_non_repairable_failure",
        repairable=False,
        reason="failure_has_no_concrete_repair_target",
        blockers=blockers,
        repair_targets=targets,
        diagnostics=diagnostics,
    )


def classify_recovery_eligibility(
    candidate: CandidateGenome,
    payload: dict[str, Any] | None = None,
    *,
    project_root: str | Path | None = None,
    max_repair_attempts: int | None = None,
) -> FailureVerdict:
    """Classify whether a dormant/archived failed candidate can seed repair.

    This is intentionally stricter than ``classify_candidate_failure``.  A
    failed offspring can be repairable because a verifier has just seen its
    local context, but a dormant/archive recovery seed must still prove it is
    grounded in the current project surface.  The predicate therefore keeps the
    existing failure taxonomy as the single source of truth, then adds two
    recovery-only guards:

    * bounded attempt count, persisted in candidate metadata;
    * at least one concrete repair target that still exists under the project
      root.
    """

    verdict = classify_candidate_failure(candidate, payload)
    if not verdict.repairable:
        return verdict
    attempts = _repair_attempt_count(candidate)
    if max_repair_attempts is not None and attempts >= max(0, int(max_repair_attempts)):
        return _verdict(
            category="terminal_repair_attempts_exhausted",
            repairable=False,
            reason="repair_attempt_cap_reached",
            blockers=[f"repair_attempts_exhausted:{attempts}"],
            repair_targets=verdict.repair_targets,
            diagnostics=verdict.diagnostics,
        )
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[2]
    existing_targets = _existing_project_targets(verdict.repair_targets, root)
    if not existing_targets:
        return _verdict(
            category="terminal_recovery_missing_existing_project_path",
            repairable=False,
            reason="dormant_recovery_requires_existing_project_relative_target",
            blockers=[*(verdict.blockers or []), "existing_project_relative_path_absent"],
            repair_targets=verdict.repair_targets,
            diagnostics=verdict.diagnostics,
        )
    if existing_targets != verdict.repair_targets:
        return _verdict(
            category=verdict.category,
            repairable=True,
            reason=verdict.reason,
            blockers=verdict.blockers,
            repair_targets=existing_targets,
            diagnostics=verdict.diagnostics,
        )
    return verdict


def is_recoverable_dormant_failure(
    candidate: CandidateGenome,
    payload: dict[str, Any] | None = None,
    *,
    project_root: str | Path | None = None,
    max_repair_attempts: int | None = None,
) -> bool:
    """Return whether a dormant/archive candidate may seed a repair lane."""

    return classify_recovery_eligibility(
        candidate,
        payload,
        project_root=project_root,
        max_repair_attempts=max_repair_attempts,
    ).repairable


def _verdict(
    *,
    category: str,
    repairable: bool,
    reason: str,
    blockers: list[str],
    repair_targets: list[str],
    diagnostics: list[str],
) -> FailureVerdict:
    clipped_blockers = [_clip(item, 180) for item in blockers if item][:8]
    clipped_targets = [_clip(item, 160) for item in repair_targets if item][:8]
    clipped_diagnostics = [_clip(item, 220) for item in diagnostics if item][:16]
    signature = _signature(category, clipped_blockers, clipped_targets)
    return FailureVerdict(
        category=category,
        repairable=repairable,
        reason=reason,
        lifecycle_action=_lifecycle_action_for_category(category, repairable=repairable),
        retention=_retention_for_category(category, repairable=repairable),
        blockers=clipped_blockers,
        repair_targets=clipped_targets,
        diagnostics=clipped_diagnostics,
        failure_signature=signature,
        failure_guidance=_failure_guidance(
            candidate_id="",
            blockers=clipped_blockers,
            targets=clipped_targets,
            repairable=repairable,
            category=category,
        ),
    )


def _failure_terms(candidate: CandidateGenome, payload: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    terms.extend(str(item or "") for item in getattr(candidate, "failure_lessons", []) or [])
    result = getattr(candidate, "verification_result", {}) or {}
    if isinstance(result, dict):
        terms.extend(str(item or "") for item in result.get("diagnostics", []) or [])
        terms.extend(str(item or "") for item in result.get("failure_lessons", []) or [])
        for feedback in result.get("tool_feedback", []) or result.get("feedback", []) or []:
            if isinstance(feedback, dict):
                terms.extend(str(item or "") for item in feedback.get("diagnostics", []) or [])
                terms.extend(str(item or "") for item in feedback.get("failed_fragments", []) or [])
    for key in ("diagnostics", "failed_files", "failure_lessons"):
        terms.extend(str(item or "") for item in payload.get(key, []) or [])
    patch_result = payload.get("patch_result")
    if not isinstance(patch_result, dict):
        patch_result = getattr(candidate, "patch_application_result", None)
    if isinstance(patch_result, dict):
        for key in ("diagnostics", "failed_files", "applied_files"):
            terms.extend(str(item or "") for item in patch_result.get(key, []) or [])
    for feedback in payload.get("tool_feedback", []) or []:
        if isinstance(feedback, dict):
            terms.extend(str(item or "") for item in feedback.get("diagnostics", []) or [])
            terms.extend(str(item or "") for item in feedback.get("failed_fragments", []) or [])
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    for guidance in metadata.get("failure_micro_guidance", []) or []:
        if isinstance(guidance, dict):
            terms.append(str(guidance.get("blocker") or ""))
            terms.append(str(guidance.get("next_action") or ""))
    return _dedupe(terms)


def _repair_targets(candidate: CandidateGenome, payload: dict[str, Any]) -> list[str]:
    targets: list[str] = []
    patch_result = payload.get("patch_result")
    if not isinstance(patch_result, dict):
        patch_result = getattr(candidate, "patch_application_result", None)
    if isinstance(patch_result, dict):
        for key in ("failed_files", "applied_files"):
            targets.extend(str(item or "") for item in patch_result.get(key, []) or [])
    for feedback in payload.get("tool_feedback", []) or []:
        if isinstance(feedback, dict):
            targets.extend(str(item or "") for item in feedback.get("failed_fragments", []) or [])
    for binding in getattr(candidate, "source_bindings", []) or []:
        if isinstance(binding, dict):
            targets.append(str(binding.get("path") or binding.get("ref") or ""))
    artifact = getattr(candidate, "artifact", None)
    if isinstance(artifact, dict):
        for key in ("path", "target_path", "file"):
            targets.append(str(artifact.get(key) or ""))
        for patch_key in ("patch", "patch_content", "diff", "unified_diff", "content"):
            value = artifact.get(patch_key)
            if isinstance(value, str):
                targets.extend(_paths_from_patch_headers(value))
    for op in getattr(candidate, "patch_set", []) or []:
        path = getattr(op, "path", "")
        if path:
            targets.append(str(path))
    return _dedupe(_normalize_path(item) for item in targets if item)


def _terminal_reason(*, text: str, targets: list[str], candidate: CandidateGenome) -> str:
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    if metadata.get("semantic_drift") or metadata.get("unrelated_drift"):
        return "terminal_unrelated_semantic_drift"
    if any(token in text for token in ("hard_reject_unrelated", "unrelated_semantic_drift", "second_runtime", "hidden_fallback", "contract_hash")):
        return "terminal_hard_reject"
    if "seed_note_only_patch" in text or any(_is_seed_note_target(path) for path in targets):
        return "terminal_seed_note_only_patch"
    if "runtime_code_change_absent:documentation_only_patch" in text:
        return "terminal_documentation_only_patch"
    if targets and all(_is_documentation_target(path) for path in targets):
        return "terminal_documentation_only_patch"
    if "source_binding_missing_path" in text or "patch_target_missing" in text:
        return "terminal_missing_project_path"
    if "source_free_final_claim" in text or "narrative_only" in text:
        return "terminal_narrative_or_source_free_final_claim"
    if "prompt_only_gate" in text or "docs_only_essay" in text:
        return "terminal_prompt_or_docs_only"
    return ""


def _repairable_category(*, text: str, diagnostics: list[str], targets: list[str]) -> str:
    if any(token in text for token in ("empty_assistant_content", "empty assistant content", "empty assistant message")):
        return "checkpoint_resume_empty_assistant"
    if any(token in text for token in ("truncated_response", "finish_reason=length", "finish reason length")):
        return "checkpoint_resume_truncated_response"
    if any(token in text for token in ("timeout", "timed out", "read timed out")):
        return "checkpoint_resume_timeout"
    if any(token in text for token in ("rate_limit_429", "http 429", " 429", "rate limit")):
        return "checkpoint_resume_rate_limit_429"
    if any(token in text for token in ("provider_5xx", "http 500", "http 502", "http 503", " 503", "service unavailable")):
        return "checkpoint_resume_provider_5xx"
    if any(token in text for token in ("modelresponseschemaerror", "schema validation", "json parse", "invalid json", "not valid json")):
        return "repairable_model_schema_or_json_contract"
    patch_tokens = (
        "patch_application_failed",
        "unified_patch_failed",
        "malformed patch",
        "unexpected eof",
        "unexpected end of file",
        "hunk",
        ".rej",
        "old_text not found",
        "patch_no_effect:no_files_applied",
        "patch_no_effect:no_files_declared",
    )
    if any(token in text for token in patch_tokens):
        if targets or not any(token in text for token in ("no_files_applied", "no_files_declared")):
            return "repairable_patch_syntax_or_context"
    if any(token in text for token in ("syntaxerror", "indentationerror", "compileall_failed", "static_check_failed", "ruff_failed", "mypy_failed")):
        return "repairable_static_or_syntax_failure"
    proof_tokens = (
        "proof_object_absent",
        "proof_object_structurally_weak",
        "evidence_ref_unverified",
        "evidence_ref_absent",
        "obligation_delta_absent",
        "ledger_non_progressing",
        "source_binding_absent",
    )
    if any(token in text for token in proof_tokens):
        return "repairable_evidence_or_obligation_gap"
    if any("failed" in item.lower() for item in diagnostics) and targets:
        return "repairable_targeted_tool_failure"
    return ""


def _lifecycle_action_for_category(category: str, *, repairable: bool) -> str:
    if not repairable:
        return "terminal_archive"
    if category.startswith("checkpoint_resume_"):
        return "checkpoint_and_resume"
    if category == "repairable_model_schema_or_json_contract":
        return "repair_output_contract"
    if category == "repairable_patch_syntax_or_context":
        return "bounded_patch_repair_lane"
    if category == "repairable_static_or_syntax_failure":
        return "bounded_static_repair_lane"
    return "bounded_repair_lane"


def _retention_for_category(category: str, *, repairable: bool) -> str:
    if not repairable:
        return "terminal"
    if category.startswith("checkpoint_resume_"):
        return "checkpoint_resume"
    return "incubating_repair_material"


def _select_blockers(diagnostics: list[str], *, terminal: bool) -> list[str]:
    preferred = []
    fallback = []
    for item in diagnostics:
        text = str(item or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if any(token in lowered for token in ("failed", "absent", "missing", "unverified", "no_effect", "hunk", "eof", "docs", "seed_note", "target")):
            preferred.append(text)
        else:
            fallback.append(text)
    chosen = preferred or fallback
    if terminal and not chosen:
        chosen = ["terminal_non_repairable_failure"]
    return _dedupe(chosen)[:8]


def _failure_guidance(
    *,
    candidate_id: str,
    blockers: list[str],
    targets: list[str],
    repairable: bool,
    category: str,
) -> list[dict[str, Any]]:
    if not repairable:
        return []
    out: list[dict[str, Any]] = []
    for blocker in blockers[:5]:
        next_action = _next_action_for_blocker(blocker, category=category, targets=targets)
        out.append(
            {
                "candidate_id": candidate_id,
                "blocker": blocker,
                "next_action": next_action,
                "evidence_needed": _evidence_needed_for_blocker(blocker, category=category),
                "source_bindings": [{"path": path, "kind": "source_file", "source": "failure_classifier"} for path in targets[:5]],
                "disallowed_repeat_pattern": _repeat_pattern_for_blocker(blocker, category=category),
                "severity": "error",
            }
        )
    return out


def _next_action_for_blocker(blocker: str, *, category: str, targets: list[str]) -> str:
    lowered = blocker.lower()
    if category.startswith("checkpoint_resume_"):
        return "persist checkpoint state, resume the same semantic step with transport-adjusted retry policy, and avoid marking candidate material terminal"
    if category == "repairable_model_schema_or_json_contract":
        return "repair the model output into the declared JSON/schema contract before lifecycle decisions"
    if category == "repairable_patch_syntax_or_context" or any(token in lowered for token in ("unified_patch_failed", "malformed patch", "hunk", "eof", "old_text")):
        target_text = ", ".join(targets[:3]) or "the declared patch targets"
        return f"regenerate a complete unified diff against exact current source context for {target_text}; do not emit narrative text in place of patch hunks"
    if category == "repairable_static_or_syntax_failure":
        target_text = ", ".join(targets[:3]) or "the changed files"
        return f"repair syntax/static-check failures in {target_text} while preserving the candidate's intended semantic delta"
    if "proof_object" in lowered:
        return "replace narrative proof text with a verifier-readable formal_artifact such as assertion_set or verification_witness"
    if "ledger" in lowered or "obligation_delta" in lowered:
        return "attach a named obligation_delta that targets or discharges a concrete active obligation"
    if "evidence_ref" in lowered or "source_binding" in lowered:
        return "bind verified evidence_refs and exact source_bindings before claiming progress"
    return "repair the failure with a concrete evidence_delta and verifier-readable artifact"


def _evidence_needed_for_blocker(blocker: str, *, category: str) -> list[str]:
    lowered = blocker.lower()
    if category.startswith("checkpoint_resume_"):
        return ["checkpoint_written", "retry_policy_adjusted", "resume_available"]
    if category == "repairable_model_schema_or_json_contract":
        return ["valid_json_object", "schema_valid_response", "candidate_contract_fields"]
    if category == "repairable_patch_syntax_or_context" or "patch" in lowered or "hunk" in lowered or "eof" in lowered:
        return ["complete_unified_diff", "existing_project_relative_path", "post_pass_local_verification"]
    if category == "repairable_static_or_syntax_failure":
        return ["compileall_or_static_check_pass", "changed_file_context", "post_pass_local_verification"]
    if "proof_object" in lowered:
        return ["formal_artifact", "structural_check"]
    if "ledger" in lowered or "obligation_delta" in lowered:
        return ["obligation_delta"]
    if "evidence_ref" in lowered or "source_binding" in lowered:
        return ["verified_evidence_ref", "source_binding"]
    return ["evidence_delta"]


def _repeat_pattern_for_blocker(blocker: str, *, category: str) -> str:
    lowered = blocker.lower()
    if category.startswith("checkpoint_resume_"):
        return "do_not_tombstone_candidate_material_for_provider_transport_failure"
    if category == "repairable_model_schema_or_json_contract":
        return "do_not_reuse_schema_invalid_or_empty_candidate_output"
    if category == "repairable_patch_syntax_or_context" or "patch" in lowered or "hunk" in lowered or "eof" in lowered:
        return "do_not_repeat_the_same_malformed_or_context_stale_diff"
    if "proof" in lowered:
        return "do_not_only_describe_the_route_without_a_checkable_formal_object"
    if "ledger" in lowered or "obligation_delta" in lowered:
        return "do_not_claim_progress_without_named_obligation_delta"
    return "do_not_repeat_the_failed_proposal_shape_without_new_evidence"


_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
_DIFF_PATH_RE = re.compile(r"^(?:---|\+\+\+)\s+(.+?)\s*$")


def _paths_from_patch_headers(text: str) -> list[str]:
    out: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        git_match = _DIFF_GIT_RE.match(stripped)
        if git_match:
            out.extend([git_match.group(1), git_match.group(2)])
            continue
        path_match = _DIFF_PATH_RE.match(stripped)
        if not path_match:
            continue
        raw = path_match.group(1).strip().split("\t", 1)[0]
        normalized = _normalize_path(raw)
        if normalized and normalized != "/dev/null":
            out.append(normalized)
    return _dedupe(out)


def _normalize_path(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if text in {"", "/dev/null"}:
        return text
    if text.startswith("file://"):
        text = text[7:]
    if text.startswith("/"):
        return text
    parts = [part for part in text.split("/") if part]
    if parts and parts[0] in {"a", "b"}:
        parts = parts[1:]
    return "/".join(parts)


def _repair_attempt_count(candidate: CandidateGenome) -> int:
    metadata = coerce_dict(getattr(candidate, "metadata", {}))
    for key in ("repair_attempts", "repair_attempt"):
        try:
            value = int(metadata.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return 0


def _existing_project_targets(targets: list[str], project_root: Path) -> list[str]:
    out: list[str] = []
    root = project_root.resolve()
    for target in targets:
        normalized = _normalize_path(target)
        if not normalized:
            continue
        path = Path(normalized)
        if path.is_absolute():
            candidate_path = path.resolve()
            try:
                rel = candidate_path.relative_to(root)
            except ValueError:
                continue
        else:
            rel = Path(normalized)
            candidate_path = (root / rel).resolve()
            try:
                candidate_path.relative_to(root)
            except ValueError:
                continue
        if _project_path_is_file(candidate_path, root=root, rel=rel):
            out.append(str(rel).replace("\\", "/"))
    return _dedupe(out)


def _project_path_is_file(candidate_path: Path, *, root: Path, rel: Path) -> bool:
    if candidate_path.is_file():
        return True
    normalized = str(rel).replace("\\", "/")
    if normalized == "cognitive_evolve_runtime/nexus/loop.py":
        return (root / "cognitive_evolve_runtime/nexus/loop/__init__.py").is_file()
    return False


def _is_seed_note_target(path: str) -> bool:
    return _normalize_path(path).lower() == "nexus_seed_note.md"


def _is_documentation_target(path: str) -> bool:
    normalized = _normalize_path(path).lower()
    return bool(
        normalized.endswith((".md", ".rst", ".txt"))
        or normalized.startswith(("docs/", "documentation/"))
        or "/docs/" in normalized
    )


def _signature(category: str, blockers: list[str], targets: list[str]) -> str:
    material = "\n".join([category, *blockers[:5], *targets[:5]])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _clip(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


__all__ = [
    "FailureVerdict",
    "classify_candidate_failure",
    "classify_recovery_eligibility",
    "is_recoverable_dormant_failure",
]
