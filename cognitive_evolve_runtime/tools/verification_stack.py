"""Nexus-native verification stack for candidate genomes.

The v1 verifier stack mixed response checks, evaluator registry, failure memory,
and human checkpoints.  This module preserves the useful behavior as structured
``ToolFeedback`` entries attached to ``CandidateGenome`` without maintaining a
separate validation architecture.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
import ast
import re
from typing import Any

from cognitive_evolve_runtime.llm.env import env_int
from cognitive_evolve_runtime.llm.governor import llm_governor

_VERIFY_MAX_WORKERS_ENV = "COGEV_VERIFY_CONCURRENCY"

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.candidates.project_candidate import ProjectCandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.obligations import (
    HARD_EVIDENCE_FAILURES,
    HARD_PROOF_FAILURES,
    evidence_obligation_summary,
    formal_signature,
    proof_progress_summary,
)
from cognitive_evolve_runtime.nexus.final_gate import final_gate_summary
from cognitive_evolve_runtime.nexus.artifact_contract import (
    contract_requires_adapter,
    evaluate_candidate_against_dynamic_contract,
    materialization_scope_from_contract,
)
from cognitive_evolve_runtime.nexus.stage_policy import annotate_stage_eligibility
from cognitive_evolve_runtime.tools.feedback import FailureMicroGuidance, ToolFeedback, failure_micro_guidance_from_diagnostics
from cognitive_evolve_runtime.nexus.source_lineage import analyze_source_lineage


@dataclass
class VerificationStackResult:
    candidate_id: str
    feedback: list[ToolFeedback] = field(default_factory=list)
    passed: bool = True
    failure_lessons: list[str] = field(default_factory=list)
    proof_progress: dict[str, Any] = field(default_factory=dict)
    evidence_obligation: dict[str, Any] = field(default_factory=dict)
    final_gate: dict[str, Any] = field(default_factory=dict)
    artifact_contract: dict[str, Any] = field(default_factory=dict)
    rank_eligible: bool = True
    final_eligible: bool = True
    failure_guidance: list[FailureMicroGuidance] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.passed:
            return "needs_evolution"
        if any(item.status == "warning" for item in self.feedback):
            return "warning"
        return "ok"

    @property
    def diagnostics(self) -> list[str]:
        out: list[str] = []
        for item in self.feedback:
            out.extend(item.diagnostics)
            if item.status != "ok":
                out.append(item.tool_id)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "passed": self.passed,
            "status": self.status,
            "diagnostics": self.diagnostics,
            "feedback": [item.to_dict() for item in self.feedback],
            "failure_lessons": list(self.failure_lessons),
            "proof_progress": dict(self.proof_progress),
            "evidence_obligation": dict(self.evidence_obligation),
            "final_gate": dict(self.final_gate),
            "artifact_contract": dict(self.artifact_contract),
            "rank_eligible": self.rank_eligible,
            "final_eligible": self.final_eligible,
            "failure_guidance": [item.to_dict() for item in self.failure_guidance],
        }


class NexusVerifierStack:
    """Small task-agnostic verifier that enriches genomes with feedback."""

    def __init__(self, *, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root).resolve() if project_root else None

    def verify(
        self,
        candidate: CandidateGenome,
        *,
        contract: NexusObjectiveContract | None = None,
        existing_formal_signatures: set[str] | None = None,
        blocking_obligation_ids: list[str] | None = None,
        current_round: int = 0,
        round_limit: int = 0,
    ) -> VerificationStackResult:
        return self.verify_candidate(
            candidate,
            contract=contract,
            existing_formal_signatures=existing_formal_signatures,
            blocking_obligation_ids=blocking_obligation_ids,
            current_round=current_round,
            round_limit=round_limit,
        )

    def verify_candidate(
        self,
        candidate: CandidateGenome,
        *,
        contract: NexusObjectiveContract | None = None,
        existing_formal_signatures: set[str] | None = None,
        blocking_obligation_ids: list[str] | None = None,
        current_round: int = 0,
        round_limit: int = 0,
    ) -> VerificationStackResult:
        feedback: list[ToolFeedback] = []
        lessons: list[str] = []
        passed = True
        if contract is not None and candidate.contract_hash and candidate.contract_hash != contract.contract_hash():
            passed = False
            lessons.append("candidate contract hash differs from current ObjectiveContract")
            feedback.append(_feedback("contract_hash", "failed", "candidate tried to evolve against a stale or modified contract"))
        if candidate.metadata.get("search_seed_not_final") and candidate.generation == 0:
            passed = False
            lessons.append("initial search seed is not a final answer; it must be evolved or synthesized")
            feedback.append(_feedback("seed_not_final", "failed", "seed candidates are search material, not final answers"))
        if candidate.current_fate == CandidateFate.AUXILIARY and candidate.multihead_scores.get("answer_likelihood", 0.0) < candidate.multihead_scores.get("auxiliary_value", 0.0):
            lessons.append("auxiliary scaffold should feed CoreExtraction rather than win as main answer")
            feedback.append(_feedback("auxiliary_guard", "warning", "auxiliary candidate should not be selected as final answer by default"))
        if candidate.missing_parts:
            feedback.append(_feedback("missing_parts", "warning", "; ".join(candidate.missing_parts[:3])))
        no_op_reason = _no_op_patch_reason(candidate)
        if no_op_reason:
            passed = False
            lessons.append("project patch verification produced no source change")
            feedback.append(_feedback("patch_no_effect", "failed", no_op_reason))
        project_scope_reason = _project_patch_scope_reason(candidate, contract=contract)
        if project_scope_reason:
            passed = False
            lessons.append("project patch did not touch the required implementation or test surface")
            feedback.append(_feedback("runtime_code_change_required", "failed", project_scope_reason))
        artifact_summary = evaluate_candidate_against_dynamic_contract(candidate, contract=contract)
        if artifact_summary.required:
            candidate.metadata["dynamic_artifact_contract_gate"] = artifact_summary.to_dict()
            if artifact_summary.diagnostics:
                lessons.append("dynamic artifact contract requires an object-level artifact, measurable delta, claim binding, and non-self-certified final gate")
                for diagnostic in artifact_summary.diagnostics:
                    status = "failed" if diagnostic in {
                        "contract_objective_absent",
                        "final_gate_self_certifying",
                        "artifact_object_absent",
                        "meta_commentary_only",
                        "required_work_product_absent",
                        "final_gate_absent",
                    } else "warning"
                    feedback.append(_feedback(diagnostic, status, diagnostic))
                    if status == "failed":
                        passed = False
            candidate.multihead_scores["artifact_contract_progress"] = 1.0 if artifact_summary.final_eligible else (0.45 if artifact_summary.artifact_present else 0.05)
            if not artifact_summary.final_eligible:
                candidate.multihead_scores["answer_likelihood"] = min(candidate.multihead_scores.get("answer_likelihood", 0.0), 0.35)
                candidate.multihead_scores["verifiability"] = min(candidate.multihead_scores.get("verifiability", 0.0), 0.35)
                candidate.multihead_scores["deferral_risk"] = max(candidate.multihead_scores.get("deferral_risk", 0.0), 0.70)

        source_adapter_required = contract_requires_adapter(contract, "source", candidate=candidate)
        patch_adapter_required = contract_requires_adapter(contract, "patch", candidate=candidate)
        source_safety_required = _is_project_patch_candidate(candidate)
        run_source_lineage = source_safety_required if source_adapter_required is None and patch_adapter_required is None else bool(source_adapter_required or patch_adapter_required or source_safety_required)
        if run_source_lineage:
            source_lineage = analyze_source_lineage(
                candidate,
                project_root=self.project_root,
                materialization_scope=materialization_scope_from_contract(contract, candidate=candidate),
            )
            if source_lineage.required:
                if source_lineage.diagnostics:
                    passed = False
                    lessons.append("project patch source bindings must match existing files, explicit extensions, or materialized new files with integration evidence")
                    for diagnostic, fragments in source_lineage.diagnostic_fragments.items():
                        feedback.append(_feedback(diagnostic, "failed", diagnostic, failed_fragments=fragments[:8]))
                    candidate.metadata["source_lineage_gate"] = source_lineage.to_dict()
                    legacy_missing_symbols = [
                        {"path": fact.path, "symbol": fact.symbol}
                        for fact in source_lineage.facts
                        if "source_binding_missing_symbol" in fact.diagnostics
                    ]
                    if legacy_missing_symbols:
                        candidate.metadata["source_patch_preflight"] = {
                            "passed": False,
                            "missing_symbols": legacy_missing_symbols[:8],
                            "diagnostics": ["source_binding_missing_symbol"],
                        }
                else:
                    candidate.metadata.pop("source_patch_preflight", None)
                    candidate.metadata["source_lineage_gate"] = source_lineage.to_dict()
        proof_adapter_required = contract_requires_adapter(contract, "proof", candidate=candidate)
        run_proof_adapter = True if proof_adapter_required is None else bool(proof_adapter_required)
        proof_summary = proof_progress_summary(
            candidate,
            contract=contract if run_proof_adapter else None,
            existing_signatures=existing_formal_signatures,
            blocking_obligation_ids=blocking_obligation_ids,
        )
        if proof_summary.required:
            for diagnostic in proof_summary.diagnostics:
                status = "failed" if diagnostic in HARD_PROOF_FAILURES else "warning"
                feedback.append(_feedback(diagnostic, status, diagnostic))
            if any(diagnostic in HARD_PROOF_FAILURES for diagnostic in proof_summary.diagnostics):
                passed = False
                lessons.append("proof-like objective requires a concrete formal object and obligation-ledger progress")
            candidate.multihead_scores["proof_progress"] = proof_summary.score
            if not proof_summary.final_eligible:
                candidate.multihead_scores["answer_likelihood"] = min(candidate.multihead_scores.get("answer_likelihood", 0.0), 0.25)
                candidate.multihead_scores["verifiability"] = min(candidate.multihead_scores.get("verifiability", 0.0), 0.20)
                candidate.multihead_scores["deferral_risk"] = max(candidate.multihead_scores.get("deferral_risk", 0.0), 0.85)
        evidence_adapter_required = contract_requires_adapter(contract, "test", candidate=candidate)
        run_evidence_contract = True if evidence_adapter_required is None else bool(evidence_adapter_required or source_adapter_required or patch_adapter_required or candidate.obligation_delta)
        evidence_summary = evidence_obligation_summary(candidate, contract=contract if run_evidence_contract else None)
        if evidence_summary.required:
            for diagnostic in evidence_summary.diagnostics:
                status = "failed" if diagnostic in HARD_EVIDENCE_FAILURES else "warning"
                feedback.append(_feedback(diagnostic, status, diagnostic))
            if any(diagnostic in HARD_EVIDENCE_FAILURES for diagnostic in evidence_summary.diagnostics):
                passed = False
                lessons.append("candidate must bind named obligations to runtime-verifiable evidence refs or source bindings")
            candidate.multihead_scores["evidence_progress"] = evidence_summary.score
            if not evidence_summary.final_eligible:
                candidate.multihead_scores["answer_likelihood"] = min(candidate.multihead_scores.get("answer_likelihood", 0.0), 0.25)
                candidate.multihead_scores["verifiability"] = min(candidate.multihead_scores.get("verifiability", 0.0), 0.20)
                candidate.multihead_scores["deferral_risk"] = max(candidate.multihead_scores.get("deferral_risk", 0.0), 0.85)
        final_summary = final_gate_summary(candidate, contract=contract, project_root=self.project_root)
        if final_summary.required:
            for diagnostic in final_summary.diagnostics:
                feedback.append(_feedback("final_gate", "warning", diagnostic))
            if not final_summary.final_eligible:
                candidate.multihead_scores["answer_likelihood"] = min(candidate.multihead_scores.get("answer_likelihood", 0.0), 0.25)
                candidate.multihead_scores["verifiability"] = min(candidate.multihead_scores.get("verifiability", 0.0), 0.20)
                candidate.multihead_scores["deferral_risk"] = max(candidate.multihead_scores.get("deferral_risk", 0.0), 0.85)
        if not feedback:
            feedback.append(_feedback("nexus_verifier_stack", "ok", "candidate passed generic Nexus checks"))
        hard_diagnostics = [
            diagnostic
            for diagnostic in dict.fromkeys(item for feedback_item in feedback for item in feedback_item.diagnostics)
            if diagnostic in (HARD_PROOF_FAILURES | HARD_EVIDENCE_FAILURES) or any(feedback_item.status == "failed" for feedback_item in feedback)
        ]
        failure_guidance = failure_micro_guidance_from_diagnostics(
            candidate_id=candidate.id,
            diagnostics=hard_diagnostics,
            source_bindings=candidate.source_bindings,
            limit=5,
        )
        result = VerificationStackResult(
            candidate.id,
            feedback=feedback,
            passed=passed,
            failure_lessons=lessons,
            proof_progress=proof_summary.to_dict(),
            evidence_obligation=evidence_summary.to_dict(),
            final_gate=final_summary.to_dict(),
            artifact_contract=artifact_summary.to_dict(),
            rank_eligible=passed and artifact_summary.rank_eligible and proof_summary.rank_eligible and evidence_summary.rank_eligible and final_summary.rank_eligible,
            final_eligible=passed and artifact_summary.final_eligible and proof_summary.final_eligible and evidence_summary.final_eligible and final_summary.final_eligible,
            failure_guidance=failure_guidance,
        )
        for item in feedback:
            candidate.add_verification_feedback(item)
        for lesson in lessons:
            if lesson not in candidate.failure_lessons:
                candidate.failure_lessons.append(lesson)
        candidate.verification_result = result.to_dict()
        if failure_guidance:
            candidate.metadata["failure_micro_guidance"] = [item.to_dict() for item in failure_guidance[:5]]
        if current_round or round_limit:
            annotate_stage_eligibility(candidate, current_round=current_round, round_limit=round_limit)
        return result

    def verify_population(
        self,
        candidates: list[CandidateGenome],
        *,
        contract: NexusObjectiveContract | None = None,
        blocking_obligation_ids: list[str] | None = None,
        current_round: int = 0,
        round_limit: int = 0,
    ) -> list[VerificationStackResult]:
        seen_formal: set[str] = set()
        seen_lock = threading.Lock()
        max_workers = env_int(_VERIFY_MAX_WORKERS_ENV, llm_governor()._max_concurrent())

        def _verify_one(candidate: CandidateGenome) -> VerificationStackResult:
            result = self.verify_candidate(
                candidate,
                contract=contract,
                existing_formal_signatures=seen_formal,
                blocking_obligation_ids=blocking_obligation_ids,
                current_round=current_round,
                round_limit=round_limit,
            )
            sig = formal_signature(candidate)
            if sig:
                with seen_lock:
                    seen_formal.add(sig)
            return result

        if max_workers <= 1 or len(candidates) <= 1:
            return [_verify_one(c) for c in candidates]

        results: list[VerificationStackResult] = [None] * len(candidates)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=min(max_workers, len(candidates))) as pool:
            future_to_idx = {pool.submit(_verify_one, c): i for i, c in enumerate(candidates)}
            for fut in as_completed(future_to_idx):
                results[future_to_idx[fut]] = fut.result()
        return results


def _feedback(tool_id: str, status: str, diagnostic: str, *, failed_fragments: list[str] | None = None) -> ToolFeedback:
    return ToolFeedback(
        tool_id=tool_id,
        status=status,
        diagnostics=[diagnostic],
        failed_fragments=list(failed_fragments or []),
        confidence=0.75 if status == "ok" else 0.5,
    )


def _no_op_patch_reason(candidate: CandidateGenome) -> str:
    patch_result: dict[str, Any] = {}
    raw_patch = getattr(candidate, "patch_application_result", None)
    if isinstance(raw_patch, dict):
        patch_result = dict(raw_patch)
    result = getattr(candidate, "verification_result", {}) or {}
    if not patch_result and isinstance(result, dict) and isinstance(result.get("patch_result"), dict):
        patch_result = dict(result.get("patch_result") or {})
    if not patch_result:
        return ""
    status = str(patch_result.get("status") or "")
    applied_files = patch_result.get("applied_files")
    applied = [str(item) for item in applied_files] if isinstance(applied_files, list) else []
    pre_hash = str(patch_result.get("pre_hash") or "")
    post_hash = str(patch_result.get("post_hash") or "")
    if status == "no_op":
        return "patch_no_effect:no_op_status"
    if status == "applied" and not applied:
        return "patch_no_effect:no_files_applied"
    if status == "applied" and pre_hash and post_hash and pre_hash == post_hash:
        return "patch_no_effect:pre_hash_equals_post_hash"
    return ""


def _project_patch_scope_reason(candidate: CandidateGenome, *, contract: NexusObjectiveContract | None = None) -> str:
    """Reject semantic no-progress project patches for runtime/code objectives.

    The sandbox can only know whether bytes changed.  For self-evolution runs,
    a markdown-only write to ``NEXUS_SEED_NOTE.md`` changes bytes but does not
    change the runtime, tests, schemas, or executable project behavior.  This
    verifier-level guard keeps early exploration permissive while preventing a
    documentation-only loop from being treated as reproducible implementation
    progress.
    """

    if not _is_project_patch_candidate(candidate):
        return ""
    changed_files = _changed_patch_files(candidate)
    if not changed_files:
        return ""
    normalized = {_normalize_patch_path(path) for path in changed_files if path}
    if normalized and normalized <= {"nexus_seed_note.md"}:
        return "seed_note_only_patch"
    if not _requires_implementation_surface(contract=contract, candidate=candidate):
        return ""
    if normalized and all(_is_documentation_only_path(path) for path in normalized):
        return "runtime_code_change_absent:documentation_only_patch"
    return ""


def _is_project_patch_candidate(candidate: CandidateGenome) -> bool:
    artifact_type = str(getattr(candidate, "artifact_type", "") or "").lower()
    if isinstance(candidate, ProjectCandidateGenome) or artifact_type in {"project_patch", "patch", "code_patch"}:
        return True
    if bool(getattr(candidate, "patch_set", None)):
        return True
    artifact = getattr(candidate, "artifact", None)
    return isinstance(artifact, dict) and any(key in artifact for key in ("path", "patch", "patch_content", "diff", "unified_diff"))


def _changed_patch_files(candidate: CandidateGenome) -> list[str]:
    raw_patch = getattr(candidate, "patch_application_result", None)
    if isinstance(raw_patch, dict):
        applied = raw_patch.get("applied_files")
        if isinstance(applied, list):
            changed = [str(item) for item in applied if item]
            if changed:
                return changed
    touched = getattr(candidate, "touched_files", None)
    if isinstance(touched, list):
        changed = [str(item) for item in touched if item]
        if changed:
            return changed
    patch_set = getattr(candidate, "patch_set", None)
    if isinstance(patch_set, list):
        changed = []
        for op in patch_set:
            path = getattr(op, "path", "") if not isinstance(op, dict) else op.get("path", "")
            if path:
                changed.append(str(path))
        if changed:
            return changed
    artifact_paths = _artifact_patch_paths(candidate)
    return artifact_paths


def _project_path_binding_failures(candidate: CandidateGenome, *, project_root: Path | None = None) -> dict[str, list[str]]:
    """Return missing-path diagnostics for project/code patch candidates.

    Model-generated project patches often include plausible but nonexistent
    source paths.  A missing source binding is different from a weak proof: it
    means the proposed repair is aimed at the wrong local surface and should be
    redirected before it is allowed to incubate.
    """

    if not _is_project_patch_candidate(candidate):
        return {}
    root = _resolved_project_root(project_root)
    if root is None:
        return {}
    failures: dict[str, list[str]] = {}
    missing_source = [
        path
        for path in _declared_source_binding_paths(candidate, root=root)
        if not _project_relative_path_exists(root, path)
    ]
    if missing_source:
        failures["source_binding_missing_path"] = sorted(set(missing_source))
    missing_patch_targets = [
        path
        for path in _declared_existing_patch_target_paths(candidate, root=root)
        if not _project_relative_path_exists(root, path)
    ]
    if missing_patch_targets:
        failures["patch_target_missing"] = sorted(set(missing_patch_targets))
    return failures


def _project_symbol_binding_failures(candidate: CandidateGenome, *, project_root: Path | None = None) -> list[dict[str, str]]:
    """Return missing-symbol diagnostics for source-bound project patches.

    The final gate already rejects hallucinated symbols, but model-backed
    self-evolution wastes rounds when such candidates survive until late
    synthesis.  This preflight is intentionally narrower than the final gate:
    it only runs for patch-like candidates and accepts symbols that either
    already exist in the real file or are explicitly introduced by the patch.
    """

    if not _is_project_patch_candidate(candidate):
        return []
    root = _resolved_project_root(project_root)
    if root is None:
        return []
    failures: list[dict[str, str]] = []
    for binding in _declared_source_symbol_bindings(candidate, root=root):
        path = binding["path"]
        symbol = binding["symbol"]
        target = root / path
        if not target.exists() or not path.endswith(".py"):
            continue
        if _python_symbol_exists(target, symbol):
            continue
        if _patch_creates_symbol(candidate, path=path, symbol=symbol, root=root):
            continue
        failures.append({"path": path, "symbol": symbol})
    return failures


def _resolved_project_root(explicit_root: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(explicit_root)
    candidates.append(Path.cwd())
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent)
    seen: set[Path] = set()
    for raw in candidates:
        root = raw.resolve()
        if root in seen:
            continue
        seen.add(root)
        if (root / "pyproject.toml").exists() and (root / "cognitive_evolve_runtime").is_dir():
            return root
    return None


def _declared_source_binding_paths(candidate: CandidateGenome, *, root: Path) -> list[str]:
    paths: list[str] = []
    for item in list(getattr(candidate, "source_bindings", []) or []):
        if isinstance(item, dict):
            raw_path = item.get("path") or item.get("file") or item.get("source_path")
            path = _project_relative_path(raw_path, root=root)
            if path:
                paths.append(path)
    metadata = getattr(candidate, "metadata", {}) or {}
    if isinstance(metadata, dict):
        for item in metadata.get("source_bindings", []) or []:
            if isinstance(item, dict):
                path = _project_relative_path(item.get("path") or item.get("file") or item.get("source_path"), root=root)
                if path:
                    paths.append(path)
    return list(dict.fromkeys(paths))


def _declared_source_symbol_bindings(candidate: CandidateGenome, *, root: Path) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []
    raw_items: list[Any] = list(getattr(candidate, "source_bindings", []) or [])
    metadata = getattr(candidate, "metadata", {}) or {}
    if isinstance(metadata, dict):
        raw_items.extend(metadata.get("source_bindings", []) or [])
        repair = metadata.get("repair_required")
        if isinstance(repair, dict):
            raw_items.extend(repair.get("source_bindings", []) or [])
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        path = _project_relative_path(item.get("path") or item.get("file") or item.get("source_path"), root=root)
        symbol = str(item.get("symbol") or item.get("name") or "").strip()
        if path and symbol:
            bindings.append({"path": path, "symbol": symbol})
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for binding in bindings:
        unique[(binding["path"], binding["symbol"])] = binding
    return list(unique.values())


def _declared_existing_patch_target_paths(candidate: CandidateGenome, *, root: Path) -> list[str]:
    paths: list[str] = []
    patch_set = getattr(candidate, "patch_set", None)
    if isinstance(patch_set, list):
        for op in patch_set:
            raw_path = getattr(op, "path", "") if not isinstance(op, dict) else op.get("path", "")
            operation = str(getattr(op, "operation", "") if not isinstance(op, dict) else op.get("operation", "") or "").lower()
            if operation in {"replace", "delete"}:
                path = _project_relative_path(raw_path, root=root)
                if path:
                    paths.append(path)
    artifact = getattr(candidate, "artifact", None)
    if isinstance(artifact, dict):
        explicit_path = _project_relative_path(artifact.get("path") or artifact.get("file") or artifact.get("target_path"), root=root)
        patch_text = _artifact_patch_text(artifact)
        created = _created_paths_from_patch_text(patch_text, root=root)
        if explicit_path and explicit_path not in created:
            paths.append(explicit_path)
        for path in _existing_paths_from_patch_text(patch_text, root=root):
            if path and path not in created:
                paths.append(path)
    return list(dict.fromkeys(paths))


def _artifact_patch_paths(candidate: CandidateGenome) -> list[str]:
    artifact = getattr(candidate, "artifact", None)
    if not isinstance(artifact, dict):
        return []
    paths: list[str] = []
    for key in ("path", "file", "target_path"):
        value = artifact.get(key)
        if value:
            paths.append(str(value))
    patch_text = _artifact_patch_text(artifact)
    paths.extend(_raw_paths_from_patch_text(patch_text))
    return list(dict.fromkeys(paths))


def _artifact_patch_text(artifact: dict[str, Any]) -> str:
    for key in ("patch", "patch_content", "diff", "unified_diff"):
        value = artifact.get(key)
        if isinstance(value, str) and value.strip():
            return value
    content = artifact.get("content")
    if isinstance(content, str) and _looks_like_patch_text(content):
        return content
    return ""


def _patch_creates_symbol(candidate: CandidateGenome, *, path: str, symbol: str, root: Path) -> bool:
    wanted = symbol.split(".", 1)[0].strip()
    if not wanted:
        return True
    patch_set = getattr(candidate, "patch_set", None)
    if isinstance(patch_set, list):
        for op in patch_set:
            op_path = getattr(op, "path", "") if not isinstance(op, dict) else op.get("path", "")
            op_path = _project_relative_path(op_path, root=root)
            if op_path and op_path != path:
                continue
            content_parts = [
                getattr(op, "content", "") if not isinstance(op, dict) else op.get("content", ""),
                getattr(op, "new_text", "") if not isinstance(op, dict) else op.get("new_text", ""),
            ]
            if any(_text_defines_symbol(str(part or ""), wanted) for part in content_parts):
                return True
    artifact = getattr(candidate, "artifact", None)
    if isinstance(artifact, dict):
        patch_text = _artifact_patch_text(artifact)
        if patch_text:
            patch_paths = {_project_relative_path(raw_path, root=root) for raw_path in _raw_paths_from_patch_text(patch_text)}
            patch_paths.discard("")
            if (not patch_paths or path in patch_paths) and _patch_text_adds_symbol(patch_text, wanted):
                return True
        for key in ("content", "replacement", "new_text"):
            value = artifact.get(key)
            if isinstance(value, str) and _text_defines_symbol(value, wanted):
                return True
    return False


def _patch_text_adds_symbol(patch_text: str, symbol: str) -> bool:
    added_lines: list[str] = []
    for line in str(patch_text or "").splitlines():
        if line.startswith("+++") or not line.startswith("+"):
            continue
        added_lines.append(line[1:])
    return _text_defines_symbol("\n".join(added_lines), symbol)


def _text_defines_symbol(text: str, symbol: str) -> bool:
    if not symbol:
        return True
    pattern = re.compile(rf"^\s*(?:async\s+def|def|class)\s+{re.escape(symbol)}\b", re.MULTILINE)
    return bool(pattern.search(str(text or "")))


def _python_symbol_exists(path: Path, symbol: str) -> bool:
    if not symbol:
        return True
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    parts = [part for part in symbol.split(".") if part]
    if not parts:
        return True
    if len(parts) == 1:
        return any(_node_declares_symbol(node, parts[0]) for node in ast.walk(tree))
    owner, member = parts[0], parts[1]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == owner:
            return any(isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == member for child in node.body)
    return False


def _node_declares_symbol(node: ast.AST, wanted: str) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == wanted:
        return True
    if isinstance(node, ast.Assign):
        return any(_assignment_target_name(target) == wanted for target in node.targets)
    if isinstance(node, ast.AnnAssign):
        return _assignment_target_name(node.target) == wanted
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            name = alias.asname or alias.name.split(".", 1)[0]
            if name == wanted:
                return True
    return False


def _assignment_target_name(target: ast.AST) -> str:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
_DIFF_PATH_RE = re.compile(r"^(?:---|\+\+\+)\s+(.+?)\s*$")




def _looks_like_patch_text(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    lines = text.splitlines()
    has_hunk = any(line.startswith("@@") for line in lines)
    has_old = any(line.startswith("--- ") for line in lines)
    has_new = any(line.startswith("+++ ") for line in lines)
    has_git = any(_DIFF_GIT_RE.match(line.strip()) for line in lines)
    return bool(has_hunk and (has_git or (has_old and has_new)))

def _raw_paths_from_patch_text(text: str) -> list[str]:
    paths: list[str] = []
    for line in str(text or "").splitlines():
        git_match = _DIFF_GIT_RE.match(line.strip())
        if git_match:
            paths.extend([git_match.group(1), git_match.group(2)])
            continue
        path_match = _DIFF_PATH_RE.match(line.strip())
        if path_match:
            paths.append(path_match.group(1))
    return [path for path in paths if path and path != "/dev/null"]


def _existing_paths_from_patch_text(text: str, *, root: Path) -> list[str]:
    paths: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        git_match = _DIFF_GIT_RE.match(stripped)
        if git_match:
            path = _project_relative_path(git_match.group(1), root=root)
            if path:
                paths.append(path)
            continue
        if stripped.startswith("--- "):
            raw = stripped[4:].strip()
            if raw != "/dev/null":
                path = _project_relative_path(raw, root=root)
                if path:
                    paths.append(path)
    return list(dict.fromkeys(paths))


def _created_paths_from_patch_text(text: str, *, root: Path) -> set[str]:
    created: set[str] = set()
    previous_was_dev_null = False
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("--- "):
            previous_was_dev_null = stripped[4:].strip() == "/dev/null"
            continue
        if previous_was_dev_null and stripped.startswith("+++ "):
            path = _project_relative_path(stripped[4:].strip(), root=root)
            if path:
                created.add(path)
            previous_was_dev_null = False
    return created


def _project_relative_path(raw_path: Any, *, root: Path) -> str:
    text = str(raw_path or "").strip().replace("\\", "/")
    if not text or text == "/dev/null" or "://" in text:
        return ""
    if text.startswith(("a/", "b/")):
        text = text[2:]
    path = Path(text)
    if path.is_absolute():
        try:
            text = path.resolve(strict=False).relative_to(root.resolve()).as_posix()
        except ValueError:
            return text
    parts = Path(text).parts
    if any(part in {"", ".", ".."} for part in parts):
        return ""
    return Path(text).as_posix()


def _project_relative_path_exists(root: Path, relative_path: str) -> bool:
    path = _project_relative_path(relative_path, root=root)
    if not path:
        return False
    target = root / path
    try:
        target.resolve(strict=False).relative_to(root.resolve())
    except ValueError:
        return False
    return target.exists()


def _normalize_patch_path(path: str) -> str:
    return str(path or "").strip().replace("\\", "/").lstrip("./").lower()


def _is_documentation_only_path(path: str) -> bool:
    normalized = _normalize_patch_path(path)
    if not normalized:
        return False
    if normalized == "nexus_seed_note.md":
        return True
    if normalized.startswith(("docs/", "doc/", "documentation/")):
        return True
    name = normalized.rsplit("/", 1)[-1]
    if name in {"readme.md", "changelog.md", "roadmap.md", "contributing.md", "license.md"}:
        return True
    return normalized.endswith((".md", ".rst", ".txt"))


def _requires_implementation_surface(*, contract: NexusObjectiveContract | None, candidate: CandidateGenome) -> bool:
    text_parts: list[str] = [
        str(getattr(candidate, "concise_claim", "") or ""),
        str(getattr(candidate, "core_mechanism", "") or ""),
    ]
    if contract is not None:
        text_parts.extend(
            [
                str(getattr(contract, "original_user_goal", "") or ""),
                str(getattr(contract, "normalized_goal", "") or ""),
                " ".join(str(item) for item in getattr(contract, "expected_output_forms", []) or []),
                " ".join(str(item) for item in getattr(contract, "verification_preferences", []) or []),
            ]
        )
    text = " ".join(text_parts).lower()
    documentation_intent = any(token in text for token in ("documentation", "docs", "readme", "文档", "说明"))
    implementation_intent = any(
        token in text
        for token in (
            "runtime",
            "implementation",
            "source",
            "code",
            "test",
            "pytest",
            "algorithm",
            "architecture",
            "core",
            "self-evolution",
            "self evolution",
            "bootstrap",
            "project",
            "代码",
            "实现",
            "运行时",
            "测试",
            "算法",
            "架构",
            "核心",
            "自举",
            "自进化",
            "项目",
        )
    )
    return implementation_intent or not documentation_intent


__all__ = ["NexusVerifierStack", "VerificationStackResult"]
