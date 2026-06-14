"""Recover repairable dormant candidates when the live parent pool collapses.

This module is deliberately small and side-effect-light.  It is not a new
global niche allocator.  It only runs after normal parent selection and the
existing ranked repair fallback both fail, and it turns archived repair material
into bounded, source-grounded repair seeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, candidate_from_dict
from cognitive_evolve_runtime.nexus.adaptive_signals import adaptive_attempt_limit
from cognitive_evolve_runtime.nexus._serde import coerce_dict, stable_hash
from cognitive_evolve_runtime.nexus.failure_classifier import FailureVerdict, classify_recovery_eligibility


@dataclass(frozen=True)
class RepairSeed:
    candidate: CandidateGenome
    verdict: FailureVerdict
    source: str
    diversity_key: tuple[str, str, str]

    def to_metadata(self, *, current_round: int) -> dict[str, Any]:
        targets = list(self.verdict.repair_targets[:6])
        blockers = list(self.verdict.blockers[:6])
        guidance = list(self.verdict.failure_guidance[:4])
        return {
            "source": self.source,
            "round": int(current_round or 0),
            "candidate_id": self.candidate.id,
            "category": self.verdict.category,
            "failure_signature": self.verdict.failure_signature,
            "target_files": targets,
            "blockers": blockers,
            "failure_lessons": list(self.candidate.failure_lessons[:6]),
            "repair_guidance": guidance,
            "disallowed_repeat_patterns": [
                str(item.get("disallowed_repeat_pattern"))
                for item in guidance
                if isinstance(item, dict) and item.get("disallowed_repeat_pattern")
            ][:4],
            "required_evidence": list(
                dict.fromkeys(
                    value
                    for item in guidance
                    if isinstance(item, dict)
                    for value in item.get("evidence_needed", []) or []
                    if value
                )
            )[:6]
            or ["complete_unified_diff", "existing_project_relative_path", "post_pass_local_verification"],
            "diversity_key": list(self.diversity_key),
        }


def recover_repairable_dormant_seeds(
    *,
    archives: Any,
    diagnosis: Any,
    policy: Any,
    limit: int,
    current_round: int,
    project_root: str | Path | None = None,
) -> list[CandidateGenome]:
    """Return ranked dormant/archive repair parents for a no-parent fallback.

    The function only recovers candidates when the diagnosis indicates a repair
    or dormancy collapse.  Recovered candidates stay blocked from final answer
    synthesis; they merely seed a targeted mutation.
    """

    target = max(0, int(limit or 0))
    if target <= 0 or not _diagnosis_requests_recovery(diagnosis):
        return []
    config = _recovery_config(policy, target=target)
    if config.get("enabled") is False:
        return []
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[2]
    target = min(target, max(0, _positive_int(config.get("max_seeds"), default=target)))
    if target <= 0:
        return []
    candidates = _archive_dormant_candidates(archives)
    max_attempts = adaptive_attempt_limit(
        population_size=len(candidates),
        distinct_blockers=_distinct_repair_blocker_count(candidates),
        configured=config.get("max_repair_attempts") if str(config.get("max_repair_attempts") or "").lower() not in {"auto", "adaptive", "model"} else None,
        fallback=_positive_int(config.get("max_attempts"), default=0),
    )
    max_per_group = _positive_int(config.get("max_per_group"), default=1)
    seeds = _rank_repair_seeds(
        candidates,
        root=root,
        max_attempts=max_attempts,
        max_per_group=max_per_group,
    )
    selected: list[CandidateGenome] = []
    for seed in seeds[:target]:
        candidate = seed.candidate
        metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
        seed_metadata = seed.to_metadata(current_round=current_round)
        metadata["dormant_repair_reactivation"] = {
            "round": int(current_round or 0),
            "reason": "selected_after_live_parent_pool_empty",
            "source": seed.source,
            "final_answer_blocked": True,
        }
        metadata["repair_seed"] = seed_metadata
        metadata["final_answer_blocked_until_repaired"] = True
        metadata["dormant_kind"] = "repairable"
        metadata["repair_required"] = _repair_required_from_seed(seed_metadata, seed.verdict)
        metadata["failure_micro_guidance"] = _dedupe_guidance(
            [dict(item) for item in metadata.get("failure_micro_guidance", []) or [] if isinstance(item, dict)]
            + [dict(item) for item in seed.verdict.failure_guidance if isinstance(item, dict)]
        )[:5]
        candidate.metadata = metadata
        candidate.mark_fate(CandidateFate.INCUBATING.value)
        if "DormantRepairReactivation" not in candidate.mutation_history:
            candidate.mutation_history.append("DormantRepairReactivation")
        selected.append(candidate)
    return selected


def recover_failure_archive_repair_seeds(
    *,
    archives: Any,
    diagnosis: Any,
    policy: Any,
    limit: int,
    current_round: int,
    project_root: str | Path | None = None,
) -> list[CandidateGenome]:
    """Synthesize bounded repair parents from terminal failure lessons.

    This is the last-resort entry survival path: it only runs after the live
    population and dormant recovery are empty.  The synthesized candidates are
    Incubating repair seeds, not final answers.  They preserve only useful,
    non-docs, non-seed-note failure lessons and force the next mutation to bind
    real source context plus post-pass verification before it can progress.
    """

    target = max(0, int(limit or 0))
    if target <= 0 or not _diagnosis_requests_recovery(diagnosis):
        return []
    config = _failure_archive_reseed_config(policy, target=target)
    if config.get("enabled") is False:
        return []
    records = _failure_records(archives)
    if not records:
        return []
    target = min(target, max(0, _positive_int(config.get("max_seeds"), default=target)))
    if target <= 0:
        return []
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[2]
    tombstones = coerce_dict(getattr(archives, "terminal_tombstones", {}))
    max_per_group = _positive_int(config.get("max_per_group"), default=1)
    scored: list[tuple[float, tuple[str, str], CandidateGenome]] = []
    for record in records:
        candidate = _candidate_from_failure_record(
            record,
            tombstone=coerce_dict(tombstones.get(str(record.get("candidate_id") or ""))),
            current_round=current_round,
            project_root=root,
            require_repair_signal=config.get("require_repair_signal", True) is not False,
        )
        if candidate is None:
            continue
        key = _failure_archive_diversity_key(candidate)
        scored.append((_failure_archive_seed_score(candidate), key, candidate))
    grouped_counts: dict[tuple[str, str], int] = {}
    selected: list[CandidateGenome] = []
    for _score, key, candidate in sorted(scored, key=lambda item: item[0], reverse=True):
        if grouped_counts.get(key, 0) >= max_per_group:
            continue
        grouped_counts[key] = grouped_counts.get(key, 0) + 1
        selected.append(candidate)
        if len(selected) >= target:
            break
    return selected


def _archive_dormant_candidates(archives: Any) -> list[CandidateGenome]:
    dormant = getattr(archives, "dormant_archive", None)
    raw = getattr(dormant, "candidates", {}) if dormant is not None else {}
    out: list[CandidateGenome] = []
    if not isinstance(raw, dict):
        return out
    for value in raw.values():
        try:
            candidate = value if isinstance(value, CandidateGenome) else candidate_from_dict(coerce_dict(value))
        except Exception:
            continue
        out.append(candidate)
    return out


def _distinct_repair_blocker_count(candidates: list[CandidateGenome]) -> int:
    blockers: set[str] = set()
    for candidate in candidates:
        metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
        repair = metadata.get("repair_required")
        if isinstance(repair, dict):
            blockers.update(str(item) for item in repair.get("blockers", []) or [] if item)
        result = candidate.verification_result if isinstance(candidate.verification_result, dict) else {}
        blockers.update(str(item) for item in result.get("diagnostics", []) or [] if item)
    return len(blockers)


def _rank_repair_seeds(
    candidates: list[CandidateGenome],
    *,
    root: Path,
    max_attempts: int,
    max_per_group: int,
) -> list[RepairSeed]:
    scored: list[tuple[float, RepairSeed]] = []
    for candidate in candidates:
        verdict = classify_recovery_eligibility(candidate, project_root=root, max_repair_attempts=max_attempts)
        if not verdict.repairable:
            candidate.metadata["dormant_recovery_reject"] = verdict.to_dict()
            continue
        key = _diversity_key(candidate, verdict)
        seed = RepairSeed(candidate=candidate, verdict=verdict, source="dormant_archive", diversity_key=key)
        scored.append((_repair_seed_score(seed), seed))
    grouped_counts: dict[tuple[str, str, str], int] = {}
    out: list[RepairSeed] = []
    for _score, seed in sorted(scored, key=lambda item: item[0], reverse=True):
        if grouped_counts.get(seed.diversity_key, 0) >= max_per_group:
            continue
        grouped_counts[seed.diversity_key] = grouped_counts.get(seed.diversity_key, 0) + 1
        out.append(seed)
    return out


def _repair_seed_score(seed: RepairSeed) -> float:
    candidate = seed.candidate
    score = 0.0
    score += 2.0 if seed.verdict.category == "repairable_patch_syntax_or_context" else 0.0
    score += 1.0 if seed.verdict.repair_targets else 0.0
    score += 0.4 * min(3, len(candidate.failure_lessons))
    score += 0.2 * min(3, len(candidate.niche_memberships))
    score += float(candidate.multihead_scores.get("objective_alignment", 0.0) or 0.0)
    score += float(candidate.multihead_scores.get("verifiability", 0.0) or 0.0)
    attempts = _positive_int(candidate.metadata.get("repair_attempts") if isinstance(candidate.metadata, dict) else None, default=0)
    score -= 0.5 * attempts
    return score


def _diversity_key(candidate: CandidateGenome, verdict: FailureVerdict) -> tuple[str, str, str]:
    family = candidate.lineage[0] if candidate.lineage else candidate.id
    target = verdict.repair_targets[0] if verdict.repair_targets else "no-target"
    return (str(family), str(target), str(verdict.category))


def _repair_required_from_seed(seed_metadata: dict[str, Any], verdict: FailureVerdict) -> dict[str, Any]:
    targets = [str(path) for path in seed_metadata.get("target_files", []) if path]
    blockers = [str(item) for item in seed_metadata.get("blockers", []) if item] or [verdict.category]
    evidence = [str(item) for item in seed_metadata.get("required_evidence", []) if item]
    return {
        "blockers": blockers,
        "evidence_needed": evidence,
        "source_bindings": [{"path": path, "kind": "source_file", "source": "dormant_repair_reactivation"} for path in targets],
        "next_actions": [
            "regenerate a complete source-grounded patch artifact, not a narrative proposal",
            "bind exact project-relative files and post-pass verification evidence",
        ],
        "acceptance_criteria": [
            "patch_applies_or_patch_set_is_structurally_valid",
            "existing_project_relative_path",
            "pre_fail_post_pass_verification_plan",
        ],
        "failure_signature": verdict.failure_signature,
        "source": "dormant_repair_reactivation",
    }


def _diagnosis_requests_recovery(diagnosis: Any) -> bool:
    actions = " ".join(str(item or "").lower() for item in getattr(diagnosis, "recommended_actions", []) or [])
    stagnation = str(getattr(diagnosis, "stagnation_type", "") or "").lower()
    notes = str(getattr(diagnosis, "notes", "") or "").lower()
    text = " ".join([actions, stagnation, notes])
    return bool(
        getattr(diagnosis, "stagnation_detected", False)
        and any(
            token in text
            for token in (
                "reactivate_dormant",
                "dormant",
                "no_parents",
                "repair",
                "patch_application",
                "malformed patch",
                "unexpected eof",
                "unexpected end of file",
                "hunk",
                "source_binding",
                "route_incomplete",
                "repeated_semantic_convergence",
                "under_explored",
                "under-explored",
                "minimal_patch",
                "rare_recall",
            )
        )
    )


def _recovery_config(policy: Any, *, target: int) -> dict[str, Any]:
    metadata = coerce_dict(getattr(policy, "metadata", {}))
    eligibility = coerce_dict(metadata.get("eligibility_policy"))
    raw = coerce_dict(eligibility.get("dormant_repair_reactivation"))
    if not raw:
        parent_preferences = coerce_dict(getattr(policy, "parent_selection_preferences", {}))
        raw = coerce_dict(parent_preferences.get("dormant_repair_reactivation"))
    out = {
        "enabled": True,
        "max_repair_attempts": eligibility.get("max_incubation_attempts", "auto"),
        "max_per_group": raw.get("max_per_group", 1),
        "max_seeds": raw.get("max_seeds", target),
    }
    out.update(raw)
    return out


def _failure_archive_reseed_config(policy: Any, *, target: int) -> dict[str, Any]:
    metadata = coerce_dict(getattr(policy, "metadata", {}))
    eligibility = coerce_dict(metadata.get("eligibility_policy"))
    raw = coerce_dict(eligibility.get("failure_archive_reseed"))
    if not raw:
        parent_preferences = coerce_dict(getattr(policy, "parent_selection_preferences", {}))
        raw = coerce_dict(parent_preferences.get("failure_archive_reseed"))
    out = {
        "enabled": True,
        "max_per_group": raw.get("max_per_group", 1),
        "max_seeds": raw.get("max_seeds", target),
        "require_repair_signal": raw.get("require_repair_signal", True),
    }
    out.update(raw)
    return out


def _failure_records(archives: Any) -> list[dict[str, Any]]:
    failure_archive = getattr(archives, "failure_archive", None)
    records = getattr(failure_archive, "records", {}) if failure_archive is not None else {}
    if not isinstance(records, dict):
        return []
    return [dict(value) for value in records.values() if isinstance(value, dict)]


def _candidate_from_failure_record(
    record: dict[str, Any],
    *,
    tombstone: dict[str, Any],
    current_round: int,
    project_root: Path,
    require_repair_signal: bool,
) -> CandidateGenome | None:
    original_id = str(record.get("candidate_id") or "").strip()
    signature = str(record.get("failure_signature") or "").strip()
    summary = str(record.get("inherited_gene_summary") or "").strip()
    text = " ".join([summary, signature, str(tombstone.get("failure_signature") or ""), str(tombstone.get("niche_key") or "")])
    if not original_id or not summary or not _failure_record_reseed_allowed(text, require_repair_signal=require_repair_signal):
        return None
    targets = _existing_targets_from_text(text, project_root)
    blockers = _failure_blockers_from_text(text)
    category = _failure_archive_category(text)
    seed_id = "FR" + stable_hash({"candidate_id": original_id, "signature": signature, "summary": summary})[:10]
    mechanism = _mechanism_from_summary(summary, tombstone)
    guidance = _failure_archive_guidance(
        candidate_id=seed_id,
        blockers=blockers,
        targets=targets,
        category=category,
    )
    repair_seed = {
        "source": "failure_archive_reseed",
        "round": int(current_round or 0),
        "candidate_id": seed_id,
        "original_candidate_id": original_id,
        "category": category,
        "failure_signature": signature[:500],
        "target_files": targets[:6],
        "blockers": blockers[:6],
        "failure_lessons": blockers[:6],
        "repair_guidance": guidance,
        "disallowed_repeat_patterns": [
            "do_not_repeat_the_same_malformed_or_context_stale_diff",
            "do_not_reseed_seed_note_or_docs_only_outputs",
            "do_not_claim_progress_without_source_grounding_and_post_pass_verification",
        ],
        "required_evidence": ["complete_unified_diff", "existing_project_relative_path", "post_pass_local_verification"],
        "diversity_key": list(_failure_archive_diversity_key_from_parts(mechanism, targets, category)),
    }
    repair_required = {
        "blockers": blockers[:6] or [category],
        "evidence_needed": list(repair_seed["required_evidence"]),
        "source_bindings": [{"path": path, "kind": "source_file", "source": "failure_archive_reseed"} for path in targets[:6]],
        "next_actions": [
            "turn the inherited mechanism into a concrete source-grounded patch",
            "derive exact current project-relative source bindings before emitting diff hunks",
            "include pre-fail/post-pass verification evidence before claiming progress",
        ],
        "acceptance_criteria": [
            "patch_applies_or_patch_set_is_structurally_valid",
            "source_binding_points_to_existing_project_file",
            "post_pass_local_verification_present",
        ],
        "failure_signature": signature[:500],
        "source": "failure_archive_reseed",
    }
    score_summary = coerce_dict(tombstone.get("score_summary"))
    rare_signal = 1.0 if _rare_reseed_text(text) else 0.0
    inherited_objective = _score(score_summary.get("objective_alignment"), default=0.0)
    inherited_verifiability = _score(score_summary.get("verifiability"), default=0.0)
    scores = {
        "objective_alignment": inherited_objective,
        "answer_likelihood": 0.0,
        "verifiability": inherited_verifiability,
        "tool_progress": 0.0,
        "evidence_progress": _score(score_summary.get("evidence_progress"), default=0.0),
        "proof_progress": _score(score_summary.get("proof_progress"), default=0.0),
        "rarity": max(_score(score_summary.get("rarity"), default=0.0), rare_signal),
        "novelty": max(_score(score_summary.get("novelty"), default=0.0), rare_signal),
    }
    return CandidateGenome(
        id=seed_id,
        parent_ids=[original_id],
        generation=max(0, int(current_round or 0)),
        lineage=[original_id, seed_id],
        artifact={
            "source": "failure_archive_reseed",
            "original_candidate_id": original_id,
            "inherited_gene_summary": summary[:1000],
            "repair_contract": repair_required,
        },
        artifact_type="repair_seed",
        concise_claim=f"Repair archived failed candidate {original_id} into a source-grounded self-evolution patch.",
        core_mechanism=mechanism,
        missing_parts=[
            "exact existing source binding",
            "complete unified diff or patch_set",
            "post-pass local verification output",
        ],
        uncertainty_notes=["failure_archive_reseed is not final-answer material; it only preserves a useful repair direction"],
        edge_knowledge_seeds=_edge_seeds_from_text(text),
        inherited_genes=[summary[:500]],
        mutation_history=["FailureArchiveRepairReseed"],
        source_bindings=[{"path": path, "kind": "source_file", "source": "failure_archive_reseed"} for path in targets[:6]],
        evidence_delta={"planned": ["convert archived failure lesson into verified source-grounded patch"]},
        verification_result={
            "passed": False,
            "rank_eligible": False,
            "final_eligible": False,
            "diagnostics": blockers[:6],
            "failure_guidance": guidance,
            "source": "failure_archive_reseed",
        },
        novelty_descriptors=["failure_archive_reseed", category],
        niche_memberships=list(dict.fromkeys(["failure_archive_reseed", *_edge_seeds_from_text(text), category])),
        failure_lessons=blockers[:6],
        current_fate=CandidateFate.INCUBATING.value,
        multihead_scores=scores,
        metadata={
            "created_in_round": int(current_round or 0),
            "incubation_started_round": int(current_round or 0),
            "repair_attempts": 0,
            "final_answer_blocked_until_repaired": True,
            "source_grounding_required": True,
            "repair_seed": repair_seed,
            "repair_required": repair_required,
            "failure_micro_guidance": guidance,
            "failure_archive_reseed": {
                "round": int(current_round or 0),
                "source_candidate_id": original_id,
                "reason": "population_empty_terminal_failure_lessons_preserved_for_repair",
                "final_answer_blocked": True,
            },
        },
    )


def _failure_record_reseed_allowed(text: str, *, require_repair_signal: bool) -> bool:
    lowered = str(text or "").lower()
    if any(
        token in lowered
        for token in (
            "seed_note_only_patch",
            "documentation_only_patch",
            "runtime_code_change_absent:documentation_only_patch",
            "docs_only_essay",
            "prompt_only_gate",
            "terminal_unrelated_semantic_drift",
            "second_runtime",
            "hidden_fallback",
            "source_binding_missing_path",
            "patch_target_missing",
            "terminal_missing_project_path",
        )
    ):
        return False
    if not require_repair_signal:
        return True
    return any(
        token in lowered
        for token in (
            "patch_application_failed",
            "unified_patch_failed",
            "malformed patch",
            "unexpected eof",
            "unexpected end of file",
            "hunk",
            ".rej",
            "old_text not found",
            "source_binding_missing_symbol",
            "final_update_artifact_absent",
            "evidence obligation",
            "evidence_obligation",
        )
    )


def _failure_archive_category(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("patch_application_failed", "unified_patch_failed", "malformed patch", "hunk", "eof")):
        return "repairable_archived_patch_syntax_or_context"
    if any(token in lowered for token in ("source_binding_missing_symbol", "source_binding")):
        return "repairable_archived_source_binding_gap"
    if any(token in lowered for token in ("evidence_obligation", "ledger", "obligation")):
        return "repairable_archived_evidence_obligation_gap"
    return "repairable_archived_failure_lesson"


def _failure_blockers_from_text(text: str) -> list[str]:
    raw_parts = re.split(r"[;\n|]+", str(text or ""))
    preferred: list[str] = []
    fallback: list[str] = []
    for part in raw_parts:
        item = " ".join(part.split()).strip()
        if not item:
            continue
        lowered = item.lower()
        if any(token in lowered for token in ("patch", "hunk", "eof", "source_binding", "final_update", "evidence", "obligation", "malformed")):
            preferred.append(item)
        else:
            fallback.append(item)
    return list(dict.fromkeys(preferred or fallback))[:8]


_PATH_TOKEN_RE = re.compile(r"(?:^|[\s:`'\"])((?:cognitive_evolve_runtime|tests|scripts|tools|src)/[A-Za-z0-9_./-]+\.(?:py|json|toml|yaml|yml|md))")


def _existing_targets_from_text(text: str, project_root: Path) -> list[str]:
    out: list[str] = []
    root = project_root.resolve()
    for match in _PATH_TOKEN_RE.finditer(str(text or "")):
        raw = match.group(1).strip().strip(".,);]")
        candidate_path = (root / raw).resolve()
        try:
            candidate_path.relative_to(root)
        except ValueError:
            continue
        if _project_path_is_file(candidate_path, root=root, rel=Path(raw)):
            out.append(raw)
    return list(dict.fromkeys(out))


def _project_path_is_file(candidate_path: Path, *, root: Path, rel: Path) -> bool:
    if candidate_path.is_file():
        return True
    normalized = str(rel).replace("\\", "/")
    if normalized == "cognitive_evolve_runtime/nexus/loop.py":
        return (root / "cognitive_evolve_runtime/nexus/loop/__init__.py").is_file()
    return False


def _mechanism_from_summary(summary: str, tombstone: dict[str, Any]) -> str:
    for source in (summary, str(tombstone.get("niche_key") or "")):
        for part in re.split(r"[;\n]+", source):
            text = " ".join(str(part or "").split()).strip()
            if len(text) >= 12 and not text.startswith("................................................................"):
                return text[:300]
    return "archived failure lesson requires source-grounded repair"


def _failure_archive_guidance(*, candidate_id: str, blockers: list[str], targets: list[str], category: str) -> list[dict[str, Any]]:
    if not blockers:
        blockers = [category]
    next_action = (
        "regenerate a complete unified diff against exact current source context"
        if "patch" in category
        else "bind exact source files and evidence obligations before proposing the patch"
    )
    return [
        {
            "candidate_id": candidate_id,
            "blocker": blocker,
            "next_action": next_action,
            "evidence_needed": ["complete_unified_diff", "existing_project_relative_path", "post_pass_local_verification"],
            "source_bindings": [{"path": path, "kind": "source_file", "source": "failure_archive_reseed"} for path in targets[:5]],
            "disallowed_repeat_pattern": "do_not_repeat_the_archived_failure_shape_without_new_source_grounded_evidence",
            "severity": "error",
        }
        for blocker in blockers[:4]
    ]


def _failure_archive_seed_score(candidate: CandidateGenome) -> float:
    scores = candidate.multihead_scores
    score = 1.0
    score += 0.8 if candidate.source_bindings else 0.0
    score += 0.4 if candidate.edge_knowledge_seeds else 0.0
    score += float(scores.get("evidence_progress", 0.0) or 0.0)
    score += float(scores.get("proof_progress", 0.0) or 0.0)
    score += float(scores.get("rarity", 0.0) or 0.0)
    score -= 0.2 * min(3, len(candidate.failure_lessons))
    return score


def _failure_archive_diversity_key(candidate: CandidateGenome) -> tuple[str, str]:
    seed = candidate.edge_knowledge_seeds[0] if candidate.edge_knowledge_seeds else "general"
    targets = candidate.metadata.get("repair_seed", {}).get("target_files", []) if isinstance(candidate.metadata, dict) else []
    target = str(targets[0]) if targets else "source_to_bind"
    return _failure_archive_diversity_key_from_parts(seed, [target], candidate.niche_memberships[-1] if candidate.niche_memberships else "")


def _failure_archive_diversity_key_from_parts(mechanism: str, targets: list[str], category: str) -> tuple[str, str]:
    primary = _tokenize_family(mechanism)
    target = targets[0] if targets else category or "source_to_bind"
    return (primary, _tokenize_family(target))


def _edge_seeds_from_text(text: str) -> list[str]:
    lowered = text.lower()
    seeds: list[str] = []
    for token in (
        "internal_forgotten_pattern",
        "rare_recall_seed",
        "evidence_obligation_tracking",
        "route_selection",
        "principled_final_gate_readiness_scoring",
        "niche_diversity",
        "lesson_aware_mutation",
        "minimal_patch",
    ):
        if token in lowered or token.replace("_", " ") in lowered:
            seeds.append(token)
    return list(dict.fromkeys(seeds))


def _rare_reseed_text(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("rare", "internal_forgotten", "under_explored", "under-explored", "diversity"))


def _tokenize_family(value: str) -> str:
    text = re.sub(r"[^a-z0-9_./-]+", "_", str(value or "").lower()).strip("_")
    return text[:80] or "general"


def _score(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < 0:
        return 0.0
    if parsed > 1:
        return 1.0
    return parsed


def _positive_int(value: Any, *, default: int) -> int:
    if isinstance(value, str) and value.strip().lower() in {"", "auto", "adaptive", "model"}:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


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


__all__ = ["RepairSeed", "recover_failure_archive_repair_seeds", "recover_repairable_dormant_seeds"]
