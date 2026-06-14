"""Project patch merge and crossover helpers for Nexus candidates."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .project_candidate import PatchOperation, ProjectCandidateGenome


@dataclass
class PatchMergeConflict:
    path: str
    reason: str
    left_operation: dict[str, Any] = field(default_factory=dict)
    right_operation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "reason": self.reason,
            "left_operation": self.left_operation,
            "right_operation": self.right_operation,
        }


@dataclass
class PatchMergeResult:
    patch_set: list[PatchOperation]
    conflicts: list[PatchMergeConflict] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.conflicts

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_set": [op.to_dict() for op in self.patch_set],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "clean": self.clean,
        }


def merge_patch_sets(left: list[PatchOperation], right: list[PatchOperation]) -> PatchMergeResult:
    """Merge conservative patch primitives and report same-file conflicts.

    The sandbox patch primitives are intentionally tiny.  A conflict is reported
    when two non-append operations target the same path with different payloads,
    or when either side deletes a path touched by the other side.  Clean append
    operations can co-exist because their order is deterministic: left, then
    right.
    """

    merged: list[PatchOperation] = []
    conflicts: list[PatchMergeConflict] = []
    by_path: dict[str, PatchOperation] = {}
    for op in left + right:
        existing = by_path.get(op.path)
        if existing is None:
            by_path[op.path] = op
            merged.append(op)
            continue
        if _compatible(existing, op):
            if existing.operation == "append" and op.operation == "append":
                merged.append(op)
            elif existing.to_dict() == op.to_dict():
                continue
            else:
                merged.append(op)
            continue
        conflicts.append(
            PatchMergeConflict(
                path=op.path,
                reason="same_path_incompatible_patch_operations",
                left_operation=existing.to_dict(),
                right_operation=op.to_dict(),
            )
        )
    return PatchMergeResult(patch_set=merged, conflicts=conflicts)


def project_patch_crossover(parent_a: ProjectCandidateGenome, parent_b: ProjectCandidateGenome, *, instruction: str = "combine complementary project patch genes") -> ProjectCandidateGenome:
    merge = merge_patch_sets(parent_a.patch_set, parent_b.patch_set)
    risk_notes = list(dict.fromkeys(parent_a.risk_notes + parent_b.risk_notes))
    if merge.conflicts:
        risk_notes.append("patch_merge_conflicts_require_model_or_human_resolution")
    return ProjectCandidateGenome(
        parent_ids=[parent_a.id, parent_b.id],
        generation=max(parent_a.generation, parent_b.generation) + 1,
        lineage=list(dict.fromkeys(parent_a.lineage + parent_b.lineage)),
        artifact={"crossover": [parent_a.id, parent_b.id], "merge": merge.to_dict()},
        artifact_type="project_patch",
        patch_set=merge.patch_set,
        concise_claim=f"Project patch hybrid of {parent_a.id} + {parent_b.id}",
        core_mechanism=" + ".join(part for part in [parent_a.core_mechanism, parent_b.core_mechanism] if part),
        assumptions=list(dict.fromkeys(parent_a.assumptions + parent_b.assumptions)),
        missing_parts=list(dict.fromkeys(parent_a.missing_parts + parent_b.missing_parts + (["resolve_patch_merge_conflicts"] if merge.conflicts else []))),
        uncertainty_notes=list(dict.fromkeys(parent_a.uncertainty_notes + parent_b.uncertainty_notes)),
        edge_knowledge_seeds=list(dict.fromkeys(parent_a.edge_knowledge_seeds + parent_b.edge_knowledge_seeds)),
        inherited_genes=[parent_a.extract_inheritable_gene_summary(), parent_b.extract_inheritable_gene_summary()],
        mutation_history=list(dict.fromkeys(parent_a.mutation_history + parent_b.mutation_history + ["CrossOver"])),
        tool_results=parent_a.tool_results + parent_b.tool_results,
        verification_trace=parent_a.verification_trace + parent_b.verification_trace,
        formal_artifacts=list(parent_a.formal_artifacts) + list(parent_b.formal_artifacts),
        proof_obligations=list(parent_a.proof_obligations) + list(parent_b.proof_obligations),
        obligation_delta={
            "introduced": list(dict.fromkeys(_delta_items(parent_a, "introduced") + _delta_items(parent_b, "introduced"))),
            "targeted": list(dict.fromkeys(_delta_items(parent_a, "targeted") + _delta_items(parent_b, "targeted"))),
            "decomposed": list(dict.fromkeys(_delta_items(parent_a, "decomposed") + _delta_items(parent_b, "decomposed"))),
            "discharged": list(dict.fromkeys(_delta_items(parent_a, "discharged") + _delta_items(parent_b, "discharged"))),
            "refuted": list(dict.fromkeys(_delta_items(parent_a, "refuted") + _delta_items(parent_b, "refuted"))),
        },
        evidence_refs=list(parent_a.evidence_refs) + list(parent_b.evidence_refs),
        source_bindings=list(parent_a.source_bindings) + list(parent_b.source_bindings),
        evidence_delta={
            "added": list(dict.fromkeys(_evidence_delta_items(parent_a, "added") + _evidence_delta_items(parent_b, "added"))),
            "verified": list(dict.fromkeys(_evidence_delta_items(parent_a, "verified") + _evidence_delta_items(parent_b, "verified"))),
            "refuted": list(dict.fromkeys(_evidence_delta_items(parent_a, "refuted") + _evidence_delta_items(parent_b, "refuted"))),
        },
        novelty_descriptors=list(dict.fromkeys(parent_a.novelty_descriptors + parent_b.novelty_descriptors + ["project_patch_crossover"])),
        niche_memberships=list(dict.fromkeys(parent_a.niche_memberships + parent_b.niche_memberships)),
        failure_lessons=list(dict.fromkeys(parent_a.failure_lessons + parent_b.failure_lessons)),
        contract_hash=parent_a.contract_hash or parent_b.contract_hash,
        touched_files=list(dict.fromkeys(parent_a.touched_files + parent_b.touched_files + [op.path for op in merge.patch_set])),
        touched_symbols=list(dict.fromkeys(parent_a.touched_symbols + parent_b.touched_symbols)),
        expected_effects=list(dict.fromkeys(parent_a.expected_effects + parent_b.expected_effects + ["merge complementary patch genes"])),
        affected_tests=list(dict.fromkeys(parent_a.affected_tests + parent_b.affected_tests)),
        risk_notes=risk_notes,
        mutation_operator="CrossOver",
        metadata={"crossover_instruction": instruction, "patch_merge_conflicts": [c.to_dict() for c in merge.conflicts]},
)


def _delta_items(candidate: ProjectCandidateGenome, key: str) -> list[str]:
    delta = candidate.obligation_delta if isinstance(candidate.obligation_delta, dict) else {}
    value = delta.get(key)
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)] if value else []


def _evidence_delta_items(candidate: ProjectCandidateGenome, key: str) -> list[str]:
    delta = candidate.evidence_delta if isinstance(candidate.evidence_delta, dict) else {}
    value = delta.get(key)
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)] if value else []


def _compatible(existing: PatchOperation, incoming: PatchOperation) -> bool:
    if existing.to_dict() == incoming.to_dict():
        return True
    if existing.operation == "append" and incoming.operation == "append":
        return True
    if "delete" in {existing.operation, incoming.operation}:
        return False
    # replace operations on distinct old_text fragments can be applied in order.
    if existing.operation == incoming.operation == "replace" and existing.old_text != incoming.old_text:
        return True
    return False


__all__ = ["PatchMergeConflict", "PatchMergeResult", "merge_patch_sets", "project_patch_crossover"]
