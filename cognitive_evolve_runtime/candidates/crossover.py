"""Genome crossover helpers."""
from __future__ import annotations

from .genome import CandidateGenome
from .project_candidate import ProjectCandidateGenome
from .patch_merge import project_patch_crossover


def crossover(parent_a: CandidateGenome, parent_b: CandidateGenome, *, instruction: str = "combine complementary genes") -> CandidateGenome:
    if isinstance(parent_a, ProjectCandidateGenome) and isinstance(parent_b, ProjectCandidateGenome):
        return project_patch_crossover(parent_a, parent_b, instruction=instruction)
    return CandidateGenome(
        parent_ids=[parent_a.id, parent_b.id],
        generation=max(parent_a.generation, parent_b.generation) + 1,
        lineage=list(dict.fromkeys(parent_a.lineage + parent_b.lineage)),
        artifact=f"Crossover of {parent_a.id} and {parent_b.id}: {parent_a.artifact}\n---\n{parent_b.artifact}",
        artifact_type=parent_a.artifact_type if parent_a.artifact_type == parent_b.artifact_type else "hybrid",
        concise_claim=f"Hybrid of {parent_a.concise_claim or parent_a.id} + {parent_b.concise_claim or parent_b.id}",
        core_mechanism=" + ".join(part for part in [parent_a.core_mechanism, parent_b.core_mechanism] if part),
        assumptions=list(dict.fromkeys(parent_a.assumptions + parent_b.assumptions)),
        missing_parts=list(dict.fromkeys(parent_a.missing_parts + parent_b.missing_parts)),
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
        novelty_descriptors=list(dict.fromkeys(parent_a.novelty_descriptors + parent_b.novelty_descriptors + ["crossover"])),
        niche_memberships=list(dict.fromkeys(parent_a.niche_memberships + parent_b.niche_memberships)),
        failure_lessons=list(dict.fromkeys(parent_a.failure_lessons + parent_b.failure_lessons)),
        contract_hash=parent_a.contract_hash or parent_b.contract_hash,
        metadata={"crossover_instruction": instruction},
)


def _delta_items(candidate: CandidateGenome, key: str) -> list[str]:
    value = getattr(candidate, "obligation_delta", {}).get(key) if isinstance(getattr(candidate, "obligation_delta", {}), dict) else []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)] if value else []


def _evidence_delta_items(candidate: CandidateGenome, key: str) -> list[str]:
    delta = candidate.evidence_delta if isinstance(candidate.evidence_delta, dict) else {}
    value = delta.get(key)
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)] if value else []


__all__ = ["crossover"]
