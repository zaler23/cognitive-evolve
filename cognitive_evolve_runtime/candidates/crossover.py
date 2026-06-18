"""Genome crossover helpers."""
from __future__ import annotations

from .genome import CandidateGenome
from .project_candidate import ProjectCandidateGenome
from .patch_merge import project_patch_crossover
from cognitive_evolve_runtime.archives.quality_diversity import candidate_final_quality, candidate_search_quality
from cognitive_evolve_runtime.nexus.v23_theory_config import CACrossoverConfig


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


def descriptor_tokens(candidate: CandidateGenome) -> set[str]:
    """Return deterministic descriptor tokens for local-neighborhood crossover."""

    values: list[str] = [
        candidate.id,
        candidate.artifact_type,
        candidate.concise_claim,
        candidate.core_mechanism,
    ]
    values.extend(candidate.novelty_descriptors)
    values.extend(candidate.niche_memberships)
    values.extend(candidate.edge_knowledge_seeds)
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    for key in ("descriptor_cell", "bin_key", "search_family", "family", "seed_type"):
        if metadata.get(key):
            values.append(str(metadata.get(key)))
    out: set[str] = set()
    for value in values:
        text = str(value or "").strip().lower()
        if not text:
            continue
        normalized = text.replace("|", " ").replace(":", " ").replace(",", " ")
        out.add("_".join(normalized.split()))
        out.update(part for part in normalized.split() if part)
    return {token for token in out if token}


def jaccard_similarity(a: set[str] | list[str] | tuple[str, ...], b: set[str] | list[str] | tuple[str, ...]) -> float:
    left = {str(item) for item in a if str(item or "").strip()}
    right = {str(item) for item in b if str(item or "").strip()}
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def neighborhood_crossover_partner(
    pivot: CandidateGenome,
    population: list[CandidateGenome],
    config: CACrossoverConfig | None = None,
) -> CandidateGenome | None:
    """Select a deterministic descriptor-neighborhood donor for ``pivot``.

    If no candidate shares descriptor tokens, the config-specified global donor
    policy chooses a deterministic donor rather than falling back to random.
    """

    cfg = config or CACrossoverConfig()
    donors = [candidate for candidate in population if candidate.id != pivot.id]
    if not donors:
        return None
    pivot_tokens = descriptor_tokens(pivot)

    def _local_key(candidate: CandidateGenome) -> tuple[float, float, float, str]:
        tokens = descriptor_tokens(candidate)
        shared = pivot_tokens & tokens
        if len(shared) < max(1, int(cfg.min_shared_descriptor_tokens or 1)):
            return (-1.0, -1.0, -1.0, candidate.id)
        return (
            jaccard_similarity(pivot_tokens, tokens),
            candidate_search_quality(candidate),
            candidate_final_quality(candidate),
            candidate.id,
        )

    local = [candidate for candidate in donors if _local_key(candidate)[0] >= 0.0]
    if local:
        return max(local, key=_local_key)
    return _global_donor(pivot, donors, cfg)


def _global_donor(pivot: CandidateGenome, donors: list[CandidateGenome], cfg: CACrossoverConfig) -> CandidateGenome | None:
    policy = str(cfg.global_donor_policy or "").strip().lower()
    if not donors:
        return None
    if policy == "highest_final_quality":
        return max(donors, key=lambda candidate: (candidate_final_quality(candidate), candidate_search_quality(candidate), candidate.id))
    if policy == "lowest_similarity":
        pivot_tokens = descriptor_tokens(pivot)
        return max(donors, key=lambda candidate: (-jaccard_similarity(pivot_tokens, descriptor_tokens(candidate)), candidate_search_quality(candidate), candidate.id))
    return max(donors, key=lambda candidate: (candidate_search_quality(candidate), candidate_final_quality(candidate), candidate.id))


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


__all__ = ["crossover", "descriptor_tokens", "jaccard_similarity", "neighborhood_crossover_partner"]
