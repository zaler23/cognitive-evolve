"""Semantic de-duplication helpers for Nexus candidate generation."""
from __future__ import annotations

from dataclasses import dataclass, field

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus.search_kernel.fingerprints import candidate_phenotype_signature, candidate_semantic_signature, normalize_text, normalize_token


@dataclass
class CandidateDeduper:
    candidates: list[CandidateGenome] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.signatures: set[str] = set()
        self.phenotypes: set[str] = set()
        self.niches: set[str] = set()
        for candidate in list(self.candidates):
            self.add(candidate)

    def add(self, candidate: CandidateGenome) -> bool:
        signature = candidate_semantic_signature(candidate)
        phenotype = candidate_phenotype_signature(candidate)
        niche = normalize_token(candidate.core_mechanism or (candidate.niche_memberships[0] if candidate.niche_memberships else ""))
        candidate.metadata.setdefault("dedupe_signature", signature)
        if phenotype:
            candidate.metadata["phenotype_signature"] = phenotype
        if phenotype and phenotype in self.phenotypes:
            candidate.metadata["dedupe_reason"] = "duplicate_materialized_artifact"
            return False
        if signature in self.signatures:
            candidate.metadata["dedupe_reason"] = "duplicate_semantic_signature"
            return False
        candidate.metadata.pop("dedupe_reason", None)
        if niche:
            self.niches.add(niche)
        self.signatures.add(signature)
        if phenotype:
            self.phenotypes.add(phenotype)
        return True

    def seen_niche(self, niche: str) -> bool:
        return normalize_token(niche) in self.niches




__all__ = ["CandidateDeduper", "candidate_semantic_signature", "normalize_text", "normalize_token"]
