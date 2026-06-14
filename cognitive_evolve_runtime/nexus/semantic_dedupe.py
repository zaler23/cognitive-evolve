"""Semantic de-duplication helpers for Nexus candidate generation."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus.obligations import formal_signature


@dataclass
class CandidateDeduper:
    candidates: list[CandidateGenome] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.signatures: set[str] = set()
        self.niches: set[str] = set()
        for candidate in list(self.candidates):
            self.add(candidate)

    def add(self, candidate: CandidateGenome) -> bool:
        signature = candidate_semantic_signature(candidate)
        niche = normalize_token(candidate.core_mechanism or (candidate.niche_memberships[0] if candidate.niche_memberships else ""))
        candidate.metadata.setdefault("dedupe_signature", signature)
        if signature in self.signatures:
            return False
        if niche:
            self.niches.add(niche)
        self.signatures.add(signature)
        return True

    def seen_niche(self, niche: str) -> bool:
        return normalize_token(niche) in self.niches


def candidate_semantic_signature(candidate: CandidateGenome) -> str:
    mechanism = normalize_token(candidate.core_mechanism or "")
    claim = normalize_text(candidate.concise_claim or "")
    artifact = normalize_text(candidate.artifact if isinstance(candidate.artifact, str) else repr(candidate.artifact))
    if artifact:
        artifact = artifact[:240]
    proof_sig = formal_signature(candidate)
    return "|".join([str(candidate.artifact_type or "answer").lower(), mechanism, claim, artifact, proof_sig])


def normalize_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", str(value or "").strip().lower())).strip()


__all__ = ["CandidateDeduper", "candidate_semantic_signature", "normalize_text", "normalize_token"]
