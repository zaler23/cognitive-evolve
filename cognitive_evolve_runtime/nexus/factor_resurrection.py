"""Tiny loser-pool factor views for prompts and parent selection."""
from __future__ import annotations

from typing import Any, Iterable

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, candidate_from_dict
from cognitive_evolve_runtime.core.serialization import coerce_dict, stable_hash


def resurrect_factor_trace(candidates: Iterable[CandidateGenome], *, limit: int = 12) -> list[dict[str, Any]]:
    factors: list[dict[str, Any]] = []
    for candidate in candidates or []:
        if not _loser_pool(candidate):
            continue
        texts = list(candidate.failure_lessons or [])
        texts.extend(str(item) for item in (candidate.edge_knowledge_seeds or [])[:3])
        texts.extend(str(item) for item in (candidate.novelty_descriptors or [])[:3])
        if candidate.core_mechanism:
            texts.append(candidate.core_mechanism)
        if candidate.concise_claim:
            texts.append(candidate.concise_claim)
        factor = _factor(candidate, texts)
        if factor:
            factors.append(factor)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in sorted(factors, key=lambda x: (float(x.get("score") or 0.0), str(x.get("candidate_id") or "")), reverse=True):
        key = str(item.get("fingerprint") or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= max(1, int(limit or 1)):
            break
    return out


def failure_factor_hints(archives: Any, *, population: Iterable[CandidateGenome] | None = None, limit: int = 8) -> list[dict[str, Any]]:
    candidates = list(population or [])
    dormant = getattr(getattr(archives, "dormant_archive", None), "candidates", {}) if archives is not None else {}
    failure_records = getattr(getattr(archives, "failure_archive", None), "records", {}) if archives is not None else {}
    for payload in list(getattr(dormant, "values", lambda: [])()):
        if isinstance(payload, dict):
            candidates.append(candidate_from_dict(payload))
    for payload in list(getattr(failure_records, "values", lambda: [])()):
        if isinstance(payload, dict) and isinstance(payload.get("candidate"), dict):
            candidates.append(candidate_from_dict(payload["candidate"]))
    return resurrect_factor_trace(candidates, limit=limit)


def _loser_pool(candidate: CandidateGenome) -> bool:
    fate = CandidateFate.normalize(getattr(candidate, "current_fate", ""), default="")
    metadata = coerce_dict(candidate.metadata)
    return fate in {CandidateFate.DORMANT.value, CandidateFate.CULLED.value, CandidateFate.FAILED.value, CandidateFate.INCUBATING.value} or bool(metadata.get("seed_reservoir"))


def _factor(candidate: CandidateGenome, texts: list[str]) -> dict[str, Any] | None:
    text = " ".join(str(item).strip() for item in texts if str(item).strip())
    if not text:
        return None
    score = 0.35
    if candidate.edge_knowledge_seeds:
        score += 0.2
    if candidate.novelty_descriptors:
        score += 0.2
    if candidate.failure_lessons:
        score += 0.1
    if coerce_dict(candidate.metadata).get("intent_binding"):
        score += 0.1
    return {
        "candidate_id": candidate.id,
        "source_pool": CandidateFate.normalize(getattr(candidate, "current_fate", ""), default="reservoir").lower(),
        "factor": " ".join(text.split())[:500],
        "score": round(min(1.0, score), 4),
        "fingerprint": stable_hash(text.lower().split()[:80]),
        "policy": "advisory_factor_only_not_verified",
    }


__all__ = ["failure_factor_hints", "resurrect_factor_trace"]
