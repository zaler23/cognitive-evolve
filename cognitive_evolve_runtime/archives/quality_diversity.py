"""Quality-diversity helpers for Nexus genomes.

This module deliberately separates *adaptive diversity pressure* from a fixed
global population cap.  A Nexus run may keep growing across genuinely different
niches, but clone-heavy bins should not keep every low-value live genome forever.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.nexus.adaptive_signals import in_top_band, score
from cognitive_evolve_runtime.core.scalars import bounded_score
from cognitive_evolve_runtime.nexus.search_kernel.descriptor_cells import behavior_descriptor, descriptor_cell_key
from cognitive_evolve_runtime.nexus.v23_theory_config import EntropyCompactionConfig, V23TheoryRuntimeConfig


def candidate_quality(candidate: CandidateGenome) -> float:
    return candidate_final_quality(candidate)


def candidate_final_quality(candidate: CandidateGenome) -> float:
    scores = candidate.multihead_scores
    axes = [
        "final_verification",
        "final_confidence",
        "objective_alignment",
        "answer_likelihood",
        "core_mechanism_strength",
        "verifiability",
        "tool_progress",
        "proof_progress",
        "evidence_progress",
        "robustness",
    ]
    if not scores:
        return 0.0
    return sum(_score(scores.get(axis, 0.0)) for axis in axes) / max(1, len(axes))


def candidate_search_quality(candidate: CandidateGenome) -> float:
    scores = candidate.multihead_scores
    axes = [
        "frontier_score",
        "challenge_pass_rate",
        "challenge_resolution",
        "schema_cleanliness",
        "continuation_value",
        "novelty",
        "rarity",
        "repair_value",
    ]
    if not scores:
        return candidate_final_quality(candidate)
    return sum(_score(scores.get(axis, 0.0)) for axis in axes) / max(1, len(axes))


def candidate_bin_key(candidate: CandidateGenome) -> str:
    """Return the MAP-Elites-style live bin for a candidate.

    The key is intentionally coarse and semantic.  It avoids creating a new bin
    for every tiny wording change while still preserving rare mechanisms,
    evidence-bearing variants, and project/source-grounded candidates.
    """

    descriptor = list(behavior_descriptor(candidate))
    primary = descriptor[0] if descriptor else "general"
    evidence_shape = "evidence" if (candidate.evidence_delta or candidate.evidence_refs or candidate.source_bindings) else "proposal"
    rare_shape = "rare" if (candidate.edge_knowledge_seeds or score(candidate, "rarity") > 0 and score(candidate, "rarity") >= score(candidate, "novelty")) else "common"
    return "|".join(_token(item) for item in ([primary] + descriptor[1:4] + [evidence_shape, rare_shape]) if item)



def descriptor_cell_distribution(candidates: list[CandidateGenome]) -> dict[str, int]:
    """Return the live-compaction descriptor distribution.

    The search-kernel ``descriptor_cell_key`` intentionally includes lineage for
    fine-grained archive analysis. Live compaction needs a coarser distribution
    or clone-heavy bins would never compact, so v2.3 entropy uses the existing
    MAP-Elites-style ``candidate_bin_key`` as the descriptor unit.
    """

    distribution: dict[str, int] = {}
    for candidate in candidates:
        key = candidate_bin_key(candidate)
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


def descriptor_population_entropy(candidates: list[CandidateGenome]) -> float:
    distribution = descriptor_cell_distribution(candidates)
    total = sum(distribution.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in distribution.values():
        probability = count / total
        if probability > 0.0:
            entropy -= probability * math.log(probability, 2)
    return entropy

def live_reproductive_candidates(candidates: list[CandidateGenome]) -> list[CandidateGenome]:
    """Candidates allowed to influence live novelty, saturation, and parents."""

    allowed = {CandidateFate.ACTIVE.value, CandidateFate.ELITE.value, CandidateFate.INCUBATING.value}
    return [candidate for candidate in candidates if CandidateFate.normalize(candidate.current_fate) in allowed]


def pareto_frontier_ids(candidates: list[CandidateGenome], *, max_axes: int = 3) -> set[str]:
    """Return a conservative low-dimensional Pareto frontier.

    High-dimensional Pareto ranking makes almost everything non-dominated.  The
    runtime therefore uses at most three aggregate axes and treats the frontier
    as a small parent/archive diversity signal, not as final-answer proof.
    """

    axes = _pareto_vectors(candidates, max_axes=max_axes)
    frontier: set[str] = set()
    for candidate in candidates:
        vector = axes.get(candidate.id)
        if vector is None:
            continue
        dominated = False
        for other in candidates:
            if other.id == candidate.id:
                continue
            other_vector = axes.get(other.id)
            if other_vector is None:
                continue
            if _dominates(other_vector, vector):
                dominated = True
                break
        if not dominated:
            frontier.add(candidate.id)
    return frontier


def entropy_diversity_survivors(
    candidates: list[CandidateGenome],
    *,
    target_k: int,
    config: EntropyCompactionConfig | None = None,
) -> tuple[list[CandidateGenome], list[CandidateGenome]]:
    """Select survivors by maximum-entropy pressure plus QD quality.

    The selector first protects the best candidate in every live compaction bin,
    then fills remaining slots with the candidates that best preserve descriptor
    entropy while retaining rare/frontier/search-quality material.
    """

    if not candidates:
        return [], []
    cfg = config or EntropyCompactionConfig()
    target = max(1, min(int(target_k or 1), len(candidates)))
    bins: dict[str, list[CandidateGenome]] = {}
    for candidate in candidates:
        bins.setdefault(candidate_bin_key(candidate), []).append(candidate)
    frontier_ids = pareto_frontier_ids(candidates)
    chosen: list[CandidateGenome] = []
    chosen_ids: set[str] = set()

    for key in sorted(bins):
        for candidate in sorted(bins[key], key=lambda item: (candidate_quality(item), candidate_search_quality(item), item.id), reverse=True)[: cfg.cell_elite_reserve]:
            if candidate.id not in chosen_ids:
                chosen.append(candidate)
                chosen_ids.add(candidate.id)

    if len(chosen) < target and cfg.rare_reserve_per_cell > 0:
        for key in sorted(bins):
            rare_candidates = [
                candidate
                for candidate in bins[key]
                if candidate.id not in chosen_ids and (candidate.edge_knowledge_seeds or float(score(candidate, "rarity") or 0.0) > 0.0)
            ]
            for candidate in sorted(rare_candidates, key=lambda item: (float(score(item, "rarity") or 0.0), candidate_search_quality(item), item.id), reverse=True)[: cfg.rare_reserve_per_cell]:
                if len(chosen) >= target:
                    break
                chosen.append(candidate)
                chosen_ids.add(candidate.id)
            if len(chosen) >= target:
                break

    def _score(candidate: CandidateGenome, current: list[CandidateGenome]) -> tuple[float, float, float, str]:
        before = descriptor_population_entropy(current)
        after = descriptor_population_entropy(current + [candidate])
        entropy_gain = after - before
        frontier = 1.0 if candidate.id in frontier_ids else 0.0
        rarity = max(float(score(candidate, "rarity") or 0.0), 1.0 if candidate.edge_knowledge_seeds else 0.0)
        search_quality = candidate_search_quality(candidate)
        weighted = (
            cfg.entropy_gain_weight * entropy_gain
            + cfg.frontier_weight * frontier
            + cfg.rarity_weight * rarity
            + cfg.search_quality_weight * search_quality
        )
        return (weighted, candidate_quality(candidate), search_quality, candidate.id)

    while len(chosen) < target:
        remaining = [candidate for candidate in candidates if candidate.id not in chosen_ids]
        if not remaining:
            break
        best = max(remaining, key=lambda candidate: _score(candidate, chosen))
        chosen.append(best)
        chosen_ids.add(best.id)

    compacted = [candidate for candidate in candidates if candidate.id not in chosen_ids]
    return chosen, compacted


def quality_diversity_survivors(
    candidates: list[CandidateGenome],
    *,
    bin_capacity: int,
    rare_reserve_per_bin: int = 1,
    config: EntropyCompactionConfig | V23TheoryRuntimeConfig | None = None,
) -> tuple[list[CandidateGenome], list[CandidateGenome]]:
    """Compatibility wrapper for v2.3 entropy-QD live survivors."""

    if isinstance(config, V23TheoryRuntimeConfig):
        entropy_config = config.entropy
    elif isinstance(config, EntropyCompactionConfig):
        entropy_config = config
    else:
        entropy_config = EntropyCompactionConfig(rare_reserve_per_cell=max(0, int(rare_reserve_per_bin or 0)))
    capacity = max(1, int(bin_capacity or 1))
    reserve = max(0, int(rare_reserve_per_bin or entropy_config.rare_reserve_per_cell or 0))
    grouped: dict[str, dict[str, int]] = {}
    for candidate in candidates:
        full_key = candidate_bin_key(candidate)
        group_key = full_key.split("|", 1)[0]
        group = grouped.setdefault(group_key, {"count": 0, "cells": 0})
        group["count"] += 1
    for key in descriptor_cell_distribution(candidates):
        group_key = key.split("|", 1)[0]
        grouped.setdefault(group_key, {"count": 0, "cells": 0})["cells"] += 1
    target = sum(max(group["cells"], min(group["count"], capacity + reserve)) for group in grouped.values())
    return entropy_diversity_survivors(candidates, target_k=target, config=entropy_config)


@dataclass
class QualityDiversityArchive:
    elites_by_niche: dict[str, dict[str, Any]] = field(default_factory=dict)
    cell_elites: dict[str, dict[str, Any]] = field(default_factory=dict)
    directive_descriptors: dict[str, dict[str, Any]] = field(default_factory=dict)
    rebalance_requests: list[dict[str, Any]] = field(default_factory=list)
    sparse_cells: list[str] = field(default_factory=list)

    def update(self, candidate: CandidateGenome) -> None:
        niches = candidate.niche_memberships or candidate.novelty_descriptors or [candidate.core_mechanism or "general"]
        final_quality = candidate_final_quality(candidate)
        search_quality = candidate_search_quality(candidate)
        quality = final_quality
        cell_key = descriptor_cell_key(candidate)
        current_cell = self.cell_elites.get(cell_key)
        current_cell_score = max(float(current_cell.get("quality", -1.0)), float(current_cell.get("search_quality", -1.0))) if current_cell else -1.0
        if current_cell is None or max(final_quality, search_quality) >= current_cell_score:
            self.cell_elites[cell_key] = {
                "candidate_id": candidate.id,
                "quality": quality,
                "search_quality": search_quality,
                "final_quality": final_quality,
                "bin_key": candidate_bin_key(candidate),
                "descriptor_cell": cell_key,
                "candidate": candidate.to_dict(),
            }
        for niche in niches:
            current = self.elites_by_niche.get(niche)
            current_score = max(float(current.get("quality", -1.0)), float(current.get("search_quality", -1.0))) if current else -1.0
            if current is None or max(final_quality, search_quality) >= current_score:
                self.elites_by_niche[niche] = {
                    "candidate_id": candidate.id,
                    "quality": quality,
                    "search_quality": search_quality,
                    "final_quality": final_quality,
                    "bin_key": candidate_bin_key(candidate),
                    "descriptor_cell": cell_key,
                    "candidate": candidate.to_dict(),
                }


    def apply_directive(self, directive: dict[str, Any], candidates: list[CandidateGenome] | None = None) -> dict[str, Any]:
        kind = str(directive.get("kind") or "")
        raw_descriptor = directive.get("descriptor")
        descriptor = _descriptor_key(raw_descriptor)
        payload = directive.get("payload") if isinstance(directive.get("payload"), dict) else {}
        candidates = list(candidates or [])
        if kind == "add_descriptor":
            source_ids = [str(item) for item in payload.get("source_candidate_ids", []) if item] if isinstance(payload.get("source_candidate_ids"), list) else []
            resolved_ids = [str(item) for item in payload.get("resolved_challenge_ids", []) if item] if isinstance(payload.get("resolved_challenge_ids"), list) else []
            descriptor_supplied = bool(raw_descriptor) if not isinstance(raw_descriptor, (list, tuple)) else bool(raw_descriptor)
            token = str(payload.get("descriptor_token") or payload.get("token") or (descriptor if descriptor_supplied else "") or "").strip()
            matched = _matched_candidate_ids(candidates, source_ids=source_ids, token=token)
            if not (source_ids or resolved_ids or token):
                return {"changed": False, "reason": "archive_directive_missing_matchable_payload", "matched_candidate_ids": []}
            record = {
                "kind": kind,
                "descriptor": descriptor,
                "payload": dict(payload),
                "source_candidate_ids": source_ids,
                "resolved_challenge_ids": resolved_ids,
                "descriptor_token": token,
                "matched_candidate_ids": matched,
            }
            changed = self.directive_descriptors.get(descriptor) != record
            self.directive_descriptors[descriptor] = record
            return {"changed": changed, "reason": "archive_descriptor_recorded" if changed else "archive_descriptor_already_present", "matched_candidate_ids": matched}
        if kind == "rebalance":
            request = {"kind": kind, "descriptor": descriptor, "payload": dict(payload)}
            existing = {_descriptor_key(item.get("descriptor")): item for item in self.rebalance_requests if isinstance(item, dict)}
            changed = existing.get(descriptor) != request
            existing[descriptor] = request
            self.rebalance_requests = list(existing.values())[-100:]
            if descriptor not in self.sparse_cells:
                self.sparse_cells.append(descriptor)
                self.sparse_cells = self.sparse_cells[-200:]
            return {"changed": changed, "reason": "archive_rebalance_recorded" if changed else "archive_rebalance_already_present", "matched_candidate_ids": []}
        return {"changed": False, "reason": "archive_directive_unknown_kind", "matched_candidate_ids": []}

    def directive_boost(self, candidate: CandidateGenome) -> float:
        boost = 0.0
        candidate_tokens = _candidate_descriptor_tokens(candidate)
        for record in self.directive_descriptors.values():
            if not isinstance(record, dict):
                continue
            if candidate.id in set(record.get("source_candidate_ids") or []) or candidate.id in set(record.get("matched_candidate_ids") or []):
                boost = max(boost, 0.2)
            token = str(record.get("descriptor_token") or "").strip().lower()
            if token and token in candidate_tokens:
                boost = max(boost, 0.12)
        candidate_cell = descriptor_cell_key(candidate).lower()
        for request in self.rebalance_requests:
            descriptor = str(request.get("descriptor") or "").strip().lower() if isinstance(request, dict) else ""
            if descriptor and (descriptor in candidate_cell or any(part and part in candidate_tokens for part in descriptor.replace("|", ":").split(":"))):
                boost = max(boost, 0.1)
        for sparse in self.sparse_cells:
            descriptor = str(sparse or "").strip().lower()
            if descriptor and (descriptor in candidate_cell or any(part and part in candidate_tokens for part in descriptor.replace("|", ":").split(":"))):
                boost = max(boost, 0.1)
        return boost

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QualityDiversityArchive":
        return cls(
            elites_by_niche=dict(data.get("elites_by_niche") or {}),
            cell_elites={str(k): dict(v) for k, v in dict(data.get("cell_elites") or {}).items() if isinstance(v, dict)},
            directive_descriptors={str(k): dict(v) for k, v in dict(data.get("directive_descriptors") or {}).items() if isinstance(v, dict)},
            rebalance_requests=[dict(item) for item in data.get("rebalance_requests", []) if isinstance(item, dict)],
            sparse_cells=[str(item) for item in data.get("sparse_cells", []) if str(item or "").strip()],
        )


def _descriptor_key(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value if str(item or "").strip()) or "general"
    return str(value or "general")


def _candidate_descriptor_tokens(candidate: CandidateGenome) -> set[str]:
    values: list[str] = [candidate.id, candidate.core_mechanism, candidate.concise_claim]
    values.extend(behavior_descriptor(candidate))
    values.extend(candidate.niche_memberships)
    values.extend(candidate.novelty_descriptors)
    out: set[str] = set()
    for value in values:
        token = str(value or "").strip().lower()
        if not token:
            continue
        out.add(token)
        out.update(part for part in token.replace("|", " ").replace(":", " ").split() if part)
    return out


def _matched_candidate_ids(candidates: list[CandidateGenome], *, source_ids: list[str], token: str) -> list[str]:
    source_set = set(source_ids)
    normalized = str(token or "").strip().lower()
    matched: list[str] = []
    for candidate in candidates:
        tokens = _candidate_descriptor_tokens(candidate)
        if candidate.id in source_set or (normalized and normalized in tokens):
            matched.append(candidate.id)
    return list(dict.fromkeys(matched))


def _score(value: Any) -> float:
    return bounded_score(value)


def _token(value: Any) -> str:
    return "_".join(str(value or "").strip().lower().replace("|", " ").split()) or "general"


def _pareto_vectors(candidates: list[CandidateGenome], *, max_axes: int) -> dict[str, tuple[float, ...]]:
    axis_count = max(1, min(3, int(max_axes or 3)))
    vectors: dict[str, tuple[float, ...]] = {}
    for candidate in candidates:
        scores = candidate.multihead_scores
        solution_quality = (_score(scores.get("objective_alignment")) + _score(scores.get("answer_likelihood"))) / 2.0
        evidence_progress = max(_score(scores.get("evidence_progress")), _score(scores.get("proof_progress")), _score(scores.get("tool_progress")))
        diversity = max(_score(scores.get("novelty")), _score(scores.get("rarity")), 1.0 if candidate.edge_knowledge_seeds else 0.0)
        vectors[candidate.id] = (evidence_progress, solution_quality, diversity)[:axis_count]
    return vectors


def _dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return all(left >= right for left, right in zip(a, b)) and any(left > right for left, right in zip(a, b))


def _dedupe_candidates(candidates: list[CandidateGenome]) -> list[CandidateGenome]:
    out: list[CandidateGenome] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.id in seen:
            continue
        seen.add(candidate.id)
        out.append(candidate)
    return out


__all__ = ["QualityDiversityArchive", "candidate_bin_key", "candidate_final_quality", "candidate_quality", "candidate_search_quality", "descriptor_cell_distribution", "descriptor_population_entropy", "entropy_diversity_survivors", "live_reproductive_candidates", "pareto_frontier_ids", "quality_diversity_survivors"]
