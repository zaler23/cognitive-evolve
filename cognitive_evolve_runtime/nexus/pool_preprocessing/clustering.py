"""Advisory candidate-pool clustering for Exploration Fabric preprocessing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.archives.quality_diversity import candidate_final_quality, candidate_search_quality
from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.fabric.config import PoolConfig
from cognitive_evolve_runtime.nexus.search_kernel.fingerprints import candidate_semantic_signature
from cognitive_evolve_runtime.nexus.search_kernel.math_model import similarity


@dataclass(frozen=True)
class PoolCluster:
    cluster_id: str
    representative_id: str
    member_ids: list[str]
    support_count: int
    semantic_signatures: list[str] = field(default_factory=list)
    duplicate_kind: str = "unique"
    average_similarity_to_representative: float = 0.0
    advisory: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["advisory"] = True
        return payload


def cluster_candidates(candidates: list[CandidateGenome], *, config: PoolConfig | None = None) -> list[PoolCluster]:
    """Cluster candidates without deleting or demoting any genome.

    Exact duplicates reuse the existing semantic signature. Near duplicates reuse
    the existing search-kernel similarity helper and the typed fabric threshold.
    The result is advisory metadata for scheduling, not verification authority.
    """

    if not candidates:
        return []
    cfg = config or PoolConfig()
    by_id = {candidate.id: candidate for candidate in candidates}
    groups: list[list[str]] = []
    group_signatures: list[set[str]] = []
    signature_to_group: dict[str, int] = {}
    for candidate in sorted(candidates, key=lambda item: item.id):
        signature = candidate_semantic_signature(candidate)
        exact_group = signature_to_group.get(signature)
        if exact_group is not None:
            groups[exact_group].append(candidate.id)
            group_signatures[exact_group].add(signature)
            continue
        near_index = _nearest_group_index(candidate, groups=groups, by_id=by_id, threshold=cfg.cluster_similarity_threshold)
        if near_index is not None:
            groups[near_index].append(candidate.id)
            group_signatures[near_index].add(signature)
            signature_to_group[signature] = near_index
            continue
        groups.append([candidate.id])
        group_signatures.append({signature})
        signature_to_group[signature] = len(groups) - 1
    clusters: list[PoolCluster] = []
    for index, member_ids in enumerate(groups):
        members = [by_id[candidate_id] for candidate_id in member_ids if candidate_id in by_id]
        representative = _representative(members)
        duplicate_kind = _duplicate_kind(members, signatures=group_signatures[index])
        clusters.append(
            PoolCluster(
                cluster_id=f"pool-cluster-{index + 1:04d}",
                representative_id=representative.id,
                member_ids=[candidate.id for candidate in members],
                support_count=len(members),
                semantic_signatures=sorted(group_signatures[index]),
                duplicate_kind=duplicate_kind,
                average_similarity_to_representative=_average_similarity(representative, members),
            )
        )
    return clusters


def annotate_candidate_clusters(candidates: list[CandidateGenome], clusters: list[PoolCluster]) -> None:
    cluster_by_member: dict[str, PoolCluster] = {}
    for cluster in clusters:
        for candidate_id in cluster.member_ids:
            cluster_by_member[candidate_id] = cluster
    for candidate in candidates:
        cluster = cluster_by_member.get(candidate.id)
        if cluster is None:
            continue
        candidate.metadata["fabric_pool_cluster"] = {
            "cluster_id": cluster.cluster_id,
            "representative_id": cluster.representative_id,
            "support_count": cluster.support_count,
            "duplicate_kind": cluster.duplicate_kind,
            "advisory": True,
        }


def representative_ids(clusters: list[PoolCluster], *, limit: int) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for cluster in sorted(clusters, key=lambda item: (item.support_count, item.representative_id), reverse=True):
        if cluster.representative_id in seen:
            continue
        ids.append(cluster.representative_id)
        seen.add(cluster.representative_id)
        if len(ids) >= max(1, int(limit or 1)):
            break
    return ids


def _nearest_group_index(candidate: CandidateGenome, *, groups: list[list[str]], by_id: dict[str, CandidateGenome], threshold: float) -> int | None:
    best_index: int | None = None
    best_similarity = 0.0
    for index, member_ids in enumerate(groups):
        representative = _representative([by_id[candidate_id] for candidate_id in member_ids if candidate_id in by_id])
        value = similarity(candidate, representative)
        if value >= threshold and value > best_similarity:
            best_index = index
            best_similarity = value
    return best_index


def _representative(candidates: list[CandidateGenome]) -> CandidateGenome:
    if not candidates:
        raise ValueError("candidate cluster cannot be empty")
    return max(candidates, key=lambda item: (candidate_search_quality(item), candidate_final_quality(item), item.id))


def _duplicate_kind(candidates: list[CandidateGenome], *, signatures: set[str]) -> str:
    if len(candidates) <= 1:
        return "unique"
    if len(signatures) == 1:
        return "exact"
    return "near"


def _average_similarity(representative: CandidateGenome, members: list[CandidateGenome]) -> float:
    if not members:
        return 0.0
    return sum(similarity(representative, member) for member in members) / max(1, len(members))


__all__ = ["PoolCluster", "annotate_candidate_clusters", "cluster_candidates", "representative_ids"]
