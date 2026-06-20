"""Advisory pool preprocessing helpers for Exploration Fabric."""
from .clustering import PoolCluster, annotate_candidate_clusters, cluster_candidates, representative_ids
from .coverage import pool_coverage_report
from .model_preprocess import build_pool_preprocess_payload, coerce_pool_preprocess_response, preprocess_candidate_pool

__all__ = [
    "PoolCluster",
    "annotate_candidate_clusters",
    "build_pool_preprocess_payload",
    "cluster_candidates",
    "coerce_pool_preprocess_response",
    "pool_coverage_report",
    "preprocess_candidate_pool",
    "representative_ids",
]
