from __future__ import annotations

import json

import pytest

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.fabric.config import PoolConfig, PreprocessConfig
from cognitive_evolve_runtime.nexus.model_adapter import StructuredModelAdapter
from cognitive_evolve_runtime.nexus.pool_preprocessing import (
    build_pool_preprocess_payload,
    cluster_candidates,
    coerce_pool_preprocess_response,
    pool_coverage_report,
    representative_ids,
)


def _candidate(candidate_id: str, *, claim: str, mechanism: str = "mechanism", descriptor: str = "direct", artifact: str = "artifact", quality: float = 0.5) -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        artifact=artifact,
        concise_claim=claim,
        core_mechanism=mechanism,
        novelty_descriptors=[descriptor],
        niche_memberships=[descriptor],
        multihead_scores={"frontier_score": quality, "objective_alignment": quality, "answer_likelihood": quality},
    )


def test_pool_clustering_exact_duplicate_support_and_no_deletion() -> None:
    left = _candidate("A", claim="same", artifact="same artifact")
    right = _candidate("B", claim="same", artifact="same artifact")
    distinct = _candidate("C", claim="different", descriptor="edge")
    candidates = [left, right, distinct]
    clusters = cluster_candidates(candidates, config=PoolConfig(cluster_similarity_threshold=0.99))
    exact = [cluster for cluster in clusters if cluster.duplicate_kind == "exact"]
    assert len(candidates) == 3
    assert exact and exact[0].support_count == 2
    assert set(exact[0].member_ids) == {"A", "B"}


def test_pool_clustering_near_duplicate_and_unique_representatives() -> None:
    left = _candidate("A", claim="alpha route", mechanism="shared", descriptor="direct", quality=0.4)
    right = _candidate("B", claim="alpha route variant", mechanism="shared", descriptor="direct", quality=0.9)
    far = _candidate("C", claim="orthogonal", mechanism="other", descriptor="edge", quality=0.7)
    clusters = cluster_candidates([left, right, far], config=PoolConfig(cluster_similarity_threshold=0.1))
    near = [cluster for cluster in clusters if cluster.support_count >= 2]
    reps = representative_ids(clusters, limit=10)
    assert near
    assert near[0].duplicate_kind in {"near", "exact"}
    assert len(reps) == len(set(reps))
    assert "B" in reps


def test_pool_coverage_detects_sparse_overrepresented_and_missing_cells() -> None:
    candidates = [
        _candidate("A", claim="a", descriptor="direct"),
        _candidate("B", claim="b", descriptor="direct"),
        _candidate("C", claim="c", descriptor="direct"),
        _candidate("D", claim="d", descriptor="edge"),
    ]
    report = pool_coverage_report(candidates, expected_cells=["missing|proposal|common"], config=PreprocessConfig(sparse_cell_max_count=1, overrepresented_cell_multiplier=1.5))
    assert report["advisory"] is True
    assert report["missing_cells"] == ["missing|proposal|common"]
    assert any("edge" in cell for cell in report["sparse_cells"])
    assert any("direct" in cell for cell in report["overrepresented_cells"])


def test_pool_preprocess_prompt_is_bounded() -> None:
    candidates = [_candidate(f"C{i}", claim=f"claim {i}", artifact="x" * 2000, descriptor=f"d{i}") for i in range(8)]
    clusters = cluster_candidates(candidates, config=PoolConfig(cluster_similarity_threshold=0.9))
    coverage = pool_coverage_report(candidates)
    cfg = PreprocessConfig(prompt_candidate_limit=8, prompt_candidate_artifact_chars=500, max_report_chars=1600)
    payload = build_pool_preprocess_payload(candidates=candidates, clusters=clusters, coverage_report=coverage, contract={"goal": "g"}, policy={}, config=cfg)
    assert len(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)) <= cfg.max_report_chars


def test_pool_preprocess_advisory_guard_rejects_authority_fields() -> None:
    with pytest.raises(ValueError, match="verification-authority"):
        coerce_pool_preprocess_response({"schedule_hints": [{"objective_solved": True}]})


def test_structured_adapter_supports_pool_preprocess_request_type() -> None:
    calls: list[str] = []

    def caller(request_type, payload, schema):
        calls.append(request_type)
        return {"schedule_hints": [{"kind": "rebalance", "target": "sparse"}], "diagnostics": []}

    adapter = StructuredModelAdapter(caller=caller)
    result = adapter.preprocess_candidate_pool(contract={}, policy={}, coverage_report={}, clusters=[], representatives=[])
    assert calls == ["nexus_pool_preprocess"]
    assert result["schedule_hints"][0]["kind"] == "rebalance"
