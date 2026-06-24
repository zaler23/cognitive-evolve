"""Model-facing advisory preprocessing for candidate pools."""
from __future__ import annotations

import json
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.fabric.advisory import assert_advisory_payload
from cognitive_evolve_runtime.fabric.config import PreprocessConfig
from cognitive_evolve_runtime.nexus.prompt_view import candidate_prompt_view
from .clustering import PoolCluster, representative_ids


def build_pool_preprocess_payload(
    *,
    candidates: list[CandidateGenome],
    clusters: list[PoolCluster],
    coverage_report: dict[str, Any],
    contract: Any,
    policy: Any,
    config: PreprocessConfig | None = None,
) -> dict[str, Any]:
    cfg = config or PreprocessConfig()
    by_id = {candidate.id: candidate for candidate in candidates}
    reps = [by_id[candidate_id] for candidate_id in representative_ids(clusters, limit=cfg.prompt_candidate_limit) if candidate_id in by_id]
    payload = {
        "request_type": "nexus_pool_preprocess",
        "contract": contract,
        "policy": policy,
        "coverage_report": coverage_report,
        "clusters": [cluster.to_dict() for cluster in clusters[: cfg.prompt_candidate_limit]],
        "representatives": [candidate_prompt_view(candidate, detail="summary", max_artifact_chars=cfg.prompt_candidate_artifact_chars) for candidate in reps],
        "instructions": {
            "advisory_only": True,
            "do_not_assign_verification_authority": True,
            "output": "schedule hints for exploration, dedupe awareness, and coverage gap filling",
        },
    }
    return _bounded_payload(payload, max_chars=cfg.max_report_chars)


def preprocess_candidate_pool(
    model: Any | None,
    *,
    candidates: list[CandidateGenome],
    clusters: list[PoolCluster],
    coverage_report: dict[str, Any],
    contract: Any,
    policy: Any,
    config: PreprocessConfig | None = None,
) -> dict[str, Any]:
    cfg = config or PreprocessConfig()
    payload = build_pool_preprocess_payload(candidates=candidates, clusters=clusters, coverage_report=coverage_report, contract=contract, policy=policy, config=cfg)
    if model is None or not hasattr(model, "preprocess_candidate_pool"):
        return {"advisory": True, "schedule_hints": [], "diagnostics": ["model_preprocess_unavailable"], "prompt_payload": payload}
    try:
        raw = model.preprocess_candidate_pool(**payload)
    except Exception as exc:
        return {
            "advisory": True,
            "schedule_hints": [],
            "diagnostics": ["model_preprocess_error"],
            "error": {"type": exc.__class__.__name__, "message": str(exc)},
            "prompt_payload": payload,
        }
    if not isinstance(raw, dict):
        return {"advisory": True, "schedule_hints": [], "diagnostics": ["model_preprocess_non_dict"], "prompt_payload": payload}
    try:
        return coerce_pool_preprocess_response(raw, prompt_payload=payload)
    except ValueError as exc:
        return {
            "advisory": True,
            "schedule_hints": [],
            "source_gap_requests": [],
            "diagnostics": ["model_preprocess_authority_payload_rejected"],
            "error": {"type": exc.__class__.__name__, "message": str(exc)},
            "prompt_payload": payload,
        }


def coerce_pool_preprocess_response(raw: dict[str, Any], *, prompt_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    assert_advisory_payload(raw)
    hints = raw.get("schedule_hints") if isinstance(raw.get("schedule_hints"), list) else []
    gap_requests = raw.get("source_gap_requests") if isinstance(raw.get("source_gap_requests"), list) else []
    diagnostics = raw.get("diagnostics") if isinstance(raw.get("diagnostics"), list) else []
    return {
        "advisory": True,
        "schedule_hints": [dict(item) for item in hints if isinstance(item, dict)],
        "source_gap_requests": [dict(item) for item in gap_requests if isinstance(item, dict)],
        "diagnostics": [str(item) for item in diagnostics],
        "prompt_payload": dict(prompt_payload or {}),
    }


def _bounded_payload(payload: dict[str, Any], *, max_chars: int) -> dict[str, Any]:
    limit = max(1, int(max_chars or 1))
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) <= limit:
        return payload
    bounded = dict(payload)
    bounded["truncated_for_prompt_bound"] = True
    while len(json.dumps(bounded, ensure_ascii=False, sort_keys=True, default=str)) > limit and len(list(bounded.get("representatives") or [])) > 1:
        reps = list(bounded.get("representatives") or [])
        bounded["representatives"] = reps[: max(1, len(reps) // 2)]
    while len(json.dumps(bounded, ensure_ascii=False, sort_keys=True, default=str)) > limit and len(list(bounded.get("clusters") or [])) > 1:
        clusters = list(bounded.get("clusters") or [])
        bounded["clusters"] = clusters[: max(1, len(clusters) // 2)]
    if len(json.dumps(bounded, ensure_ascii=False, sort_keys=True, default=str)) > limit:
        bounded["representatives"] = []
        bounded["clusters"] = []
    return bounded


__all__ = ["build_pool_preprocess_payload", "coerce_pool_preprocess_response", "preprocess_candidate_pool"]
