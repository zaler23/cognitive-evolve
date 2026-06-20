"""Pool preprocessing facet for StructuredModelAdapter."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.nexus.model_adapter_schemas import _pool_preprocess_schema


class PreprocessFacet:
    def preprocess_candidate_pool(
        self,
        *,
        request_type: str = "nexus_pool_preprocess",
        contract: Any,
        policy: Any,
        coverage_report: dict[str, Any],
        clusters: list[dict[str, Any]],
        representatives: list[dict[str, Any]],
        instructions: dict[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        schema = _pool_preprocess_schema()
        payload = {
            "request_type": request_type,
            "contract": contract,
            "policy": policy,
            "coverage_report": coverage_report,
            "clusters": clusters,
            "representatives": representatives,
            "instructions": dict(instructions or {}),
        }
        if extra:
            payload["extra"] = dict(extra)
        return self._call("nexus_pool_preprocess", payload, schema)


__all__ = ["PreprocessFacet"]
