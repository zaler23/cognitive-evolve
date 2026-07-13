"""Typed configuration for connected Exploration Fabric runtime features."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.core.serialization import coerce_dict


@dataclass(frozen=True)
class PoolConfig:
    cluster_similarity_threshold: float = 0.85


@dataclass(frozen=True)
class PreprocessConfig:
    prompt_candidate_limit: int = 48
    max_report_chars: int = 500000
    prompt_candidate_artifact_chars: int = 64000
    sparse_cell_max_count: int = 1
    overrepresented_cell_multiplier: float = 2.0


@dataclass(frozen=True)
class FabricRuntimeConfig:
    pool: PoolConfig = field(default_factory=PoolConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    diagnostics: list[str] = field(default_factory=list)

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(include_hash=False), ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if include_hash:
            payload["config_hash"] = self.config_hash
        return payload

    @classmethod
    def from_runtime_context(cls, *, policy: Any | None = None, contract: Any | None = None) -> "FabricRuntimeConfig":
        data: dict[str, Any] = {}
        diagnostics: list[str] = []
        for source_name, source in (("policy", policy), ("contract", contract)):
            metadata = getattr(source, "metadata", {}) if source is not None else {}
            if isinstance(metadata, dict) and isinstance(metadata.get("fabric_runtime"), dict):
                data = _deep_merge(data, metadata["fabric_runtime"])
                diagnostics.append(f"fabric_config_loaded_from_{source_name}_metadata")
        cfg = _config_from_mapping(data)
        return cls(
            pool=cfg.pool,
            preprocess=cfg.preprocess,
            diagnostics=[*cfg.diagnostics, *diagnostics],
        )


def _config_from_mapping(data: dict[str, Any]) -> FabricRuntimeConfig:
    pool = coerce_dict(data.get("pool"))
    preprocess = coerce_dict(data.get("preprocess"))
    diagnostics = [str(item) for item in data.get("diagnostics", [])] if isinstance(data.get("diagnostics"), list) else []
    return FabricRuntimeConfig(
        pool=PoolConfig(
            cluster_similarity_threshold=float(pool.get("cluster_similarity_threshold") or 0.85),
        ),
        preprocess=PreprocessConfig(
            prompt_candidate_limit=max(1, int(preprocess.get("prompt_candidate_limit") or 48)),
            max_report_chars=max(1, int(preprocess.get("max_report_chars") or 500000)),
            prompt_candidate_artifact_chars=max(1, int(preprocess.get("prompt_candidate_artifact_chars") or 64000)),
            sparse_cell_max_count=max(1, int(preprocess.get("sparse_cell_max_count") or 1)),
            overrepresented_cell_multiplier=max(1.0, float(preprocess.get("overrepresented_cell_multiplier") or 2.0)),
        ),
        diagnostics=diagnostics,
    )


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


__all__ = [
    "FabricRuntimeConfig",
    "PoolConfig",
    "PreprocessConfig",
]
