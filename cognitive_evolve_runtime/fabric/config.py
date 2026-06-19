"""Typed configuration for Exploration Fabric runtime features."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict

FABRIC_ENV_PREFIX = "COGEV_FABRIC_"


@dataclass(frozen=True)
class SchedulerConfig:
    epoch_barrier: str = "full"
    max_active_tasks: int = 4
    pool_concurrency: dict[str, int] = field(default_factory=lambda: {"default": 3, "seed": 1, "verify": 3, "local": 4})


@dataclass(frozen=True)
class PoolConfig:
    cluster_similarity_threshold: float = 0.85
    representative_limit: int = 48


@dataclass(frozen=True)
class PreprocessConfig:
    prompt_candidate_limit: int = 48
    max_report_chars: int = 12000
    prompt_candidate_artifact_chars: int = 800
    sparse_cell_max_count: int = 1
    overrepresented_cell_multiplier: float = 2.0
    run_each_epoch: bool = False


@dataclass(frozen=True)
class ExpansionConfig:
    representative_artifact_chars: int = 4000
    max_sidecar_chars: int = 200000
    checkpoint_index_summary_chars: int = 1000


@dataclass(frozen=True)
class BootstrapConfig:
    mode: str = "hybrid"
    coverage_target: float = 0.9
    max_batches: int = 12
    stagnation_windows: int = 2


@dataclass(frozen=True)
class StreamingConfig:
    enabled: bool = False
    cluster_parallelism_limit: int = 2
    best_stream_path: str = "best-so-far.jsonl"


@dataclass(frozen=True)
class FabricRuntimeConfig:
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    diagnostics: list[str] = field(default_factory=list)

    @property
    def pool_concurrency(self) -> dict[str, int]:
        return dict(self.scheduler.pool_concurrency)

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
        env_overlay = _env_overlay()
        if env_overlay:
            data = _deep_merge(data, env_overlay)
            diagnostics.append("fabric_config_loaded_from_env_overlay")
        cfg = _config_from_mapping(data)
        return cls(
            scheduler=cfg.scheduler,
            pool=cfg.pool,
            preprocess=cfg.preprocess,
            expansion=cfg.expansion,
            bootstrap=cfg.bootstrap,
            streaming=cfg.streaming,
            diagnostics=[*cfg.diagnostics, *diagnostics],
        )


def _config_from_mapping(data: dict[str, Any]) -> FabricRuntimeConfig:
    scheduler = coerce_dict(data.get("scheduler"))
    pool = coerce_dict(data.get("pool"))
    preprocess = coerce_dict(data.get("preprocess"))
    expansion = coerce_dict(data.get("expansion"))
    bootstrap = coerce_dict(data.get("bootstrap"))
    streaming = coerce_dict(data.get("streaming"))
    diagnostics = [str(item) for item in data.get("diagnostics", [])] if isinstance(data.get("diagnostics"), list) else []
    return FabricRuntimeConfig(
        scheduler=SchedulerConfig(
            epoch_barrier=str(scheduler.get("epoch_barrier") or scheduler.get("barrier") or "full"),
            max_active_tasks=max(1, int(scheduler.get("max_active_tasks") or 4)),
            pool_concurrency={str(k): max(1, int(v)) for k, v in coerce_dict(scheduler.get("pool_concurrency")).items()} or SchedulerConfig().pool_concurrency,
        ),
        pool=PoolConfig(
            cluster_similarity_threshold=float(pool.get("cluster_similarity_threshold") or 0.85),
            representative_limit=max(1, int(pool.get("representative_limit") or 48)),
        ),
        preprocess=PreprocessConfig(
            prompt_candidate_limit=max(1, int(preprocess.get("prompt_candidate_limit") or 48)),
            max_report_chars=max(1, int(preprocess.get("max_report_chars") or 12000)),
            prompt_candidate_artifact_chars=max(1, int(preprocess.get("prompt_candidate_artifact_chars") or 800)),
            sparse_cell_max_count=max(1, int(preprocess.get("sparse_cell_max_count") or 1)),
            overrepresented_cell_multiplier=max(1.0, float(preprocess.get("overrepresented_cell_multiplier") or 2.0)),
            run_each_epoch=bool(preprocess.get("run_each_epoch") or False),
        ),
        expansion=ExpansionConfig(
            representative_artifact_chars=max(1, int(expansion.get("representative_artifact_chars") or 4000)),
            max_sidecar_chars=max(1, int(expansion.get("max_sidecar_chars") or 200000)),
            checkpoint_index_summary_chars=max(1, int(expansion.get("checkpoint_index_summary_chars") or 1000)),
        ),
        bootstrap=BootstrapConfig(
            mode=str(bootstrap.get("mode") or "hybrid"),
            coverage_target=float(bootstrap.get("coverage_target") or 0.9),
            max_batches=max(1, int(bootstrap.get("max_batches") or 12)),
            stagnation_windows=max(1, int(bootstrap.get("stagnation_windows") or 2)),
        ),
        streaming=StreamingConfig(
            enabled=bool(streaming.get("enabled") or False),
            cluster_parallelism_limit=max(1, int(streaming.get("cluster_parallelism_limit") or 2)),
            best_stream_path=str(streaming.get("best_stream_path") or "best-so-far.jsonl"),
        ),
        diagnostics=diagnostics,
    )


def _env_overlay() -> dict[str, Any]:
    overlay: dict[str, Any] = {}
    if "COGEV_FABRIC_MAX_ACTIVE_TASKS" in os.environ:
        overlay.setdefault("scheduler", {})["max_active_tasks"] = os.environ["COGEV_FABRIC_MAX_ACTIVE_TASKS"]
    if "COGEV_FABRIC_EPOCH_BARRIER" in os.environ:
        overlay.setdefault("scheduler", {})["epoch_barrier"] = os.environ["COGEV_FABRIC_EPOCH_BARRIER"]
    if "COGEV_FABRIC_POOL_CONCURRENCY" in os.environ:
        raw = os.environ["COGEV_FABRIC_POOL_CONCURRENCY"]
        parsed: dict[str, int] = {}
        for item in raw.split(","):
            if ":" not in item:
                continue
            key, value = item.split(":", 1)
            try:
                parsed[key.strip()] = max(1, int(value.strip()))
            except ValueError:
                continue
        if parsed:
            overlay.setdefault("scheduler", {})["pool_concurrency"] = parsed
    if "COGEV_FABRIC_BOOTSTRAP_MODE" in os.environ:
        overlay.setdefault("bootstrap", {})["mode"] = os.environ["COGEV_FABRIC_BOOTSTRAP_MODE"]
    return overlay


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


__all__ = [
    "FABRIC_ENV_PREFIX",
    "BootstrapConfig",
    "ExpansionConfig",
    "FabricRuntimeConfig",
    "PoolConfig",
    "PreprocessConfig",
    "SchedulerConfig",
    "StreamingConfig",
]
