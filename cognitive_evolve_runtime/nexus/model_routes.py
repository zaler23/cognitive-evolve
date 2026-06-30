"""Explicit model routing for Nexus runtime roles."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from cognitive_evolve_runtime.llm.fanout import run_ordered_fanout
from cognitive_evolve_runtime.nexus._shared import call_with_optional_context
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike, NexusSeedModelProtocol


class NexusModelRole(StrEnum):
    DEFAULT = "default"
    SEED = "seed"


@dataclass(frozen=True)
class NexusModelRoutes:
    default_model: NexusModelLike | None = None
    seed_model: NexusSeedModelProtocol | list[NexusSeedModelProtocol] | None = None

    def model_for(self, role: NexusModelRole | str) -> NexusModelLike | None:
        normalized = NexusModelRole(str(role)) if not isinstance(role, NexusModelRole) else role
        if normalized == NexusModelRole.SEED and self.seed_model is not None:
            if isinstance(self.seed_model, list):
                return SeedModelEnsembleAdapter(self.seed_model)
            return self.seed_model
        return self.default_model

    def public_summary(self) -> dict[str, Any]:
        return {
            "default": _model_summary(self.default_model),
            "seed": _model_summary(self.seed_model if self.seed_model is not None else self.default_model),
            "seed_uses_default": self.seed_model is None,
        }


def coerce_model_routes(*, model: NexusModelLike | None = None, model_routes: NexusModelRoutes | dict[str, Any] | None = None) -> NexusModelRoutes:
    if model_routes is None:
        return NexusModelRoutes(default_model=model)
    if isinstance(model_routes, NexusModelRoutes):
        routes = model_routes
    elif isinstance(model_routes, dict):
        routes = NexusModelRoutes(default_model=model_routes.get("default_model"), seed_model=model_routes.get("seed_model"))
    else:
        raise TypeError("model_routes must be NexusModelRoutes, dict, or None")
    if model is not None and routes.default_model is not None and model is not routes.default_model:
        raise ValueError("model and model_routes.default_model refer to different adapters; pass only model_routes to avoid silent mismatch")
    if model is not None and routes.default_model is None:
        return NexusModelRoutes(default_model=model, seed_model=routes.seed_model)
    return routes


def _model_summary(model: Any) -> dict[str, Any]:
    if isinstance(model, list):
        return {"configured": bool(model), "ensemble_size": len(model), "models": [_model_summary(item) for item in model]}
    if model is None:
        return {"configured": False}
    metadata = getattr(model, "metadata", {}) if isinstance(getattr(model, "metadata", None), dict) else {}
    spec = metadata.get("model_spec") if isinstance(metadata.get("model_spec"), dict) else None
    summary = {
        "configured": True,
        "adapter_type": type(model).__name__,
    }
    if spec:
        public = dict(spec)
        public.pop("api_key", None)
        summary["model_spec"] = public
    elif metadata.get("transport"):
        summary["transport"] = str(metadata.get("transport"))
    return summary


@dataclass(frozen=True)
class SeedModelEnsembleAdapter:
    models: list[NexusSeedModelProtocol]

    @property
    def metadata(self) -> dict[str, Any]:
        return {"adapter_type": "SeedModelEnsembleAdapter", "ensemble_size": len(self.models)}

    def seed_population(self, *, contract: Any, world: Any, policy: Any, provided_context: dict[str, Any] | None = None) -> list[Any]:
        def _call(model: NexusSeedModelProtocol) -> list[Any]:
            return list(call_with_optional_context(model.seed_population, contract=contract, world=world, policy=policy, provided_context=provided_context) or [])

        batches = run_ordered_fanout(self.models, _call, max_workers=len(self.models))
        out: list[Any] = []
        for index, batch in enumerate(batches):
            for item in batch:
                if isinstance(item, dict):
                    metadata = dict(item.get("metadata") or {}) if isinstance(item.get("metadata"), dict) else {}
                    metadata.setdefault("origin_model_index", index)
                    item = {**item, "metadata": metadata}
                elif hasattr(item, "metadata"):
                    metadata = item.metadata if isinstance(getattr(item, "metadata", None), dict) else {}
                    metadata.setdefault("origin_model_index", index)
                    item.metadata = metadata
                out.append(item)
        return out


__all__ = ["NexusModelRole", "NexusModelRoutes", "SeedModelEnsembleAdapter", "coerce_model_routes"]
