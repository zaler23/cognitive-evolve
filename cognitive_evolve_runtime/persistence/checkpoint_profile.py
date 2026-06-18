"""Checkpoint profile helpers for long Nexus runs."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

CHECKPOINT_PROFILE_ENV = "COGEV_CHECKPOINT_PROFILE"


@dataclass(frozen=True)
class CheckpointProfile:
    name: str = "thin"
    max_verification_trace: int = 3
    max_budget_history: int = 200

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def checkpoint_profile_from_env() -> CheckpointProfile:
    name = str(os.environ.get(CHECKPOINT_PROFILE_ENV) or "thin").strip().lower() or "thin"
    if name in {"full", "legacy"}:
        return CheckpointProfile(name="full", max_verification_trace=100, max_budget_history=1000)
    return CheckpointProfile(name="thin")


def apply_checkpoint_profile_to_population(population: dict[str, Any], profile: CheckpointProfile) -> dict[str, Any]:
    if profile.name == "full":
        return population
    data = _clone(population)
    candidates = data.get("candidates") if isinstance(data, dict) else None
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            trace = candidate.get("verification_trace")
            if isinstance(trace, list):
                candidate["verification_trace"] = _summarize_trace(trace[-max(0, profile.max_verification_trace):])
                candidate.setdefault("checkpoint_thinning", {})["verification_trace_original_count"] = len(trace)
            metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
            if isinstance(metadata.get("offspring_harvest"), dict):
                metadata["offspring_harvest_summary"] = {k: metadata["offspring_harvest"].get(k) for k in ("accepted_count", "rejected_count", "stage") if k in metadata["offspring_harvest"]}
                metadata.pop("offspring_harvest", None)
            candidate["metadata"] = metadata
    return data


def apply_checkpoint_profile_to_history(history: list[dict[str, Any]], profile: CheckpointProfile) -> list[dict[str, Any]]:
    if profile.name == "full":
        return list(history or [])
    return [dict(item) for item in (history or [])[-max(0, profile.max_budget_history):] if isinstance(item, dict)]


def _summarize_trace(trace: list[Any]) -> list[Any]:
    out: list[Any] = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        out.append(
            {
                "passed": item.get("passed"),
                "score": item.get("score"),
                "strength": item.get("strength"),
                "measured_strength": item.get("measured_strength") or metadata.get("measured_strength"),
                "evidence_ref": item.get("evidence_ref"),
                "replayable": item.get("replayable"),
                "metadata": {k: metadata.get(k) for k in ("cache_key", "verifier_fingerprint", "artifact_sha256", "grounding_regime_id") if k in metadata},
            }
        )
    return out


def _clone(value: dict[str, Any]) -> dict[str, Any]:
    import json

    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return dict(value)


__all__ = [
    "CHECKPOINT_PROFILE_ENV",
    "CheckpointProfile",
    "apply_checkpoint_profile_to_history",
    "apply_checkpoint_profile_to_population",
    "checkpoint_profile_from_env",
]
