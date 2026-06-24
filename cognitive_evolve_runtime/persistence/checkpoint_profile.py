"""Checkpoint profile helpers for long Nexus runs."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.core.serialization import coerce_dict, stable_hash

CHECKPOINT_PROFILE_ENV = "COGEV_CHECKPOINT_PROFILE"
ARCHIVE_STORAGE_SCHEMA = "cogev.archive_checkpoint_refs.v1"

_CANDIDATE_FIELD_NAMES = {item.name for item in fields(CandidateGenome)}


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


def apply_checkpoint_profile_to_archives(
    archives: dict[str, Any],
    population: dict[str, Any],
    profile: CheckpointProfile,
) -> dict[str, Any]:
    """Replace repeated archived candidate bodies with population refs.

    This is a storage-only optimization. Runtime archives still hold full
    candidates; thin checkpoints keep one canonical candidate body in
    ``population`` and store archive memberships as refs that are hydrated on
    restore. The search pool, dormant/failure lanes, and final-selection
    behavior are therefore unchanged.
    """

    if profile.name == "full":
        return archives
    data = _clone(archives)
    population_index = _population_index(population)
    if not population_index:
        return data
    stats = {"schema": ARCHIVE_STORAGE_SCHEMA, "mode": "population_ref", "candidate_refs": 0, "missing_refs": 0}

    def compact_direct(value: Any) -> Any:
        compacted, changed = _compact_candidate_record(value, population_index)
        if changed:
            stats["candidate_refs"] += 1
        elif _looks_like_candidate_record(value):
            stats["missing_refs"] += 1
        return compacted

    for key in ("answer_archive", "mechanism_archive", "novelty_archive", "project_patch_archive"):
        lane = data.get(key)
        if isinstance(lane, dict):
            data[key] = {str(item_key): compact_direct(item_value) for item_key, item_value in lane.items()}

    for key in ("latent_pareto_archive", "rarity_archive", "auxiliary_archive", "dormant_archive"):
        lane = data.get(key)
        if isinstance(lane, dict) and isinstance(lane.get("candidates"), dict):
            lane["candidates"] = {str(item_key): compact_direct(item_value) for item_key, item_value in lane["candidates"].items()}

    qd = data.get("quality_diversity")
    if isinstance(qd, dict):
        for key in ("elites_by_niche", "cell_elites"):
            records = qd.get(key)
            if not isinstance(records, dict):
                continue
            compacted_records: dict[str, Any] = {}
            for item_key, item_value in records.items():
                if isinstance(item_value, dict) and isinstance(item_value.get("candidate"), dict):
                    compacted_candidate, changed = _compact_candidate_record(item_value.get("candidate"), population_index)
                    item_value = dict(item_value)
                    item_value["candidate"] = compacted_candidate
                    if changed:
                        stats["candidate_refs"] += 1
                compacted_records[str(item_key)] = item_value
            qd[key] = compacted_records

    data["archive_storage_profile"] = stats
    return data


def hydrate_checkpoint_archives(archives: dict[str, Any], population: dict[str, Any]) -> dict[str, Any]:
    """Hydrate thin archive candidate refs from checkpoint population."""

    data = _clone(archives)
    population_index = _population_index(population)
    if not population_index:
        return data
    stats = {"schema": ARCHIVE_STORAGE_SCHEMA, "hydrated_refs": 0, "missing_refs": 0}

    def hydrate_direct(value: Any) -> Any:
        hydrated, changed, missing = _hydrate_candidate_record(value, population_index)
        if changed:
            stats["hydrated_refs"] += 1
        elif missing:
            stats["missing_refs"] += 1
        return hydrated

    for key in ("answer_archive", "mechanism_archive", "novelty_archive", "project_patch_archive"):
        lane = data.get(key)
        if isinstance(lane, dict):
            data[key] = {str(item_key): hydrate_direct(item_value) for item_key, item_value in lane.items()}

    for key in ("latent_pareto_archive", "rarity_archive", "auxiliary_archive", "dormant_archive"):
        lane = data.get(key)
        if isinstance(lane, dict) and isinstance(lane.get("candidates"), dict):
            lane["candidates"] = {str(item_key): hydrate_direct(item_value) for item_key, item_value in lane["candidates"].items()}

    qd = data.get("quality_diversity")
    if isinstance(qd, dict):
        for key in ("elites_by_niche", "cell_elites"):
            records = qd.get(key)
            if not isinstance(records, dict):
                continue
            hydrated_records: dict[str, Any] = {}
            for item_key, item_value in records.items():
                if isinstance(item_value, dict):
                    candidate, changed, missing = _hydrate_candidate_record(item_value.get("candidate"), population_index)
                    if changed or missing:
                        item_value = dict(item_value)
                        item_value["candidate"] = candidate
                        stats["hydrated_refs"] += int(changed)
                        stats["missing_refs"] += int(missing)
                hydrated_records[str(item_key)] = item_value
            qd[key] = hydrated_records

    data["archive_storage_profile"] = {**coerce_dict(data.get("archive_storage_profile")), **stats}
    return data


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


def _population_index(population: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = population.get("candidates") if isinstance(population, dict) else None
    if not isinstance(candidates, list):
        return {}
    return {
        str(item.get("id")): dict(item)
        for item in candidates
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }


def _looks_like_candidate_record(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if "candidate_ref" in value:
        return True
    if not str(value.get("id") or "").strip():
        return False
    return any(key in value for key in ("artifact", "artifact_type", "core_mechanism", "concise_claim", "current_fate", "metadata"))


def _compact_candidate_record(value: Any, population_index: dict[str, dict[str, Any]]) -> tuple[Any, bool]:
    if not _looks_like_candidate_record(value) or not isinstance(value, dict):
        return value, False
    candidate_id = str(value.get("id") or "").strip()
    if not candidate_id or candidate_id not in population_index:
        return value, False
    extras = {key: _clone_value(item) for key, item in value.items() if key not in _CANDIDATE_FIELD_NAMES}
    compacted = {
        **extras,
        "candidate_ref": candidate_id,
        "candidate_hash": stable_hash(population_index[candidate_id]),
        "candidate_preview": _candidate_preview(population_index[candidate_id]),
    }
    return compacted, True


def _hydrate_candidate_record(value: Any, population_index: dict[str, dict[str, Any]]) -> tuple[Any, bool, bool]:
    if not isinstance(value, dict) or "candidate_ref" not in value:
        return value, False, False
    candidate_id = str(value.get("candidate_ref") or "").strip()
    candidate = population_index.get(candidate_id)
    if candidate is None:
        preview = coerce_dict(value.get("candidate_preview"))
        if preview:
            return {"id": candidate_id, **preview}, False, True
        return value, False, True
    hydrated = _clone(candidate)
    for key, item in value.items():
        if key in {"candidate_ref", "candidate_hash", "candidate_preview"}:
            continue
        hydrated[key] = _clone_value(item)
    return hydrated, True, False


def _candidate_preview(candidate: dict[str, Any]) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    for key in (
        "id",
        "generation",
        "current_fate",
        "artifact_type",
        "core_mechanism",
        "concise_claim",
        "parent_ids",
        "niche_memberships",
        "novelty_descriptors",
        "edge_knowledge_seeds",
    ):
        if key in candidate:
            preview[key] = _clone_value(candidate.get(key))
    for key in ("core_mechanism", "concise_claim"):
        if isinstance(preview.get(key), str) and len(preview[key]) > 240:
            preview[key] = preview[key][:237] + "..."
    return preview


def _clone_value(value: Any) -> Any:
    import json

    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return value


__all__ = [
    "ARCHIVE_STORAGE_SCHEMA",
    "CHECKPOINT_PROFILE_ENV",
    "CheckpointProfile",
    "apply_checkpoint_profile_to_archives",
    "apply_checkpoint_profile_to_history",
    "apply_checkpoint_profile_to_population",
    "checkpoint_profile_from_env",
    "hydrate_checkpoint_archives",
]
