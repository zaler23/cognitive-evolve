"""Lossless, caller-selected inheritance between Nexus runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.core.serialization import coerce_dict, stable_hash
from cognitive_evolve_runtime.evaluators.evidence_authority import stable_artifact_hash


HANDOFF_SCHEMA_VERSION = "inheritable-handoff/v1"


def build_inheritable_handoff(
    *,
    population: Any,
    failure_archive: Any,
    source_run_id: str,
    project_signature: str,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for candidate in getattr(population, "candidates", []) or []:
        candidate_id = str(getattr(candidate, "id", "") or "")
        if not candidate_id:
            continue
        entry = {
            "candidate_id": candidate_id,
            "sources": ["final_population"],
            "artifact_digest": stable_artifact_hash(getattr(candidate, "artifact", None)),
            "gene_summary": str(candidate.extract_inheritable_gene_summary()),
            "failure_signature": "",
            "future_reactivation_condition": "",
        }
        entries.append(entry)
        by_id[candidate_id] = entry

    raw_records = coerce_dict(getattr(failure_archive, "records", failure_archive))
    failure_records = sorted(
        (coerce_dict(record) for record in raw_records.values() if isinstance(record, dict)),
        key=lambda record: str(record.get("candidate_id") or ""),
    )
    for record in failure_records:
        candidate_id = str(record.get("candidate_id") or "")
        if not candidate_id:
            continue
        entry = by_id.get(candidate_id)
        if entry is None:
            entry = {
                "candidate_id": candidate_id,
                "sources": ["failure_archive"],
                "artifact_digest": None,
                "gene_summary": str(record.get("inherited_gene_summary") or ""),
                "failure_signature": str(record.get("failure_signature") or ""),
                "future_reactivation_condition": str(record.get("future_reactivation_condition") or ""),
            }
            entries.append(entry)
            by_id[candidate_id] = entry
        else:
            entry["sources"] = ["final_population", "failure_archive"]
            entry["failure_signature"] = str(record.get("failure_signature") or "")
            entry["future_reactivation_condition"] = str(record.get("future_reactivation_condition") or "")

    for entry in entries:
        entry["entry_digest"] = _entry_digest(entry)
    return {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "source_run_id": str(source_run_id or ""),
        "project_signature": str(project_signature or ""),
        "entries": entries,
    }


def load_inherited_gene_entries(
    inherited_handoff_path: str | Path | None,
    inherited_candidate_ids: list[str] | None,
) -> list[dict[str, str]]:
    if (inherited_handoff_path is None) != (inherited_candidate_ids is None):
        raise ValueError("inherited_handoff_path and inherited_candidate_ids must be provided together")
    if inherited_handoff_path is None:
        return []
    payload = json.loads(Path(inherited_handoff_path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != HANDOFF_SCHEMA_VERSION:
        raise ValueError("inherited handoff must use schema inheritable-handoff/v1")
    by_id: dict[str, dict[str, Any]] = {}
    for raw_entry in payload.get("entries", []):
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        candidate_id = str(entry.get("candidate_id") or "")
        if not candidate_id:
            continue
        if candidate_id in by_id:
            raise ValueError(f"duplicate inherited handoff candidate_id: {candidate_id}")
        by_id[candidate_id] = entry

    selected: list[dict[str, str]] = []
    for requested_id in inherited_candidate_ids or []:
        candidate_id = str(requested_id)
        if candidate_id not in by_id:
            raise KeyError(f"inherited handoff candidate not found: {candidate_id}")
        entry = by_id[candidate_id]
        if str(entry.get("entry_digest") or "") != _entry_digest(entry):
            raise ValueError(f"entry_digest mismatch: {candidate_id}")
        selected.append(
            {
                "candidate_id": candidate_id,
                "gene_summary": str(entry.get("gene_summary") or ""),
            }
        )
    return selected


def _entry_digest(entry: dict[str, Any]) -> str:
    payload = {key: value for key, value in entry.items() if key != "entry_digest"}
    return "entry:" + stable_hash(payload)


__all__ = [
    "HANDOFF_SCHEMA_VERSION",
    "build_inheritable_handoff",
    "load_inherited_gene_entries",
]
