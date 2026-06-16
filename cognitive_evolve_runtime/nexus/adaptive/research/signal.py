"""Research extension signals with deterministic, pure merge semantics.

This module intentionally stays a pure data/serialization layer.  It must not
import ``cognitive_evolve_runtime.concepts``; authority enforcement happens at
manager/test boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cognitive_evolve_runtime.evaluators.evidence import EvidenceRecord, SearchPressure
from cognitive_evolve_runtime.nexus._serde import coerce_dict
from cognitive_evolve_runtime.core.scalars import bounded_score


_EFFECT_CHANNELS = (
    "verification_obligations",
    "archive_directives",
    "budget_directives",
    "context_transforms",
    "candidate_transforms",
    "contract_delta_proposals",
)


@dataclass(frozen=True)
class ResearchSignal:
    source: str
    round_index: int = 0
    selection_advisory: dict[str, dict[str, float]] = field(default_factory=dict)
    search_pressures: list[SearchPressure] = field(default_factory=list)
    evidence_records: list[EvidenceRecord] = field(default_factory=list)
    final_gate_directives: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    verification_obligations: list[Any] = field(default_factory=list)
    archive_directives: list[Any] = field(default_factory=list)
    budget_directives: list[Any] = field(default_factory=list)
    context_transforms: list[Any] = field(default_factory=list)
    candidate_transforms: list[Any] = field(default_factory=list)
    contract_delta_proposals: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "round_index": int(self.round_index or 0),
            "selection_advisory": {str(k): dict(v) for k, v in self.selection_advisory.items()},
            "search_pressures": [p.to_dict() if hasattr(p, "to_dict") else _item_dict(p) for p in self.search_pressures],
            "evidence_records": [r.to_dict() if hasattr(r, "to_dict") else _item_dict(r) for r in self.evidence_records],
            "final_gate_directives": [dict(item) for item in self.final_gate_directives if isinstance(item, dict)],
            "metrics": dict(self.metrics),
            "warnings": list(self.warnings),
            "verification_obligations": [_item_dict(item) for item in self.verification_obligations],
            "archive_directives": [_item_dict(item) for item in self.archive_directives],
            "budget_directives": [_item_dict(item) for item in self.budget_directives],
            "context_transforms": [_item_dict(item) for item in self.context_transforms],
            "candidate_transforms": [_item_dict(item) for item in self.candidate_transforms],
            "contract_delta_proposals": [_item_dict(item) for item in self.contract_delta_proposals],
        }

    @classmethod
    def empty(cls, *, source: str = "research", round_index: int = 0, warnings: list[str] | None = None) -> "ResearchSignal":
        return cls(source=source, round_index=round_index, warnings=list(warnings or []))

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ResearchSignal":
        if not isinstance(data, dict):
            return cls.empty()
        allowed = {
            "source",
            "round_index",
            "selection_advisory",
            "search_pressures",
            "evidence_records",
            "final_gate_directives",
            "metrics",
            "warnings",
            *_EFFECT_CHANNELS,
        }
        unknown = sorted(str(key) for key in data.keys() if str(key) not in allowed)
        pressures = [SearchPressure.from_dict(item) for item in data.get("search_pressures", []) if isinstance(item, dict)]
        records = [item for item in (EvidenceRecord.from_dict(raw) for raw in data.get("evidence_records", []) if isinstance(raw, dict)) if item is not None]
        effects = {channel: [_item_dict(item) for item in data.get(channel, []) if _item_dict(item)] for channel in _EFFECT_CHANNELS}
        return cls(
            source=str(data.get("source") or "research"),
            round_index=_int(data.get("round_index"), 0),
            selection_advisory=_coerce_advisory(data.get("selection_advisory")),
            search_pressures=pressures,
            evidence_records=records,
            final_gate_directives=[dict(item) for item in data.get("final_gate_directives", []) if isinstance(item, dict)],
            metrics=coerce_dict(data.get("metrics")),
            warnings=[str(item) for item in data.get("warnings", []) if item] + (["unknown_research_signal_fields:" + ",".join(unknown)] if unknown else []),
            **effects,
        )


def merge_research_signals(signals: list[ResearchSignal]) -> ResearchSignal:
    """Pure, total, deterministic merge; never performs authority checks."""

    if not signals:
        return ResearchSignal.empty()
    ordered = sorted(signals, key=lambda item: (str(item.source), int(item.round_index or 0)))
    advisory: dict[str, dict[str, float]] = {}
    pressures: list[SearchPressure] = []
    records: list[EvidenceRecord] = []
    directives: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    warnings: list[str] = []
    effect_lists: dict[str, list[Any]] = {channel: [] for channel in _EFFECT_CHANNELS}
    for signal in ordered:
        for candidate_id, features in signal.selection_advisory.items():
            current = dict(advisory.get(candidate_id) or {})
            for key, value in (features or {}).items():
                current[key] = max(float(current.get(key, 0.0) or 0.0), float(value or 0.0))
            advisory[candidate_id] = {k: bounded_score(v) for k, v in sorted(current.items())}
        pressures.extend(signal.search_pressures)
        records.extend(signal.evidence_records)
        directives.extend(signal.final_gate_directives)
        for channel in _EFFECT_CHANNELS:
            effect_lists[channel].extend(list(getattr(signal, channel, []) or []))
        for key, value in sorted(signal.metrics.items()):
            metrics[f"{signal.source}.{key}"] = value
        warnings.extend(str(item) for item in signal.warnings if item)
    return ResearchSignal(
        source="merged_research_signal",
        round_index=max(int(item.round_index or 0) for item in ordered),
        selection_advisory=dict(sorted(advisory.items())),
        search_pressures=_dedupe_pressures(pressures),
        evidence_records=records,
        final_gate_directives=directives,
        metrics=metrics,
        warnings=list(dict.fromkeys(warnings)),
        verification_obligations=_dedupe_items(effect_lists["verification_obligations"], key_fields=("id",)),
        archive_directives=_dedupe_items(effect_lists["archive_directives"], key_fields=("kind", "descriptor")),
        budget_directives=_dedupe_items(effect_lists["budget_directives"], key_fields=("target", "reason")),
        context_transforms=_dedupe_items(effect_lists["context_transforms"], key_fields=("view_hash",)),
        candidate_transforms=_dedupe_items(effect_lists["candidate_transforms"], key_fields=("candidate_id", "kind")),
        contract_delta_proposals=_dedupe_items(effect_lists["contract_delta_proposals"], key_fields=("delta_id",)),
    )


def _dedupe_pressures(pressures: list[SearchPressure]) -> list[SearchPressure]:
    out: list[SearchPressure] = []
    seen: set[str] = set()
    for pressure in pressures:
        if pressure.id in seen:
            continue
        seen.add(pressure.id)
        out.append(pressure)
    return out


def _dedupe_items(items: list[Any], *, key_fields: tuple[str, ...]) -> list[Any]:
    out: list[Any] = []
    seen: set[tuple[Any, ...]] = set()
    for item in items:
        data = _item_dict(item)
        if not data:
            continue
        key = tuple(_hashable(data.get(field)) for field in key_fields)
        if not any(key):
            key = (repr(sorted(data.items())),)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _coerce_advisory(value: Any) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for candidate_id, raw in coerce_dict(value).items():
        features = coerce_dict(raw)
        out[str(candidate_id)] = {str(k): bounded_score(v) for k, v in features.items()}
    return out


def _item_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "to_dict"):
        try:
            value = item.to_dict()
            return dict(value) if isinstance(value, dict) else {}
        except Exception:
            return {}
    if isinstance(item, dict):
        return dict(item)
    try:
        return dict(item)
    except Exception:
        return {}


def _hashable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _hashable(v)) for k, v in value.items()))
    return value


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = ["ResearchSignal", "merge_research_signals"]
