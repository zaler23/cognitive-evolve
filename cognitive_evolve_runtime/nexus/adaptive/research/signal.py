"""Research extension signals with deterministic merge semantics."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.evaluators.evidence import EvidenceRecord, SearchPressure
from cognitive_evolve_runtime.nexus._serde import coerce_dict
from cognitive_evolve_runtime.core.scalars import bounded_score


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

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["search_pressures"] = [p.to_dict() if hasattr(p, "to_dict") else dict(p) for p in self.search_pressures]
        data["evidence_records"] = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in self.evidence_records]
        return data

    @classmethod
    def empty(cls, *, source: str = "research", round_index: int = 0) -> "ResearchSignal":
        return cls(source=source, round_index=round_index)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ResearchSignal":
        if not isinstance(data, dict):
            return cls.empty()
        allowed = {"source", "round_index", "selection_advisory", "search_pressures", "evidence_records", "final_gate_directives", "metrics", "warnings"}
        unknown = sorted(str(key) for key in data.keys() if str(key) not in allowed)
        pressures = [SearchPressure.from_dict(item) for item in data.get("search_pressures", []) if isinstance(item, dict)]
        records = [item for item in (EvidenceRecord.from_dict(raw) for raw in data.get("evidence_records", []) if isinstance(raw, dict)) if item is not None]
        return cls(
            source=str(data.get("source") or "research"),
            round_index=_int(data.get("round_index"), 0),
            selection_advisory=_coerce_advisory(data.get("selection_advisory")),
            search_pressures=pressures,
            evidence_records=records,
            final_gate_directives=[dict(item) for item in data.get("final_gate_directives", []) if isinstance(item, dict)],
            metrics=coerce_dict(data.get("metrics")),
            warnings=[str(item) for item in data.get("warnings", []) if item] + (["unknown_research_signal_fields:" + ",".join(unknown)] if unknown else []),
        )


def merge_research_signals(signals: list[ResearchSignal]) -> ResearchSignal:
    if not signals:
        return ResearchSignal.empty()
    ordered = sorted(signals, key=lambda item: (str(item.source), int(item.round_index or 0)))
    advisory: dict[str, dict[str, float]] = {}
    pressures: list[SearchPressure] = []
    records: list[EvidenceRecord] = []
    directives: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    warnings: list[str] = []
    for signal in ordered:
        for candidate_id, features in signal.selection_advisory.items():
            current = dict(advisory.get(candidate_id) or {})
            for key, value in (features or {}).items():
                if key == "risk":
                    current[key] = max(float(current.get(key, 0.0) or 0.0), float(value or 0.0))
                else:
                    current[key] = max(float(current.get(key, 0.0) or 0.0), float(value or 0.0))
            advisory[candidate_id] = {k: bounded_score(v) for k, v in sorted(current.items())}
        pressures.extend(signal.search_pressures)
        records.extend(signal.evidence_records)
        directives.extend(signal.final_gate_directives)
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


def _coerce_advisory(value: Any) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for candidate_id, raw in coerce_dict(value).items():
        features = coerce_dict(raw)
        out[str(candidate_id)] = {str(k): bounded_score(v) for k, v in features.items()}
    return out


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = ["ResearchSignal", "merge_research_signals"]
