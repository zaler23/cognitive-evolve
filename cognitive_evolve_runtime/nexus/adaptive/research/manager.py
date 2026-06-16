"""Adaptive-internal research extension manager.

The manager applies ResearchSignal outputs back into the existing Evidence
Control Plane.  It does not own runtime orchestration, archive fate, challenge
truth, or final solved authority.
"""
from __future__ import annotations

from typing import Any, Callable

from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.evaluators.challenge_memory import ChallengeMemory
from cognitive_evolve_runtime.evaluators.evidence import EvidenceRecord, SearchPressure, apply_evidence_record
from cognitive_evolve_runtime.nexus._serde import coerce_dict, utc_now
from cognitive_evolve_runtime.nexus.adaptive.research.protocol import ResearchContext, ResearchExtension
from cognitive_evolve_runtime.nexus.adaptive.research.registry import ResearchConfig, build_research_extensions
from cognitive_evolve_runtime.nexus.adaptive.research.signal import ResearchSignal, merge_research_signals
from cognitive_evolve_runtime.nexus.adaptive.research.state import ResearchRegistryState


class ResearchExtensionRegistry:
    def __init__(self, *, config: ResearchConfig, state: ResearchRegistryState | None = None, extensions: list[ResearchExtension] | None = None) -> None:
        self.config = config
        self.state = state or ResearchRegistryState()
        self.extensions = extensions if extensions is not None else build_research_extensions(config)
        for extension in self.extensions:
            extension.restore(coerce_dict(self.state.extensions.get(extension.extension_id)))
        self._last_advisory: dict[str, dict[str, float]] = {}

    @classmethod
    def from_config(cls, config: ResearchConfig, *, restored_state: dict[str, Any] | None = None) -> "ResearchExtensionRegistry":
        return cls(config=config, state=ResearchRegistryState.from_dict(restored_state))

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and self.extensions)

    @property
    def mode(self) -> str:
        return str(getattr(self.config, "mode", "observe") or "observe").strip().lower() or "observe"

    def after_evidence(self, ctx: ResearchContext) -> ResearchSignal:
        return self._run_hook("after_evidence", ctx)

    def before_parent_selection(self, ctx: ResearchContext) -> ResearchSignal:
        signal = self._run_hook("before_parent_selection", ctx)
        self._last_advisory = dict(signal.selection_advisory)
        return signal

    def before_mutation_planning(self, ctx: ResearchContext) -> ResearchSignal:
        return self._run_hook("before_mutation_planning", ctx)

    def before_final_projection(self, ctx: ResearchContext) -> ResearchSignal:
        return self._run_hook("before_final_projection", ctx)

    def advisory_features(self) -> dict[str, dict[str, float]]:
        return dict(self._last_advisory)

    def pending_search_pressures(self, *, parent_id: str | None = None) -> list[SearchPressure]:
        out: list[SearchPressure] = []
        for raw in self.state.pending_search_pressures:
            pressure = SearchPressure.from_dict(raw)
            if parent_id and pressure.parent_id not in {None, "", parent_id}:
                continue
            out.append(pressure)
        return out

    def apply_signal(self, signal: ResearchSignal, *, candidates: list[CandidateGenome] | None = None, challenge_memory: ChallengeMemory | None = None) -> None:
        self._apply_filtered_signal(self._filter_signal_for_mode(signal), candidates=candidates, challenge_memory=challenge_memory)

    def _apply_filtered_signal(self, signal: ResearchSignal, *, candidates: list[CandidateGenome] | None = None, challenge_memory: ChallengeMemory | None = None) -> None:
        by_id = {candidate.id: candidate for candidate in candidates or []}
        for record in signal.evidence_records:
            candidate = by_id.get(record.candidate_id)
            if candidate is not None:
                apply_evidence_record(candidate, record)
                if challenge_memory is not None:
                    challenge_memory.ingest(record, round_index=signal.round_index)
        if signal.search_pressures:
            existing = {str(item.get("id") or ""): dict(item) for item in self.state.pending_search_pressures if isinstance(item, dict)}
            for pressure in signal.search_pressures:
                existing[pressure.id] = pressure.to_dict()
            self.state.pending_search_pressures = list(existing.values())[-100:]
        if signal.final_gate_directives:
            self.state.final_gate_directives.extend(dict(item) for item in signal.final_gate_directives if isinstance(item, dict) and item.get("kind"))
            self.state.final_gate_directives = self.state.final_gate_directives[-50:]
        if signal.metrics:
            self.state.metrics.update(signal.metrics)
        if signal.warnings:
            self.state.warnings = list(dict.fromkeys([*self.state.warnings, *signal.warnings]))[-100:]
        self.state.metrics["research_signal_count"] = int(self.state.metrics.get("research_signal_count") or 0) + 1
        self.state.record_event({"event": "research_signal", "source": signal.source, "round": signal.round_index, "mode": self.mode, "warnings": signal.warnings[:5]})

    def snapshot(self) -> dict[str, Any]:
        snapshots: dict[str, dict[str, Any]] = {}
        for extension in self.extensions:
            payload = extension.snapshot()
            if payload:
                snapshots[extension.extension_id] = dict(payload)
        self.state.extensions = snapshots
        self.state.metrics["active_extensions"] = len(self.extensions)
        self.state.metrics["research_registry_enabled"] = bool(self.enabled)
        return self.state.to_dict()

    def record_generated_targets(self, *, candidate_id: str, challenge_ids: list[str], pressure_id: str, round_index: int) -> None:
        self._record_target_event("generated", candidate_id=candidate_id, challenge_ids=challenge_ids, pressure_id=pressure_id, round_index=round_index)

    def record_evaluated_targets(self, *, candidate_id: str, challenge_ids: list[str], record: EvidenceRecord | None, round_index: int) -> None:
        self._record_target_event("evaluated", candidate_id=candidate_id, challenge_ids=challenge_ids, pressure_id="", round_index=round_index)

    def record_resolved_targets(self, *, candidate_id: str, challenge_ids: list[str], record: EvidenceRecord | None, round_index: int) -> None:
        self._record_target_event("resolved", candidate_id=candidate_id, challenge_ids=challenge_ids, pressure_id="", round_index=round_index)

    def _record_target_event(self, kind: str, *, candidate_id: str, challenge_ids: list[str], pressure_id: str, round_index: int) -> None:
        ids = [str(item) for item in challenge_ids if str(item or "").strip()]
        if not ids:
            return
        tracking = dict(self.state.target_tracking or {})
        events = [dict(item) for item in tracking.get("events", []) if isinstance(item, dict)]
        existing = {
            (str(item.get("kind") or ""), str(item.get("candidate_id") or ""), str(item.get("challenge_id") or ""))
            for item in events
        }
        for cid in ids:
            key = (kind, str(candidate_id), cid)
            if key in existing:
                continue
            existing.add(key)
            events.append({"kind": kind, "candidate_id": str(candidate_id), "challenge_id": cid, "pressure_id": str(pressure_id or ""), "round": int(round_index or 0), "at": utc_now()})
        tracking["events"] = events[-500:]
        self.state.target_tracking = tracking
        self._update_target_metrics()

    def _update_target_metrics(self) -> None:
        events = [dict(item) for item in self.state.target_tracking.get("events", []) if isinstance(item, dict)]
        generated = [item for item in events if item.get("kind") == "generated"]
        evaluated = [item for item in events if item.get("kind") == "evaluated"]
        resolved = [item for item in events if item.get("kind") == "resolved"]
        generated_keys = {(item.get("candidate_id"), item.get("challenge_id")) for item in generated}
        evaluated_keys = {(item.get("candidate_id"), item.get("challenge_id")) for item in evaluated}
        resolved_keys = {(item.get("candidate_id"), item.get("challenge_id")) for item in resolved}
        evaluated_generated = evaluated_keys & generated_keys
        resolved_evaluated = resolved_keys & evaluated_keys
        resolved_generated = resolved_keys & generated_keys
        self.state.metrics.update({
            "generated_targeted_count": len(generated_keys),
            "evaluated_targeted_count": len(evaluated_keys),
            "resolved_targeted_count": len(resolved_keys),
            "generated_target_evaluation_rate": len(evaluated_generated) / max(1, len(generated_keys)),
            "evaluated_target_resolution_rate": len(resolved_evaluated) / max(1, len(evaluated_keys)),
            "generated_target_resolution_rate": len(resolved_generated) / max(1, len(generated_keys)),
            "targeted_evaluation_rate": len(evaluated_generated) / max(1, len(generated_keys)),
            "targeted_resolution_rate": len(resolved_evaluated) / max(1, len(evaluated_keys)),
        })

    def _run_hook(self, hook: str, ctx: ResearchContext) -> ResearchSignal:
        if not self.enabled:
            return ResearchSignal.empty(source=f"research.{hook}", round_index=ctx.round_index)
        signals: list[ResearchSignal] = []
        for extension in self.extensions:
            fn: Callable[[ResearchContext], ResearchSignal] = getattr(extension, hook)
            try:
                signal = fn(ctx)
            except Exception as exc:  # Extensions may not break NexusRuntime.
                signal = ResearchSignal(source=extension.extension_id, round_index=ctx.round_index, warnings=[f"{hook}_extension_error:{extension.extension_id}:{exc.__class__.__name__}"])
            signals.append(signal)
        merged = merge_research_signals(signals)
        filtered = self._filter_signal_for_mode(merged, hook=hook)
        self._apply_filtered_signal(filtered, candidates=ctx.candidates, challenge_memory=ctx.challenge_memory)
        return filtered

    def _filter_signal_for_mode(self, signal: ResearchSignal, *, hook: str = "") -> ResearchSignal:
        mode = self.mode if self.mode in {"observe", "advisory", "active"} else "observe"
        metrics = dict(signal.metrics or {})
        warnings = list(signal.warnings or [])
        dropped: dict[str, int] = {}

        selection_advisory = signal.selection_advisory if mode in {"advisory", "active"} else {}
        search_pressures = signal.search_pressures if mode in {"advisory", "active"} else []
        evidence_records = signal.evidence_records if mode == "active" else []
        final_gate_directives = _normalize_final_gate_directives(signal.final_gate_directives, mode=mode, round_index=signal.round_index)

        if mode == "observe":
            dropped = {
                "selection_advisory": len(signal.selection_advisory),
                "search_pressures": len(signal.search_pressures),
                "evidence_records": len(signal.evidence_records),
                "final_gate_directives": len(signal.final_gate_directives),
            }
            dropped = {key: value for key, value in dropped.items() if value}
            if dropped:
                warnings.append("research_observe_mode_dropped_effective_signal")
        elif mode == "advisory":
            if signal.evidence_records:
                dropped["evidence_records"] = len(signal.evidence_records)
                warnings.append("research_advisory_mode_dropped_evidence_records")

        unknown_count = sum(1 for item in signal.final_gate_directives if _unknown_final_directive(item))
        if unknown_count:
            metrics["research_signal_unknown_count"] = int(metrics.get("research_signal_unknown_count") or 0) + unknown_count
            warnings.append("unknown_research_final_gate_directive")
        if dropped:
            metrics["research_signal_dropped_effect_count"] = int(metrics.get("research_signal_dropped_effect_count") or 0) + sum(dropped.values())
            metrics["research_signal_dropped_effects"] = dropped

        return ResearchSignal(
            source=signal.source,
            round_index=signal.round_index,
            selection_advisory=dict(selection_advisory),
            search_pressures=list(search_pressures),
            evidence_records=list(evidence_records),
            final_gate_directives=final_gate_directives,
            metrics=metrics,
            warnings=list(dict.fromkeys(warnings)),
        )


_KNOWN_FINAL_DIRECTIVE_KINDS = {
    "bft_quorum_report",
    "chaos_replay_required_if_configured",
    "contract_refinement_proposal",
    "immune_necropsy_report",
    "parametric_candidate_not_final",
}


def _unknown_final_directive(item: Any) -> bool:
    return not isinstance(item, dict) or str(item.get("kind") or "") not in _KNOWN_FINAL_DIRECTIVE_KINDS


def _normalize_final_gate_directives(items: list[dict[str, Any]], *, mode: str, round_index: int) -> list[dict[str, Any]]:
    if mode == "observe":
        return []
    directives: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            if mode == "active":
                directives.append({"kind": "unknown_research_final_gate_directive", "enforcement": "blocking", "round_index": int(round_index or 0)})
            continue
        kind = str(item.get("kind") or "")
        if kind not in _KNOWN_FINAL_DIRECTIVE_KINDS:
            if mode == "active":
                directives.append({"kind": "unknown_research_final_gate_directive", "original_kind": kind[:120], "enforcement": "blocking", "round_index": int(round_index or 0)})
            continue
        directive = dict(item)
        directive["enforcement"] = "report" if mode == "advisory" else str(directive.get("enforcement") or "blocking")
        directive["research_mode"] = mode
        directive["round_index"] = int(round_index or 0)
        directives.append(directive)
    return directives


__all__ = ["ResearchExtensionRegistry"]
