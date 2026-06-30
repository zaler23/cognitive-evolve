"""Context-request protocol for project evolution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.inputs.context_selector import ContextPacket, ContextRequest, ContextSelector
from cognitive_evolve_runtime.inputs.project_map import ProjectWorldModel
from cognitive_evolve_runtime.inputs.project_snapshot import ProjectSnapshot
from cognitive_evolve_runtime.nexus.obligations import candidate_obligation_delta, candidate_source_bindings
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike


@dataclass
class ContextProtocolResult:
    requests: list[ContextRequest] = field(default_factory=list)
    packets: list[ContextPacket] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": [req.__dict__ for req in self.requests],
            "packets": [packet.to_dict() for packet in self.packets],
        }

    def to_source_context(self, *, max_files: int = 3, max_chars: int = 4000) -> dict[str, Any]:
        """Bounded real source slices for downstream seed/mutation/offspring model calls.

        Flattens packet ``raw_file_slices`` into a compact, char-capped structure so the
        model sees actual current file contents (not summaries) without exceeding the
        prompt budget.  Deduplicates by path and stops at ``max_files``.
        """
        slices: list[dict[str, Any]] = []
        seen: set[str] = set()
        for packet in self.packets:
            for path, text in packet.raw_file_slices.items():
                if path in seen:
                    continue
                seen.add(path)
                slices.append({"path": path, "text": (text or "")[:max_chars], "hash": packet.source_hashes.get(path, "")})
                if len(slices) >= max_files:
                    break
            if len(slices) >= max_files:
                break
        return {
            "selected_files": [item["path"] for item in slices],
            "slices": slices,
            "budget_policy": f"top_{max_files}_files_capped_{max_chars}chars_from_context_packets",
        }


class ContextOrchestrator:
    """Ask a model for bounded project context, then materialize slices locally."""

    def __init__(self, *, selector: ContextSelector | None = None) -> None:
        self.selector = selector or ContextSelector()

    def build_for_parents(
        self,
        *,
        contract: Any,
        snapshot: ProjectSnapshot,
        world: ProjectWorldModel,
        parents: list[CandidateGenome],
        archives: ArchiveManager,
        model: NexusModelLike | None = None,
        mutation_instruction: str = "",
        max_requests: int = 3,
    ) -> ContextProtocolResult:
        requests: list[ContextRequest] = []
        if model is not None and hasattr(model, "request_context"):
            raw = model.request_context(contract=contract, world=world, parents=parents, archives=archives, mutation_instruction=mutation_instruction)
            raw_requests = raw if isinstance(raw, list) else [raw]
            for item in raw_requests[:max_requests]:
                if isinstance(item, ContextRequest):
                    requests.append(item)
                elif isinstance(item, dict):
                    requests.append(ContextRequest.from_dict(item))
        if not requests:
            requests.append(_fallback_request(world, parents))
        packets = [
            self.selector.build_context_packet(
                contract=contract,
                snapshot=snapshot,
                world=world,
                request=request,
                parent_candidates=parents,
                archive_hints=archives.summary(),
                mutation_instruction=mutation_instruction,
            )
            for request in requests
        ]
        return ContextProtocolResult(requests=requests, packets=packets)


def _fallback_request(world: ProjectWorldModel, parents: list[CandidateGenome]) -> ContextRequest:
    touched = []
    target_obligation_ids: list[str] = []
    evidence_needs: list[str] = []
    for parent in parents:
        for binding in candidate_source_bindings(parent):
            path = binding.get("path")
            if path:
                touched.append(str(path))
        touched.extend(str(path) for path in getattr(parent, "touched_files", []) if path)
        delta = candidate_obligation_delta(parent)
        for key in ("targeted", "blocked", "introduced"):
            value = delta.get(key)
            if isinstance(value, list):
                target_obligation_ids.extend(str(item) for item in value if item)
            elif value:
                target_obligation_ids.append(str(value))
        metadata = getattr(parent, "metadata", {}) or {}
        if isinstance(metadata, dict):
            target_obligation_ids.extend(str(item) for item in metadata.get("target_obligation_ids", []) if item)
            if metadata.get("evidence_need"):
                evidence_needs.append(str(metadata.get("evidence_need")))
    selected = list(dict.fromkeys(touched)) or _top_relevant_files(world)
    tests: list[str] = []
    for rel in selected:
        tests.extend(world.test_map.get(rel, []))
    return ContextRequest(
        need_files=selected[:5],
        need_tests=list(dict.fromkeys(tests))[:5],
        target_obligation_ids=list(dict.fromkeys(target_obligation_ids))[:12],
        evidence_need=(evidence_needs[0] if evidence_needs else "minimal source-grounded context for named obligation evidence"),
        reason="obligation_targeted_context_for_project_evolution",
    )


def _top_relevant_files(world: ProjectWorldModel) -> list[str]:
    ranked = sorted(world.objective_relevance_map, key=lambda path: world.objective_relevance_map[path], reverse=True)
    if ranked and world.objective_relevance_map.get(ranked[0], 0.0) > 0:
        return ranked[:5]
    return [path for path, role in world.file_roles.items() if role in {"implementation", "test", "config"}][:5]


__all__ = ["ContextProtocolResult", "ContextOrchestrator"]
