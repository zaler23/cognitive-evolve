"""Context-request protocol for project evolution."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
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

    def to_source_context(self, *, max_files: int | None = None, max_chars: int | None = None) -> dict[str, Any]:
        """Real source slices for downstream seed/mutation/offspring model calls.

        Flattens already selected packet slices without a second default text clip.
        Deduplicates by path. Optional caller-provided limits are recorded rather
        than silently changing the model-visible context.
        """
        all_slices: list[dict[str, Any]] = []
        seen: set[str] = set()
        char_limit = int(max_chars) if max_chars is not None and int(max_chars) > 0 else None
        file_limit = int(max_files) if max_files is not None and int(max_files) > 0 else None
        for packet in self.packets:
            for path, text in packet.raw_file_slices.items():
                if path in seen:
                    continue
                seen.add(path)
                source_text = text or ""
                all_slices.append({"path": path, "text": source_text, "hash": packet.source_hashes.get(path, "")})
        slices = [dict(item) for item in (all_slices[:file_limit] if file_limit is not None else all_slices)]
        omitted_files = [item["path"] for item in all_slices[len(slices) :]]
        truncated_files: list[str] = []
        for item in slices:
            original_chars = len(str(item.get("text") or ""))
            if char_limit is None or original_chars <= char_limit:
                continue
            item["text"] = str(item.get("text") or "")[:char_limit]
            item.update({"truncated": True, "original_chars": original_chars, "selected_chars": len(item["text"])})
            truncated_files.append(str(item.get("path") or ""))
        file_policy = f"top_{file_limit}_files" if file_limit is not None else "all_selected_files"
        char_policy = f"capped_{char_limit}chars_from_context_packets" if char_limit is not None else "full_selected_slices"
        return {
            "selected_files": [item["path"] for item in slices],
            "slices": slices,
            "budget_policy": f"{file_policy}_{char_policy}",
            "context_limits": {
                "max_files": file_limit,
                "max_chars_per_file": char_limit,
                "available_file_count": len(all_slices),
                "selected_file_count": len(slices),
                "omitted_files": omitted_files,
                "truncated_files": truncated_files,
            },
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
            requests.append(_fallback_request(world, parents, mutation_instruction=mutation_instruction))
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


def _fallback_request(
    world: ProjectWorldModel,
    parents: list[CandidateGenome],
    *,
    mutation_instruction: str = "",
) -> ContextRequest:
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
    action_paths = _instruction_paths(world, mutation_instruction)
    parent_paths = _resolve_manifest_paths(world, touched)
    if action_paths:
        action_path_set = set(action_paths)
        parent_fillers = [path for path in parent_paths if path not in action_path_set]
        selected = action_paths + parent_fillers[: max(0, 5 - len(action_paths))]
    else:
        selected = parent_paths[:5] or _top_relevant_files(world)
    tests: list[str] = []
    for rel in selected:
        tests.extend(world.test_map.get(rel, []))
    return ContextRequest(
        need_files=selected,
        need_tests=list(dict.fromkeys(tests))[:5],
        target_obligation_ids=list(dict.fromkeys(target_obligation_ids))[:12],
        evidence_need=(evidence_needs[0] if evidence_needs else "minimal source-grounded context for named obligation evidence"),
        reason="obligation_targeted_context_for_project_evolution",
    )


def _instruction_paths(world: ProjectWorldModel, instruction: str) -> list[str]:
    text = str(instruction or "")
    if not text:
        return []
    manifest = list(world.file_roles)
    basename_counts: dict[str, int] = {}
    for path in manifest:
        name = Path(path).name
        basename_counts[name] = basename_counts.get(name, 0) + 1
    mentioned: list[tuple[int, int, str]] = []
    for manifest_index, path in enumerate(manifest):
        names = [path]
        basename = Path(path).name
        if basename_counts[basename] == 1:
            names.append(basename)
        positions = [
            match.start()
            for name in names
            if (match := re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(name)}(?![A-Za-z0-9_.-])", text))
        ]
        if positions:
            mentioned.append((min(positions), manifest_index, path))
    return [path for _, _, path in sorted(mentioned)]


def _resolve_manifest_paths(world: ProjectWorldModel, paths: list[str]) -> list[str]:
    manifest = list(world.file_roles)
    manifest_set = set(manifest)
    resolved: list[str] = []
    for raw in paths:
        item = str(raw or "").strip()
        if item.startswith("./"):
            item = item[2:]
        if not item or item.startswith(("/", "~")) or any(part in {"", ".", ".."} for part in Path(item).parts):
            continue
        match = item if item in manifest_set else ""
        if not match:
            basename_matches = [path for path in manifest if Path(path).name == Path(item).name]
            if len(basename_matches) == 1:
                match = basename_matches[0]
        if not match and "/" in item:
            suffix_matches = [path for path in manifest if path.endswith(item)]
            if len(suffix_matches) == 1:
                match = suffix_matches[0]
        if match and match not in resolved:
            resolved.append(match)
    return resolved


def _top_relevant_files(world: ProjectWorldModel) -> list[str]:
    ranked = sorted(world.objective_relevance_map, key=lambda path: world.objective_relevance_map[path], reverse=True)
    if ranked and world.objective_relevance_map.get(ranked[0], 0.0) > 0:
        return ranked[:5]
    return [path for path, role in world.file_roles.items() if role in {"implementation", "test", "config"}][:5]


__all__ = ["ContextProtocolResult", "ContextOrchestrator"]
