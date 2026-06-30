"""Registry-style archive lane routing for ArchiveManager."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.candidates.project_candidate import ProjectCandidateGenome
from cognitive_evolve_runtime.core.serialization import coerce_str_list
from cognitive_evolve_runtime.nexus.source_binding_resolver import annotate_candidate_source_bindings, candidate_admission_route, candidate_source_binding_class

from .latent_pareto import _candidate_is_latent_pareto_frontier
from .quality_diversity import candidate_quality
from .types import FateAssignment

TERMINAL_FAILURE_FATES = {CandidateFate.CULLED.value, CandidateFate.FAILED.value}


@dataclass
class ArchiveRegistry:
    """Route candidates into archive lanes while ArchiveManager owns state."""

    manager: Any

    def route_candidate(self, candidate: CandidateGenome, assignment: FateAssignment) -> None:
        fate = CandidateFate.normalize(assignment.fate)
        try:
            annotate_candidate_source_bindings(candidate, project_root=getattr(self.manager, "project_root", "") or None)
        except Exception:
            if isinstance(candidate.metadata, dict):
                candidate.metadata.setdefault("source_binding_manifest", {"binding_class": "no_binding", "admission_route": "repair_only", "diagnostics": ["source_binding_annotation_failed"]})
        admission_route = candidate_admission_route(candidate)
        binding_class = candidate_source_binding_class(candidate)
        if fate in TERMINAL_FAILURE_FATES or admission_route == "negative_archive_only":
            self.manager.failure_archive.add(candidate, signature=assignment.failure_signature)
            assignment.archive_targets.append("FailureArchive")
            return
    
        if _candidate_is_latent_pareto_frontier(candidate):
            self.manager.latent_pareto_archive.add(candidate)
            assignment.archive_targets.append("LatentParetoIntentArchive")
    
        self.manager.quality_diversity.update(candidate)
        assignment.archive_targets.append("QualityDiversityArchive")
        self.manager.rarity_archive.add(candidate)
        if candidate.id in self.manager.rarity_archive.candidates:
            assignment.archive_targets.append("RarityArchive")
        if candidate.multihead_scores.get("novelty", 0.0) > 0 or candidate.novelty_descriptors:
            self.manager.novelty_archive[candidate.id] = candidate.to_dict()
            assignment.archive_targets.append("NoveltyArchive")
        mechanism_key = candidate.core_mechanism or candidate.concise_claim or candidate.id
        if mechanism_key:
            current = self.manager.mechanism_archive.get(mechanism_key)
            if current is None or candidate_quality(candidate) > float(current.get("quality", -1.0)):
                data = candidate.to_dict()
                data["quality"] = candidate_quality(candidate)
                self.manager.mechanism_archive[mechanism_key] = data
                assignment.archive_targets.append("MechanismArchive")
        source_blocked = binding_class in {"invented", "unresolved"}
        if not source_blocked and (isinstance(candidate, ProjectCandidateGenome) or candidate.artifact_type in {"project_patch", "patch", "code_patch"}):
            self.manager.project_patch_archive[candidate.id] = candidate.to_dict()
            assignment.archive_targets.append("ProjectPatchArchive")
        elif isinstance(candidate, ProjectCandidateGenome) or candidate.artifact_type in {"project_patch", "patch", "code_patch"}:
            assignment.archive_targets.append(f"ProjectPatchArchiveBlocked:{binding_class}")
        if fate == CandidateFate.ELITE.value and not source_blocked:
            self.manager.answer_archive[candidate.id] = candidate.to_dict()
            assignment.archive_targets.append("AnswerArchive")
        elif fate == CandidateFate.ELITE.value:
            assignment.archive_targets.append(f"AnswerArchiveBlocked:{binding_class}")
        elif fate == CandidateFate.AUXILIARY.value:
            self.manager.auxiliary_archive.add(candidate)
            assignment.archive_targets.append("AuxiliaryArchive")
        elif fate == CandidateFate.DORMANT.value:
            self.manager.dormant_archive.add(candidate, condition=assignment.future_reactivation_condition or "reactivate_when_search_needs_diversity_or_complementarity")
            assignment.archive_targets.append("DormantArchive")
        elif fate == CandidateFate.INCUBATING.value:
            assignment.archive_targets.append("IncubatingRepairLane")

    def remove_candidate_from_archives(self, candidate_id: str) -> None:
        """Remove stale memberships before re-routing a mutable candidate."""
    
        self.manager.answer_archive.pop(candidate_id, None)
        self.manager.novelty_archive.pop(candidate_id, None)
        self.manager.latent_pareto_archive.discard(candidate_id, record=False)
        self.manager.project_patch_archive.pop(candidate_id, None)
        self.manager.rarity_archive.candidates.pop(candidate_id, None)
        self.manager.auxiliary_archive.candidates.pop(candidate_id, None)
        self.manager.dormant_archive.candidates.pop(candidate_id, None)
        self.manager.dormant_archive.reactivation_conditions.pop(candidate_id, None)
        self.manager.failure_archive.records.pop(candidate_id, None)
        self.manager.terminal_tombstones.pop(candidate_id, None)
        self._remove_from_quality_diversity(candidate_id)
        self._remove_from_mechanism_archive(candidate_id)
        self._rebuild_rarity_seeds()

    def _remove_from_quality_diversity(self, candidate_id: str) -> None:
        stale_niches = [
            niche
            for niche, data in self.manager.quality_diversity.elites_by_niche.items()
            if data.get("candidate_id") == candidate_id
            or (isinstance(data.get("candidate"), dict) and data["candidate"].get("id") == candidate_id)
        ]
        for niche in stale_niches:
            self.manager.quality_diversity.elites_by_niche.pop(niche, None)

    def _remove_from_mechanism_archive(self, candidate_id: str) -> None:
        stale_keys = [
            key
            for key, data in self.manager.mechanism_archive.items()
            if key == candidate_id
            or data.get("id") == candidate_id
            or (isinstance(data.get("candidate"), dict) and data["candidate"].get("id") == candidate_id)
        ]
        for key in stale_keys:
            self.manager.mechanism_archive.pop(key, None)

    def _rebuild_rarity_seeds(self) -> None:
        seeds: list[str] = []
        for data in self.manager.rarity_archive.candidates.values():
            for seed in coerce_str_list(data.get("edge_knowledge_seeds")):
                if seed not in seeds:
                    seeds.append(seed)
        self.manager.rarity_archive.seeds = seeds

__all__ = ["ArchiveRegistry"]
