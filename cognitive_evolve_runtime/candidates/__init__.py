"""Canonical Nexus candidate primitives."""
from __future__ import annotations

from .crossover import crossover
from .genome import CandidateFate, CandidateGenome, CandidatePopulation, candidate_from_dict
from .lineage import lineage_counts, lineage_family, saturated_lineages
from .mutation import MutationEngine, MutationOperator, MutationPlan, MutationPlanner
from .patch_merge import PatchMergeConflict, PatchMergeResult, merge_patch_sets, project_patch_crossover
from .project_candidate import PatchApplicationResult, PatchOperation, ProjectCandidateGenome

__all__ = [
    "CandidateFate",
    "CandidateGenome",
    "CandidatePopulation",
    "candidate_from_dict",
    "ProjectCandidateGenome",
    "PatchApplicationResult",
    "PatchOperation",
    "MutationEngine",
    "MutationOperator",
    "MutationPlan",
    "MutationPlanner",
    "crossover",
    "lineage_family",
    "lineage_counts",
    "saturated_lineages",
    "PatchMergeConflict",
    "PatchMergeResult",
    "merge_patch_sets",
    "project_patch_crossover",
]
