"""Nexus archive system."""
from __future__ import annotations

from .manager import ARCHIVE_NAMES, ArchiveManager, FateAssignment
from .quality_diversity import QualityDiversityArchive
from .rarity import RarityArchive
from .dormant import DormantArchive
from .failure import FailureArchive, FailureRecord
from .auxiliary import AuxiliaryArchive

__all__ = [
    "ARCHIVE_NAMES",
    "ArchiveManager",
    "FateAssignment",
    "QualityDiversityArchive",
    "RarityArchive",
    "DormantArchive",
    "FailureArchive",
    "FailureRecord",
    "AuxiliaryArchive",
]
