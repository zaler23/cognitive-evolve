"""Stage policy constants and diagnostic classes."""
from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateFate
from cognitive_evolve_runtime.nexus.obligations import HARD_EVIDENCE_FAILURES, HARD_PROOF_FAILURES
from cognitive_evolve_runtime.nexus.source_lineage import MATERIALIZATION_HARD_DIAGNOSTICS, MATERIALIZATION_REPAIR_DIAGNOSTICS

EARLY_STAGE = "early"
MIDDLE_STAGE = "middle"
LATE_STAGE = "late"
FINAL_STAGE = "final"
STAGE_ORDER = {EARLY_STAGE: 0, MIDDLE_STAGE: 1, LATE_STAGE: 2, FINAL_STAGE: 3}

PREFINAL_REPAIR_DIAGNOSTICS: set[str] = set()

# Answer-first mode retires proof/source/final-gate repair obligations.  Tool
# diagnostics may still be recorded, but they must not push candidates into
# repair lanes or hard-reject them unless another subsystem marks a true
# terminal/safety failure in metadata.
REPAIRABLE_DIAGNOSTICS = {
    "missing_parts",
    "auxiliary_guard",
}

HARD_REJECT_DIAGNOSTICS: set[str] = set()

TERMINAL_FATES = {CandidateFate.CULLED.value, CandidateFate.FAILED.value}

__all__ = [
    "EARLY_STAGE", "MIDDLE_STAGE", "LATE_STAGE", "FINAL_STAGE", "STAGE_ORDER",
    "PREFINAL_REPAIR_DIAGNOSTICS", "REPAIRABLE_DIAGNOSTICS", "HARD_REJECT_DIAGNOSTICS", "TERMINAL_FATES",
]
