"""Canonical fate semantics for Nexus candidate lifecycle decisions."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from cognitive_evolve_runtime.candidates.genome import CandidateFate


@dataclass(frozen=True)
class FateSemantics:
    fate: str
    role: str
    may_be_parent: bool
    may_reactivate: bool
    terminal: bool
    final_answer_candidate: bool
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


FATE_SEMANTICS: dict[str, FateSemantics] = {
    CandidateFate.ACTIVE.value: FateSemantics(
        fate=CandidateFate.ACTIVE.value,
        role="live_search_candidate",
        may_be_parent=True,
        may_reactivate=False,
        terminal=False,
        final_answer_candidate=True,
        notes="A live candidate that can rank, reproduce, and mature toward final selection.",
    ),
    CandidateFate.INCUBATING.value: FateSemantics(
        fate=CandidateFate.INCUBATING.value,
        role="bounded_repair_lane",
        may_be_parent=True,
        may_reactivate=False,
        terminal=False,
        final_answer_candidate=False,
        notes="Promising but incomplete material kept alive for explicit repair obligations; not final until repaired and reverified.",
    ),
    CandidateFate.DORMANT.value: FateSemantics(
        fate=CandidateFate.DORMANT.value,
        role="parked_reactivation_pool",
        may_be_parent=False,
        may_reactivate=True,
        terminal=False,
        final_answer_candidate=False,
        notes="Parked material. It is not a winner or parent by default, but can be reactivated through a named repair/reactivation path.",
    ),
    CandidateFate.ELITE.value: FateSemantics(
        fate=CandidateFate.ELITE.value,
        role="current_best_material",
        may_be_parent=True,
        may_reactivate=False,
        terminal=False,
        final_answer_candidate=True,
        notes="Best current material; still subject to the strict final gate before being treated as solved or merge-ready.",
    ),
    CandidateFate.AUXILIARY.value: FateSemantics(
        fate=CandidateFate.AUXILIARY.value,
        role="supporting_scaffold",
        may_be_parent=False,
        may_reactivate=True,
        terminal=False,
        final_answer_candidate=False,
        notes="Useful tool/scaffold material that supports search but is not a main answer by default.",
    ),
    CandidateFate.CULLED.value: FateSemantics(
        fate=CandidateFate.CULLED.value,
        role="removed_nonterminal_lesson",
        may_be_parent=False,
        may_reactivate=False,
        terminal=True,
        final_answer_candidate=False,
        notes="Removed from live search; lessons may remain in archives but the candidate itself is not reactivated.",
    ),
    CandidateFate.FAILED.value: FateSemantics(
        fate=CandidateFate.FAILED.value,
        role="terminal_or_repair_seed_material",
        may_be_parent=False,
        may_reactivate=False,
        terminal=True,
        final_answer_candidate=False,
        notes="Failed candidate. Only a separate repair-seed extraction may create new Incubating material; the failed candidate is not a live parent.",
    ),
}


def fate_semantics(fate: str) -> FateSemantics:
    normalized = CandidateFate.normalize(fate)
    return FATE_SEMANTICS[normalized]


def fate_semantics_table() -> dict[str, dict[str, object]]:
    return {fate: semantics.to_dict() for fate, semantics in FATE_SEMANTICS.items()}


__all__ = ["FATE_SEMANTICS", "FateSemantics", "fate_semantics", "fate_semantics_table"]
