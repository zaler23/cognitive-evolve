"""Reusable search-skill descriptors passed to model-facing policy metadata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SearchSkill:
    skill_id: str
    description: str
    applies_when: str
    expected_effect: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "description": self.description,
            "applies_when": self.applies_when,
            "expected_effect": self.expected_effect,
        }


DEFAULT_SEARCH_SKILLS = (
    SearchSkill("lens_transplant", "reinterpret the mechanism through a distant discipline", "descriptor cells are collapsing", "new mechanism family"),
    SearchSkill("oracle_grounding", "turn a claim into a tool or replayable verifier target", "verification strength is low", "higher measured checkability"),
    SearchSkill("failure_inversion", "invert a repeated failure into a constraint or counterexample", "same failure class repeats", "ruled-out region or repair route"),
    SearchSkill("parameter_freeze", "replace unconstrained knobs with explicit assignments", "parameter space remains unresolved", "lower final-gate risk"),
)


def search_skill_payload(limit: int | None = None) -> list[dict[str, Any]]:
    skills = list(DEFAULT_SEARCH_SKILLS)
    if limit is not None:
        skills = skills[: max(0, int(limit))]
    return [skill.to_dict() for skill in skills]


__all__ = ["DEFAULT_SEARCH_SKILLS", "SearchSkill", "search_skill_payload"]
