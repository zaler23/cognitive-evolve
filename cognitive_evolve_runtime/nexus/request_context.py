"""Request-local Nexus and LLM knobs used by API and CLI runs."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Iterator

_INTERNAL_ROUND_CAP: ContextVar[int | None] = ContextVar("cogev_internal_round_cap", default=None)
_EVOLUTION_PROFILE: ContextVar[str | None] = ContextVar("cogev_evolution_profile", default=None)
_LLM_STAGE: ContextVar[str | None] = ContextVar("cogev_llm_stage", default=None)
_LLM_STAGE_BUDGET_PERCENTAGES: ContextVar[dict[str, float] | None] = ContextVar("cogev_llm_stage_budget_percentages", default=None)

LLM_STAGE_BUDGET_PERCENTAGES = MappingProxyType({"seed": 0.20, "evolution": 0.60, "synthesis_verification": 0.20})
LLM_STAGE_GROUPS = MappingProxyType({
    "seed": frozenset({
        "semantic_reconstruction",
        "budget_planning",
        "evidence_planning",
        "evidence_execution",
        "uncertainty_policy",
        "candidate_seed_generation",
        "candidate_seed_scoring",
        "nexus_contract",
        "nexus_policy",
    }),
    "evolution": frozenset({
        "candidate_population",
        "candidate_critique",
        "candidate_reflection",
        "candidate_genome_mutation",
        "candidate_genome_mutation_scoring",
        "candidate_round_artifact",
        "candidate_final_judge_rerank",
        "candidate_pairwise_judge",
        "pairwise_judge",
        "nexus_relative_rank",
        "nexus_diagnosis",
    }),
    "synthesis_verification": frozenset({"synthesis", "uncertainty_fuse", "local_verification", "nexus_synthesis"}),
})
LLM_STAGE_BUDGET_PROFILES = MappingProxyType({
    "default": LLM_STAGE_BUDGET_PERCENTAGES,
    "direct_answer_or_small_edit": MappingProxyType({"seed": 0.15, "evolution": 0.35, "synthesis_verification": 0.50}),
    "research_or_evidence_dependent_plan": MappingProxyType({"seed": 0.30, "evolution": 0.45, "synthesis_verification": 0.25}),
    "technical_execution_or_codebase_task": MappingProxyType({"seed": 0.25, "evolution": 0.50, "synthesis_verification": 0.25}),
    "architecture_refactor_or_migration": MappingProxyType({"seed": 0.22, "evolution": 0.58, "synthesis_verification": 0.20}),
    "governed_safe_plan": MappingProxyType({"seed": 0.20, "evolution": 0.40, "synthesis_verification": 0.40}),
    "structured_decision_or_design": MappingProxyType({"seed": 0.20, "evolution": 0.55, "synthesis_verification": 0.25}),
    "proof_resolution": MappingProxyType({"seed": 0.25, "evolution": 0.60, "synthesis_verification": 0.15}),
    "open_conjecture": MappingProxyType({"seed": 0.25, "evolution": 0.60, "synthesis_verification": 0.15}),
    "mechanism_discovery": MappingProxyType({"seed": 0.25, "evolution": 0.58, "synthesis_verification": 0.17}),
    "novel_algorithm_design": MappingProxyType({"seed": 0.22, "evolution": 0.60, "synthesis_verification": 0.18}),
})


def llm_stage_group(stage: str | None) -> str | None:
    normalized = str(stage or "")
    for group, stages in LLM_STAGE_GROUPS.items():
        if normalized in stages:
            return group
    return None


def _normalize_stage_budget_percentages(raw: dict[str, float]) -> dict[str, float]:
    groups = tuple(LLM_STAGE_BUDGET_PERCENTAGES.keys())
    values = {group: max(0.0, float(raw.get(group, LLM_STAGE_BUDGET_PERCENTAGES[group]))) for group in groups}
    total = sum(values.values())
    if total <= 0:
        return dict(LLM_STAGE_BUDGET_PERCENTAGES)
    return {group: round(values[group] / total, 6) for group in groups}


def _profile_for_task_type(task_type: str | None) -> dict[str, float]:
    key = str(task_type or "default").strip() or "default"
    profile = LLM_STAGE_BUDGET_PROFILES.get(key) or LLM_STAGE_BUDGET_PROFILES["default"]
    return _normalize_stage_budget_percentages(dict(profile))


def llm_stage_budget_percentages(task_type: str | None = None) -> dict[str, float]:
    if task_type is not None:
        return _profile_for_task_type(task_type)
    scoped = _LLM_STAGE_BUDGET_PERCENTAGES.get()
    return dict(scoped) if scoped else dict(LLM_STAGE_BUDGET_PERCENTAGES)


@contextmanager
def llm_stage_budget_profile(task_type: str | dict[str, float] | None) -> Iterator[None]:
    profile = _normalize_stage_budget_percentages(task_type) if isinstance(task_type, dict) else _profile_for_task_type(task_type)
    token = _LLM_STAGE_BUDGET_PERCENTAGES.set(profile)
    try:
        yield
    finally:
        _LLM_STAGE_BUDGET_PERCENTAGES.reset(token)


def llm_stage_groups() -> dict[str, list[str]]:
    return {group: sorted(stages) for group, stages in LLM_STAGE_GROUPS.items()}


def get_internal_round_cap() -> int | None:
    return _INTERNAL_ROUND_CAP.get()


@contextmanager
def internal_round_cap(cap: int | str | None) -> Iterator[None]:
    try:
        normalized = None if cap in {None, ""} else int(cap)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        normalized = None
    token = _INTERNAL_ROUND_CAP.set(normalized)
    try:
        yield
    finally:
        _INTERNAL_ROUND_CAP.reset(token)


def get_evolution_profile() -> str | None:
    return _EVOLUTION_PROFILE.get()


@contextmanager
def evolution_profile(profile: str | None) -> Iterator[None]:
    normalized = str(profile).strip().lower() if profile else None
    token = _EVOLUTION_PROFILE.set(normalized or None)
    try:
        yield
    finally:
        _EVOLUTION_PROFILE.reset(token)


def get_llm_stage() -> str | None:
    return _LLM_STAGE.get()


@contextmanager
def llm_stage(stage: str | None) -> Iterator[None]:
    token = _LLM_STAGE.set(str(stage) if stage else None)
    try:
        yield
    finally:
        _LLM_STAGE.reset(token)


__all__ = [
    "LLM_STAGE_BUDGET_PERCENTAGES",
    "LLM_STAGE_GROUPS",
    "LLM_STAGE_BUDGET_PROFILES",
    "get_internal_round_cap",
    "internal_round_cap",
    "get_evolution_profile",
    "evolution_profile",
    "get_llm_stage",
    "llm_stage",
    "llm_stage_budget_percentages",
    "llm_stage_budget_profile",
    "llm_stage_group",
    "llm_stage_groups",
]
