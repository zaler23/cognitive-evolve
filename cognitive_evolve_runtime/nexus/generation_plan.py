"""Verifiable per-generation transition plans for Nexus evolution."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.archives.manager import ArchiveManager, FateAssignment
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.nexus._serde import coerce_dict, coerce_str_list, stable_hash, utc_now

KNOWN_STAGE_OPS = {
    "critique_and_verify",
    "rank",
    "archive_assign",
    "generation_plan_validate",
    "archive_update",
    "compact",
    "diagnose",
    "stop_check",
    "select_parents",
    "plan_mutations",
    "generate_offspring",
    "verify_offspring",
    "synthesize",
}

STAGE_ORDER = [
    "critique_and_verify",
    "rank",
    "archive_assign",
    "generation_plan_validate",
    "archive_update",
    "compact",
    "diagnose",
    "stop_check",
    "select_parents",
    "plan_mutations",
    "generate_offspring",
    "verify_offspring",
    "synthesize",
]

STAGE_PREREQUISITES = {
    "rank": {"critique_and_verify"},
    "archive_assign": {"rank"},
    "generation_plan_validate": {"archive_assign"},
    "archive_update": {"generation_plan_validate"},
    "compact": {"archive_update"},
    "diagnose": {"compact"},
    "stop_check": {"diagnose"},
    "select_parents": {"stop_check"},
    "plan_mutations": {"select_parents"},
    "generate_offspring": {"plan_mutations"},
    "verify_offspring": {"generate_offspring"},
}

_STAGE_INDEX = {op: index for index, op in enumerate(STAGE_ORDER)}


class GenerationPlanError(ValueError):
    """Raised when a generation transition plan is not admissible."""


@dataclass(frozen=True)
class GenerationPlan:
    """A single authority for one generation's state transition.

    The first production slice owns the rank → fate → archive transition.  The
    schema already reserves parent, mutation, and stage-graph fields so later
    slices extend the same authority instead of adding another competing one.
    """

    plan_id: str
    round_index: int
    source: str
    fate_assignments: list[dict[str, Any]]
    parent_ids: list[str] = field(default_factory=list)
    mutation_objectives: list[str] = field(default_factory=list)
    archive_writes: list[dict[str, Any]] = field(default_factory=list)
    stage_graph: list[dict[str, Any]] = field(default_factory=list)
    ranking_summary: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationPlan":
        return cls(
            plan_id=str(data.get("plan_id") or ""),
            round_index=int(data.get("round_index") or 0),
            source=str(data.get("source") or "unknown"),
            fate_assignments=[_coerce_assignment_dict(item) for item in data.get("fate_assignments", []) if isinstance(item, (dict, FateAssignment))],
            parent_ids=coerce_str_list(data.get("parent_ids")),
            mutation_objectives=coerce_str_list(data.get("mutation_objectives")),
            archive_writes=[dict(item) for item in data.get("archive_writes", []) if isinstance(item, dict)],
            stage_graph=_coerce_stage_graph(data.get("stage_graph")),
            ranking_summary=coerce_dict(data.get("ranking_summary")),
            created_at=str(data.get("created_at") or utc_now()),
        )


def build_generation_plan(
    *,
    round_index: int,
    candidates: list[CandidateGenome],
    fate_assignments: list[FateAssignment] | list[dict[str, Any]],
    ranking: Any | None = None,
    parent_ids: list[str] | None = None,
    mutation_objectives: list[str] | None = None,
    archive_writes: list[dict[str, Any]] | None = None,
    stage_graph: list[dict[str, Any]] | None = None,
    source: str = "runtime_default_generation_transition",
) -> GenerationPlan:
    """Build and validate the authoritative transition for a generation."""

    assignment_dicts = [_coerce_assignment_dict(item) for item in fate_assignments]
    stage_ops = _coerce_stage_graph(stage_graph or [])
    writes = [dict(item) for item in archive_writes] if archive_writes is not None else _archive_writes_from_assignments(assignment_dicts)
    payload = {
        "round_index": int(round_index or 0),
        "source": str(source or "runtime_default_generation_transition"),
        "fate_assignments": assignment_dicts,
        "parent_ids": coerce_str_list(parent_ids),
        "mutation_objectives": coerce_str_list(mutation_objectives),
        "archive_writes": writes,
        "stage_graph": stage_ops,
        "ranking_summary": _ranking_summary(ranking),
    }
    plan = GenerationPlan(plan_id=stable_hash(payload)[:20], **payload)
    validate_generation_plan(plan, candidates)
    return plan


def validate_generation_plan(plan: GenerationPlan, candidates: list[CandidateGenome]) -> None:
    """Reject plans that cannot truthfully describe the candidate transition."""

    candidate_ids = {candidate.id for candidate in candidates}
    seen: set[str] = set()
    assignments: list[FateAssignment] = []
    for item in plan.fate_assignments:
        assignment = _coerce_fate_assignment(item)
        if not assignment.candidate_id:
            raise GenerationPlanError("generation plan contains fate assignment without candidate_id")
        if assignment.candidate_id not in candidate_ids:
            raise GenerationPlanError(f"generation plan references unknown candidate: {assignment.candidate_id}")
        if assignment.candidate_id in seen:
            raise GenerationPlanError(f"generation plan contains duplicate fate assignment: {assignment.candidate_id}")
        seen.add(assignment.candidate_id)
        assignments.append(assignment)
    missing = sorted(candidate_ids - seen)
    if missing:
        raise GenerationPlanError(f"generation plan missing fate assignment for candidates: {', '.join(missing[:8])}")
    for parent_id in plan.parent_ids:
        if parent_id not in candidate_ids:
            raise GenerationPlanError(f"generation plan references unknown parent: {parent_id}")
    assigned_ids = {assignment.candidate_id for assignment in assignments}
    for write in plan.archive_writes:
        candidate_id = str(write.get("candidate_id") or "")
        if candidate_id and candidate_id not in assigned_ids:
            raise GenerationPlanError(f"generation plan archive write is not backed by fate assignment: {candidate_id}")
    _validate_stage_graph(plan.stage_graph)


def validate_generation_plan_record(plan_data: dict[str, Any]) -> GenerationPlan:
    """Validate a persisted generation plan without trusting live objects."""

    plan = GenerationPlan.from_dict(plan_data)
    if not plan.plan_id:
        raise GenerationPlanError("persisted generation plan is missing plan_id")
    expected = expected_generation_plan_id(plan)
    if plan.plan_id != expected:
        raise GenerationPlanError(f"persisted generation plan_id mismatch: {plan.plan_id} != {expected}")
    assignment_ids = _validate_persisted_assignments(plan)
    _validate_stage_graph(plan.stage_graph)
    for parent_id in plan.parent_ids:
        if parent_id not in assignment_ids:
            raise GenerationPlanError(f"persisted generation plan references unknown parent: {parent_id}")
    for write in plan.archive_writes:
        candidate_id = str(write.get("candidate_id") or "")
        if candidate_id and candidate_id not in assignment_ids:
            raise GenerationPlanError(f"persisted generation plan archive write is not backed by fate assignment: {candidate_id}")
    _validate_completed_stage_ops(plan, coerce_str_list(plan_data.get("completed_stage_ops")))
    return plan


def validate_generation_plan_history(
    records: list[dict[str, Any]],
    *,
    archive_history: list[dict[str, Any]] | None = None,
) -> None:
    """Replay persisted generation-plan evidence enough to reject corrupt resumes."""

    archive_plan_ids = {
        str(item.get("generation_plan_id") or "")
        for item in archive_history or []
        if isinstance(item, dict) and item.get("generation_plan_id")
    }
    for record in records:
        if not isinstance(record, dict):
            continue
        plan_data = record.get("generation_plan")
        if not isinstance(plan_data, dict) or not plan_data:
            continue
        plan = validate_generation_plan_record(plan_data)
        if archive_plan_ids and plan.plan_id not in archive_plan_ids:
            raise GenerationPlanError(f"persisted generation plan has no matching archive history entry: {plan.plan_id}")


def expected_generation_plan_id(plan: GenerationPlan) -> str:
    payload = {
        "round_index": int(plan.round_index or 0),
        "source": str(plan.source or "runtime_default_generation_transition"),
        "fate_assignments": [_coerce_assignment_dict(item) for item in plan.fate_assignments],
        "parent_ids": coerce_str_list(plan.parent_ids),
        "mutation_objectives": coerce_str_list(plan.mutation_objectives),
        "archive_writes": [dict(item) for item in plan.archive_writes],
        "stage_graph": _coerce_stage_graph(plan.stage_graph),
        "ranking_summary": coerce_dict(plan.ranking_summary),
    }
    return stable_hash(payload)[:20]


def apply_generation_plan(plan: GenerationPlan, candidates: list[CandidateGenome], archives: ArchiveManager) -> list[FateAssignment]:
    """Apply a validated transition plan and annotate durable evidence."""

    validate_generation_plan(plan, candidates)
    assert_stage_ready(plan, "archive_update", ["critique_and_verify", "rank", "archive_assign", "generation_plan_validate"])
    candidate_by_id = {candidate.id: candidate for candidate in candidates}
    assignments: list[FateAssignment] = []
    for item in plan.fate_assignments:
        assignment = _coerce_fate_assignment(item)
        candidate = candidate_by_id[assignment.candidate_id]
        candidate.metadata["generation_plan_id"] = plan.plan_id
        candidate.metadata["generation_plan_source"] = plan.source
        candidate.metadata["generation_plan_round"] = int(plan.round_index or 0)
        candidate.metadata["generation_plan_fate"] = assignment.fate
        assignments.append(assignment)
    applied = archives.update(assignments, candidates=candidates)
    if archives.history:
        archives.history[-1]["generation_plan_id"] = plan.plan_id
        archives.history[-1]["generation_plan_round"] = int(plan.round_index or 0)
        archives.history[-1]["generation_plan_source"] = plan.source
        archives.history[-1]["stage_graph"] = [dict(item) for item in plan.stage_graph]
    return applied


def assert_stage_ready(plan: GenerationPlan, op: str, completed_stage_ops: list[str] | None = None) -> None:
    """Assert that ``op`` is authorized by the plan and its prerequisites ran."""

    if not plan.stage_graph:
        return
    stage_op = str(op or "").strip()
    plan_ops = [str(stage.get("op") or "").strip() for stage in plan.stage_graph]
    if stage_op not in plan_ops:
        raise GenerationPlanError(f"generation plan does not authorize stage op: {stage_op}")
    completed = set(completed_stage_ops or [])
    unknown_completed = sorted(item for item in completed if item not in KNOWN_STAGE_OPS)
    if unknown_completed:
        raise GenerationPlanError(f"generation plan completed unknown stage op: {', '.join(unknown_completed[:8])}")
    missing = sorted(STAGE_PREREQUISITES.get(stage_op, set()) - completed)
    if missing:
        raise GenerationPlanError(f"generation plan stage op missing completed prerequisite for {stage_op}: {', '.join(missing)}")


def _coerce_assignment_dict(value: FateAssignment | dict[str, Any]) -> dict[str, Any]:
    assignment = _coerce_fate_assignment(value)
    return assignment.to_dict()


def _coerce_fate_assignment(value: FateAssignment | dict[str, Any]) -> FateAssignment:
    data = value.to_dict() if isinstance(value, FateAssignment) else dict(value)
    candidate_id = str(data.get("candidate_id") or "")
    fate = CandidateFate.normalize(data.get("fate"), default="")
    if fate not in CandidateFate.ALL:
        raise GenerationPlanError(f"generation plan contains unknown fate: {data.get('fate')}")
    return FateAssignment(
        candidate_id=candidate_id,
        fate=fate,
        archive_targets=coerce_str_list(data.get("archive_targets")),
        failure_signature=str(data.get("failure_signature") or ""),
        inherited_gene_summary=str(data.get("inherited_gene_summary") or ""),
        covered_by=str(data.get("covered_by") or ""),
        future_reactivation_condition=str(data.get("future_reactivation_condition") or ""),
    )


def _coerce_stage_graph(value: Any) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    for item in value or []:
        if not isinstance(item, dict):
            raise GenerationPlanError("generation plan stage must be an object")
        stage = dict(item)
        stage["op"] = str(stage.get("op") or "").strip()
        stages.append(stage)
    return stages


def _validate_stage_graph(stages: list[dict[str, Any]]) -> None:
    previous_index = -1
    seen: set[str] = set()
    for stage in stages:
        op = str(stage.get("op") or "").strip()
        if not op:
            raise GenerationPlanError("generation plan stage is missing op")
        if op not in KNOWN_STAGE_OPS:
            raise GenerationPlanError(f"generation plan stage op is not registered: {op}")
        if op in seen:
            raise GenerationPlanError(f"generation plan stage op is duplicated: {op}")
        current_index = _STAGE_INDEX[op]
        if current_index < previous_index:
            raise GenerationPlanError(f"generation plan stage op is out of order: {op}")
        missing = sorted(STAGE_PREREQUISITES.get(op, set()) - seen)
        if missing:
            raise GenerationPlanError(f"generation plan stage op missing prerequisite for {op}: {', '.join(missing)}")
        seen.add(op)
        previous_index = current_index


def _validate_persisted_assignments(plan: GenerationPlan) -> set[str]:
    if not plan.fate_assignments:
        raise GenerationPlanError("persisted generation plan has no fate assignments")
    seen: set[str] = set()
    for item in plan.fate_assignments:
        assignment = _coerce_fate_assignment(item)
        if not assignment.candidate_id:
            raise GenerationPlanError("persisted generation plan contains fate assignment without candidate_id")
        if assignment.candidate_id in seen:
            raise GenerationPlanError(f"persisted generation plan contains duplicate fate assignment: {assignment.candidate_id}")
        seen.add(assignment.candidate_id)
    return seen


def _validate_completed_stage_ops(plan: GenerationPlan, completed: list[str]) -> None:
    if not completed:
        return
    if not plan.stage_graph:
        raise GenerationPlanError("persisted generation plan completed stages require stage_graph")
    done: list[str] = []
    for op in completed:
        if op in done:
            raise GenerationPlanError(f"persisted generation plan completed stage is duplicated: {op}")
        assert_stage_ready(plan, op, done)
        done.append(op)


def _archive_writes_from_assignments(assignments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": str(item.get("candidate_id") or ""),
            "fate": CandidateFate.normalize(item.get("fate"), default=""),
            "archive_targets": coerce_str_list(item.get("archive_targets")),
        }
        for item in assignments
    ]


def _ranking_summary(ranking: Any | None) -> dict[str, Any]:
    if ranking is None:
        return {}
    if hasattr(ranking, "to_dict"):
        data = ranking.to_dict()
    elif isinstance(ranking, dict):
        data = dict(ranking)
    else:
        return {"raw_type": ranking.__class__.__name__}
    keys = [
        "best_final_answer_id",
        "strongest_mechanism_id",
        "mutation_worthy_ids",
        "edge_value_ids",
        "auxiliary_ids",
        "dormant_ids",
        "crossover_pairs",
        "preserve_incomplete_ids",
        "raw_notes",
    ]
    return {key: data.get(key) for key in keys if key in data}


__all__ = [
    "GenerationPlan",
    "GenerationPlanError",
    "KNOWN_STAGE_OPS",
    "STAGE_ORDER",
    "STAGE_PREREQUISITES",
    "apply_generation_plan",
    "assert_stage_ready",
    "build_generation_plan",
    "expected_generation_plan_id",
    "validate_generation_plan",
    "validate_generation_plan_history",
    "validate_generation_plan_record",
]
