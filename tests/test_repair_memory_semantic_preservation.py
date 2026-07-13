from __future__ import annotations

from pathlib import Path

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.archives.types import FateAssignment
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationOperator, MutationPlan
from cognitive_evolve_runtime.nexus.failure_classifier import FailureVerdict, classify_candidate_failure
from cognitive_evolve_runtime.nexus.project_verification import _route_failed_project_candidate
from cognitive_evolve_runtime.nexus.repair_reactivation import RepairSeed, _tokenize_family
from cognitive_evolve_runtime.nexus.reproduction import _mark_offspring_repair_incubating


def _guidance(index: int, targets: list[str]) -> dict[str, object]:
    return {
        "blocker": f"blocker-{index}",
        "next_action": f"repair-action-{index}",
        "evidence_needed": [f"evidence-{index}"],
        "source_bindings": [{"path": path, "kind": "source_file"} for path in targets],
        "disallowed_repeat_pattern": f"repeat-{index}",
    }


def test_repair_seed_metadata_preserves_every_semantic_entry() -> None:
    targets = [f"target-{index}.py" for index in range(9)]
    blockers = [f"blocker-{index}" for index in range(9)]
    guidance = [_guidance(index, targets) for index in range(9)]
    lessons = [f"lesson-{index}" for index in range(9)]
    candidate = CandidateGenome(id="repair-seed", failure_lessons=lessons)
    verdict = FailureVerdict(
        category="repairable_patch_syntax_or_context",
        repairable=True,
        reason="repair",
        blockers=blockers,
        repair_targets=targets,
        diagnostics=blockers,
        failure_signature="signature",
        failure_guidance=guidance,
    )

    metadata = RepairSeed(candidate, verdict, "dormant_archive", ("family", "target", "repair")).to_metadata(current_round=1)

    assert metadata["target_files"] == targets
    assert metadata["blockers"] == blockers
    assert metadata["failure_lessons"] == lessons
    assert metadata["repair_guidance"] == guidance
    assert metadata["disallowed_repeat_patterns"] == [f"repeat-{index}" for index in range(9)]
    assert metadata["required_evidence"] == [f"evidence-{index}" for index in range(9)]


def test_failure_classifier_preserves_nth_plus_one_blocker_target_and_signature() -> None:
    targets = [f"module-{index}.py" for index in range(8)]
    blockers = [f"patch_application_failed:{index}" for index in range(8)]
    first = CandidateGenome(id="first", failure_lessons=blockers, source_bindings=[{"path": path} for path in targets])
    second = CandidateGenome(id="second", failure_lessons=[*blockers[:-1], "patch_application_failed:tail-b"], source_bindings=[{"path": path} for path in targets])

    first_verdict = classify_candidate_failure(first)
    second_verdict = classify_candidate_failure(second)

    assert first_verdict.blockers == blockers
    assert first_verdict.repair_targets == targets
    assert [item["blocker"] for item in first_verdict.failure_guidance] == blockers
    assert first_verdict.failure_guidance[-1]["source_bindings"][-1]["path"] == targets[-1]
    assert targets[-1] in first_verdict.failure_guidance[-1]["next_action"]
    assert first_verdict.failure_signature != second_verdict.failure_signature
    assert len(first_verdict.failure_signature) == 64


def test_repair_lanes_preserve_all_blockers_targets_and_guidance(tmp_path: Path) -> None:
    targets = [f"module-{index}.py" for index in range(9)]
    for target in targets:
        (tmp_path / target).write_text("value = 1\n", encoding="utf-8")
    blockers = [f"patch_application_failed:{index}" for index in range(9)]
    guidance = [_guidance(index, targets) for index in range(9)]
    verdict = FailureVerdict(
        category="repairable_patch_syntax_or_context",
        repairable=True,
        reason="repair",
        blockers=blockers,
        repair_targets=targets,
        diagnostics=blockers,
        failure_signature="signature",
        failure_guidance=guidance,
    )
    offspring = CandidateGenome(id="offspring", failure_lessons=blockers)

    _mark_offspring_repair_incubating(offspring, {}, verdict)

    assert offspring.metadata["repair_required"]["blockers"] == blockers
    assert [item["blocker"] for item in offspring.metadata["failure_micro_guidance"]] == [f"blocker-{index}" for index in range(9)]
    assert [item["path"] for item in offspring.metadata["repair_required"]["source_bindings"]] == targets

    project = CandidateGenome(
        id="project",
        artifact_type="code_patch",
        failure_lessons=blockers,
        source_bindings=[{"path": path, "kind": "source_file"} for path in targets],
    )
    _route_failed_project_candidate(
        project,
        {
            "source_root": str(tmp_path),
            "patch_result": {"status": "failed", "diagnostics": blockers, "failed_files": targets},
        },
    )

    assert project.metadata["repair_required"]["blockers"][: len(blockers)] == blockers
    assert project.metadata["repair_required"]["blockers"][-1] == targets[-1]
    assert [item["blocker"] for item in project.metadata["failure_micro_guidance"]] == blockers
    assert [item["path"] for item in project.metadata["repair_required"]["source_bindings"]] == targets


def test_mutation_repair_note_preserves_all_blockers_and_acceptance_criteria() -> None:
    long_blocker = "long-blocker:" + ("X" * 2_000) + ":END"
    blockers = [f"blocker-{index}" for index in range(5)] + [long_blocker]
    criteria = [f"criterion-{index}" for index in range(5)] + ["criterion-tail"]
    parent = CandidateGenome(id="parent", artifact="base artifact")
    plan = MutationPlan(
        operator=MutationOperator.REPAIR,
        metadata={"repair_required": {"blockers": blockers, "acceptance_criteria": criteria}},
    )

    child = MutationEngine().mutate(parent, plan)

    assert long_blocker in child.artifact
    assert criteria[-1] in child.artifact
    assert child.metadata["repair_required"]["blockers"] == blockers
    assert child.metadata["repair_required"]["acceptance_criteria"] == criteria


def test_archive_retains_every_failure_lesson_without_character_or_count_clipping() -> None:
    long_lesson = "long-lesson:" + ("Y" * 3_000) + ":END"
    lessons = [f"lesson-{index}" for index in range(6)] + [long_lesson]
    candidate = CandidateGenome(id="failed", failure_lessons=lessons, current_fate=CandidateFate.FAILED.value)
    archives = ArchiveManager()

    archives.update([FateAssignment(candidate.id, CandidateFate.FAILED.value)], candidates=[candidate])

    lesson_records = [item for item in archives.constraint_records if item["kind"] == "failure_lesson_constraint"]
    assert [item["rule"] for item in lesson_records] == lessons
    assert lesson_records[-1]["rule"].endswith(":END")


def test_archive_constraint_history_is_not_silently_dropped_after_200_records() -> None:
    archives = ArchiveManager()
    for index in range(205):
        candidate = CandidateGenome(
            id=f"failed-{index}",
            failure_lessons=[f"lesson-{index}"],
            current_fate=CandidateFate.FAILED.value,
        )
        archives.update([FateAssignment(candidate.id, CandidateFate.FAILED.value)], candidates=[candidate])

    assert len(archives.constraint_records) == 205
    assert archives.constraint_records[0]["rule"] == "lesson-0"
    assert archives.constraint_records[-1]["rule"] == "lesson-204"


def test_failure_archive_diversity_identity_hashes_the_complete_canonical_value() -> None:
    prefix = "shared-" + ("Z" * 400)

    assert _tokenize_family(prefix + "-tail-a") != _tokenize_family(prefix + "-tail-b")
