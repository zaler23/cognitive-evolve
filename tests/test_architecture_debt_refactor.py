from __future__ import annotations

from pathlib import Path
import json

import pytest

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidatePopulation
from cognitive_evolve_runtime.candidates.fate_machine import IllegalFateTransition, transition_candidate_fate, validate_transition
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.candidates.metadata_schema import metadata_audit, unknown_metadata_keys
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.fallbacks import finish_fallback_capture, record_fallback, start_fallback_capture
from cognitive_evolve_runtime.nexus.loop import EvolutionLoopResult
from cognitive_evolve_runtime.nexus import EvolutionBudget, NexusRuntime, evolve_once, seed_population
from cognitive_evolve_runtime.nexus._shared import MODEL_BOUNDARY_ERRORS, bounded_score, bounded_score_or_none, positive_int
from cognitive_evolve_runtime.nexus.budget_factory import evolution_budget_from_round_budget, route_incomplete_round_budget
from cognitive_evolve_runtime.nexus.budgeting import NexusRoundBudget
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.runtime import NexusRunResult, _attach_fallback_events
from cognitive_evolve_runtime.nexus.runtime_services import NexusPersistenceService
from cognitive_evolve_runtime.nexus.synthesis import SynthesizedResult

ROOT = Path(__file__).resolve().parents[1]


def test_nexus_loop_is_split_package_with_compat_exports() -> None:
    assert not (ROOT / "cognitive_evolve_runtime" / "nexus" / "loop.py").exists()
    loop_dir = ROOT / "cognitive_evolve_runtime" / "nexus" / "loop"
    expected = {"budget.py", "round.py", "controller.py", "seeding.py", "closure.py", "offspring.py", "policy_directives.py", "repair_guidance.py", "stage_helpers.py"}
    assert expected.issubset({path.name for path in loop_dir.glob("*.py")})
    assert EvolutionBudget.__name__ == "EvolutionBudget"
    assert NexusRuntime.__name__ == "NexusRuntime"
    assert callable(evolve_once)
    assert callable(seed_population)


def test_model_boundary_errors_are_centralized() -> None:
    occurrences = []
    for path in (ROOT / "cognitive_evolve_runtime").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "MODEL_BOUNDARY_ERRORS" in text and "LLMConfigurationError, LLMResponseError, ModelResponseSchemaError" in text:
            occurrences.append(path.relative_to(ROOT).as_posix())
    assert occurrences == ["cognitive_evolve_runtime/nexus/_shared.py"]
    assert len(MODEL_BOUNDARY_ERRORS) == 3


def test_shared_positive_int_and_bounded_score_semantics() -> None:
    assert positive_int("3") == 3
    assert positive_int(0) is None
    assert bounded_score(2.0) == 1.0
    assert bounded_score(-1.0) == 0.0
    assert bounded_score(float("nan"), default=0.25) == 0.25
    assert bounded_score_or_none(float("inf")) is None


def test_budget_factory_unifies_route_incomplete_shape() -> None:
    base = NexusRoundBudget(max_rounds=48, profile="deep", source="test", initial_candidate_count=7, mutation_branches_per_round=3, stop_policy="adaptive_until_solved", min_rounds_before_stop=4, adaptive=True, round_safety_limit=48, completion_requires_stop_signal=True)
    diagnostic = route_incomplete_round_budget(base)
    budget = evolution_budget_from_round_budget(diagnostic)

    assert diagnostic.max_rounds == 1
    assert diagnostic.stop_policy == "route_incomplete_single_diagnostic"
    assert budget.max_rounds == 1
    assert budget.branch_factor == 3
    assert budget.initial_candidate_count == 7
    assert budget.adaptive is False


def test_fate_machine_validates_terminal_and_semantic_transitions() -> None:
    assert validate_transition(CandidateFate.ACTIVE.value, CandidateFate.INCUBATING.value).target == CandidateFate.INCUBATING.value
    with pytest.raises(IllegalFateTransition):
        validate_transition(CandidateFate.CULLED.value, CandidateFate.ACTIVE.value)
    with pytest.raises(IllegalFateTransition):
        validate_transition(CandidateFate.ELITE.value, CandidateFate.INCUBATING.value)

    candidate = CandidateGenome(id="C-fate", current_fate=CandidateFate.FAILED.value)
    transition_candidate_fate(candidate, CandidateFate.DORMANT.value, reason="repair archive material")
    assert candidate.current_fate == CandidateFate.DORMANT.value
    assert candidate.metadata["fate_transition_history"][-1]["reason"] == "repair archive material"


def test_candidate_metadata_schema_audit_keeps_extension_visible() -> None:
    assert unknown_metadata_keys({"repair_required": {}, "new_experimental_key": True}) == ("new_experimental_key",)
    audit = metadata_audit({"repair_required": {}, "new_experimental_key": True})
    assert audit["has_unknown_keys"] is True
    assert "new_experimental_key" in audit["unknown_keys"]


def test_runtime_delegates_difficulty_estimation_to_focused_module() -> None:
    runtime_text = (ROOT / "cognitive_evolve_runtime" / "runtime.py").read_text(encoding="utf-8")
    estimator_text = (ROOT / "cognitive_evolve_runtime" / "nexus" / "difficulty_estimator.py").read_text(encoding="utf-8")

    assert "difficulty_estimator import" in runtime_text
    assert "_DIFFICULTY_ROUND_BANDS" not in runtime_text
    assert "_MODEL_CAPABILITY_SCORE_MULTIPLIERS" not in runtime_text
    assert "def _round_estimate_payload" not in runtime_text
    assert "runtime_round_budget" in estimator_text
    assert "_DIFFICULTY_ROUND_BANDS" in estimator_text


def test_model_adapter_facade_is_not_schema_or_repair_host() -> None:
    facade = ROOT / "cognitive_evolve_runtime" / "nexus" / "model_adapter.py"
    text = facade.read_text(encoding="utf-8")

    assert len(text.splitlines()) <= 90
    assert "StructuredModelAdapterCore" in text
    assert "model_adapter_facets" in text
    assert "Draft202012Validator" not in text
    assert "def _objective_contract_schema" not in text
    assert "def _repair_objective_contract_response" not in text
    assert (ROOT / "cognitive_evolve_runtime" / "nexus" / "model_adapter_core.py").exists()
    assert (ROOT / "cognitive_evolve_runtime" / "nexus" / "model_adapter_schemas.py").exists()
    assert (ROOT / "cognitive_evolve_runtime" / "nexus" / "model_adapter_repair.py").exists()
    assert (ROOT / "cognitive_evolve_runtime" / "nexus" / "model_adapter_facets").is_dir()


def test_stage_policy_is_package_with_compat_exports() -> None:
    assert not (ROOT / "cognitive_evolve_runtime" / "nexus" / "stage_policy.py").exists()
    stage_dir = ROOT / "cognitive_evolve_runtime" / "nexus" / "stage_policy"
    expected = {"__init__.py", "constants.py", "metrics.py", "types.py", "stages.py", "eligibility.py", "repair.py", "metadata.py"}
    assert expected.issubset({path.name for path in stage_dir.glob("*.py")})

    from cognitive_evolve_runtime.nexus.stage_policy import EligibilityDecision, RepairRequirement, annotate_stage_eligibility

    assert EligibilityDecision.__name__ == "EligibilityDecision"
    assert RepairRequirement.__name__ == "RepairRequirement"
    assert callable(annotate_stage_eligibility)


def test_archive_manager_uses_registry_not_duplicate_routing_helpers() -> None:
    manager_text = (ROOT / "cognitive_evolve_runtime" / "archives" / "manager.py").read_text(encoding="utf-8")
    registry_text = (ROOT / "cognitive_evolve_runtime" / "archives" / "registry.py").read_text(encoding="utf-8")

    assert "ArchiveRegistry(self).route_candidate" in manager_text
    assert "ArchiveRegistry(self).remove_candidate_from_archives" in manager_text
    assert "def _candidate_is_verified_dormant_frontier" not in manager_text
    assert "def _constraint_id" not in manager_text
    assert "def route_candidate" in registry_text
    assert "def remove_candidate_from_archives" in registry_text


def test_archive_manager_reads_legacy_snapshot_without_new_lane_payloads() -> None:
    legacy_candidate = CandidateGenome(id="legacy-elite", current_fate=CandidateFate.ELITE.value, concise_claim="legacy answer").to_dict()
    legacy = {
        "archive_schema": {"AnswerArchive": {"enabled": True}},
        "answer_archive": {"legacy-elite": legacy_candidate},
        "mechanism_archive": {},
        "novelty_archive": {},
        "project_patch_archive": {},
        "fates": {"legacy-elite": CandidateFate.ELITE.value},
        "history": [{"candidate_id": "legacy-elite", "fate": CandidateFate.ELITE.value}],
    }

    restored = ArchiveManager.from_dict(legacy)

    assert restored.answer_archive["legacy-elite"]["id"] == "legacy-elite"
    assert restored.fates["legacy-elite"] == CandidateFate.ELITE.value
    assert restored.latent_pareto_archive.summary()["candidates"] == 0
    assert restored.constraint_records == []


def test_fallback_events_are_captured_and_persisted(tmp_path: Path) -> None:
    token = start_fallback_capture()
    record_fallback(
        stage="model_schema_repair",
        reason="FixtureError",
        detail="/Users/private/project/path with secret API_KEY should be summarized",
        target="/Users/private/project/runtime-state.json",
    )
    events = finish_fallback_capture(token)

    evolution: dict[str, object] = {}
    _attach_fallback_events(evolution, events)
    run = NexusRunResult(mode="text", contract={}, policy={}, world={}, evolution=evolution)
    result = EvolutionLoopResult(
        population=CandidatePopulation(),
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        diagnosis=SearchDiagnosis(),
        synthesis=SynthesizedResult(status="completed", final_answer="candidate output"),
        current_round=0,
        max_rounds=1,
        stop_reason="max_rounds",
        completion_status="completed",
    )

    NexusPersistenceService(output_dir=tmp_path).persist(run, result, contract={}, world={}, budget_history=[], budget=EvolutionBudget(max_rounds=1))

    run_result = json.loads((tmp_path / "run-result.json").read_text(encoding="utf-8"))
    events_jsonl = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()

    assert run_result["evolution"]["fallback_event_count"] == 1
    persisted_event = run_result["evolution"]["fallback_events"][0]
    assert persisted_event["type"] == "nexus_fallback"
    assert "/Users/" not in json.dumps(persisted_event)
    assert "API_KEY" not in json.dumps(persisted_event)
    assert any(json.loads(line)["type"] == "nexus_fallback" for line in events_jsonl)
