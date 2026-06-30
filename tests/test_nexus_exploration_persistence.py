from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.engine.orchestrator import EngineOrchestrator
from cognitive_evolve_runtime.llm.env import LLMResponseError
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.runtime import NexusRunResult, NexusRuntime
from cognitive_evolve_runtime.persistence.checkpoint import CheckpointStore


class NarrowSeedModel:
    def seed_population(self, **_: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": "M0",
                "generation": 0,
                "artifact": "model seed",
                "artifact_type": "answer",
                "concise_claim": "model seed",
                "core_mechanism": "model_route",
                "multihead_scores": {"objective_alignment": 0.7, "answer_likelihood": 0.7},
            }
        ]

    def relative_rank(self, *, candidates: list[CandidateGenome], **_: Any) -> dict[str, Any]:
        return {
            "best_final_answer_id": "M0",
            "strongest_mechanism_id": "M0",
            "mutation_worthy_ids": ["M0"],
            "edge_value_ids": [candidate.id for candidate in candidates if candidate.edge_knowledge_seeds],
            "auxiliary_ids": [],
            "dormant_ids": [],
            "dominated_pairs": [],
            "crossover_pairs": [],
            "preserve_incomplete_ids": [],
            "pairwise_preferences": [],
            "multihead_observations": {},
        }

    def diagnose_search_state(self, **_: Any) -> dict[str, Any]:
        return {"stagnation_detected": False, "stagnation_type": "None", "recommended_actions": ["continue"]}

    def update_policy(self, *, policy: Any, **_: Any) -> dict[str, Any]:
        return policy.to_dict()

    def synthesize_result(self, *, population: list[CandidateGenome], **_: Any) -> dict[str, Any]:
        return {"status": "ok", "final_answer": f"synthesized from {len(population)} candidates"}


class BadOffspringModel(NarrowSeedModel):
    def __init__(self) -> None:
        self.offspring_attempted = False

    def generate_offspring(self, **_: Any) -> list[dict[str, Any]]:
        self.offspring_attempted = True
        raise LLMResponseError("offspring schema drift")


class InterruptingModel(NarrowSeedModel):
    def relative_rank(self, **_: Any) -> dict[str, Any]:
        raise LLMResponseError("quota exhausted during relative rank")


class DiagnoseInterruptsSecondRoundModel(NarrowSeedModel):
    def __init__(self) -> None:
        self.diagnose_calls = 0

    def diagnose_search_state(self, **_: Any) -> dict[str, Any]:
        self.diagnose_calls += 1
        if self.diagnose_calls >= 2:
            raise LLMResponseError("provider 5xx during nexus_diagnose_search_state")
        return super().diagnose_search_state(**_)


def _write_resume_fixture(
    out: Path,
    *,
    stop_reason: str,
    checkpoint_round: int = 1,
    checkpoint_max_rounds: int = 2,
    phase: str = "terminal",
    write_run_result: bool = True,
) -> dict[str, Any]:
    contract = NexusObjectiveContract(original_user_goal="resume goal", normalized_goal="resume goal")
    policy = EvolutionPolicy()
    world = {"kind": "text", "goal_summary": "resume goal"}
    population = CandidatePopulation(
        [
            CandidateGenome(
                id="C0",
                artifact="persisted answer",
                concise_claim="persisted answer",
                core_mechanism="fixture",
                multihead_scores={"objective_alignment": 0.9, "answer_likelihood": 0.9},
            )
        ]
    )
    CheckpointStore(out / "checkpoint.json").save_state(
        round=checkpoint_round,
        max_rounds=checkpoint_max_rounds,
        population=population,
        archives=ArchiveManager(),
        policy=policy,
        contract=contract,
        world=world,
        mode="text",
        progress_event={"type": "evolution_progress", "round": checkpoint_round, "max_rounds": checkpoint_max_rounds, "phase": phase},
        budget={"current_round": checkpoint_round, "max_rounds": checkpoint_max_rounds, "stop_reason": stop_reason},
        verification_plan={"verifier_id": "noop", "strength": "NONE", "modality": "none", "verifier_fingerprint": "fixture"},
    )
    payload = NexusRunResult(
        mode="text",
        contract=contract.to_dict(),
        policy=policy.to_dict(),
        world=world,
        evolution={"synthesis": {"final_answer": "persisted answer"}, "stop_reason": stop_reason, "persisted": True},
        artifacts={"run_result": str(out / "run-result.json")},
    ).to_dict()
    if write_run_result:
        (out / "run-result.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_post_seeding_checkpoint_survives_verification_synthesizer_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail_synthesize(self: object, *_args: Any, **_kwargs: Any) -> object:
        raise LLMResponseError("verification synthesize failed after seed")

    monkeypatch.setattr("cognitive_evolve_runtime.nexus.runtime.VerificationSynthesizer.synthesize", fail_synthesize)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def value():\n    return 1\n", encoding="utf-8")

    cases = [
        ("text", tmp_path / "text", lambda out: NexusRuntime(model=NarrowSeedModel(), output_dir=out).run_text("seed then fail", max_rounds=4, min_population_size=3)),
        ("project", tmp_path / "project", lambda out: NexusRuntime(output_dir=out).run_project(repo, user_goal="seed then fail", max_rounds=4, min_population_size=3)),
    ]
    for mode, out, run in cases:
        with pytest.raises(LLMResponseError, match="verification synthesize failed after seed"):
            run(out)

        checkpoint_path = out / "checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        restored = CheckpointStore(checkpoint_path).restore_state()
        assert restored is not None
        assert checkpoint["mode"] == mode
        assert checkpoint["round"] == 0
        assert checkpoint["progress_event"]["phase"] == "post_seeding"
        assert checkpoint["budget"]["current_round"] == 0
        assert checkpoint["policy"]["policy_id"] == "nexus-evolution-policy"
        assert checkpoint["archives"]["archive_schema"]
        assert len(checkpoint["population"]["candidates"]) >= 3
        assert len(restored["population"].candidates) >= 3


def test_terminal_resume_reuses_persisted_run_result_without_evolving(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[bool] = []
    payload = _write_resume_fixture(tmp_path, stop_reason="candidate_ready_for_external_review")

    def fail_evolve_once(**_: Any) -> None:
        calls.append(True)
        pytest.fail("terminal resume should not call evolve_once")

    monkeypatch.setattr("cognitive_evolve_runtime.nexus.runtime.evolve_once", fail_evolve_once)

    resumed = NexusRuntime(output_dir=tmp_path).resume_from_checkpoint(max_rounds=2)

    assert calls == []
    assert resumed.to_dict() == payload


def test_terminal_resume_extending_rounds_enters_evolve_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[int] = []
    _write_resume_fixture(tmp_path, stop_reason="candidate_ready_for_external_review")

    class ReachedEvolveOnce(Exception):
        pass

    def stop_at_evolve_once(**kwargs: Any) -> None:
        calls.append(kwargs["budget"].max_rounds)
        raise ReachedEvolveOnce

    monkeypatch.setattr("cognitive_evolve_runtime.nexus.runtime.evolve_once", stop_at_evolve_once)

    with pytest.raises(ReachedEvolveOnce):
        NexusRuntime(output_dir=tmp_path).resume_from_checkpoint(max_rounds=3)

    assert calls == [3]


def test_post_seeding_resume_does_not_use_terminal_short_circuit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[int] = []
    _write_resume_fixture(tmp_path, stop_reason="", checkpoint_round=0, phase="post_seeding")

    class ReachedEvolveOnce(Exception):
        pass

    def stop_at_evolve_once(**kwargs: Any) -> None:
        calls.append(kwargs["budget"].current_round)
        raise ReachedEvolveOnce

    monkeypatch.setattr("cognitive_evolve_runtime.nexus.runtime.evolve_once", stop_at_evolve_once)

    with pytest.raises(ReachedEvolveOnce):
        NexusRuntime(output_dir=tmp_path).resume_from_checkpoint(max_rounds=2)

    assert calls == [0]


def test_terminal_resume_requires_persisted_run_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[bool] = []
    _write_resume_fixture(tmp_path, stop_reason="candidate_ready_for_external_review", write_run_result=False)

    def fail_evolve_once(**_: Any) -> None:
        calls.append(True)
        pytest.fail("terminal resume should fail closed before evolve_once")

    monkeypatch.setattr("cognitive_evolve_runtime.nexus.runtime.evolve_once", fail_evolve_once)

    with pytest.raises(FileNotFoundError, match="terminal checkpoint resume requires persisted run-result.json"):
        NexusRuntime(output_dir=tmp_path).resume_from_checkpoint()

    assert calls == []


def test_nexus_amplifies_narrow_model_seed_pool_and_marks_search_seeds(tmp_path: Path) -> None:
    result = NexusRuntime(model=NarrowSeedModel(), output_dir=tmp_path).run_text(
        "Solve a hard math problem.",
        max_rounds=1,
        min_population_size=12,
    )

    candidates = result.evolution["population"]["candidates"]
    assert len(candidates) >= 8
    assert candidates[0]["id"] == "M0"
    assert any(candidate.get("metadata", {}).get("search_seed_not_final") for candidate in candidates)
    assert any("negative" in candidate.get("core_mechanism", "") or "dual" in candidate.get("core_mechanism", "") for candidate in candidates)


def test_model_offspring_failure_falls_back_to_deterministic_mutation(tmp_path: Path) -> None:
    model = BadOffspringModel()
    result = NexusRuntime(model=model, output_dir=tmp_path).run_text(
        "Keep evolving even if model offspring response is malformed.",
        max_rounds=2,
        min_population_size=10,
    )

    assert result.evolution.get("interrupted") is not True
    assert "interrupted before final convergence" not in result.final_answer
    assert model.offspring_attempted is True
    assert result.evolution["progress_events"][-1]["round"] == 2


def test_nexus_live_persistence_survives_mid_round_model_interruption(tmp_path: Path) -> None:
    result = NexusRuntime(model=InterruptingModel(), output_dir=tmp_path).run_text(
        "Hard problem that will hit quota.",
        max_rounds=4,
        min_population_size=10,
    )

    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert result.evolution["interrupted"] is True
    assert result.evolution["completion_status"] == "paused_quota"
    assert "paused on provider quota" in result.final_answer
    assert result.final_answer != "model seed"
    assert checkpoint["round"] == 1
    assert checkpoint["max_rounds"] == 4
    assert checkpoint["budget"]["current_round"] == 1
    assert len(checkpoint["population"]["candidates"]) >= 10
    assert (tmp_path / "candidate-journal.jsonl").exists()
    assert any(path.name.startswith("round-0001-error_checkpoint") for path in (tmp_path / "rounds").iterdir())


def test_error_checkpoint_reconciles_previous_progress_round_on_second_round_failure(tmp_path: Path) -> None:
    result = NexusRuntime(model=DiagnoseInterruptsSecondRoundModel(), output_dir=tmp_path).run_text(
        "Hard problem with provider 5xx during second-round diagnosis.",
        max_rounds=4,
        min_population_size=10,
    )

    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert result.evolution["interrupted"] is True
    assert result.evolution["completion_status"] == "interrupted_checkpointed"
    assert checkpoint["round"] == 2
    assert checkpoint["progress_event"]["round"] == 2
    assert checkpoint["progress_event"]["metadata"]["previous_progress_round"] == 1
    assert any(path.name.startswith("round-0002-error_checkpoint") for path in (tmp_path / "rounds").iterdir())
    assert (tmp_path / "run-result.json").exists()
    assert (tmp_path / "final-answer.md").exists()


def test_exhaustive_profile_carries_old_branch_and_candidate_width(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COGEV_NEXUS_PROFILE_EXHAUSTIVE_SAFETY_ROUNDS", "2")
    monkeypatch.setenv("COGEV_NEXUS_PROFILE_EXHAUSTIVE_MIN_CANDIDATES", "14")
    monkeypatch.setenv("COGEV_NEXUS_PROFILE_EXHAUSTIVE_BRANCH_FACTOR", "4")

    result = EngineOrchestrator(model=NarrowSeedModel()).run(
        "Difficult exhaustive task",
        context={
            "task_dir": str(tmp_path),
            "interface": "openai_compatible_api",
            "openai_compatible_model": "cognitive-evolve-one-shot-exhaustive",
            "evolution_profile": "exhaustive",
        },
    )

    budget = result.evolution["runtime_metadata"]["round_budget"]
    assert budget["initial_candidate_count"] == 14
    assert budget["mutation_branches_per_round"] == 4
    assert result.evolution["progress_events"][-1]["max_rounds"] == 2
    assert len(result.evolution["population"]["candidates"]) >= 14
