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
from cognitive_evolve_runtime.nexus.live_store import LiveNexusStore
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.runtime import NexusRunResult, NexusRuntime
from cognitive_evolve_runtime.persistence.checkpoint import CheckpointStore
from cognitive_evolve_runtime.persistence.transactional_snapshot import NexusSnapshotTransaction, SnapshotWrite


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
        best_id = candidates[0].id if candidates else ""
        return {
            "best_final_answer_id": best_id,
            "strongest_mechanism_id": best_id,
            "mutation_worthy_ids": [best_id] if best_id else [],
            "edge_value_ids": [candidate.id for candidate in candidates if candidate.edge_knowledge_seeds],
            "auxiliary_ids": [],
            "dormant_ids": [],
            "dominated_pairs": [],
            "crossover_pairs": [],
            "preserve_incomplete_ids": [],
            "pairwise_preferences": [],
            "multihead_observations": {},
        }

    def generate_offspring(self, *, plans: list[Any], parents: list[CandidateGenome], **_: Any) -> list[dict[str, Any]]:
        parent_by_id = {parent.id: parent for parent in parents}
        offspring: list[dict[str, Any]] = []
        for index, plan in enumerate(plans):
            parent_ids = list(getattr(plan, "parent_ids", []) or [])
            parent = parent_by_id[parent_ids[0]]
            plan_id = str((getattr(plan, "metadata", {}) or {}).get("plan_id") or "")
            offspring.append(
                {
                    "id": f"{parent.id}-model-g{parent.generation + 1}-{index}",
                    "parent_ids": parent_ids,
                    "artifact": f"model refinement of {parent.artifact}",
                    "artifact_type": parent.artifact_type,
                    "concise_claim": f"model refinement of {parent.concise_claim}",
                    "core_mechanism": "model_refinement",
                    "metadata": {"plan_id": plan_id},
                }
            )
        return offspring

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


class SeedQuotaModel(NarrowSeedModel):
    def __init__(self) -> None:
        self.seed_calls = 0

    def seed_population(self, **_: Any) -> list[dict[str, Any]]:
        self.seed_calls += 1
        raise LLMResponseError("You've hit your usage limit. Try again later.")


class EmptySeedModel(NarrowSeedModel):
    def seed_population(self, **_: Any) -> list[dict[str, Any]]:
        return []


class PartialSeedQuotaModel(NarrowSeedModel):
    def __init__(self) -> None:
        self.seed_calls: list[int] = []

    def seed_population(self, *, policy: Any, **_: Any) -> list[dict[str, Any]]:
        batch = int((policy.metadata or {}).get("seed_batch_index") or 0)
        self.seed_calls.append(batch)
        if batch == 1:
            raise LLMResponseError("You've hit your usage limit. Try again later.")
        return [
            {
                "id": f"S{batch}",
                "generation": 0,
                "artifact": f"model seed {batch}",
                "artifact_type": "answer",
                "concise_claim": f"model seed {batch}",
                "core_mechanism": "model_route",
            }
        ]


class ResumeWithoutReseedModel(NarrowSeedModel):
    def seed_population(self, **_: Any) -> list[dict[str, Any]]:
        raise AssertionError("partial quota resume must not reseed")

    def relative_rank(self, *, candidates: list[CandidateGenome], **_: Any) -> dict[str, Any]:
        candidate_id = candidates[0].id if candidates else ""
        return {
            "best_final_answer_id": candidate_id,
            "strongest_mechanism_id": candidate_id,
            "mutation_worthy_ids": [candidate_id] if candidate_id else [],
            "edge_value_ids": [],
            "auxiliary_ids": [],
            "dormant_ids": [],
            "dominated_pairs": [],
            "crossover_pairs": [],
            "preserve_incomplete_ids": [],
            "pairwise_preferences": [],
            "multihead_observations": {},
        }


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
        expected_minimum = 1 if mode == "text" else 3
        assert len(checkpoint["population"]["candidates"]) >= expected_minimum
        assert len(restored["population"].candidates) >= expected_minimum
        if mode == "text":
            assert [candidate["metadata"]["model_claimed_candidate_id"] for candidate in checkpoint["population"]["candidates"]] == ["M0"]


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


def test_resume_reads_checkpoint_from_latest_live_generation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_resume_fixture(tmp_path, stop_reason="", checkpoint_round=1, checkpoint_max_rounds=3, phase="round_end")
    NexusSnapshotTransaction(tmp_path).commit(
        [
            SnapshotWrite("checkpoint.json", "json", json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))),
            SnapshotWrite("run-result.json", "json", json.loads((tmp_path / "run-result.json").read_text(encoding="utf-8"))),
        ]
    )
    contract = NexusObjectiveContract(original_user_goal="resume goal", normalized_goal="resume goal")
    LiveNexusStore(tmp_path, mode="text", contract=contract, world={"kind": "text"}, max_rounds=3)(
        {
            "population": CandidatePopulation([CandidateGenome(id="C2", concise_claim="latest")]),
            "archives": ArchiveManager(),
            "policy": EvolutionPolicy(),
            "phase": "round_end",
            "round": 2,
            "progress_event": {"type": "evolution_progress", "round": 2},
        }
    )
    calls: list[int] = []

    class ReachedEvolveOnce(Exception):
        pass

    def stop_at_evolve_once(**kwargs: Any) -> None:
        calls.append(kwargs["budget"].current_round)
        raise ReachedEvolveOnce

    monkeypatch.setattr("cognitive_evolve_runtime.nexus.runtime.evolve_once", stop_at_evolve_once)

    with pytest.raises(ReachedEvolveOnce):
        NexusRuntime(output_dir=tmp_path).resume_from_checkpoint(max_rounds=4)

    assert calls == [2]


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


def test_nexus_keeps_successful_model_seed_pool_pure_and_undersized(tmp_path: Path) -> None:
    result = NexusRuntime(model=NarrowSeedModel(), output_dir=tmp_path).run_text(
        "Solve a hard math problem.",
        max_rounds=1,
        min_population_size=12,
    )

    candidates = result.evolution["population"]["candidates"]
    assert [candidate["metadata"]["model_claimed_candidate_id"] for candidate in candidates] == ["M0"]
    assert all(not candidate.get("metadata", {}).get("search_seed_not_final") for candidate in candidates)


def test_model_offspring_failure_checkpoints_without_deterministic_mutation(tmp_path: Path) -> None:
    model = BadOffspringModel()
    result = NexusRuntime(model=model, output_dir=tmp_path).run_text(
        "Keep evolving even if model offspring response is malformed.",
        max_rounds=2,
        min_population_size=10,
    )

    assert result.evolution["interrupted"] is True
    assert result.evolution["completion_status"] == "interrupted_checkpointed"
    assert result.evolution["stop_reason"] == "model_schema_repair_checkpointed"
    assert "interrupted before final convergence" in result.final_answer
    assert model.offspring_attempted is True
    assert not any(candidate["generation"] > 0 for candidate in result.evolution["population"]["candidates"])


def test_empty_model_seed_checkpoints_without_mechanical_candidate(tmp_path: Path) -> None:
    result = NexusRuntime(model=EmptySeedModel(), output_dir=tmp_path).run_text(
        "Do not fabricate a candidate when the configured model returns no seed.",
        max_rounds=2,
        min_population_size=4,
    )

    assert result.evolution["interrupted"] is True
    assert result.evolution["completion_status"] == "interrupted_checkpointed"
    assert result.evolution["stop_reason"] == "model_boundary_error_checkpointed"
    assert result.evolution["population"]["candidates"] == []
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["round"] == 0
    assert any(path.name.startswith("round-0000-error_checkpoint") for path in (tmp_path / "rounds").iterdir())

    resumed = NexusRuntime(model=NarrowSeedModel(), output_dir=tmp_path).resume_from_checkpoint(max_rounds=3)
    resumed_candidates = resumed.evolution["population"]["candidates"]
    assert any(candidate.get("metadata", {}).get("model_claimed_candidate_id") == "M0" for candidate in resumed_candidates)
    assert not any(
        candidate.get("metadata", {}).get("exploration_source") in {"emergency_activation_reseed", "activation_repair_seed"}
        for candidate in resumed_candidates
    )


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
    assert [candidate["metadata"]["model_claimed_candidate_id"] for candidate in checkpoint["population"]["candidates"]] == ["M0"]
    assert (tmp_path / "candidate-journal.jsonl").exists()
    assert any(path.name.startswith("round-0001-error_checkpoint") for path in (tmp_path / "rounds").iterdir())


def test_seed_usage_limit_pauses_after_first_model_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "1")
    model = SeedQuotaModel()
    result = NexusRuntime(model=model, output_dir=tmp_path).run_text(
        "Hard problem that hits usage quota during seeding.",
        max_rounds=4,
        min_population_size=4,
    )

    assert model.seed_calls == 1
    assert result.evolution["interrupted"] is True
    assert result.evolution["completion_status"] == "paused_quota"
    assert result.evolution["synthesis"]["status"] == "paused_quota"
    assert result.evolution["stop_reason"] == "model_quota_pause_checkpointed"
    assert result.evolution["population"]["candidates"] == []
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["round"] == 0
    assert checkpoint["budget"]["current_round"] == 0
    assert any(path.name.startswith("round-0000-error_checkpoint") for path in (tmp_path / "rounds").iterdir())

    resumed = NexusRuntime(model=NarrowSeedModel(), output_dir=tmp_path).resume_from_checkpoint(max_rounds=5)
    assert resumed.evolution["completion_status"] != "paused_quota"
    assert resumed.evolution["current_round"] > 0
    assert any(candidate.get("metadata", {}).get("model_claimed_candidate_id") == "M0" for candidate in resumed.evolution["population"]["candidates"])


def test_single_seed_batch_avoids_opening_a_second_quota_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "1")
    model = PartialSeedQuotaModel()
    result = NexusRuntime(model=model, output_dir=tmp_path).run_text(
        "Hard problem that hits quota after one accepted model seed.",
        max_rounds=4,
        min_population_size=8,
    )

    assert model.seed_calls == [0]
    assert result.evolution["completion_status"] != "paused_quota"
    candidates = result.evolution["population"]["candidates"]
    generation_zero = [candidate for candidate in candidates if candidate["generation"] == 0]
    assert [candidate["metadata"]["model_claimed_candidate_id"] for candidate in generation_zero] == ["S0"]
    assert any(candidate["generation"] > 0 for candidate in candidates)
    assert all(not candidate.get("metadata", {}).get("search_seed_not_final") for candidate in candidates)


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


def test_exhaustive_profile_carries_width_request_without_mechanical_seed_padding(monkeypatch, tmp_path: Path) -> None:
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
    generation_zero = [candidate for candidate in result.evolution["population"]["candidates"] if candidate["generation"] == 0]
    assert [candidate["metadata"]["model_claimed_candidate_id"] for candidate in generation_zero] == ["M0"]
    assert any(candidate["generation"] > 0 for candidate in result.evolution["population"]["candidates"])
