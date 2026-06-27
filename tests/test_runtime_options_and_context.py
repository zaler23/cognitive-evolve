from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.nexus.context_protocol import ContextProtocolResult
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.fallbacks import capture_fallback_events
from cognitive_evolve_runtime.nexus.loop.budget import EvolutionBudget
from cognitive_evolve_runtime.nexus.model_routes import NexusModelRoutes
from cognitive_evolve_runtime.nexus.loop.seeding import _policy_for_seed_batch
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.runtime import NexusRuntime, _build_text_world_model
from cognitive_evolve_runtime.nexus.runtime_options import option_bool, restore_runtime_options
from cognitive_evolve_runtime.persistence.checkpoint import CheckpointStore, build_checkpoint_state
from cognitive_evolve_runtime.inputs.text_packet import TextInputPacket


class _Result:
    def __init__(self, *, population, archives, policy, budget):
        self.population = population
        self.archives = archives
        self.policy = policy
        self.diagnosis = SearchDiagnosis()
        self.progress_events = []
        self.pipeline_events = []
        self.budget_history = []
        self.adaptive_state = {}
        self.fabric_state = {}
        self.graded_output = {}
        self.current_round = 0
        self.max_rounds = budget.max_rounds
        self.interrupted = False
        self.stop_reason = "test_complete"
        self.completion_status = "completed"
        self.synthesis = SimpleNamespace(final_answer="ok", closure_certificate={})

    def to_dict(self):
        return {"synthesis": {"final_answer": "ok"}}


def test_run_project_passes_effective_options_and_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("x = 1\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class SeedModel:
        provided_context: dict[str, object] | None = None

        def seed_population(self, *, contract, world, policy, provided_context=None):
            self.provided_context = provided_context
            return []

    seed_model = SeedModel()

    def fake_evolve_once(**kwargs):
        captured.update(kwargs)
        return _Result(population=kwargs["population"], archives=kwargs["archives"], policy=kwargs["policy"], budget=kwargs["budget"])

    monkeypatch.setattr("cognitive_evolve_runtime.nexus.runtime.evolve_once", fake_evolve_once)
    monkeypatch.setattr(NexusRuntime, "_persist", lambda self, *args, **kwargs: {})
    seen_include_tests: list[bool] = []
    seen_overlays: list[dict[str, object]] = []

    def fake_verify(self, snapshot, candidates, *, include_tests=False, contract=None, applied_overlays=None):
        seen_include_tests.append(include_tests)
        seen_overlays.append(dict(applied_overlays or {}))
        return []

    monkeypatch.setattr(NexusRuntime, "_verify_project_population", fake_verify)
    run = NexusRuntime(model_routes=NexusModelRoutes(seed_model=seed_model), output_dir=tmp_path / "out").run_project(
        repo,
        user_goal="improve",
        include_tests=True,
        max_rounds=1,
        adaptive_config={"evidence": {"allow_docs_only": True}},
    )

    assert seen_include_tests == [True]
    assert seen_overlays and "artifact-policy" in seen_overlays[0]
    assert run.evolution["runtime_options"]["verification.include_tests"] is True
    provided = captured["provided_context"]
    assert isinstance(provided, dict)
    assert isinstance(provided["ContextProtocolResult"], ContextProtocolResult)
    assert provided["context_protocol"]["packets"]
    assert provided["source_context"]["slices"][0]["text"].startswith("x = 1")
    assert seed_model.provided_context is not None
    assert seed_model.provided_context["slices"][0]["text"].startswith("x = 1")


def test_checkpoint_roundtrips_runtime_options(tmp_path: Path) -> None:
    checkpoint = build_checkpoint_state(
        round=1,
        max_rounds=2,
        population=CandidatePopulation([CandidateGenome(id="C1")]),
        archives=ArchiveManager(),
        runtime_options={"verification.include_tests": True, "x.component": {"ok": 1}},
    )
    store = CheckpointStore(tmp_path / "checkpoint.json")
    store.save(checkpoint)
    restored = store.restore_state()
    assert restored is not None
    assert restored["runtime_options"]["verification.include_tests"] is True
    assert restore_runtime_options(persisted=restored["runtime_options"], overrides={"verification.include_tests": False})["verification.include_tests"] is False
    assert option_bool(restored["runtime_options"], "verification.include_tests") is True


def test_text_world_model_fallback_only_catches_declared_exceptions() -> None:
    packet = TextInputPacket.from_text("hello")

    class Undeclared:
        def build_text_world_model(self, *, packet):
            raise ValueError("programming bug")

    with pytest.raises(ValueError):
        _build_text_world_model(packet, model=Undeclared())

    class Declared:
        identity = "declared-world-builder"
        fallback_exceptions = (ValueError,)

        def build_text_world_model(self, *, packet):
            raise ValueError("provider degraded")

    with capture_fallback_events() as events:
        world = _build_text_world_model(packet, model=Declared())
    assert world.kind == "text"
    assert events and events[0]["stage"] == "text_world_model"
    assert events[0]["target"] == "declared-world-builder"


def test_seed_policy_prioritizes_undercovered_model_authored_families() -> None:
    policy = EvolutionPolicy(
        metadata={
            "search_space_plan": {
                "source": "model_authored_search_space",
                "candidate_families": [{"id": "covered"}, {"id": "rare"}],
            }
        }
    )
    accepted = [CandidateGenome(id="C1", metadata={"search_space": {"family_id": "covered"}})]
    batch_policy = _policy_for_seed_batch(policy, batch_index=2, accepted=accepted, rejected=[])
    priority = batch_policy.metadata["seed_family_priority"]
    assert priority[0]["id"] == "rare"
    assert batch_policy.metadata["seed_family_priority_source"] == "model_authored_search_space"
    assert batch_policy.metadata["seed_family_coverage_snapshot"] == {"covered": 1}


def test_runtime_no_longer_contains_contract_hash_rebase_helper() -> None:
    source = Path("cognitive_evolve_runtime/nexus/runtime.py").read_text(encoding="utf-8")
    assert "_rebase_population_contract_hashes" not in source
    assert "contract_hash_overlay_rebased" not in source
