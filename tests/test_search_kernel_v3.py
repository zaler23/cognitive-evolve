from __future__ import annotations

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.archives.quality_diversity import QualityDiversityArchive
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.candidates.project_candidate import PatchOperation, ProjectCandidateGenome
from cognitive_evolve_runtime.nexus.loop.seeding import _generate_model_seed_batches
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.search_kernel.harvesting import _unbounded_seed_handoff_exhausted
from cognitive_evolve_runtime.nexus.search_kernel.descriptor_cells import behavior_descriptor, descriptor_cell_key
from cognitive_evolve_runtime.nexus.search_kernel.fingerprints import normalized_ast_signature
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector


class _SeedModel:
    def __init__(self) -> None:
        self.calls = 0

    def seed_population(self, *, contract, world, policy):
        self.calls += 1
        batch = getattr(policy, "metadata", {}).get("seed_batch_index", self.calls - 1)
        if batch == 0:
            return [
                {"id": "A", "artifact": "def f(x):\n    return x + 1", "concise_claim": "increment", "core_mechanism": "increment", "metadata": {}},
                {"id": "B", "artifact": "def g(y):\n    return y + 1", "concise_claim": "increment", "core_mechanism": "increment", "metadata": {}},
            ]
        return [
            {"id": "C", "artifact": "def h(y):\n    return y * 2", "concise_claim": "double", "core_mechanism": "doubling", "metadata": {}},
        ]


class _RoutedSeedModel(_SeedModel):
    metadata = {
        "transport": "cogev_llm_json",
        "model_spec": {"profile_id": "default-high", "provider": "openai", "model": "gpt-5.5"},
        "model_spec_hash": "llm-model-spec-test",
    }


class _LongTailSeedModel:
    def __init__(self, *, unique_batches: int) -> None:
        self.calls = 0
        self.unique_batches = unique_batches

    def seed_population(self, *, contract, world, policy):
        self.calls += 1
        batch = getattr(policy, "metadata", {}).get("seed_batch_index", self.calls - 1)
        if batch >= self.unique_batches:
            return []
        return [
            {
                "id": f"LT-{batch}",
                "artifact": f"long tail seed {batch}",
                "concise_claim": f"long tail mechanism {batch}",
                "core_mechanism": f"long tail mechanism {batch}",
                "metadata": {},
            }
        ]


class _MarginalFamilySeedModel:
    def __init__(self) -> None:
        self.calls = 0

    def seed_population(self, *, contract, world, policy):
        self.calls += 1
        batch = getattr(policy, "metadata", {}).get("seed_batch_index", self.calls - 1)
        out = []
        for offset in range(2):
            idx = batch * 2 + offset
            family = f"F{idx}" if batch < 3 else f"F{idx % 6}"
            out.append(
                {
                    "id": f"MG-{idx}",
                    "artifact": f"marginal family seed {idx}",
                    "concise_claim": f"marginal family mechanism {idx}",
                    "core_mechanism": f"marginal family mechanism {idx}",
                    "metadata": {"search_space": {"family_id": family}},
                }
            )
        return out


class _World:
    kind = "text"


class _Contract:
    objective = "write small numeric transforms"

    def to_dict(self):
        return {"objective": self.objective}



def test_behavior_descriptor_uses_search_space_and_real_project_paths() -> None:
    candidate = ProjectCandidateGenome(
        id="P3D",
        core_mechanism="fallback",
        metadata={"search_space": {"family_id": "adapter repair"}, "target_files": ["tests/test_adapter.py"]},
        source_bindings=[{"path": "cognitive_evolve_runtime/nexus/model_adapter.py"}],
        patch_set=[PatchOperation(path="cognitive_evolve_runtime/nexus/model_adapter.py", operation="append", content="# x")],
        patch_application_result={"status": "applied", "applied_files": ["cognitive_evolve_runtime/nexus/model_adapter.py"]},
    )

    descriptor = behavior_descriptor(candidate)

    assert descriptor[0].startswith("adapter_repair#m")
    assert any(item.startswith("src:") and "model_adapter" in item for item in descriptor)
    assert any(item.startswith("patch:") and "model_adapter" in item for item in descriptor)
    assert any(item.startswith("target:") and "test_adapter" in item for item in descriptor)
    assert "applied" in descriptor


def test_behavior_descriptor_initializes_canonical_family_without_nextgen_metadata() -> None:
    candidate = CandidateGenome(id="B1", core_mechanism="alpha clone", niche_memberships=["alpha"])

    descriptor = behavior_descriptor(candidate)

    assert descriptor[0].startswith("alpha#m")
    assert not descriptor[0].startswith("b1")
    assert candidate.metadata["nextgen"]["canonical_mechanism_family_id"] == descriptor[0]


def test_search_kernel_ast_signature_ignores_variable_rename() -> None:
    left = "def solve(x):\n    y = x + 1\n    return y\n"
    right = "def other(a):\n    b = a + 1\n    return b\n"
    assert normalized_ast_signature(left) == normalized_ast_signature(right)


def test_seed_generation_forces_multishot_and_dedupes_semantics(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_SEED_MIN_BATCHES", "2")
    model = _SeedModel()
    accepted, rejected, error = _generate_model_seed_batches(
        model=model,
        contract=_Contract(),
        world=_World(),
        policy=EvolutionPolicy(metadata={"initial_candidate_count": 1}),
        target_size=1,
    )
    assert error is None
    assert model.calls >= 2
    assert [candidate.id for candidate in accepted] == ["A", "C"]
    assert any(item["reason"] == "duplicate_semantic_signature" for item in rejected)
    assert all("descriptor_cell" in candidate.metadata for candidate in accepted)


def test_seed_generation_records_single_model_origin_without_changing_acceptance(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_SEED_MIN_BATCHES", "2")
    model = _RoutedSeedModel()

    accepted, rejected, error = _generate_model_seed_batches(
        model=model,
        contract=_Contract(),
        world=_World(),
        policy=EvolutionPolicy(metadata={"initial_candidate_count": 1}),
        target_size=1,
    )

    assert error is None
    assert [candidate.id for candidate in accepted] == ["A", "C"]
    assert rejected
    assert {candidate.metadata["origin_model"] for candidate in accepted} == {"openai/gpt-5.5"}
    assert {candidate.metadata["model_profile_id"] for candidate in accepted} == {"default-high"}
    assert {candidate.metadata["origin_model_spec_hash"] for candidate in accepted} == {"llm-model-spec-test"}


def test_parent_selector_uses_descriptor_diversity_over_top_clone() -> None:
    candidates = [
        CandidateGenome(id="A", core_mechanism="alpha", concise_claim="alpha", niche_memberships=["same"], lineage=["root1", "A"], multihead_scores={"objective_alignment": 1.0, "answer_likelihood": 1.0, "verifiability": 1.0}),
        CandidateGenome(id="B", core_mechanism="alpha", concise_claim="alpha variant", niche_memberships=["same"], lineage=["root1", "B"], multihead_scores={"objective_alignment": 0.99, "answer_likelihood": 0.99, "verifiability": 0.99}),
        CandidateGenome(id="C", core_mechanism="beta", concise_claim="beta", niche_memberships=["different"], lineage=["root2", "C"], multihead_scores={"objective_alignment": 0.70, "answer_likelihood": 0.70, "verifiability": 0.70, "novelty": 1.0}),
    ]
    selected = ParentSelector().select(candidates, ArchiveManager(), limit=2, eligibility_policy={"max_per_lineage": 1, "max_per_descriptor_cell": 1})
    assert {candidate.id for candidate in selected} == {"A", "C"}


def test_quality_diversity_archive_tracks_descriptor_cells_and_directive_boost() -> None:
    archive = QualityDiversityArchive()
    target = CandidateGenome(id="T", core_mechanism="immune", concise_claim="regression", niche_memberships=["immune"], multihead_scores={"frontier_score": 0.8})
    archive.update(target)
    cell = descriptor_cell_key(target)
    assert cell in archive.cell_elites
    result = archive.apply_directive({"kind": "rebalance", "descriptor": cell, "payload": {"reason": "sparse"}}, [target])
    assert result["changed"] is True
    assert archive.directive_boost(target) > 0.0

from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationPlan, MutationOperator
from cognitive_evolve_runtime.llm.env import LLMResponseError
from cognitive_evolve_runtime.nexus.loop.offspring import _generate_offspring, _mutation_plan_batch_limit, _offspring_batch_limit
from cognitive_evolve_runtime.nexus.loop.seeding import _seed_safety_batch_limit


class _PartialOffspringModel:
    def __init__(self, exc: Exception) -> None:
        self.calls = 0
        self.exc = exc

    def generate_offspring(self, *, plans, parents, world, contract, policy):
        self.calls += 1
        if self.calls == 1:
            return [CandidateGenome(id="M1", parent_ids=[parents[0].id], artifact="model child", concise_claim="model child", core_mechanism="model child")]
        raise self.exc


def test_offspring_partial_model_success_survives_recoverable_later_error(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_OFFSPRING_MIN_BATCHES", "2")
    monkeypatch.setenv("COGEV_NEXUS_OFFSPRING_BATCH_LIMIT", "2")
    parent = CandidateGenome(id="P", artifact="parent", concise_claim="parent", core_mechanism="parent")
    plan = MutationPlan(operator=MutationOperator.DEEPEN, parent_ids=["P"], instruction="deepen")
    offspring = _generate_offspring(
        model=_PartialOffspringModel(LLMResponseError("recoverable provider json error")),
        mutation_engine=MutationEngine(),
        parents=[parent],
        plans=[plan],
        world=_World(),
        contract=_Contract(),
        policy=EvolutionPolicy(),
    )
    assert any(candidate.id == "M1" for candidate in offspring)
    assert any(candidate.metadata.get("partial_model_offspring_error") for candidate in offspring)


def test_offspring_non_boundary_error_is_not_silently_fallback(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_OFFSPRING_MIN_BATCHES", "2")
    monkeypatch.setenv("COGEV_NEXUS_OFFSPRING_BATCH_LIMIT", "2")
    parent = CandidateGenome(id="P", artifact="parent", concise_claim="parent", core_mechanism="parent")
    plan = MutationPlan(operator=MutationOperator.DEEPEN, parent_ids=["P"], instruction="deepen")
    try:
        _generate_offspring(
            model=_PartialOffspringModel(ValueError("programming bug")),
            mutation_engine=MutationEngine(),
            parents=[parent],
            plans=[plan],
            world=_World(),
            contract=_Contract(),
            policy=EvolutionPolicy(),
        )
    except ValueError as exc:
        assert "programming bug" in str(exc)
    else:  # pragma: no cover - explicit failure message for regression readability
        raise AssertionError("non-boundary exception must propagate")


def test_batch_limit_environment_values_are_bounded(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_SEED_BATCH_LIMIT", "999")
    monkeypatch.setenv("COGEV_NEXUS_MUTATION_PLAN_BATCH_LIMIT", "999")
    monkeypatch.setenv("COGEV_NEXUS_OFFSPRING_BATCH_LIMIT", "999")
    assert _seed_safety_batch_limit(policy=EvolutionPolicy()) == 999
    assert _mutation_plan_batch_limit(3) == 16
    assert _offspring_batch_limit(3) == 16


def test_seed_batch_limit_honors_explicit_24_batch_policy(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_SEED_BATCH_LIMIT", "16")
    policy = EvolutionPolicy(
        metadata={
            "seed_safety_max_batches": 24,
            "seed_harvest_safety_max_batches": 24,
            "seed_max_batches": 24,
        }
    )
    assert _seed_safety_batch_limit(policy=policy) == 24


def test_seed_batch_limit_can_be_unbounded_until_exhausted(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_SEED_BATCH_LIMIT", "unbounded")
    monkeypatch.setenv("COGEV_NEXUS_SEED_LOW_NOVELTY_PATIENCE", "1")
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "1")
    model = _LongTailSeedModel(unique_batches=10)
    policy = EvolutionPolicy(metadata={"initial_candidate_count": 1})

    accepted, rejected, error = _generate_model_seed_batches(
        model=model,
        contract=_Contract(),
        world=_World(),
        policy=policy,
        target_size=1,
    )

    assert _seed_safety_batch_limit(policy=policy) is None
    assert error is None
    assert len(accepted) == 10
    assert model.calls == 11
    assert len(accepted) > 8


def test_unbounded_seed_handoff_uses_marginal_family_yield_not_low_static_cap(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_NEXUS_SEED_BATCH_LIMIT", "unbounded")
    monkeypatch.setenv("COGEV_NEXUS_SEED_MIN_BATCHES", "2")
    monkeypatch.setenv("COGEV_NEXUS_SEED_LOW_NOVELTY_PATIENCE", "2")
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "1")
    model = _MarginalFamilySeedModel()
    policy = EvolutionPolicy(metadata={"initial_candidate_count": 4})

    accepted, rejected, error = _generate_model_seed_batches(
        model=model,
        contract=_Contract(),
        world=_World(),
        policy=policy,
        target_size=4,
    )

    assert error is None
    assert len(accepted) > 8
    assert model.calls < 20
    assert accepted[0].metadata["seed_harvest"]["stopped_reason"] == "low_gain_patience"
    assert "accepted_ids" not in accepted[0].metadata["seed_harvest"]
    assert policy.metadata["seed_harvest"]["accepted_count"] == len(accepted)


def test_unbounded_seed_handoff_floor_does_not_double_count_patience() -> None:
    assert _unbounded_seed_handoff_exhausted(
        accepted_before=247,
        accepted_delta=40,
        family_count_before=32,
        family_delta=1,
        accepted_count=287,
        family_count=33,
        batches=44,
        target_size=64,
        min_batches=4,
        low_gain_patience=2,
    )


def test_unbounded_seed_handoff_broadness_uses_target_not_accepted_count() -> None:
    assert _unbounded_seed_handoff_exhausted(
        accepted_before=368,
        accepted_delta=6,
        family_count_before=9,
        family_delta=0,
        accepted_count=374,
        family_count=9,
        batches=62,
        target_size=64,
        min_batches=4,
        low_gain_patience=2,
    )
    assert not _unbounded_seed_handoff_exhausted(
        accepted_before=368,
        accepted_delta=6,
        family_count_before=2,
        family_delta=0,
        accepted_count=374,
        family_count=2,
        batches=62,
        target_size=64,
        min_batches=4,
        low_gain_patience=2,
    )
