from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationOperator, MutationPlan
from cognitive_evolve_runtime.llm.env import LLMResponseError
from cognitive_evolve_runtime.nexus.model_errors import is_quota_error
from cognitive_evolve_runtime.nexus.loop.offspring import _generate_offspring
from cognitive_evolve_runtime.nexus.loop.offspring import _merge_plan_metadata_into_model_offspring
from cognitive_evolve_runtime.nexus.model_adapter import ModelResponseSchemaError
from cognitive_evolve_runtime.nexus.loop.seeding import _generate_model_seed_batches
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.search_kernel.harvesting import CandidateHarvester, HarvestPolicy


class _World:
    kind = "text"


class _Contract:
    objective = "probe model fanout"

    def to_dict(self) -> dict[str, Any]:
        return {"objective": self.objective}


class _ConcurrencyProbe:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def enter(self) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def exit(self) -> None:
        with self.lock:
            self.active -= 1


class _FanoutSeedModel:
    def __init__(self, probe: _ConcurrencyProbe) -> None:
        self.probe = probe

    def seed_population(self, *, contract: Any, world: Any, policy: EvolutionPolicy) -> list[dict[str, Any]]:
        batch = int((policy.metadata or {}).get("seed_batch_index") or 0)
        self.probe.enter()
        try:
            time.sleep(0.03)
            return [
                {
                    "id": f"S{batch}",
                    "artifact": f"seed artifact {batch}",
                    "concise_claim": f"seed claim {batch}",
                    "core_mechanism": f"seed mechanism {batch}",
                }
            ]
        finally:
            self.probe.exit()


class _NoveltyExhaustionSeedModel:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def seed_population(self, *, contract: Any, world: Any, policy: EvolutionPolicy) -> list[dict[str, Any]]:
        batch = int((policy.metadata or {}).get("seed_batch_index") or 0)
        self.calls.append(batch)
        if batch < 3:
            return [
                {
                    "id": f"N{batch}",
                    "artifact": f"novel seed artifact {batch}",
                    "concise_claim": f"novel seed claim {batch}",
                    "core_mechanism": f"novel seed mechanism {batch}",
                }
            ]
        return [
            {
                "id": "N0-duplicate",
                "artifact": "novel seed artifact 0",
                "concise_claim": "novel seed claim 0",
                "core_mechanism": "novel seed mechanism 0",
            }
        ]


class _PartialErrorSeedModel:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def seed_population(self, *, contract: Any, world: Any, policy: EvolutionPolicy) -> list[dict[str, Any]]:
        batch = int((policy.metadata or {}).get("seed_batch_index") or 0)
        self.calls.append(batch)
        if batch == 1:
            raise LLMResponseError("recoverable provider hiccup")
        return [
            {
                "id": f"P{batch}",
                "artifact": f"partial seed artifact {batch}",
                "concise_claim": f"partial seed claim {batch}",
                "core_mechanism": f"partial seed mechanism {batch}",
            }
        ]


class _FanoutOffspringModel:
    def __init__(self, probe: _ConcurrencyProbe) -> None:
        self.probe = probe

    def generate_offspring(self, *, plans: list[MutationPlan], parents: list[CandidateGenome], world: Any, contract: Any, policy: EvolutionPolicy) -> list[dict[str, Any]]:
        batch = int((policy.metadata or {}).get("offspring_batch_index") or 0)
        self.probe.enter()
        try:
            time.sleep(0.03)
            parent = parents[batch % len(parents)]
            return [
                {
                    "id": f"O{batch}",
                    "parent_ids": [parent.id],
                    "generation": parent.generation + 1,
                    "artifact": f"offspring artifact {batch}",
                    "concise_claim": f"offspring claim {batch}",
                    "core_mechanism": f"offspring mechanism {batch}",
                }
            ]
        finally:
            self.probe.exit()


class _SlotOffspringModel:
    def __init__(self, probe: _ConcurrencyProbe, failures: set[str] | None = None) -> None:
        self.probe = probe
        self.failures = failures or set()
        self.calls: list[str] = []

    def generate_offspring(self, *, plans: list[MutationPlan], parents: list[CandidateGenome], world: Any, contract: Any, policy: EvolutionPolicy) -> list[dict[str, Any]]:
        assert len(plans) == 1
        assert len(plans[0].metadata["branch_slots"]) == 1
        assert policy.metadata["requested_candidate_count"] == 1
        slot_id = str(plans[0].metadata["branch_slots"][0]["slot_id"])
        self.calls.append(slot_id)
        self.probe.enter()
        try:
            time.sleep(0.03)
            if slot_id in self.failures:
                raise LLMResponseError(f"failed {slot_id}")
            return [
                {
                    "id": f"child-{slot_id}",
                    "parent_ids": [parents[0].id],
                    "artifact": f"artifact-{slot_id}",
                    "concise_claim": slot_id,
                    "core_mechanism": f"mechanism-{slot_id}",
                }
            ]
        finally:
            self.probe.exit()


def _candidate(cid: str) -> CandidateGenome:
    return CandidateGenome(id=cid, artifact=f"artifact {cid}", concise_claim=f"claim {cid}", core_mechanism=f"mechanism {cid}")


def test_candidate_harvester_model_fanout_overlaps_and_preserves_batch_order(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "3")
    probe = _ConcurrencyProbe()
    harvester = CandidateHarvester(
        policy=HarvestPolicy(target_size=4, max_batches=4, min_batches=4, low_gain_patience=8, relevance_floor=0.0)
    )

    def _request(batch_index: int, accepted: list[CandidateGenome], rejected: list[dict[str, Any]]) -> list[CandidateGenome]:
        probe.enter()
        try:
            time.sleep(0.03)
            return [_candidate(f"C{batch_index}")]
        finally:
            probe.exit()

    result = harvester.harvest(request_batch=_request)
    assert probe.max_active > 1
    assert result.batches == 4
    assert [candidate.id for candidate in result.accepted[:4]] == ["C0", "C1", "C2", "C3"]
    assert [candidate.metadata["search_kernel_batch"] for candidate in result.accepted[:4]] == [0, 1, 2, 3]


def test_candidate_harvester_model_fanout_serial_fallback(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "1")
    probe = _ConcurrencyProbe()
    harvester = CandidateHarvester(
        policy=HarvestPolicy(target_size=3, max_batches=3, min_batches=3, low_gain_patience=8, relevance_floor=0.0)
    )

    def _request(batch_index: int, accepted: list[CandidateGenome], rejected: list[dict[str, Any]]) -> list[CandidateGenome]:
        probe.enter()
        try:
            time.sleep(0.01)
            return [_candidate(f"C{batch_index}")]
        finally:
            probe.exit()

    result = harvester.harvest(request_batch=_request)
    assert probe.max_active == 1
    assert [candidate.id for candidate in result.accepted[:3]] == ["C0", "C1", "C2"]


def test_harvester_records_later_fanout_error_after_target_reached(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "2")
    harvester = CandidateHarvester(
        policy=HarvestPolicy(target_size=1, max_batches=2, min_batches=1, relevance_floor=0.0),
    )

    def request(index: int, *_args):
        if index == 0:
            return [_candidate("accepted")]
        raise LLMResponseError("later in-flight quota failure")

    result = harvester.harvest(
        request_batch=request,
        on_error=lambda _error: True,
        recoverable_errors=(LLMResponseError,),
    )

    assert [candidate.id for candidate in result.accepted] == ["accepted"]
    assert isinstance(result.fatal_model_error, LLMResponseError)
    assert result.failed_batch_ids == [1]
    assert result.stopped_reason == "model_error"


def test_candidate_harvester_quota_stops_after_inflight_window(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "3")
    started: list[int] = []
    harvester = CandidateHarvester(
        policy=HarvestPolicy(target_size=6, max_batches=6, min_batches=6, low_gain_patience=8, relevance_floor=0.0)
    )

    def _request(batch_index: int, accepted: list[CandidateGenome], rejected: list[dict[str, Any]]) -> list[CandidateGenome]:
        started.append(batch_index)
        if batch_index == 1:
            raise LLMResponseError("You've hit your usage limit. Try again later.")
        return [_candidate(f"C{batch_index}")]

    result = harvester.harvest(
        request_batch=_request,
        on_error=is_quota_error,
        recoverable_errors=(LLMResponseError,),
    )

    assert sorted(started) == [0, 1, 2]
    assert [candidate.id for candidate in result.accepted] == ["C0", "C2"]
    assert result.fatal_model_error is not None
    assert result.stopped_reason == "model_error"


def test_seed_generation_default_follows_global_model_fanout(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "3")
    monkeypatch.setenv("COGEV_NEXUS_SEED_MIN_BATCHES", "4")
    monkeypatch.setenv("COGEV_NEXUS_SEED_BATCH_LIMIT", "4")
    probe = _ConcurrencyProbe()

    accepted, rejected, error = _generate_model_seed_batches(
        model=_FanoutSeedModel(probe),
        contract=_Contract(),
        world=_World(),
        policy=EvolutionPolicy(),
        target_size=4,
    )

    assert error is None
    assert probe.max_active > 1
    assert rejected == []
    assert [candidate.metadata["model_claimed_candidate_id"] for candidate in accepted[:4]] == ["S0", "S1", "S2", "S3"]


def test_seed_generation_serial_override_still_disables_seed_fanout(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "3")
    monkeypatch.setenv("COGEV_NEXUS_SEED_MIN_BATCHES", "4")
    monkeypatch.setenv("COGEV_NEXUS_SEED_BATCH_LIMIT", "4")
    probe = _ConcurrencyProbe()

    accepted, rejected, error = _generate_model_seed_batches(
        model=_FanoutSeedModel(probe),
        contract=_Contract(),
        world=_World(),
        policy=EvolutionPolicy(metadata={"seed_batch_concurrency": 1}),
        target_size=4,
    )

    assert error is None
    assert probe.max_active == 1
    assert rejected == []
    assert [candidate.metadata["model_claimed_candidate_id"] for candidate in accepted[:4]] == ["S0", "S1", "S2", "S3"]


def test_seed_generation_preserves_successful_batches_after_recoverable_batch_error(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "4")
    monkeypatch.setenv("COGEV_NEXUS_SEED_MIN_BATCHES", "4")
    monkeypatch.setenv("COGEV_NEXUS_SEED_BATCH_LIMIT", "4")
    model = _PartialErrorSeedModel()

    accepted, rejected, error = _generate_model_seed_batches(
        model=model,
        contract=_Contract(),
        world=_World(),
        policy=EvolutionPolicy(),
        target_size=4,
    )

    assert error is None
    assert sorted(model.calls) == [0, 1, 2, 3]
    assert [candidate.metadata["model_claimed_candidate_id"] for candidate in accepted] == ["P0", "P2", "P3"]
    assert all("model_seed_error" not in candidate.metadata for candidate in accepted)
    assert any(item.get("reason") == "recoverable_model_error" and item.get("batch") == 1 for item in rejected)
    assert accepted[0].metadata["seed_harvest"]["failed_batch_ids"] == [1]


def test_seed_generation_stops_at_target_without_novelty_tail_calls(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "1")
    monkeypatch.setenv("COGEV_NEXUS_SEED_BATCH_LIMIT", "8")
    monkeypatch.delenv("COGEV_NEXUS_SEED_MIN_BATCHES", raising=False)
    monkeypatch.delenv("COGEV_NEXUS_SEED_LOW_NOVELTY_PATIENCE", raising=False)
    model = _NoveltyExhaustionSeedModel()

    accepted, rejected, error = _generate_model_seed_batches(
        model=model,
        contract=_Contract(),
        world=_World(),
        policy=EvolutionPolicy(),
        target_size=1,
    )

    assert error is None
    assert model.calls == [0]
    assert [candidate.metadata["model_claimed_candidate_id"] for candidate in accepted] == ["N0"]
    assert accepted[0].metadata["seed_harvest"]["stopped_reason"] == "target_reached"
    assert rejected == []


def test_seed_generation_can_opt_into_seed_specific_fanout(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "1")
    monkeypatch.setenv("COGEV_NEXUS_SEED_MIN_BATCHES", "4")
    monkeypatch.setenv("COGEV_NEXUS_SEED_BATCH_LIMIT", "4")
    probe = _ConcurrencyProbe()

    accepted, rejected, error = _generate_model_seed_batches(
        model=_FanoutSeedModel(probe),
        contract=_Contract(),
        world=_World(),
        policy=EvolutionPolicy(metadata={"seed_fanout_concurrency": 3}),
        target_size=4,
    )

    assert error is None
    assert probe.max_active > 1
    assert rejected == []
    assert [candidate.metadata["model_claimed_candidate_id"] for candidate in accepted[:4]] == ["S0", "S1", "S2", "S3"]


def test_offspring_generation_uses_bounded_model_fanout(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "3")
    monkeypatch.setenv("COGEV_NEXUS_OFFSPRING_MIN_BATCHES", "4")
    monkeypatch.setenv("COGEV_NEXUS_OFFSPRING_BATCH_LIMIT", "4")
    probe = _ConcurrencyProbe()
    parents = [_candidate(f"P{i}") for i in range(4)]
    plans = [
        MutationPlan(operator=MutationOperator.DEEPEN, parent_ids=[parent.id], instruction=f"deepen {parent.id}")
        for parent in parents
    ]

    offspring = _generate_offspring(
        model=_FanoutOffspringModel(probe),
        mutation_engine=MutationEngine(),
        parents=parents,
        plans=plans,
        world=_World(),
        contract=_Contract(),
        policy=EvolutionPolicy(),
    )

    assert probe.max_active > 1
    assert {"O0", "O1", "O2", "O3"}.issubset(
        {str(candidate.metadata.get("model_claimed_candidate_id") or "") for candidate in offspring}
    )
    assert len({candidate.id for candidate in offspring}) == len(offspring)


def test_runtime_branch_slots_are_independent_ordered_model_calls(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "3")
    monkeypatch.delenv("COGEV_OFFSPRING_PARALLEL_MODE", raising=False)
    probe = _ConcurrencyProbe()
    parents = [_candidate(f"P{i}") for i in range(3)]
    slots = [
        {"slot_id": f"slot-{index}", "arm_id": parent.id, "parent_id": parent.id, "intent": "explore_fresh"}
        for index, parent in enumerate(parents)
    ]
    plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=[parent.id for parent in parents],
        metadata={"plan_id": "runtime-plan", "plan_source": "runtime_lineage_envelope", "branch_slots": slots},
    )
    model = _SlotOffspringModel(probe)

    offspring = _generate_offspring(
        model=model,
        mutation_engine=MutationEngine(),
        parents=parents,
        plans=[plan],
        world=_World(),
        contract=_Contract(),
        policy=EvolutionPolicy(),
        target_size=3,
    )

    assert probe.max_active > 1
    assert sorted(model.calls) == ["slot-0", "slot-1", "slot-2"]
    assert [candidate.metadata["branch_slot_id"] for candidate in offspring] == ["slot-0", "slot-1", "slot-2"]
    assert [candidate.parent_ids for candidate in offspring] == [["P0"], ["P1"], ["P2"]]


def test_runtime_branch_slots_keep_partial_success_without_batch_fallback(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "3")
    monkeypatch.delenv("COGEV_OFFSPRING_PARALLEL_MODE", raising=False)
    probe = _ConcurrencyProbe()
    parents = [_candidate(f"P{i}") for i in range(3)]
    slots = [
        {"slot_id": f"slot-{index}", "arm_id": parent.id, "parent_id": parent.id, "intent": "explore_fresh"}
        for index, parent in enumerate(parents)
    ]
    plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=[parent.id for parent in parents],
        metadata={"plan_id": "runtime-plan", "plan_source": "runtime_lineage_envelope", "branch_slots": slots},
    )
    model = _SlotOffspringModel(probe, {"slot-1"})

    offspring = _generate_offspring(
        model=model,
        mutation_engine=MutationEngine(),
        parents=parents,
        plans=[plan],
        world=_World(),
        contract=_Contract(),
        policy=EvolutionPolicy(),
        target_size=3,
    )

    assert sorted(model.calls) == ["slot-0", "slot-1", "slot-2"]
    assert [candidate.metadata["branch_slot_id"] for candidate in offspring] == ["slot-0", "slot-2"]
    assert all("failed slot-1" in candidate.metadata["partial_model_offspring_error"] for candidate in offspring)


def test_runtime_branch_slots_all_model_errors_propagate_without_batch_fallback(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "3")
    probe = _ConcurrencyProbe()
    parents = [_candidate("P0"), _candidate("P1")]
    slots = [
        {"slot_id": "slot-0", "arm_id": "P0", "parent_id": "P0"},
        {"slot_id": "slot-1", "arm_id": "P1", "parent_id": "P1"},
    ]
    plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=[parent.id for parent in parents],
        metadata={"plan_id": "runtime-plan", "branch_slots": slots},
    )
    model = _SlotOffspringModel(probe, {"slot-0", "slot-1"})

    with pytest.raises(LLMResponseError, match="failed slot-0"):
        _generate_offspring(
            model=model,
            mutation_engine=MutationEngine(),
            parents=parents,
            plans=[plan],
            world=_World(),
            contract=_Contract(),
            policy=EvolutionPolicy(),
            candidate_pool=parents,
            target_size=2,
        )

    assert sorted(model.calls) == ["slot-0", "slot-1"]


def test_runtime_branch_slots_do_not_swallow_programming_errors(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "2")
    parent = _candidate("P0")
    plan = MutationPlan(
        operator="ModelDirected",
        parent_ids=[parent.id],
        metadata={
            "plan_id": "runtime-plan",
            "branch_slots": [{"slot_id": "slot-0", "arm_id": parent.id, "parent_id": parent.id}],
        },
    )

    class BrokenModel:
        def generate_offspring(self, **_kwargs):
            raise RuntimeError("programming bug")

    with pytest.raises(RuntimeError, match="programming bug"):
        _generate_offspring(
            model=BrokenModel(),
            mutation_engine=MutationEngine(),
            parents=[parent],
            plans=[plan],
            world=_World(),
            contract=_Contract(),
            policy=EvolutionPolicy(),
            candidate_pool=[parent],
            target_size=1,
        )


def test_model_offspring_merges_edge_lineage_by_parent_or_plan_not_modulo_index() -> None:
    parent_a = CandidateGenome(
        id="PA",
        edge_knowledge_seeds=["edge-a"],
        inherited_genes=["gene-a"],
        novelty_descriptors=["novel-a"],
        niche_memberships=["niche-a"],
    )
    parent_b = CandidateGenome(
        id="PB",
        edge_knowledge_seeds=["edge-b"],
        inherited_genes=["gene-b"],
        novelty_descriptors=["novel-b"],
        niche_memberships=["niche-b"],
    )
    plans = [
        MutationPlan(operator=MutationOperator.DEEPEN, parent_ids=["PA"], metadata={"plan_id": "plan-a", "edge_knowledge_seeds": ["plan-edge-a"]}),
        MutationPlan(operator=MutationOperator.REPAIR, parent_ids=["PB"], metadata={"plan_id": "plan-b", "niche_memberships": ["plan-niche-b"]}),
    ]
    child_b = CandidateGenome(id="CB", parent_ids=["PB"], artifact="child b", metadata={"plan_id": "plan-b"}, novelty_descriptors=["child-novel"])
    child_a = CandidateGenome(id="CA", parent_ids=["PA"], artifact="child a", metadata={"plan_id": "plan-a"})
    child_unmapped = CandidateGenome(id="CU", parent_ids=["MISSING"], artifact="unmapped")

    _merge_plan_metadata_into_model_offspring([child_b, child_a], plans, [parent_a, parent_b])

    assert child_b.edge_knowledge_seeds == ["edge-b"]
    assert child_b.inherited_genes == ["gene-b"]
    assert child_b.novelty_descriptors == ["child-novel", "novel-b"]
    assert child_b.niche_memberships == ["plan-niche-b", "niche-b"]
    assert child_a.edge_knowledge_seeds == ["plan-edge-a", "edge-a"]
    assert child_a.inherited_genes == ["gene-a"]
    assert child_a.generation == 1
    assert child_a.lineage == ["PA", "CA"]
    with pytest.raises(ModelResponseSchemaError, match="cannot be resolved"):
        _merge_plan_metadata_into_model_offspring([child_unmapped], plans, [parent_a, parent_b])
