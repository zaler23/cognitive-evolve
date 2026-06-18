from __future__ import annotations

import threading
import time
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationEngine, MutationOperator, MutationPlan
from cognitive_evolve_runtime.nexus.loop.offspring import _generate_offspring
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


def test_seed_generation_uses_bounded_model_fanout(monkeypatch) -> None:
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
    assert [candidate.id for candidate in accepted[:4]] == ["S0", "S1", "S2", "S3"]


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
    assert {"O0", "O1", "O2", "O3"}.issubset({candidate.id for candidate in offspring})
