#!/usr/bin/env python3
"""Deterministic mechanical benchmark for the Nexus search kernel."""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.durable.async_writer import WriteBehindObserver
from cognitive_evolve_runtime.durable.file_lock import atomic_write_json
from cognitive_evolve_runtime.llm import LLMSession, llm_json, llm_session, logical_llm_call
from cognitive_evolve_runtime.llm.fanout import run_ordered_fanout
from cognitive_evolve_runtime.llm.mock_provider import MockProviderResponse
from cognitive_evolve_runtime.llm.provider_interface import LLMProviderResult
from cognitive_evolve_runtime.nexus.diagnosis import SearchDiagnosis
from cognitive_evolve_runtime.nexus.loop.adaptive_stop import adaptive_stagnation_exhausted
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.search_kernel.islands import allocate_logical_islands
from cognitive_evolve_runtime.ranking.novelty import novelty_distance
from cognitive_evolve_runtime.ranking.parent_selection import evaluator_selection_key


def run(output: Path) -> dict[str, Any]:
    output = output.expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    if output == repo_root or repo_root in output.parents:
        raise ValueError("benchmark output must be outside the source tree")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "cogev.search-mechanics.v1",
        "fanout": _fanout_probe(),
        "replay": _replay_probe(output),
        "persistence": _persistence_probe(),
        "islands": _island_probe(),
        "selection_novelty_stop": _selection_novelty_stop_probe(),
    }
    atomic_write_json(output / "search-mechanics.v1.json", report, sort_keys=True)
    return report


def _fanout_probe() -> dict[str, Any]:
    values = list(range(3))
    lock = threading.Lock()
    active = 0
    maximum = 0
    barrier = threading.Barrier(len(values))

    def call(value: int) -> int:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        barrier.wait()
        with lock:
            active -= 1
        return value

    ordered = run_ordered_fanout(values, call, max_workers=len(values), thread_name_prefix="mechanics")
    return {"call_count": len(ordered), "stable_order": ordered == values, "max_concurrent": maximum}


def _replay_probe(output: Path) -> dict[str, Any]:
    class Provider:
        provider_id = "mechanics"

        def __init__(self) -> None:
            self.calls = 0

        def complete_json(self, **_: Any) -> LLMProviderResult:
            self.calls += 1
            return LLMProviderResult(response=MockProviderResponse({"ok": True}), estimated_cost_usd=None)

    env = {
        "COGEV_LLM_PROVIDER": "litellm",
        "COGEV_LLM_MODEL": "mechanics/model",
        "COGEV_LLM_API_KEY": "mechanics-not-a-secret",
    }
    previous = {key: os.environ.get(key) for key in env}
    try:
        os.environ.update(env)
        provider = Provider()
        session = LLMSession(run_id="mechanics", journal_dir=str(output / "replay"))
        with llm_session(session):
            for _attempt in range(2):
                with logical_llm_call("round/plan/slot"):
                    llm_json("mechanics_replay", {"payload": "same"}, system="Return JSON", schema_hint={}, provider=provider)
        return {
            "logical_attempts": 2,
            "physical_provider_calls": provider.calls,
            "cache_replayed": sum(1 for event in session.snapshot() if event.get("cache_replayed") is True),
        }
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _Frozen:
    serialized_bytes = 1


class _ProbeStore:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.writes = 0

    def freeze(self, _update: dict[str, Any]) -> _Frozen:
        return _Frozen()

    def write(self, _batch: _Frozen) -> None:
        self.started.set()
        self.release.wait()
        self.writes += 1


def _persistence_probe() -> dict[str, Any]:
    store = _ProbeStore()
    observer = WriteBehindObserver(store)
    started = time.perf_counter()
    observer({"phase": "post_mutation"})
    callback_ms = (time.perf_counter() - started) * 1000.0
    write_started = store.started.wait(1)
    callback_returned_before_write = store.writes == 0
    store.release.set()
    observer.close()
    return {
        "callback_ms": round(callback_ms, 3),
        "write_started": write_started,
        "callback_returned_before_write": callback_returned_before_write,
        "fifo_write_count": store.writes,
        **observer.telemetry(),
    }


def _island_probe() -> dict[str, Any]:
    parents = [CandidateGenome(id=f"c{index}", lineage=[f"root{index}", f"c{index}"], artifact={"value": index}) for index in range(9)]
    allocation = allocate_logical_islands(parents=parents, candidates=parents, total_slots=5)
    return {
        "slot_count": len(allocation.branches.slots),
        "lineage_count": len(parents),
        "island_count": len(set(allocation.slot_islands.values())),
        "all_islands_covered": set(allocation.slot_islands.values()) == set(allocation.candidate_islands.values()),
    }


def _selection_novelty_stop_probe() -> dict[str, Any]:
    passed = CandidateGenome(id="passed", artifact="alpha beta gamma complete mechanism")
    passed.metadata["evaluator"] = {"status": "passed", "passed": True, "metrics": {"score": 0.2}}
    unmeasured = CandidateGenome(id="unmeasured", artifact="different graph search")
    failed = CandidateGenome(id="failed", artifact="failed")
    failed.metadata["evaluator"] = {"status": "failed", "passed": False, "metrics": {"score": 1.0}}
    near = CandidateGenome(id="near", artifact="alpha beta gamma complete mechanism!")
    far = CandidateGenome(id="far", artifact="orthogonal counterexample graph construction")
    policy = EvolutionPolicy(
        stagnation_actions=["widen", "repair"],
        metadata={"adaptive_stagnation_patience": 2},
    )
    stopped = adaptive_stagnation_exhausted(
        adaptive=True,
        best_answer_id="passed",
        history=[{"ranking": {"best_final_answer_id": "passed"}, "diagnosis": {"recommended_actions": ["widen"]}}],
        diagnosis=SearchDiagnosis(recommended_actions=["repair"]),
        policy=policy,
    )
    return {
        "selection_tiers": [evaluator_selection_key(item)[0] for item in (passed, unmeasured, failed)],
        "near_novelty": novelty_distance(passed, near),
        "far_novelty": novelty_distance(passed, far),
        "adaptive_stop_after_interventions": stopped,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
