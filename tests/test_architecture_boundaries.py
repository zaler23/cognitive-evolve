from __future__ import annotations

from pathlib import Path

from cognitive_evolve_runtime.api.models import ChatCompletionRequest
from cognitive_evolve_runtime.api.payloads import _completion_payload
from cognitive_evolve_runtime.api.prompting import build_one_shot_prompt
from cognitive_evolve_runtime.api.streaming import _stream_chunks
from cognitive_evolve_runtime.engine.orchestrator import EngineOrchestrator
from cognitive_evolve_runtime.engine.result import NexusEngineResult
from cognitive_evolve_runtime.nexus.runtime import NexusRuntime

ROOT = Path(__file__).resolve().parents[1]


def test_engine_orchestrator_returns_nexus_result(tmp_path: Path) -> None:
    result = EngineOrchestrator().run("Audit this architecture.", context={"task_dir": str(tmp_path), "rounds": 1})
    assert isinstance(result, NexusEngineResult)
    data = result.to_dict()
    assert data["runtime_architecture"] == "nexus"
    assert data["runtime_path"] == "nexus"
    assert result.final_answer
    assert (tmp_path / "nexus-runtime" / "run-result.json").exists()


def test_parallel_entrypoints_are_absent() -> None:
    for relative in [
        "cognitive_evolve_runtime/adaptive_engine.py",
        "cognitive_evolve_runtime/candidate_search.py",
        "cognitive_evolve_runtime/multi_agent_optimizer.py",
        "cognitive_evolve_runtime/nexus/projection.py",
        "cognitive_evolve_runtime/objective_contract.py",
        "cognitive_evolve_runtime/evidence_planner.py",
        "cognitive_evolve_runtime/evidence_ledger.py",
        "cognitive_evolve_runtime/archive",
        "cognitive_evolve_runtime/optimizer",
        "cognitive_evolve_runtime/selection_contracts.py",
        "cognitive_evolve_runtime/core/evolve.py",
        "cognitive_evolve_runtime/core/evaluator.py",
        "cognitive_evolve_runtime/core/goal_builder.py",
        "cognitive_evolve_runtime/core/failure_classifier.py",
        "cognitive_evolve_runtime/core/prompt_templates.py",
    ]:
        assert not (ROOT / relative).exists(), relative


def test_current_runtime_modules_expose_nexus_boundaries() -> None:
    from cognitive_evolve_runtime.archives.manager import ArchiveManager
    from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
    from cognitive_evolve_runtime.engine.orchestrator import EngineOrchestrator
    from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, evolve_once, seed_population
    from cognitive_evolve_runtime.nexus.runtime import NexusRunResult, NexusRuntime
    from cognitive_evolve_runtime.ranking.relative_rater import RelativeRater

    assert EngineOrchestrator.__name__ == "EngineOrchestrator"
    assert NexusRuntime.__name__ == "NexusRuntime"
    assert NexusRunResult.__name__ == "NexusRunResult"
    assert CandidateGenome.__name__ == "CandidateGenome"
    assert CandidatePopulation.__name__ == "CandidatePopulation"
    assert ArchiveManager.__name__ == "ArchiveManager"
    assert RelativeRater.__name__ == "RelativeRater"
    assert EvolutionBudget.__name__ == "EvolutionBudget"
    assert callable(seed_population) and callable(evolve_once)


def test_api_prompt_payload_and_streaming_are_nexus_shaped() -> None:
    request = ChatCompletionRequest(
        model="cognitive-evolve-nexus",
        messages=[
            {"role": "system", "content": "Be strict."},
            {"role": "user", "content": "Audit this architecture."},
        ],
    )
    prompt = build_one_shot_prompt(request.messages)
    payload = _completion_payload(
        request_id="chatcmpl-test",
        model="cognitive-evolve-nexus",
        prompt=prompt,
        answer="done",
        nexus_data={"version": "2.0", "interaction_mode": "one_shot", "evolution": {"progress_events": [{"round": 1}]}, "verification_summaries": []},
    )
    chunks = list(_stream_chunks(payload))
    assert payload["system_fingerprint"] == "cogev-v2-nexus"
    assert payload["cognitive_evolve"]["actual_rounds"] == 1
    assert chunks[-1] == b"data: [DONE]\n\n"


def test_direct_nexus_runtime_still_works(tmp_path: Path) -> None:
    run = NexusRuntime(output_dir=tmp_path / "nexus-runtime").run_text("Solve a small test task", max_rounds=1)
    assert run.final_answer
    assert (tmp_path / "nexus-runtime" / "checkpoint.json").exists()
    assert (tmp_path / "nexus-runtime" / "snapshot-transaction.json").exists()
    assert "snapshot_transaction" in run.artifacts
