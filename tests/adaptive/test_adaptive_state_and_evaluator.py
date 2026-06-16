from __future__ import annotations

import json
import sys
from pathlib import Path

from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.evaluators import EvaluatorSpec, ExternalEvaluatorRunner
from cognitive_evolve_runtime.nexus.adaptive import AdaptiveRuntimeState, build_final_certificate, apply_final_certificate_to_closure
from cognitive_evolve_runtime.nexus.adaptive.spatial_population import build_or_update_spatial_state
from cognitive_evolve_runtime.nexus.loop import EvolutionBudget
from cognitive_evolve_runtime.nexus.runtime import NexusRuntime
from cognitive_evolve_runtime.nexus.synthesis import SynthesizedResult
from cognitive_evolve_runtime.persistence.checkpoint import NexusCheckpoint, build_checkpoint_state
from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy


def test_adaptive_state_roundtrip_and_old_checkpoint_default() -> None:
    state = AdaptiveRuntimeState(round_index=3, enabled_features={"external_evaluator": True}, metrics={"x": 1})
    state.record_event({"type": "unit", "prompt": "must be redacted", "path": "/" + "Users/example/private"})

    restored = AdaptiveRuntimeState.from_dict(state.to_dict())

    assert restored.round_index == 3
    assert restored.enabled_features["external_evaluator"] is True
    assert "prompt" not in restored.events[-1]
    assert restored.events[-1]["path"] == "[redacted-path]"
    assert NexusCheckpoint.from_dict({"round": 1, "max_rounds": 2, "population": {}, "archives": {}}).adaptive_state == {}


def test_adaptive_state_reads_old_challenge_bank_snapshot_as_challenge_memory() -> None:
    restored = AdaptiveRuntimeState.from_dict(
        {
            "challenge_bank": {
                "cases": {
                    "case-old": {
                        "id": "case-old",
                        "summary": "old snapshot case",
                        "payload": {"candidate_id": "C1"},
                    }
                }
            }
        }
    )

    assert restored.challenge_memory is not None
    assert "case-old" in restored.challenge_memory["cases"]


def test_checkpoint_persists_adaptive_state() -> None:
    population = CandidatePopulation([CandidateGenome(id="C1")])
    checkpoint = build_checkpoint_state(
        round=1,
        max_rounds=2,
        population=population,
        archives=ArchiveManager(),
        policy=EvolutionPolicy(),
        adaptive_state={"version": "adaptive-runtime-state/v1", "round_index": 1},
    )

    assert checkpoint.to_dict()["adaptive_state"]["round_index"] == 1
    assert NexusCheckpoint.from_dict(checkpoint.to_dict()).adaptive_state["version"] == "adaptive-runtime-state/v1"


def test_external_evaluator_applies_metadata_and_scores(tmp_path: Path) -> None:
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(
        "import json, sys\n"
        "data=json.load(open(sys.argv[1]))\n"
        "ok=data['id']=='C-pass'\n"
        "print(json.dumps({'passed': ok, 'metrics': {'score': 0.87, 'runtime_ms': 12}, 'diagnostics': [] if ok else ['nope']}))\n",
        encoding="utf-8",
    )
    spec = EvaluatorSpec.from_mapping(
        {
            "enabled": True,
            "command": f"{sys.executable} evaluator.py {{candidate_path}}",
            "cwd": str(tmp_path),
            "timeout_seconds": 5,
        }
    )
    candidate = CandidateGenome(id="C-pass")

    result = ExternalEvaluatorRunner().evaluate_candidate(candidate, spec=spec)

    assert result.passed is True
    ExternalEvaluatorRunner().evaluate_population_if_configured([candidate], spec=spec)
    assert candidate.metadata["evaluator"]["passed"] is True
    assert candidate.multihead_scores["objective_score"] == 0.87
    assert candidate.multihead_scores["correctness"] == 1.0
    assert any(item.get("tool_id") == "external_evaluator" for item in candidate.verification_trace)


def test_final_certificate_blocks_model_solved_when_evaluator_failed() -> None:
    candidate = CandidateGenome(id="C1", verification_result={"passed": True, "final_eligible": True})
    candidate.metadata["evaluator"] = {"status": "failed", "passed": False}
    synthesis = SynthesizedResult(status="completed", final_answer="answer", best_candidate_id="C1")
    closure = {"objective_solved": True, "critical_failures": []}

    certificate = build_final_certificate(
        population=CandidatePopulation([candidate]),
        synthesis=synthesis,
        closure_certificate=closure,
        evaluator_required=True,
    )
    updated = apply_final_certificate_to_closure(closure, certificate)

    assert certificate["objective_solved"] is False
    assert "external_evaluator_not_passed" in certificate["blocking_reasons"]
    assert updated["objective_solved"] is False
    assert "adaptive_final_certificate_gate_failed" in updated["critical_failures"]


def test_spatial_observe_writes_metadata_without_changing_scores() -> None:
    candidate = CandidateGenome(id="C1", current_fate="Active", multihead_scores={"objective_score": 0.8, "novelty": 0.2})
    before = dict(candidate.multihead_scores)

    state = build_or_update_spatial_state([candidate], existing=None, round_index=1)

    assert state.candidate_to_coord["C1"].x == 0
    assert candidate.metadata["spatial"]["mode"] == "observe"
    assert candidate.multihead_scores == before


def test_runtime_writes_adaptive_artifacts_when_enabled(tmp_path: Path) -> None:
    result = NexusRuntime(output_dir=tmp_path).run_text(
        "Return a compact answer with external review labels.",
        max_rounds=1,
        adaptive_config={"enabled": True, "spatial": {"enabled": True, "mode": "observe"}},
    )

    adaptive_dir = tmp_path / "adaptive"
    assert result.evolution["adaptive_state"]["enabled_features"]["spatial_observe"] is True
    assert (adaptive_dir / "adaptive-state.json").exists()
    assert (adaptive_dir / "spatial-topology.json").exists()
    assert (adaptive_dir / "final-certificate.json").exists()
    assert (adaptive_dir / "final-projection.json").exists()
    final_projection = json.loads((adaptive_dir / "final-projection.json").read_text(encoding="utf-8"))
    assert final_projection["status"] in {"best_current", "no_candidate", "solved"}
    assert "objective_solved" in final_projection
    assert (tmp_path / "challenge-memory.json").exists()
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["adaptive_state"]["enabled_features"]["spatial_observe"] is True
    assert checkpoint["adaptive_state"]["enabled_features"]["evidence_control_plane"] is True


def test_task_adaptive_config_sets_nested_evaluator_cwd(tmp_path: Path) -> None:
    from cognitive_evolve_runtime.runtime import _task_adaptive_config

    (tmp_path / "task.yaml").write_text(
        "adaptive:\n"
        "  enabled: true\n"
        "  evaluator:\n"
        "    enabled: true\n"
        "    command: python3 evaluator.py {candidate_path}\n",
        encoding="utf-8",
    )

    config = _task_adaptive_config(tmp_path)

    assert config["enabled"] is True
    assert config["evaluator"]["command"] == "python3 evaluator.py {candidate_path}"
    assert config["evaluator"]["cwd"] == str(tmp_path)


def test_adaptive_state_persists_config_for_resume() -> None:
    from cognitive_evolve_runtime.nexus.adaptive import AdaptiveRuntimeController

    controller = AdaptiveRuntimeController.from_sources(
        explicit={
            "enabled": True,
            "evaluator": {"enabled": True, "command": "python evaluator.py {candidate_path}", "timeout_seconds": 7},
            "evidence": {
                "machine_artifact_required": True,
                "artifact_type": "cache_policy",
                "artifact_type_aliases": {"cache_policy_json": "cache_policy"},
                "required_fields": ["admission"],
            },
            "spatial": {"enabled": True, "mode": "observe"},
        }
    )
    payload = controller.to_dict()

    restored = AdaptiveRuntimeController.from_sources(restored_state=payload)

    assert restored.enabled is True
    assert restored.evaluator_enabled is True
    assert restored.config.evaluator["command"] == "python evaluator.py {candidate_path}"
    assert restored.config.evidence["artifact_type"] == "cache_policy"
    assert restored.config.evidence["artifact_type_aliases"]["cache_policy_json"] == "cache_policy"
    assert restored.config.spatial.enabled is True
    assert restored.to_dict()["enabled_features"]["evidence_control_plane"] is True


def test_adaptive_resume_explicit_config_overrides_restored_config() -> None:
    from cognitive_evolve_runtime.nexus.adaptive import AdaptiveRuntimeController

    restored = AdaptiveRuntimeController.from_sources(
        restored_state={
            "config": {
                "enabled": True,
                "evaluator": {"enabled": True, "command": "old {candidate_path}"},
                "evidence": {"artifact_type": "old_type"},
            }
        },
        explicit={"evaluator": {"command": "new {candidate_path}"}, "evidence": {"artifact_type": "new_type"}},
    )

    assert restored.config.evaluator["command"] == "new {candidate_path}"
    assert restored.config.evidence["artifact_type"] == "new_type"
    assert restored.evaluator_enabled is True
