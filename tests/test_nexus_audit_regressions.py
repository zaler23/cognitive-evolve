from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cognitive_evolve_runtime.api.config import get_service_config
from cognitive_evolve_runtime.api.jobs import _rehydrate_job_from_artifact
from cognitive_evolve_runtime.api.server import serve
from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.candidates.project_candidate import PatchOperation, ProjectCandidateGenome
from cognitive_evolve_runtime.configuration import parse_simple_yaml
from cognitive_evolve_runtime.durable.idempotency import stable_hash as durable_stable_hash
from cognitive_evolve_runtime.nexus.project_verification import ProjectCandidateVerifier
from cognitive_evolve_runtime.nexus._serde import stable_hash as nexus_stable_hash
from cognitive_evolve_runtime.nexus.fallbacks import capture_fallback_events
from cognitive_evolve_runtime.nexus.runtime import _world_from_checkpoint
from cognitive_evolve_runtime.nexus.state import nexus_verification_results
from cognitive_evolve_runtime.nexus.synthesis import synthesize_result
from cognitive_evolve_runtime.persistence.event_store import EventStore
from cognitive_evolve_runtime.tools.patch_sandbox import PatchSandbox
from cognitive_evolve_runtime.tools.runner import ToolRunner
from cognitive_evolve_runtime.tools.verifier_environment import VerifierEnvironment
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.types import GradedOutput, VerifiedResult


def test_failed_candidate_is_removed_from_answer_archive_but_can_be_best_current() -> None:
    candidate = CandidateGenome(
        id="candidate",
        artifact="bad answer",
        current_fate=CandidateFate.ELITE,
        multihead_scores={"answer_likelihood": 0.95, "objective_alignment": 0.9},
    )
    archives = ArchiveManager()
    archives.update([candidate])
    assert "candidate" in archives.answer_archive

    candidate.mark_fate(CandidateFate.FAILED.value)
    archives.update([candidate])

    assert "candidate" not in archives.answer_archive
    assert "candidate" in archives.failure_archive.records
    result = synthesize_result(population=CandidatePopulation([candidate]), archives=archives)
    assert result.status == "synthesized"
    assert result.best_candidate_id == "candidate"
    assert result.best_current_direction["route"] == "best_current"
    assert result.best_current_direction["validation_status"] == "preliminary_failed"
    assert result.objective_solved is False


def test_incubating_failed_patch_cannot_be_final_answer_fallback() -> None:
    candidate = CandidateGenome(
        id="patch",
        artifact="diff --git a/app.py b/app.py\nbroken patch",
        current_fate=CandidateFate.INCUBATING.value,
        metadata={"final_answer_blocked_until_repaired": True},
        multihead_scores={"answer_likelihood": 0.95, "objective_alignment": 0.9},
    )

    result = synthesize_result(population=CandidatePopulation([candidate]), archives=ArchiveManager())

    assert result.status == "route_incomplete"
    assert result.best_candidate_id == ""
    assert result.final_answer != candidate.artifact
    assert result.best_current_direction["candidate_id"] == "patch"


def test_failed_candidate_does_not_pollute_general_archives() -> None:
    candidate = CandidateGenome(
        id="failed",
        current_fate=CandidateFate.FAILED,
        core_mechanism="stale",
        novelty_descriptors=["novel"],
        edge_knowledge_seeds=["rare"],
        multihead_scores={"rarity": 1.0, "novelty": 1.0, "answer_likelihood": 0.9},
    )
    archives = ArchiveManager()
    archives.update([candidate])

    assert archives.summary()["failure_records"] == 1
    assert archives.summary()["answer_candidates"] == 0
    assert archives.summary()["rarity_candidates"] == 0
    assert archives.summary()["novelty_candidates"] == 0
    assert archives.summary()["mechanism_elites"] == 0


def test_candidate_scores_reject_non_finite_values_and_json_is_strict() -> None:
    candidate = CandidateGenome(multihead_scores={"nan": "nan", "inf": "inf", "ok": "0.4", "high": "2"})

    assert candidate.multihead_scores == {"ok": 0.4, "high": 1.0}
    encoded = candidate.to_json()
    assert "NaN" not in encoded
    assert json.loads(encoded)["multihead_scores"] == {"high": 1.0, "ok": 0.4}


def test_nexus_verification_results_does_not_treat_completion_status_as_solved_authority() -> None:
    run_data = {
        "evolution": {
            "completion_status": "solved",
            "synthesis": {
                "completion_status": "solved",
                "objective_solved": False,
                "graded_output": {"mode": "graded_portfolio", "verification_strength": "NONE", "verification_strength_value": 0},
            },
        },
        "verification_summaries": [{"passed": True}],
    }

    assert nexus_verification_results(run_data)["objective_solved"] is False


def test_nexus_verification_results_never_claims_objective_solved() -> None:
    graded = GradedOutput(
        mode="verified_result",
        verification_strength=VerificationStrength.FORMAL,
        result=VerifiedResult(answer="verified", replayable=True, evidence_ref="e1", verifier_fingerprint="vf"),
        replay_certificate={"scope": "verifier_on_frozen_artifact_only", "measured_strength_value": 4},
    ).to_dict()
    run_data = {
        "evolution": {
            "completion_status": "solved",
            "synthesis": {"objective_solved": True, "graded_output": graded},
        },
        "verification_summaries": [{"passed": True}],
    }

    assert nexus_verification_results(run_data)["objective_solved"] is False


def test_model_synthesis_rejects_invalid_best_candidate_id() -> None:
    class BadSynthModel:
        def synthesize_result(self, **_: object) -> dict[str, object]:
            return {"final_answer": "unbacked answer", "best_candidate_id": "missing"}

    candidate = CandidateGenome(id="good", artifact="verified", current_fate=CandidateFate.ELITE, multihead_scores={"answer_likelihood": 0.9})
    archives = ArchiveManager()
    archives.update([candidate])

    with capture_fallback_events() as events:
        result = synthesize_result(population=CandidatePopulation([candidate]), archives=archives, model=BadSynthModel())

    assert result.best_candidate_id == "good"
    assert result.final_answer == "verified"
    assert [(event["stage"], event["reason"]) for event in events] == [("final_synthesis", "unknown_candidate_id")]


def test_incomplete_model_synthesis_preserves_blocked_best_current_without_local_substitution() -> None:
    class IncompleteSynthModel:
        def synthesize_result(self, **_: object) -> dict[str, object]:
            return {
                "status": "incomplete",
                "final_answer": "Repair the smaller derived-copy patch, then rerun preliminary checks.",
                "best_candidate_id": "new",
            }

    old = CandidateGenome(
        id="old",
        artifact="older blocked patch",
        current_fate=CandidateFate.ELITE,
        metadata={"final_answer_blocked_until_repaired": True},
        multihead_scores={"answer_likelihood": 0.95},
    )
    new = CandidateGenome(
        id="new",
        artifact="newer blocked patch",
        current_fate=CandidateFate.ACTIVE,
        metadata={"final_answer_blocked_until_repaired": True},
        multihead_scores={"answer_likelihood": 0.9},
    )
    archives = ArchiveManager()
    archives.update([old, new])

    with capture_fallback_events() as events:
        result = synthesize_result(
            population=CandidatePopulation([old, new]),
            archives=archives,
            model=IncompleteSynthModel(),
        )

    assert result.status == "incomplete"
    assert result.final_answer == "Repair the smaller derived-copy patch, then rerun preliminary checks."
    assert result.best_candidate_id == ""
    assert result.best_current_direction["candidate_id"] == "new"
    assert "model_synthesis_requested_candidate_best_current_only" in result.warnings
    assert events == []


def test_model_synthesis_failure_falls_back_to_reference_summary() -> None:
    class EmptySynthModel:
        def synthesize_result(self, **_: object) -> dict[str, object]:
            return {}

    candidate = CandidateGenome(
        id="reference",
        artifact="Use a source-aware patch preflight before expensive ranking.",
        current_fate=CandidateFate.DORMANT,
        source_bindings=[{"path": "cognitive_evolve_runtime/nexus/project_verification.py", "symbol": "ProjectCandidateVerifier"}],
        evidence_refs=[{"kind": "test", "path": "tests/test_nexus_audit_regressions.py", "status": "planned"}],
        obligation_delta={"targeted": ["obl_source_preflight"]},
        multihead_scores={"objective_alignment": 0.8, "answer_likelihood": 0.55, "verifiability": 0.4},
        verification_result={
            "passed": True,
            "rank_eligible": True,
            "final_eligible": False,
            "diagnostics": [],
            "final_gate": {"diagnostics": ["final_update_artifact_absent"]},
        },
    )

    result = synthesize_result(
        population=CandidatePopulation([candidate]),
        archives=ArchiveManager(),
        contract={"outcome_policy": {"accepts_answer_first_output": False, "requires_verified_solution": True}},
        model=EmptySynthModel(),
    )

    assert result.status == "final_synthesis_local_fallback"
    assert result.best_candidate_id == "reference"
    assert not hasattr(result, "reference" + "_candidate_id")
    assert "model_synthesis_local_fallback:empty_final_answer" in result.warnings
    assert result.final_answer == "Use a source-aware patch preflight before expensive ranking."


def test_model_synthesis_ordinary_exception_still_falls_back() -> None:
    class BuggySynthModel:
        def synthesize_result(self, **_: object) -> dict[str, object]:
            raise RuntimeError("local bug")

    candidate = CandidateGenome(id="reference", artifact="fallback answer", multihead_scores={"answer_likelihood": 0.8})

    result = synthesize_result(population=CandidatePopulation([candidate]), archives=ArchiveManager(), model=BuggySynthModel())

    assert result.status == "final_synthesis_local_fallback"
    assert result.final_answer == "fallback answer"
    assert "model_synthesis_local_fallback:RuntimeError" in result.warnings



def test_model_synthesis_uses_reference_not_seed_or_final_id_for_unverified_candidate() -> None:
    class ReviewSynthModel:
        def synthesize_result(self, **_: object) -> dict[str, object]:
            return {
                "status": "model_synthesized",
                "final_answer": "Use sliding-window parity counting; please review externally.",
                "best_candidate_id": "ANS1",
            }

    seed = CandidateGenome(
        id="SEED0",
        artifact="initial seed",
        current_fate=CandidateFate.ELITE,
        metadata={"search_seed_not_final": True},
        multihead_scores={"answer_likelihood": 0.99, "objective_alignment": 0.99},
    )
    answer = CandidateGenome(
        id="ANS1",
        artifact="Use sliding-window parity counting.",
        current_fate=CandidateFate.DORMANT,
        evidence_refs=[{"kind": "manual", "status": "pending"}],
        multihead_scores={"objective_alignment": 0.8, "answer_likelihood": 0.75, "verifiability": 0.4},
        verification_result={
            "passed": True,
            "rank_eligible": True,
            "final_eligible": False,
            "diagnostics": [],
            "final_gate": {"diagnostics": ["external_validation_not_completed"]},
        },
    )
    archives = ArchiveManager()
    archives.update([seed, answer])

    result = synthesize_result(population=CandidatePopulation([seed, answer]), archives=archives, model=ReviewSynthModel())

    assert result.status == "model_synthesized"
    assert result.best_candidate_id == "ANS1"
    assert result.best_current_direction["candidate_id"] == "ANS1"
    assert not hasattr(result, "reference" + "_candidate_id")
    assert "review externally" in result.final_answer.lower()
    assert "model_final_answer_differs_from_selected_candidate_artifact" in result.warnings


def test_model_requested_failed_candidate_keeps_unbound_final_answer() -> None:
    class FailedCandidateSynthModel:
        def synthesize_result(self, **_: object) -> dict[str, object]:
            return {
                "status": "model_synthesized",
                "final_answer": "Model-authored final answer that should not be replaced.",
                "best_candidate_id": "failed",
            }

    candidate = CandidateGenome(
        id="failed",
        artifact="failed candidate answer material",
        current_fate=CandidateFate.FAILED.value,
        multihead_scores={"answer_likelihood": 0.9, "objective_alignment": 0.8},
    )

    result = synthesize_result(population=CandidatePopulation([candidate]), archives=ArchiveManager(), model=FailedCandidateSynthModel())

    assert result.status == "model_synthesized"
    assert result.final_answer == "Model-authored final answer that should not be replaced."
    assert result.best_candidate_id == ""
    assert result.best_current_direction["candidate_id"] == ""
    assert "model_synthesis_requested_candidate_unbound_ineligible_candidate_id" in result.warnings


def test_model_requested_empty_candidate_keeps_unbound_final_answer() -> None:
    class EmptyCandidateSynthModel:
        def synthesize_result(self, **_: object) -> dict[str, object]:
            return {
                "status": "model_synthesized",
                "final_answer": "Model text survives even when the requested candidate has no answer material.",
                "best_candidate_id": "empty",
            }

    candidate = CandidateGenome(
        id="empty",
        artifact="",
        concise_claim="",
        core_mechanism="",
        current_fate=CandidateFate.DORMANT.value,
    )

    result = synthesize_result(population=CandidatePopulation([candidate]), archives=ArchiveManager(), model=EmptyCandidateSynthModel())

    assert result.status == "model_synthesized"
    assert result.final_answer == "Model text survives even when the requested candidate has no answer material."
    assert result.best_candidate_id == ""
    assert result.best_current_direction["candidate_id"] == ""
    assert "model_synthesis_requested_candidate_unbound_empty_answer_material" in result.warnings


def test_answer_candidate_ranking_can_prioritize_high_answer_score_without_source_gate() -> None:
    hallucinated = CandidateGenome(
        id="Z-hallucinated",
        artifact="Patch an invented select_parents symbol.",
        current_fate=CandidateFate.DORMANT,
        source_bindings=[{"path": "cognitive_evolve_runtime/api/engine_runner.py", "symbol": "select_parents"}],
        multihead_scores={"objective_alignment": 0.95, "answer_likelihood": 0.9, "verifiability": 0.8},
        verification_result={
            "passed": True,
            "rank_eligible": True,
            "final_eligible": False,
            "diagnostics": [],
            "final_gate": {"diagnostics": ["source_binding_missing_symbol"]},
        },
    )
    grounded = CandidateGenome(
        id="A-grounded",
        artifact="Add source-aware preflight in the existing project verifier.",
        current_fate=CandidateFate.DORMANT,
        source_bindings=[{"path": "cognitive_evolve_runtime/nexus/project_verification.py", "symbol": "ProjectCandidateVerifier"}],
        evidence_refs=[{"kind": "test", "path": "tests/test_nexus_audit_regressions.py", "status": "planned"}],
        obligation_delta={"targeted": ["obl_source_preflight"]},
        multihead_scores={"objective_alignment": 0.75, "answer_likelihood": 0.55, "verifiability": 0.35},
        verification_result={"passed": True, "rank_eligible": True, "final_eligible": False, "diagnostics": []},
    )

    result = synthesize_result(
        population=CandidatePopulation([hallucinated, grounded]),
        archives=ArchiveManager(),
        contract={"outcome_policy": {"accepts_answer_first_output": False, "requires_verified_solution": True}},
    )

    assert result.best_candidate_id == "Z-hallucinated"
    assert "invented select_parents" in result.final_answer


def test_patch_sandbox_rejects_symlink_escape(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("safe", encoding="utf-8")
    (source / "escape").symlink_to(outside)

    candidate = ProjectCandidateGenome(id="patch", patch_set=[PatchOperation(path="escape", operation="write", content="owned")])
    result = PatchSandbox(source, tmp_path / "sandboxes").apply(candidate)

    assert result.status == "failed"
    assert "unsafe" in " ".join(result.diagnostics)
    assert outside.read_text(encoding="utf-8") == "safe"


def test_patch_sandbox_enforces_allowed_patch_scope(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "pkg").mkdir(parents=True)
    (source / "docs").mkdir()
    (source / "pkg" / "module.py").write_text("value = 1\n", encoding="utf-8")
    (source / "docs" / "note.md").write_text("note\n", encoding="utf-8")

    candidate = ProjectCandidateGenome(id="out-of-scope", patch_set=[PatchOperation(path="docs/note.md", operation="append", content="x")])
    result = PatchSandbox(source, tmp_path / "sandboxes", allowed_patch_scope=["pkg/**/*.py"]).apply(candidate)

    assert result.status == "failed"
    assert "outside allowed_patch_scope" in " ".join(result.diagnostics)
    assert (source / "docs" / "note.md").read_text(encoding="utf-8") == "note\n"
    assert not (tmp_path / "sandboxes" / "out-of-scope").exists()


def test_project_verification_failure_marks_candidate_failed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    candidate = ProjectCandidateGenome(id="bad", patch_set=[PatchOperation(path="../escape", operation="write", content="bad")])

    summary = ProjectCandidateVerifier(source_root=source, sandbox_root=tmp_path / "sandboxes").verify(candidate)

    assert summary.passed is False
    assert candidate.current_fate == CandidateFate.FAILED


def test_server_refuses_public_bind_without_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COGEV_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("COGEV_SERVER_HOST", "0.0.0.0")
    monkeypatch.setenv("COGEV_SERVER_REQUIRE_AUTH", "false")
    monkeypatch.delenv("COGEV_ALLOW_INSECURE_BIND", raising=False)

    with pytest.raises(RuntimeError, match="Refusing to serve"):
        serve()


def test_cors_defaults_to_localhost_and_wildcard_disables_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COGEV_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.delenv("COGEV_CORS_ALLOW_ORIGINS", raising=False)
    config = get_service_config()
    assert "*" not in config.cors_allow_origins
    assert any(origin.startswith("http://127.0.0.1") for origin in config.cors_allow_origins)

    monkeypatch.setenv("COGEV_CORS_ALLOW_ORIGINS", "*")
    wildcard = get_service_config()
    assert wildcard.cors_allow_origins == ("*",)
    assert wildcard.cors_allow_credentials is False


def test_runner_blocks_non_allowlisted_commands(tmp_path: Path) -> None:
    result = ToolRunner(allowed_executables={"python"}).run(["sh", "-c", "echo bad"], cwd=tmp_path)

    assert result.status == "blocked"
    assert "not allowlisted" in result.diagnostics[0]


def test_runner_rejects_path_spoofed_allowlisted_command() -> None:
    allowed, reason = ToolRunner(allowed_executables={"pytest"})._command_allowed(["/tmp/pytest"])

    assert allowed is False
    assert "path not allowlisted" in reason


def test_runner_accepts_only_an_explicit_absolute_executable_path() -> None:
    runner = ToolRunner(
        allowed_executables={Path(sys.executable).name},
        allowed_executable_paths={sys.executable},
    )

    allowed, _ = runner._command_allowed([sys.executable, "-c", "print('ok')"])
    other_allowed, _ = runner._command_allowed(["/tmp/" + Path(sys.executable).name, "-c", "print('bad')"])

    assert allowed is True
    assert other_allowed is False


def test_verifier_environment_filters_secret_and_protected_extra_env(tmp_path: Path) -> None:
    env = VerifierEnvironment.for_path(
        tmp_path,
        extra_env={"API_KEY": "secret", "COGEV_HERMETIC_TEST": "0", "SAFE_FLAG": "1"},
        hermetic=True,
    )

    assert "API_KEY" not in env.env
    assert env.env["COGEV_HERMETIC_TEST"] != "0"
    assert env.env["SAFE_FLAG"] == "1"


def test_runner_uses_process_group_and_resource_limiter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):  # noqa: ANN001
        captured.update(kwargs)
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ToolRunner(allowed_executables={"python"}).run(["python", "-c", "print('ok')"], cwd=tmp_path)

    assert result.status == "passed"
    assert captured["start_new_session"] is True
    assert callable(captured["preexec_fn"])


def test_yaml_config_uses_real_yaml_semantics() -> None:
    data = parse_simple_yaml(
        """
        server:
          api_keys:
            - one
            - two
          public_base_url: "http://example.test/#fragment"
        """
    )

    assert data["server"]["api_keys"] == ["one", "two"]
    assert data["server"]["public_base_url"].endswith("#fragment")


def test_durable_stable_hash_uses_same_recursive_canonicalization_as_nexus() -> None:
    value = {"path": Path("a/b"), "items": {"z", "a"}}

    assert durable_stable_hash(value) == nexus_stable_hash(value)


def test_event_store_append_many_once_deduplicates_in_one_pass(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.jsonl")
    events = [{"type": "progress", "round": 1}, {"type": "progress", "round": 1}, {"type": "progress", "round": 2}]

    appended = store.append_many_once(events)

    assert len(appended) == 2
    assert [event["round"] for event in store.read_all()] == [1, 2]


def test_rehydrated_running_job_is_marked_interrupted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "api-runs"
    job_dir = root / "job-test"
    job_dir.mkdir(parents=True)
    (job_dir / "job-status.json").write_text(json.dumps({"id": "job-test", "status": "running"}), encoding="utf-8")
    monkeypatch.setenv("COGEV_API_TASK_ROOT", str(root))

    job = _rehydrate_job_from_artifact("job-test")

    assert job is not None
    assert job["status"] == "interrupted"


def test_project_checkpoint_resume_preserves_world_envelope() -> None:
    world = _world_from_checkpoint(
        "project",
        {
            "snapshot": {"root_path": "redacted-test-root", "root_hash": "abc", "file_manifest": []},
            "project_world_model": {"kind": "project", "snapshot_id": "snapshot-abc", "project_summary": "summary"},
        },
    )

    assert isinstance(world, dict)
    assert "snapshot" in world
    assert "project_world_model" in world
    assert world["project_world_model"]["kind"] == "project"
