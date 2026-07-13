from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.core.serialization import stable_hash
from cognitive_evolve_runtime.llm.session import LLMSession, llm_session
from cognitive_evolve_runtime.nexus.final_projection import FinalProjection
from cognitive_evolve_runtime.nexus.runtime import NexusRuntime
from cognitive_evolve_runtime.outcomes.external_review_bundle import (
    build_external_review_bundle,
    verify_external_review_bundle,
)


def _bundle() -> dict[str, Any]:
    contract = NexusObjectiveContract(
        original_user_goal="Return a reviewable answer.",
        normalized_goal="return a reviewable answer",
        outcome_policy={"requires_verified_solution": True},
        dynamic_artifact_contract={"artifact_type": "answer", "acceptance_criteria": ["independent replay passes"]},
    )
    candidate = CandidateGenome(
        id="C-review",
        artifact={"answer": "candidate material", "token": "identifier", "session_id": "42"},
        concise_claim="candidate material",
        contract_hash=contract.contract_hash(),
        evidence_refs=[{"ref": "evidence/result.json", "sha256": "abc"}],
        verification_result={"status": "preliminary_pass"},
        metadata={
            "objective_solved": True,
            "parent_verification_summary": {"verification_result": {"passed": True, "status": "verified"}},
        },
        multihead_scores={"rank_score": 0.8},
        missing_parts=["independent external replay", "/" + "Users/alice/private/run.log"],
    )
    return build_external_review_bundle(
        candidate=candidate,
        objective_contract=contract,
        world={"kind": "text", "input_packet_id": "textpkt-123"},
        mode="text",
        final_projection={
            "status": "completed",
            "candidate_id": candidate.id,
            "artifact": candidate.artifact,
            "objective_solved": False,
            "blocking_issues": ["external_review_required"],
        },
        verification_summaries=[{"candidate_id": candidate.id, "passed": True, "tool_feedback": []}],
        producer_artifact_refs={"checkpoint": "checkpoint.json", "events": "events.jsonl"},
        usage={
            "total_tokens": {"value": 100, "provenance": "unavailable", "source": "provider_record_without_declared_provenance"},
            "cost_usd": {"value": 0.01, "provenance": "estimated", "source": "llm_session.estimated_cost_usd"},
        },
    )


def test_external_review_bundle_is_export_only_and_content_addressed() -> None:
    bundle = _bundle()

    assert verify_external_review_bundle(bundle)
    assert bundle["external_review_required"] is True
    assert bundle["producer_correctness_claim"] == "not_claimed"
    assert bundle["candidate"]["id"] == "C-review"
    assert bundle["candidate"]["genome_hash"] == stable_hash(bundle["candidate"]["genome"])
    assert bundle["objective_contract"]["normalized_goal"] == "return a reviewable answer"
    assert bundle["acceptance_contract"]["dynamic_artifact_contract"]["artifact_type"] == "answer"
    assert bundle["input_identity"]["input_packet_id"] == "textpkt-123"
    assert {item["signal_class"] for item in bundle["preliminary_checks"]} == {"static", "model_judged", "measured"}
    assert bundle["usage"]["total_tokens"]["provenance"] == "unavailable"
    assert "external_result" not in bundle
    assert "signature" not in bundle
    assert "/" + "Users/alice" not in json.dumps(bundle)
    assert bundle["candidate"]["genome"]["artifact"]["token"] == "identifier"
    assert bundle["candidate"]["genome"]["artifact"]["session_id"] == "42"
    assert bundle["candidate"]["genome"]["metadata"]["objective_solved"] is False
    assert "verification_result" not in bundle["candidate"]["genome"]["metadata"]["parent_verification_summary"]
    assert "unbound_producer_artifact_hints" in bundle["replay_recipe"]

    mutations = [
        ("external_review_required",),
        ("producer_correctness_claim",),
        ("candidate", "genome", "artifact"),
        ("objective_contract", "normalized_goal"),
        ("acceptance_contract", "input_constraints"),
        ("input_identity", "input_packet_id"),
        ("preliminary_checks", 0, "signal_class"),
        ("evidence_refs", 0, "ref"),
        ("known_limitations", 0),
        ("replay_recipe", "steps", 0, "action"),
        ("usage", "total_tokens", "value"),
    ]
    for path in mutations:
        tampered = deepcopy(bundle)
        target: Any = tampered
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = "tampered"
        assert not verify_external_review_bundle(tampered), path

    shell = {
        "version": bundle["version"],
        "bundle_role": bundle["bundle_role"],
        "external_review_required": True,
        "producer_correctness_claim": "not_claimed",
    }
    shell["canonical_bundle_hash"] = "erb:" + stable_hash(shell)
    assert not verify_external_review_bundle(shell)


def test_runtime_persists_external_review_bundle_in_snapshot(tmp_path) -> None:
    with llm_session(LLMSession()):
        run = NexusRuntime(output_dir=tmp_path).run_text("Answer with a falsifiable example.", max_rounds=1)

    bundle_path = tmp_path / "external-review-bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    run_result = json.loads((tmp_path / "run-result.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "snapshot-transaction.json").read_text(encoding="utf-8"))

    assert verify_external_review_bundle(bundle)
    assert run.artifacts["external_review_bundle"] == "external-review-bundle.json"
    assert run_result["artifacts"]["external_review_bundle"] == "external-review-bundle.json"
    assert bundle["candidate"]["id"]
    assert bundle["usage"]["total_tokens"] == {
        "value": None,
        "provenance": "unavailable",
        "source": "llm_session_usage_records",
    }
    assert "external-review-bundle.json" in manifest["files"]
    assert "external_review_bundle" not in bundle["replay_recipe"]["unbound_producer_artifact_hints"]


def test_runtime_omits_invalid_external_review_bundle_when_projection_is_unbound(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "cognitive_evolve_runtime.nexus.runtime_services.build_final_projection",
        lambda **_kwargs: FinalProjection(
            status="completed",
            title="Unbound synthesis answer",
            artifact_type="answer",
            artifact="answer text not bound to a candidate",
            advisory_issues=["answer_unbound_to_candidate_artifact"],
            objective_solved=False,
        ),
    )

    with llm_session(LLMSession()):
        run = NexusRuntime(output_dir=tmp_path).run_text("Answer with a falsifiable example.", max_rounds=1)

    run_result = json.loads((tmp_path / "run-result.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "snapshot-transaction.json").read_text(encoding="utf-8"))

    assert not (tmp_path / "external-review-bundle.json").exists()
    assert "external_review_bundle" not in run.artifacts
    assert "external_review_bundle" not in run_result["artifacts"]
    assert "external-review-bundle.json" not in manifest["files"]
    assert run_result["evolution"]["runtime_metadata"]["external_review_bundle_unavailable_reason"] == "candidate_not_bound_to_final_projection"
