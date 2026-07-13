from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus.obligations import formal_artifact_kind, formal_signature, looks_like_formal_artifact


def test_formal_artifact_type_alias_accepts_assertion_set() -> None:
    artifact = {
        "type": "assertion_set",
        "target_obligation_id": "obl_metric_parser",
        "assertions": [
            "assert parse_metric_value('1.2e-4') == 0.00012",
            "assert parse_metric_value('malformed') is None",
        ],
    }

    assert formal_artifact_kind(artifact) == "assertion_set"
    assert looks_like_formal_artifact(artifact) is True


def test_formal_artifact_artifact_type_alias_accepts_assertion_set() -> None:
    artifact = {
        "artifact_type": "assertion_set",
        "target_obligation_id": "obl_runtime_assertion",
        "assertions": [{"expression": "assert Path('tests/test_project_candidate_patch_sandbox.py').exists()"}],
    }

    assert formal_artifact_kind(artifact) == "assertion_set"
    assert looks_like_formal_artifact(artifact) is True


def test_type_alias_unblocks_proof_progress_structural_gate() -> None:
    candidate = CandidateGenome(
        id="C-type-alias",
        artifact="Runtime parser proof witness.",
        concise_claim="Executable assertion_set witness discharges parser obligation.",
        core_mechanism="The formal artifact uses type=assertion_set, not kind=assertion_set.",
        formal_artifacts=[
            {
                "type": "assertion_set",
                "target_obligation_id": "obl_metric_parser",
                "assertions": [
                    "assert parse_metric_value('1.2e-4') == 0.00012",
                    "assert parse_metric_value('malformed') is None",
                ],
            }
        ],
        proof_obligations=[{"id": "obl_metric_parser", "status": "discharged", "description": "verify parser"}],
        obligation_delta={"targeted": ["obl_metric_parser"], "discharged": ["obl_metric_parser"]},
        source_bindings=[{"path": "cognitive_evolve_runtime/nexus/stage_policy.py", "kind": "source_file"}],
    )

    assert formal_signature(candidate)
