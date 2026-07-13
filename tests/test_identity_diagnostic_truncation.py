from __future__ import annotations

import json

from cognitive_evolve_runtime.archives.manager import _evidence_failure_signature
from cognitive_evolve_runtime.archives.types import TerminalCandidateTombstone
from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.core.serialization import stable_hash
from cognitive_evolve_runtime.evaluators.evidence import EvidenceRecord
from cognitive_evolve_runtime.evaluators.runner import ExternalEvaluatorRunner, _parse_evaluator_output
from cognitive_evolve_runtime.evaluators.spec import EvaluatorSpec
from cognitive_evolve_runtime.nexus.search_kernel.fingerprints import candidate_fingerprint
from cognitive_evolve_runtime.tools.feedback import ToolFeedback


def test_failure_signatures_hash_complete_canonical_inputs() -> None:
    common_lessons = [f"lesson-{index}" for index in range(8)]
    first = CandidateGenome(id="first", failure_lessons=[*common_lessons, "tail-a"])
    second = CandidateGenome(id="second", failure_lessons=[*common_lessons, "tail-b"])

    assert candidate_fingerprint(first).failure_signature != candidate_fingerprint(second).failure_signature

    tombstone_first = TerminalCandidateTombstone.from_candidate(first, fate="Failed")
    tombstone_second = TerminalCandidateTombstone.from_candidate(second, fate="Failed")
    assert tombstone_first.failure_signature == "failure_lessons:" + stable_hash({"failure_lessons": first.failure_lessons})
    assert tombstone_first.failure_signature != tombstone_second.failure_signature

    common_challenges = [f"challenge-{index}" for index in range(4)]
    first.metadata["evidence_records"] = [
        EvidenceRecord(
            candidate_id=first.id,
            source="external",
            stage="probe",
            emitted_challenge_ids=[*common_challenges, "tail-a"],
        ).to_dict()
    ]
    second.metadata["evidence_records"] = [
        EvidenceRecord(
            candidate_id=second.id,
            source="external",
            stage="probe",
            emitted_challenge_ids=[*common_challenges, "tail-b"],
        ).to_dict()
    ]
    expected = stable_hash(
        {
            "stage": "probe",
            "source": "external",
            "emitted_challenge_ids": [*common_challenges, "tail-a"],
        }
    )
    assert _evidence_failure_signature(first) == "evidence:" + expected
    assert _evidence_failure_signature(first) != _evidence_failure_signature(second)


def test_invalid_evaluator_output_preserves_every_diagnostic_line() -> None:
    lines = [f"diagnostic-{index}" for index in range(35)]

    assert _parse_evaluator_output("\n".join(lines))["diagnostics"] == lines


class _NonListDiagnosticRunner:
    def run(self, *args: object, **kwargs: object) -> ToolFeedback:
        diagnostics = [f"diagnostic-{index}" for index in range(25)]
        return ToolFeedback(
            tool_id="external-evaluator",
            status="passed",
            diagnostics=diagnostics,
            raw_output_ref=json.dumps({"passed": False, "diagnostics": "malformed"}),
        )


def test_non_list_evaluator_diagnostics_preserve_complete_safe_runner_feedback(tmp_path) -> None:
    result = ExternalEvaluatorRunner(runner=_NonListDiagnosticRunner()).evaluate_candidate(
        CandidateGenome(id="candidate"),
        spec=EvaluatorSpec.from_mapping(
            {
                "enabled": True,
                "command": "python evaluator.py {candidate_path}",
                "cwd": str(tmp_path),
            }
        ),
    )

    assert result.diagnostics == [f"diagnostic-{index}" for index in range(25)]
