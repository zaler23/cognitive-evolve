from __future__ import annotations

import json
import sys
from pathlib import Path

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.archives.quality_diversity import QualityDiversityArchive, candidate_final_quality, candidate_search_quality
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.evaluators import EvaluatorSpec, ExternalEvaluatorRunner
from cognitive_evolve_runtime.evaluators.artifact_normalizer import normalize_artifact
from cognitive_evolve_runtime.evaluators.challenge_bank import ChallengeBank
from cognitive_evolve_runtime.evaluators.evidence import EvidenceResult, progressive_evidence
from cognitive_evolve_runtime.nexus.adaptive.config import AdaptiveConfig
from cognitive_evolve_runtime.nexus.final_projection import build_final_projection
from cognitive_evolve_runtime.nexus.synthesis import SynthesizedResult


def test_artifact_normalizer_marks_refolded_artifact_probe_only() -> None:
    candidate = CandidateGenome(
        id="C1",
        artifact="Repair mutation: {'n': 4, 'layers': [[[0, 1], [2, 3]]]}",
        artifact_type="sorting_network",
    )

    view = normalize_artifact(candidate, artifact_type="sorting_network", machine_artifact_required=True)

    assert view.status == "refolded"
    assert view.probe_eligible is True
    assert view.final_eligible is False
    assert isinstance(view.normalized_artifact, dict)


def test_legacy_evaluator_writes_progressive_evidence_and_scores(tmp_path: Path) -> None:
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(
        "import json, sys\n"
        "data=json.load(open(sys.argv[1]))\n"
        "print(json.dumps({'passed': False, 'metrics': {'score': 0.42, 'challenge_pass_rate': 0.5}, 'diagnostics': ['case-23 failed']}))\n",
        encoding="utf-8",
    )
    candidate = CandidateGenome(id="C1", artifact={"answer": 1}, artifact_type="machine")
    spec = EvaluatorSpec.from_mapping(
        {
            "enabled": True,
            "command": f"{sys.executable} evaluator.py {{candidate_path}}",
            "cwd": str(tmp_path),
            "evidence": {"machine_artifact_required": True, "artifact_type": "machine"},
        }
    )

    results = ExternalEvaluatorRunner().evaluate_population_if_configured([candidate], spec=spec, round_index=3)

    assert len(results) == 1
    evidence = progressive_evidence(candidate)
    assert evidence is not None
    assert evidence.level == "L2"
    assert evidence.status == "challenge_failed"
    assert evidence.artifact_view is not None and evidence.artifact_view.status == "clean"
    assert candidate.multihead_scores["frontier_score"] > 0
    assert candidate.multihead_scores["challenge_pass_rate"] == 0.5
    assert candidate.metadata["repair_value"] > 0
    assert candidate.metadata["challenge_failures"]


def test_challenge_bank_dedupes_and_tracks_resolution() -> None:
    case_payload = {"mask": 23}
    result = EvidenceResult(
        candidate_id="C1",
        domain_id="combo",
        status="challenge_failed",
        challenge_cases=[],
    )
    # Use evaluator helper path by ingesting two identical diagnostics from metadata-shaped cases.
    from cognitive_evolve_runtime.evaluators.challenge_bank import challenge_from_diagnostic

    case = challenge_from_diagnostic(candidate_id="C1", domain_id="combo", diagnostic=json.dumps(case_payload), kind="counterexample", round_index=1)
    result = EvidenceResult(candidate_id="C1", domain_id="combo", status="challenge_failed", challenge_cases=[case], score=0.8)
    bank = ChallengeBank()

    bank.ingest(result, round_index=1)
    bank.ingest(result, round_index=2)
    bank.mark_resolved("C2", [case.id])

    assert len(bank.cases) == 1
    stored = next(iter(bank.cases.values()))
    assert stored["kill_count"] == 2
    assert stored["frontier_kill_count"] == 2
    assert stored["resolved_by_candidate_ids"] == ["C2"]


def test_archive_keeps_repairable_challenge_failure_out_of_terminal_cull() -> None:
    candidate = CandidateGenome(
        id="C1",
        current_fate=CandidateFate.ACTIVE.value,
        artifact={"x": 1},
        multihead_scores={"objective_alignment": 0.0, "frontier_score": 0.7},
    )
    candidate.metadata["progressive_evidence"] = EvidenceResult(
        candidate_id="C1",
        status="challenge_failed",
        score=0.7,
        hard_reject=False,
        final_eligible=False,
        repair_hints=["repair case"],
    ).to_dict()
    candidate.metadata["repair_value"] = 0.7
    candidate.failure_lessons.append("probe failed")

    assignments = ArchiveManager().assign_by_policy([candidate], current_round=1, round_limit=5, branch_factor=2)

    assert assignments[0].fate in {CandidateFate.INCUBATING.value, CandidateFate.DORMANT.value, CandidateFate.ACTIVE.value}
    assert assignments[0].fate not in {CandidateFate.CULLED.value, CandidateFate.FAILED.value}


def test_quality_diversity_separates_search_and_final_quality() -> None:
    candidate = CandidateGenome(id="C1", niche_memberships=["niche"], multihead_scores={"frontier_score": 0.9, "repair_value": 0.8, "final_verification": 0.0})
    archive = QualityDiversityArchive()

    archive.update(candidate)

    entry = archive.elites_by_niche["niche"]
    assert entry["search_quality"] > entry["final_quality"]
    assert candidate_search_quality(candidate) > candidate_final_quality(candidate)


def test_final_projection_returns_best_current_without_internal_directives() -> None:
    candidate = CandidateGenome(id="C1", artifact={"answer": 1}, artifact_type="machine", multihead_scores={"frontier_score": 0.8, "challenge_pass_rate": 0.5, "schema_cleanliness": 1.0})
    candidate.metadata["progressive_evidence"] = EvidenceResult(candidate_id="C1", status="challenge_failed", score=0.8, final_eligible=False, repair_hints=["fix challenge"] ).to_dict()
    synthesis = SynthesizedResult(status="best_current_route", final_answer="internal repair directive should not be reused", best_candidate_id="C1")

    projection = build_final_projection(population=CandidatePopulation([candidate]), synthesis=synthesis, final_certificate={"blocking_reasons": ["external_evaluator_not_passed"]})
    markdown = projection.to_markdown()

    assert projection.status == "best_current"
    assert projection.objective_solved is False
    assert "internal repair directive should not be reused" not in markdown
    assert "Best current artifact" in markdown


def test_adaptive_config_exposes_progressive_evidence_feature() -> None:
    config = AdaptiveConfig.from_sources(explicit={"enabled": True, "evidence": {"machine_artifact_required": True}})

    assert config.evidence["machine_artifact_required"] is True
    assert config.enabled_features["progressive_evidence"] is True
