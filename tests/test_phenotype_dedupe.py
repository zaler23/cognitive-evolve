from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.nexus.loop.round import _canonical_family_metrics
from cognitive_evolve_runtime.nexus.reproduction import dedupe_offspring_against_population
from cognitive_evolve_runtime.nexus.search_kernel.harvesting import CandidateHarvester, HarvestPolicy
from cognitive_evolve_runtime.nexus.search_kernel.fingerprints import candidate_phenotype_signature


def _candidate(candidate_id: str, artifact: dict[str, object], *, narrative: str) -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        artifact=artifact,
        artifact_type="decision",
        concise_claim=f"claim {narrative}",
        core_mechanism=f"mechanism {narrative}",
        niche_memberships=[f"niche-{narrative}"],
        novelty_descriptors=[f"novelty-{narrative}"],
    )


def test_harvesting_uses_materialized_artifact_as_exact_clone_key() -> None:
    first = _candidate("first", {"pump": "off", "radio": "on"}, narrative="alpha")
    clone = _candidate("clone", {"radio": "on", "pump": "off"}, narrative="beta")
    distinct = _candidate("distinct", {"pump": "on", "radio": "on"}, narrative="alpha")
    harvester = CandidateHarvester(
        policy=HarvestPolicy(target_size=3, max_batches=1, relevance_floor=0.0),
    )

    result = harvester.harvest(request_batch=lambda *_args: [first, clone, distinct])

    assert [candidate.id for candidate in result.accepted] == ["first", "distinct"]
    assert result.rejected[0]["reason"] == "duplicate_materialized_artifact"
    assert first.metadata["phenotype_signature"] == clone.metadata["phenotype_signature"]
    assert distinct.metadata["phenotype_signature"] != first.metadata["phenotype_signature"]


def test_population_integration_rejects_same_artifact_with_different_narrative() -> None:
    parent = _candidate("parent", {"schedule": ["clinic", "radio"]}, narrative="baseline")
    clone = _candidate("clone", {"schedule": ["clinic", "radio"]}, narrative="new-story")
    clone.parent_ids = [parent.id]

    kept = dedupe_offspring_against_population([clone], CandidatePopulation([parent]))

    assert kept == []
    assert clone.metadata["candidate_budget_decision"]["reason"] == "duplicate_materialized_artifact"


def test_canonical_metrics_count_identical_artifacts_as_one_phenotype() -> None:
    candidates = [
        _candidate(
            f"clone-{index}",
            {"clinic": "on", "pump": "off", "radio": "on"},
            narrative=str(index),
        )
        for index in range(5)
    ]

    metrics = _canonical_family_metrics(candidates)

    assert metrics["population_count"] == 5
    assert metrics["phenotype_count"] == 1
    assert metrics["exact_clone_excess"] == 4
    assert metrics["distinct_canonical_family_count"] == 1
    assert metrics["candidate_bin_count"] == 1
    assert metrics["descriptive_canonical_family_count"] == 5
    assert metrics["descriptive_candidate_bin_count"] == 5


def test_phenotype_key_preserves_exact_model_owned_text_bytes() -> None:
    compact = CandidateGenome(id="compact", artifact="answer\n", artifact_type="text")
    spaced = CandidateGenome(id="spaced", artifact=" answer\r\n", artifact_type="text")

    assert candidate_phenotype_signature(compact) != candidate_phenotype_signature(spaced)


def test_artifact_type_label_cannot_escape_exact_artifact_dedupe() -> None:
    first = CandidateGenome(id="first", artifact={"answer": 1}, artifact_type="answer")
    relabelled = CandidateGenome(id="relabelled", artifact={"answer": 1}, artifact_type="novel-story")

    kept = dedupe_offspring_against_population([relabelled], CandidatePopulation([first]))

    assert kept == []
    assert relabelled.metadata["duplicate_offspring_reason"] == "duplicate_materialized_artifact"


def test_distinct_artifacts_with_same_evaluator_outcome_share_behavior_family() -> None:
    first = _candidate("first", {"path": [1, 2, 3]}, narrative="alpha")
    second = _candidate("second", {"path": [3, 2, 1]}, narrative="beta")
    for candidate in (first, second):
        candidate.metadata["evaluator"] = {
            "status": "passed",
            "passed": True,
            "metrics": {"length": 873, "coverage": 720, "distinct": 720},
            "diagnostics": [],
        }

    metrics = _canonical_family_metrics([first, second])

    assert metrics["phenotype_count"] == 2
    assert metrics["exact_clone_excess"] == 0
    assert metrics["evaluator_outcome_signature_count"] == 1
    assert metrics["distinct_canonical_family_count"] == 1
    assert metrics["candidate_bin_count"] == 1
    assert metrics["descriptive_canonical_family_count"] == 2
