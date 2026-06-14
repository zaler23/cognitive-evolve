from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome


def test_nexus_search_space_and_prompt_compaction_contract() -> None:
    from cognitive_evolve_runtime.nexus.search_space import analyze_coverage, build_search_space_map, classify_candidate
    from cognitive_evolve_runtime.nexus.prompt_view import candidate_prompt_view

    search_map = build_search_space_map(
        {
            "task_type": "model_defined",
            "real_objective": "Study a frontier theorem carefully.",
            "search_space": {
                "candidate_families": [
                    {"id": "global_mechanism", "description": "the whole-objective mechanism"},
                    {"id": "failure_mode", "description": "ways the route can fail"},
                    {"id": "evidence_object", "description": "objects that would test or compare progress"},
                ]
            },
        },
        6,
    )
    assert search_map["candidate_target_count"] >= 6
    assert search_map["source"] == "model_authored_search_space"
    assert "global_mechanism" in search_map["route_family"]

    candidate = CandidateGenome(
        id="C1",
        artifact="long proof artifact " * 100,
        concise_claim="A verifier-backed construction.",
        core_mechanism="tool-grounded construction",
        novelty_descriptors=["tool_grounded"],
        multihead_scores={"objective_alignment": 0.8, "answer_likelihood": 0.7, "novelty": 0.5, "verifiability": 0.9},
    )
    assert classify_candidate(candidate.to_dict(), search_map)["family_id"] in search_map["route_family"]
    coverage = analyze_coverage([{ "search_space": classify_candidate(candidate.to_dict(), search_map) }], search_map)
    assert coverage["covered_count"] >= 1
    compacted = candidate_prompt_view(candidate, max_artifact_chars=60)
    assert compacted["artifact_summary"]["chars"] > len(compacted["artifact_summary"]["preview"])


def test_search_space_fallback_is_objective_derived_not_domain_taxonomy() -> None:
    from cognitive_evolve_runtime.nexus.search_space import build_search_space_map

    search_map = build_search_space_map({"task_type": "proof_resolution", "real_objective": "Study a frontier theorem carefully."}, 6)

    assert search_map["needs_model_authored_search_space"] is True
    assert all(route.startswith("model_defined_focus_") for route in search_map["route_family"])
    assert "duality_or_reduction" not in search_map["route_family"]


def test_archive_evolution_and_verification_stack_paths() -> None:
    from cognitive_evolve_runtime.archives.quality_diversity import QualityDiversityArchive
    from cognitive_evolve_runtime.evolution import DriftDetector, ProgressMonitor, StagnationDetector
    from cognitive_evolve_runtime.tools.verification_stack import NexusVerifierStack

    candidate = CandidateGenome(
        id="C1",
        artifact="Verifier-backed construction",
        concise_claim="A construction with pytest evidence.",
        core_mechanism="tool-grounded construction",
        novelty_descriptors=["tool_grounded"],
        multihead_scores={"objective_alignment": 0.8, "answer_likelihood": 0.7, "novelty": 0.5, "verifiability": 0.9},
    )
    archive = QualityDiversityArchive()
    archive.update(candidate)
    assert archive.to_dict()["elites_by_niche"]
    detector_candidates = [{"title": "Verifier-backed construction", "summary": "A construction with pytest evidence.", "validation": ["pytest verifier"]}]
    assert DriftDetector().detect(detector_candidates)["status"] == "ok"
    assert ProgressMonitor().summarize([{"new_verifier_result": True}])["round_count"] == 1
    assert StagnationDetector().detect([{}, {}])["status"] == "stagnation_detected"

    seed = CandidateGenome(
        id="Seed",
        artifact="Direct Solver Seed: pursue the prompt",
        concise_claim="seed",
        core_mechanism="seed",
        novelty_descriptors=["direct"],
    )
    seed.metadata["search_seed_not_final"] = True
    result = NexusVerifierStack().verify(seed)
    assert result.status == "needs_evolution"
    assert "seed_not_final" in result.diagnostics


def test_nexus_capability_selection_replaces_legacy_capability_runtime() -> None:
    from cognitive_evolve_runtime.nexus.semantics import select_capability_ids

    selected = select_capability_ids("Use external evidence and independent review for this architecture refactor")
    assert "tool_boundary" in selected
    assert "independent_review" in selected
