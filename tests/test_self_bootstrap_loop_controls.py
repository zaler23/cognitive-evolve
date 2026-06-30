from __future__ import annotations

import json
from pathlib import Path

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.candidates.mutation import _remove_scaffold_terms
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.live_store import LiveNexusStore
from cognitive_evolve_runtime.nexus.minimal_core import (
    ABLATION_PROFILES,
    apply_seed_active_frontier,
    effective_project_attachment_inventory,
    estimate_reproduction_pressure,
    extract_failure_theorem,
    run_core_ablation,
    single_promotion_gate,
    useful_attachment_stack,
)
from cognitive_evolve_runtime.nexus.nextgen import _looks_like_engineering_noise, false_cull_monitor, resurrection_quota
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.diagnosis import PolicyUpdater, SearchDiagnosis
from cognitive_evolve_runtime.nexus.prompt_view import archive_prompt_view, build_prompt_view
from cognitive_evolve_runtime.nexus.seed_coverage import SEED_RESERVOIR_SIDECAR_PAYLOAD_KEY, assess_seed_coverage, seed_reservoir_sidecar_payload, target_perturb_seed_judgment
from cognitive_evolve_runtime.nexus.strategy_comparison import strategy_comparison_context


def _candidate(candidate_id: str, family: str, *, fate: str = CandidateFate.ACTIVE.value) -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        concise_claim=f"claim {candidate_id}",
        core_mechanism=f"mechanism {family}",
        niche_memberships=[family],
        current_fate=fate,
        metadata={"search_space": {"family_id": family}},
    )


def test_seed_coverage_reports_broad_and_thin_without_fixed_taxonomy() -> None:
    broad = [_candidate(f"C{i}", f"F{i % 8}") for i in range(32)]
    coverage = assess_seed_coverage(broad, policy=EvolutionPolicy(metadata={"initial_candidate_count": 32}))

    assert coverage["accepted_count"] == 32
    assert coverage["family_count"] == 8
    assert coverage["status"] == "broad"
    assert coverage["needs_more_seed"] is False

    thin = assess_seed_coverage([_candidate("A", "same"), _candidate("B", "same")], policy=EvolutionPolicy(metadata={"initial_candidate_count": 32}))
    assert thin["status"] == "thin"
    assert thin["needs_more_seed"] is True
    capped_thin = assess_seed_coverage([_candidate("A", "same")], harvest_summary={"stopped_reason": "max_batches"})
    assert capped_thin["needs_more_seed"] is True
    concentrated = assess_seed_coverage([_candidate(f"C{i}", "same") for i in range(20)])
    assert concentrated["top1_family_share"] == 1.0
    assert concentrated["needs_target_perturb"] is True


def test_target_perturb_seed_judgment_recommends_only_after_stuck_round() -> None:
    candidates = [_candidate(f"C{i}", "same") for i in range(10)]
    diagnosis = {"stagnation_type": "SemanticLooping"}

    early = target_perturb_seed_judgment(candidates, baseline_family_count=14, baseline_seed_count=219, current_round=9, diagnosis=diagnosis)
    late = target_perturb_seed_judgment(
        candidates,
        coverage={"coverage_status": "thin", "top1_family_share": 0.5},
        baseline_family_count=14,
        baseline_seed_count=219,
        current_round=10,
        diagnosis=diagnosis,
        best_current_history=["A", "A", "A"],
        generation_stats={"new_generation_novelty": 0.0},
    )

    assert early["judgment"] == "watch"
    assert late["judgment"] == "trigger_recommended"
    assert late["policy"] == "recommend_only_resume_from_latest_checkpoint"
    assert "suggested_prompt_delta" in late


def test_resurrection_quota_scales_with_large_loser_pool_but_keeps_budget_floor() -> None:
    assert resurrection_quota(6, pool_size=400) > 3
    assert resurrection_quota(6, pool_size=400) <= 6
    assert resurrection_quota(1, pool_size=0) == 1


def test_general_search_wordlists_no_longer_strip_or_flag_terms() -> None:
    text = "router validator framework classification layer scaffold remain if the goal asks for them"
    assert _remove_scaffold_terms(text) == text
    assert _looks_like_engineering_noise("schema repair metadata only") is False
    monitor = false_cull_monitor([_candidate("F", "framework", fate=CandidateFate.DORMANT.value)])
    assert "boundary_loop_candidate_count" not in monitor
    assert "high_intent_nonactive_count" in monitor


def test_strategy_comparison_is_open_carrier_not_named_architecture_gate() -> None:
    policy = EvolutionPolicy(metadata={"strategy_comparison": {"open_hypotheses": ["any free text"], "decision_questions": ["which works?"]}})
    candidate = _candidate("S", "free")
    candidate.metadata["strategy_observation"] = {"claim": "observed tradeoff"}

    context = strategy_comparison_context(policy, [candidate])

    assert context["open_hypotheses"] == ["any free text"]
    assert context["observations"][0]["candidate_id"] == "S"


def test_minimal_core_ablation_compares_profiles_without_provider_calls() -> None:
    score_only = _candidate("score", "common")
    score_only.multihead_scores = {"objective_alignment": 0.9, "core_mechanism_strength": 0.8}
    minimal = _candidate("minimal", "rare")
    minimal.multihead_scores = {"objective_alignment": 0.72, "rarity": 0.9, "novelty": 0.8}
    minimal.failure_lessons.append("fails when rare branch pressure is ignored")
    fusion = _candidate("fusion", "rare")
    fusion.multihead_scores = {"objective_alignment": 0.72, "rarity": 0.9, "novelty": 0.8}
    fusion.formal_artifacts.append({"kind": "proof_adapter_note"})

    report = run_core_ablation([score_only, minimal, fusion])

    assert set(report["profiles"]) == set(ABLATION_PROFILES)
    assert report["profiles"]["score_only"]["best_candidate_id"] == "score"
    assert report["profiles"]["minimal_active_core"]["best_candidate_id"] in {"minimal", "fusion"}
    assert report["efficiency_metrics"]["provider_calls_added"] == 0
    assert report["verification_status"] == "advisory"


def test_minimal_core_plus_useful_attachments_keeps_all_positive_support() -> None:
    bare = _candidate("bare", "rare")
    bare.multihead_scores = {"objective_alignment": 0.72, "rarity": 0.9, "novelty": 0.8}
    rich = _candidate("rich", "rare")
    rich.multihead_scores = {"objective_alignment": 0.72, "rarity": 0.9, "novelty": 0.8}
    rich.failure_lessons.append("reentry fails without dormant factor handles")
    rich.formal_artifacts.append({"kind": "proof_object"})
    rich.proof_obligations.append("bind reentry to changed next action")
    rich.source_bindings.append({"source": "runroot"})
    rich.edge_knowledge_seeds.append("loser factor can become future search gradient")
    rich.metadata["intent_binding"] = {"direct_answer_score": 0.8, "alignment_rationale": "directly answers frozen goal"}
    rich.metadata["resurrection_score"] = 0.7
    rich.metadata["resurrection_reason"] = "dormant factor covers current mechanism gap"
    rich.metadata["open_signal"] = {"score": 0.5, "rationale": "arbitrary model-authored support signal"}

    stack = useful_attachment_stack(rich, factors=[extract_failure_theorem(rich)])
    report = run_core_ablation([bare, rich])

    assert {item["source"] for item in stack} >= {
        "candidate.failure_lessons",
        "candidate.formal_artifacts",
        "candidate.proof_obligations",
        "candidate.source_bindings",
        "candidate.edge_knowledge_seeds",
        "metadata.intent_binding.direct_answer_score",
        "metadata.resurrection",
        "metadata.open_signal",
    }
    stacked = report["profiles"]["minimal_core_plus_useful_attachments"]
    assert stacked["best_candidate_id"] == "rich"
    assert stacked["best_score"] > report["profiles"]["minimal_active_core"]["best_score"]
    assert stacked["active_support_stack"]
    assert report["profiles"]["current_project_incumbent"]["effective_attachment_inventory"]
    assert report["current_project_comparison"]["effective_attachment_coverage"]["coverage_complete"] is True
    assert report["recommendation"] == "minimal_core_with_all_effective_attachments_beats_current_project"


def test_minimal_core_does_not_claim_success_when_project_effective_attachment_is_missing() -> None:
    rich = _candidate("rich", "rare")
    rich.multihead_scores = {"objective_alignment": 0.72, "rarity": 0.9, "novelty": 0.8}
    rich.failure_lessons.append("reentry fails without dormant factor handles")
    policy = EvolutionPolicy(metadata={"incumbent_signal": {"score": 0.9, "rationale": "current project has a useful external support path"}})

    report = run_core_ablation([rich], policy=policy)
    coverage = report["current_project_comparison"]["effective_attachment_coverage"]

    assert "current_project_incumbent" in report["profiles"]
    assert coverage["coverage_complete"] is False
    assert coverage["missing_effective_attachment_count"] >= 1
    assert report["recommendation"] == "current_project_incumbent_until_stacked_core_covers_effective_attachments"


def test_effective_project_attachment_inventory_is_open_field_path_inventory() -> None:
    candidate = _candidate("rich", "rare")
    candidate.failure_lessons.append("failure lesson")
    candidate.metadata["arbitrary_new_signal"] = {"score": 0.8, "rationale": "future component support"}
    policy = EvolutionPolicy(metadata={"another_component": {"status": "active"}})

    inventory = effective_project_attachment_inventory([candidate], factors=[extract_failure_theorem(candidate)], policy=policy)
    sources = {item["source"] for item in inventory}

    assert "candidate.failure_lessons" in sources
    assert "metadata.arbitrary_new_signal" in sources
    assert "policy.metadata.another_component" in sources


def test_r_eff_failure_theorem_and_single_gate_are_real_artifacts() -> None:
    failed = _candidate("F", "rare", fate=CandidateFate.FAILED.value)
    failed.failure_lessons.append("mechanism fails when counterexample pressure is missing")
    failed.multihead_scores = {"objective_alignment": 0.7, "rarity": 0.8}

    theorem = extract_failure_theorem(failed)
    pressure = estimate_reproduction_pressure(failed, [failed], factor_count=1)
    gate = single_promotion_gate(failed)

    assert theorem and theorem["schema"] == "failure_theorem.v1"
    assert pressure["schema"] == "r_eff.v1"
    assert pressure["R_eff"] > 0
    assert gate["schema"] == "single_promotion_gate.v1"
    assert gate["verified_claim_allowed"] is False


def test_large_seed_pool_marks_small_active_frontier_without_deleting_candidates() -> None:
    candidates = [_candidate(f"C{i}", f"F{i}") for i in range(10)]

    trace = apply_seed_active_frontier(candidates, limit=4)

    assert len(trace["selected_ids"]) == 4
    assert trace["dormant_count"] == 6
    assert len(candidates) == 10
    assert sum(1 for candidate in candidates if candidate.current_fate == CandidateFate.DORMANT.value) == 6


def test_live_checkpoint_persists_monitor_search_kernel_and_runtime_options(tmp_path: Path) -> None:
    reservoir = [_candidate("R1", "reserve")]
    policy = EvolutionPolicy(
        metadata={
            "seed_coverage": {"status": "broad"},
            "algorithm_efficiency": {"duplicate_calls_avoided": 2},
            SEED_RESERVOIR_SIDECAR_PAYLOAD_KEY: seed_reservoir_sidecar_payload(reservoir),
        }
    )
    contract = NexusObjectiveContract(original_user_goal="goal", normalized_goal="goal")
    store = LiveNexusStore(tmp_path, mode="test", contract=contract, world={"kind": "test"}, max_rounds=2)

    store(
        {
            "population": CandidatePopulation([_candidate("C", "family")]),
            "archives": ArchiveManager(),
            "policy": policy,
            "phase": "round_end",
            "round": 1,
            "runtime_options": {"model.route.default": "high-capability"},
        }
    )

    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    round_snapshot = json.loads((tmp_path / "rounds" / "round-0001-round_end.json").read_text(encoding="utf-8"))

    assert checkpoint["search_kernel"]["seed_coverage"]["status"] == "broad"
    assert checkpoint["search_kernel"]["seed_reservoir_ref"]["count"] == 1
    assert checkpoint["runtime_options"]["model.route.default"] == "high-capability"
    assert SEED_RESERVOIR_SIDECAR_PAYLOAD_KEY not in str(checkpoint)
    assert Path(checkpoint["search_kernel"]["seed_reservoir_ref"]["path"]).exists()
    assert round_snapshot["monitor_state"]["population_size"] == 1


def test_policy_update_preserves_seed_search_kernel_metadata_from_model_policy() -> None:
    class _PolicyModel:
        def update_policy(self, *, policy: EvolutionPolicy, diagnosis: SearchDiagnosis) -> dict:
            return EvolutionPolicy(metadata={"updated_by_model": True}).to_dict()

    policy = EvolutionPolicy(
        metadata={
            "seed_coverage": {"status": "broad", "fingerprint": "abc"},
            "seed_active_frontier": {"selected_ids": ["C1"]},
            "algorithm_efficiency": {"seed_batches": 12},
        }
    )
    diagnosis = SearchDiagnosis(stagnation_type="healthy")

    updated = PolicyUpdater().update(policy, diagnosis, model=_PolicyModel())

    assert updated.metadata["updated_by_model"] is True
    assert updated.metadata["seed_coverage"]["fingerprint"] == "abc"
    assert updated.metadata["seed_active_frontier"]["selected_ids"] == ["C1"]
    assert updated.metadata["algorithm_efficiency"]["seed_batches"] == 12
