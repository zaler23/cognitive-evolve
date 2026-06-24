from __future__ import annotations

import json
import inspect
from pathlib import Path
from types import SimpleNamespace

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome, CandidatePopulation
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.llm.call_identity import identity_from_status
from cognitive_evolve_runtime.llm.model_spec import LLMModelSpec
from cognitive_evolve_runtime.llm.provider_interface import LLMProviderResult
from cognitive_evolve_runtime.llm.session import LLMSession, llm_session
from cognitive_evolve_runtime.llm.transport import llm_json
from cognitive_evolve_runtime.nexus import nextgen
from cognitive_evolve_runtime.nexus.final_projection import build_final_projection
from cognitive_evolve_runtime.nexus.nextgen import (
    best_current_direction_payload,
    budget_eligible_candidates,
    cbt_soft_budget_adjustment,
    observe_productive_child,
    select_best_current_direction,
    structurally_blocked,
)
from cognitive_evolve_runtime.nexus.prompt_view import build_prompt_view
from cognitive_evolve_runtime.nexus.reproduction import dedupe_offspring_against_population, verify_offspring
from cognitive_evolve_runtime.nexus.search_kernel.harvesting import CandidateHarvester, HarvestPolicy
from cognitive_evolve_runtime.nexus.search_space import build_search_space_map, classify_candidate
from cognitive_evolve_runtime.nexus.model_routes import SeedModelEnsembleAdapter
from cognitive_evolve_runtime.nexus.synthesis import SynthesizedResult, synthesize_result
from cognitive_evolve_runtime.ranking.parent_selection import ParentSelector
from cognitive_evolve_runtime.verification.ladder import VerificationStrength
from cognitive_evolve_runtime.verification.types import GradedOutput, VerifiedResult


def _graded_portfolio() -> GradedOutput:
    return GradedOutput(mode="graded_portfolio", verification_strength=VerificationStrength.NONE)


def test_final_best_current_direction_survives_failed_source_free_candidate() -> None:
    candidate = CandidateGenome(
        id="critical-failed",
        current_fate=CandidateFate.FAILED.value,
        concise_claim="Critical Branching should split proof search at unstable branch points.",
        core_mechanism="Critical Branching theorem discovery",
        verification_result={"passed": False, "diagnostics": ["source_free_final_claim"]},
        multihead_scores={"answer_likelihood": 0.9, "novelty": 0.8, "rarity": 0.7},
    )

    result = synthesize_result(population=CandidatePopulation([candidate]), archives=ArchiveManager())
    projection = build_final_projection(
        population=CandidatePopulation([candidate]),
        synthesis=result,
        graded_output=_graded_portfolio(),
    )

    assert result.answer_produced is True
    assert result.objective_solved is False
    assert result.best_current_direction["candidate_id"] == "critical-failed"
    assert result.best_current_direction["verification_status"] == "failed"
    assert projection.candidate_id == "critical-failed"
    assert projection.best_current_direction["route"] == "best_current"


def test_structural_or_safety_candidate_never_becomes_best_current() -> None:
    structural = CandidateGenome(
        id="bad",
        concise_claim="Critical Branching but corrupt artifact",
        current_fate=CandidateFate.FAILED.value,
        metadata={"structural_failure": True},
    )

    assert select_best_current_direction([structural]) is None
    projection = build_final_projection(
        population=CandidatePopulation([structural]),
        synthesis=SynthesizedResult(status="completed", final_answer="", best_candidate_id="bad"),
        graded_output=_graded_portfolio(),
    )

    assert projection.status == "no_candidate"
    assert projection.candidate_id == ""


def test_stage_hard_reject_reason_is_structural_for_best_current() -> None:
    candidate = CandidateGenome(
        id="second-authority",
        concise_claim="Add a hidden fallback router as a second runtime authority.",
        metadata={"stage_eligibility": {"hard_reject_reason": "second_runtime_or_ranking_authority"}},
    )

    assert structurally_blocked(candidate) is True
    assert select_best_current_direction([candidate]) is None
    result = synthesize_result(population=CandidatePopulation([candidate]), archives=ArchiveManager())
    assert result.status == "failure_report"
    assert result.best_candidate_id == ""


def test_verified_candidate_wins_before_best_current_fallback() -> None:
    verified = CandidateGenome(
        id="verified",
        current_fate=CandidateFate.ACTIVE.value,
        concise_claim="Verified compact answer",
        verification_result={"passed": True},
        multihead_scores={"answer_likelihood": 0.4},
    )
    failed_critical = CandidateGenome(
        id="critical-failed",
        current_fate=CandidateFate.FAILED.value,
        concise_claim="Critical Branching with very novel branch-point search",
        core_mechanism="Critical Branching",
        verification_result={"passed": False},
        multihead_scores={"answer_likelihood": 1.0, "novelty": 1.0, "rarity": 1.0},
    )

    result = synthesize_result(population=CandidatePopulation([failed_critical, verified]), archives=ArchiveManager())
    projection = build_final_projection(
        population=CandidatePopulation([failed_critical, verified]),
        synthesis=SynthesizedResult(status="completed", final_answer="", best_candidate_id=""),
        graded_output=_graded_portfolio(),
    )

    assert result.best_candidate_id == "verified"
    assert result.best_current_direction["route"] == "final"
    assert projection.candidate_id == "verified"
    assert projection.best_current_direction["route"] == "final"


def test_intent_binding_prefers_direct_goal_claim_over_supporting_artifact() -> None:
    contract = NexusObjectiveContract(
        original_user_goal="Explore Critical Branching search dynamics as the next-generation model.",
        normalized_goal="Explore Critical Branching search dynamics as the next-generation model.",
    )
    direct = CandidateGenome(
        id="direct",
        concise_claim="Use lineage divergence and branch survival probability to model critical search bifurcations.",
        core_mechanism="Critical Branching search dynamics model",
        metadata={"intent_binding": {"direct_answer_score": 0.92, "alignment_rationale": "directly answers the frozen goal"}},
        multihead_scores={"answer_likelihood": 0.4, "novelty": 0.6},
    )
    support = CandidateGenome(
        id="support",
        concise_claim="Record the branch exploration in an audit ledger.",
        core_mechanism="branch exploration record",
        metadata={"intent_binding": {"direct_answer_score": 0.2, "alignment_rationale": "supporting material for the goal"}},
        multihead_scores={"answer_likelihood": 1.0, "verifiability": 1.0},
    )

    selected = select_best_current_direction([support, direct], contract=contract)
    result = synthesize_result(population=CandidatePopulation([support, direct]), archives=ArchiveManager(), contract=contract)

    assert selected is direct
    assert result.best_candidate_id == "direct"
    assert result.best_current_direction["candidate_main_claim"] == "Critical Branching search dynamics model"
    assert result.best_current_direction["supporting_claims"]


def test_intent_binding_does_not_blacklist_support_artifacts_when_goal_asks_for_them() -> None:
    contract = NexusObjectiveContract(
        original_user_goal="Explore a better audit record mechanism for candidate evolution.",
        normalized_goal="Explore a better audit record mechanism for candidate evolution.",
    )
    audit_record = CandidateGenome(
        id="audit-record",
        concise_claim="A reversible candidate-evolution audit record with concise replay handles.",
        core_mechanism="audit record mechanism",
        metadata={"intent_binding": {"direct_answer_score": 0.95}},
        multihead_scores={"answer_likelihood": 0.5},
    )
    dynamics = CandidateGenome(
        id="dynamics",
        concise_claim="A branch dynamics hypothesis unrelated to audit records.",
        core_mechanism="branch dynamics hypothesis",
        metadata={"intent_binding": {"direct_answer_score": 0.25}},
        multihead_scores={"answer_likelihood": 0.9},
    )

    assert select_best_current_direction([dynamics, audit_record], contract=contract) is audit_record


def test_intent_binding_falls_back_without_contract_or_metadata() -> None:
    candidate = CandidateGenome(id="fallback", concise_claim="useful direction", core_mechanism="open exploration")

    selected = select_best_current_direction([candidate])

    assert selected is candidate
    assert candidate.metadata["intent_binding"]["direct_answer_score"] == 0.5


def test_intent_binding_recomputes_stale_no_contract_fallback_when_goal_arrives() -> None:
    candidate = CandidateGenome(
        id="stale",
        concise_claim="Use branching survival dynamics to compare architecture directions.",
        core_mechanism="branching survival dynamics architecture search",
    )
    stale = nextgen.bind_candidate_intent(candidate)
    assert stale["alignment_rationale"].startswith("no frozen search intent supplied")

    contract = NexusObjectiveContract(
        original_user_goal="Explore branching survival dynamics for CognitiveEvolve architecture search.",
        normalized_goal="Explore branching survival dynamics for CognitiveEvolve architecture search.",
    )
    refreshed = nextgen.bind_candidate_intent(candidate, contract=contract)

    assert refreshed["search_intent"] == contract.normalized_goal
    assert not refreshed["alignment_rationale"].startswith("no frozen search intent supplied")
    assert refreshed["direct_answer_score"] > 0.5


def test_final_scoring_has_no_domain_hardcoded_resurrection_terms() -> None:
    source = "\n".join(
        inspect.getsource(obj)
        for obj in (
            nextgen.best_current_direction_score,
            nextgen.select_best_current_direction,
            nextgen.resurrection_score,
            nextgen.mark_resurrection_candidate,
        )
    ).lower()

    assert "critical branching" not in source
    assert "critical_branching" not in source
    assert "framework_noise" not in source
    assert "target_kind" not in source
    assert "engineering_type" not in source
    assert "mechanism_type" not in source


def test_user_facing_verification_does_not_trust_candidate_metadata_verified() -> None:
    candidate = CandidateGenome(
        id="meta-verified",
        concise_claim="answer",
        metadata={"verification_status": "verified", "intent_binding": {"direct_answer_score": 0.8}},
    )

    payload = best_current_direction_payload(candidate, graded_output=_graded_portfolio().to_dict())
    projection = build_final_projection(
        population=CandidatePopulation([candidate]),
        synthesis=SynthesizedResult(status="completed", final_answer="answer", best_candidate_id="meta-verified"),
        graded_output=_graded_portfolio(),
    )

    assert payload["verification_status"] != "verified"
    assert projection.best_current_direction["verification_status"] != "verified"
    assert projection.best_current_direction["route"] == "best_current"


def test_final_projection_binds_synthesis_answer_to_best_current_candidate_id() -> None:
    candidate = CandidateGenome(
        id="chosen",
        artifact={"direction": "recoverable evidence-debt frontier"},
        concise_claim="Recoverable evidence-debt frontier",
    )
    synthesis = SynthesizedResult(
        status="completed",
        final_answer="Synthesis text should stay bound to the selected candidate.",
        best_current_direction={"candidate_id": "chosen"},
    )

    projection = build_final_projection(
        population=CandidatePopulation([candidate]),
        synthesis=synthesis,
        graded_output=_graded_portfolio(),
    )

    assert projection.candidate_id == "chosen"
    assert projection.best_current_direction["candidate_id"] == "chosen"
    assert "answer_unbound_to_candidate_artifact" not in projection.advisory_issues


def test_final_projection_unwraps_best_current_direction_carrier_to_real_direction() -> None:
    direction = CandidateGenome(
        id="direction",
        artifact={"mechanism": "recover dormant factors when they change the next search action"},
        concise_claim="Dormant factor resurrection is the best current direction.",
        metadata={"intent_binding": {"direct_answer_score": 0.94}},
    )
    carrier = CandidateGenome(
        id="carrier",
        artifact={
            "best_current_direction": {
                "candidate_id": "direction",
                "direction_name": "minimal active core with useful resurrection attachment",
            },
            "claim_permissions": {"may_claim_verified": False},
        },
        artifact_type="best_current_direction_status_contract",
        concise_claim="Keep direction as best current but block verified claims.",
        metadata={"intent_binding": {"direct_answer_score": 0.14}},
    )
    synthesis = SynthesizedResult(
        status="completed",
        final_answer="carrier explains status; direction is the mechanism",
        best_current_direction={"candidate_id": "carrier"},
    )

    projection = build_final_projection(
        population=CandidatePopulation([direction, carrier]),
        synthesis=synthesis,
        graded_output=_graded_portfolio(),
    )

    assert projection.candidate_id == "direction"
    assert projection.best_current_direction["candidate_id"] == "direction"
    assert direction.metadata["best_current_direction_carriers"] == ["carrier"]


def test_user_facing_verification_accepts_graded_verified_result() -> None:
    candidate = CandidateGenome(id="verified", concise_claim="answer")
    graded = GradedOutput(
        mode="verified_result",
        verification_strength=VerificationStrength.FORMAL,
        result=VerifiedResult(answer="answer", replayable=True, evidence_ref="e1", verifier_fingerprint="vf"),
        replay_certificate={"scope": "verifier_on_frozen_artifact_only", "measured_strength_value": 4},
    )

    payload = best_current_direction_payload(candidate, graded_output=graded.to_dict())

    assert payload["verification_status"] == "verified"



def test_seed_reservoir_soft_retains_low_relevance_and_duplicates() -> None:
    duplicate = CandidateGenome(id="dup2", concise_claim="same", core_mechanism="same", artifact="same")

    harvester = CandidateHarvester(
        policy=HarvestPolicy(target_size=4, max_batches=1, relevance_floor=0.1, reservoir_mode=True)
    )
    result = harvester.harvest(
        request_batch=lambda *_args: [
            CandidateGenome(id="dup1", concise_claim="same", core_mechanism="same", artifact="same"),
            duplicate,
            CandidateGenome(id="low", concise_claim="", core_mechanism="", artifact=""),
        ],
        context={},
    )

    assert [candidate.id for candidate in result.accepted] == ["dup1"]
    assert {candidate.id for candidate in result.reservoir} == {"dup2", "low"}
    assert {item["reason"] for item in result.rejected} >= {"duplicate_semantic_signature", "low_relevance"}
    assert duplicate.metadata["candidate_budget_decision"]["action"] == "soft_reservoir"


def test_duplicate_offspring_is_soft_retained_with_budget_trace() -> None:
    parent = CandidateGenome(id="p", artifact="same", concise_claim="same", core_mechanism="same")
    child = CandidateGenome(id="c", parent_ids=["p"], artifact="same", concise_claim="same", core_mechanism="same")

    kept = dedupe_offspring_against_population([child], CandidatePopulation([parent]))

    assert kept == [child]
    assert child.metadata["candidate_budget_decision"]["reason"] == "duplicate_semantic_signature"
    assert "productive_child_observation" in child.metadata["nextgen"]


def test_source_free_verifier_failure_is_final_advisory_not_terminal_fate() -> None:
    child = CandidateGenome(id="child", concise_claim="bold hypothesis", core_mechanism="critical branching")

    verify_offspring([child], lambda items: [{"candidate_id": items[0].id, "passed": False, "diagnostics": ["source_free_final_claim"]}])

    assert child.current_fate == CandidateFate.ACTIVE.value
    assert child.metadata["final_answer_advisory"]["final_eligible"] is False
    assert child.metadata["candidate_budget_decision"]["action"] == "soft_retain"


def test_budget_eligible_lane_includes_nonstructural_reserve_but_excludes_structural() -> None:
    reserve = CandidateGenome(id="reserve", current_fate=CandidateFate.FAILED, concise_claim="keep exploring")
    structural = CandidateGenome(id="bad", current_fate=CandidateFate.FAILED, concise_claim="unsafe", metadata={"structural_failure": True})

    assert budget_eligible_candidates([reserve, structural]) == [reserve]
    assert structurally_blocked(structural) is True


def test_unknown_search_space_family_becomes_singleton_not_first_family() -> None:
    search_map = build_search_space_map({"candidate_families": [{"id": "known", "description": "known"}]}, requested_candidate_count=1)
    result = classify_candidate({"core_mechanism": "orthogonal new mechanism", "metadata": {"search_space": {"family_id": "alien"}}}, search_map)

    assert result["family_id"].startswith("singleton_")
    assert result["family_id"] != "known"
    assert result["classification_reason"] == "nextgen_provisional_singleton_family"


def test_productive_observation_has_no_gate_booleans() -> None:
    parent = CandidateGenome(id="p", concise_claim="critical branching")
    child = CandidateGenome(id="c", parent_ids=["p"], concise_claim="critical branching with new theorem")

    payload = observe_productive_child(parent, child).to_dict()

    assert "passed" not in payload
    assert "must_not_block" not in payload
    assert child.metadata["nextgen"]["productive_child_observation"] == payload


def test_cbt_soft_quota_protects_singletons_and_never_blocks() -> None:
    singleton = CandidateGenome(id="s", concise_claim="cross-family jump", multihead_scores={"rarity": 0.8})

    adjustment = cbt_soft_budget_adjustment(singleton, [singleton])

    assert adjustment > 0
    assert singleton.metadata["nextgen"]["cbt_soft_quota"]["floor"] == 1
    assert singleton.metadata["candidate_budget_decision"]["action"] == "soft_boost"


def test_prompt_protection_keeps_frontier_candidate_under_tiny_budget() -> None:
    candidates = [
        CandidateGenome(id=f"c{i}", concise_claim=f"candidate {i}", core_mechanism="ordinary")
        for i in range(12)
    ]
    protected = CandidateGenome(id="critical", concise_claim="critical branching", core_mechanism="protected")
    candidates.append(protected)

    view = build_prompt_view(
        "nexus_diagnose_search_state",
        {"candidates": candidates, "_prompt_context_controls": {"protect_candidate_ids": ["critical"]}},
        max_chars=900,
    )

    assert "critical" in {item.get("id") for item in view.payload.get("candidates", []) if isinstance(item, dict)}
    assert view.metadata["protected_paths_applied"] == []


def test_llm_model_spec_profile_identity_is_profile_scoped() -> None:
    status = LLMModelSpec(profile_id="seed-explorer", provider="direct_http", model="model-a").apply_to_status({})
    identity = identity_from_status(status, request_type="nexus_seed_population")

    assert identity.profile_id == "seed-explorer"
    assert identity.breaker_key == "seed-explorer"
    assert identity.provider == "direct_http"


def test_llm_profile_identity_reaches_ledger_journal_and_telemetry(tmp_path, monkeypatch) -> None:
    class Provider:
        provider_id = "unit"

        def __init__(self) -> None:
            self.seen_kwargs = {}

        def complete_json(self, **_kwargs):
            self.seen_kwargs = dict(_kwargs)
            return LLMProviderResult(
                response=SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                ),
                attempts=1,
                estimated_cost_usd=0.25,
            )

    monkeypatch.setenv("COGEV_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("COGEV_LLM_MODEL", "default-model")
    monkeypatch.setenv("COGEV_LLM_RETRY_ATTEMPTS", "1")
    journal = tmp_path / "journal"
    ledger = tmp_path / "call-ledger.jsonl"

    provider = Provider()
    with llm_session(LLMSession(journal_dir=str(journal), call_ledger_path=str(ledger))) as session:
        response = llm_json(
            "nexus_seed_population",
            {"x": 1},
            system="Return JSON",
            schema_hint={},
            provider=provider,
            model_spec=LLMModelSpec(profile_id="seed-explorer", provider="litellm", model="same-provider-model"),
        )
        events = session.snapshot()

    assert response["ok"] is True
    assert provider.seen_kwargs["model"] == "same-provider-model"
    assert events[-1]["provider"] == "litellm"
    assert events[-1]["model_profile_id"] == "seed-explorer"
    assert events[-1]["llm_call_identity"]["route_role"] == "nexus_seed_population"
    ledger_rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert {row["model_profile_id"] for row in ledger_rows} == {"seed-explorer"}
    assert ledger_rows[-1]["estimated_cost_usd"] == 0.25
    journal_rows = [json.loads(line) for line in (journal / "llm-calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert journal_rows[-1]["model_profile_id"] == "seed-explorer"
    assert journal_rows[-1]["status"] == "ok"


def test_seed_model_ensemble_returns_input_order_with_origin_trace() -> None:
    class SeedModel:
        def __init__(self, label: str) -> None:
            self.label = label

        def seed_population(self, **_kwargs):
            return [{"id": self.label, "concise_claim": self.label}]

    result = SeedModelEnsembleAdapter([SeedModel("a"), SeedModel("b")]).seed_population(contract=None, world=None, policy=None)

    assert [item["id"] for item in result] == ["a", "b"]
    assert [item["metadata"]["origin_model_index"] for item in result] == [0, 1]


def test_resurrection_lane_prefers_intent_aligned_loser_pool_candidate() -> None:
    active = CandidateGenome(id="active", current_fate=CandidateFate.ACTIVE.value, concise_claim="baseline answer", multihead_scores={"answer_likelihood": 0.5})
    aligned = CandidateGenome(
        id="aligned",
        current_fate=CandidateFate.FAILED.value,
        concise_claim="The candidate directly explores the frozen search objective.",
        core_mechanism="intent aligned high-variance direction",
        metadata={"intent_binding": {"direct_answer_score": 0.9}},
        multihead_scores={"novelty": 0.8, "rarity": 0.7, "answer_likelihood": 0.7},
    )
    support = CandidateGenome(
        id="support",
        current_fate=CandidateFate.FAILED.value,
        concise_claim="Supporting record around the same run.",
        core_mechanism="supporting record",
        metadata={"intent_binding": {"direct_answer_score": 0.1}},
        multihead_scores={"answer_likelihood": 0.95},
    )

    selected = ParentSelector().select([active, aligned, support], limit=2, eligibility_policy={"current_round": 9})

    assert [candidate.id for candidate in selected] == ["active", "aligned"]
    assert aligned.metadata["resurrection_lane"] is True
    assert aligned.metadata["candidate_budget_decision"]["hard_gate"] is False
    assert aligned.metadata["resurrection_reason"] == "intent_aligned_resurrection"
    assert aligned.metadata["resurrection_round"] == 9


def test_seed_reservoir_limit_truncates_with_summary() -> None:
    harvester = CandidateHarvester(
        policy=HarvestPolicy(target_size=1, max_batches=1, relevance_floor=1.1, reservoir_mode=True, reservoir_limit=2)
    )
    batch = [CandidateGenome(id=f"low-{i}", concise_claim=f"candidate {i}") for i in range(5)]

    result = harvester.harvest(request_batch=lambda *_args: batch, context={})

    assert [candidate.id for candidate in result.reservoir] == ["low-0", "low-1"]
    assert result.reservoir_truncated_count == 3
    assert [item["candidate_id"] for item in result.reservoir_truncated_summaries] == ["low-2", "low-3", "low-4"]
    assert result.to_dict()["reservoir_truncated_count"] == 3


def test_seed_model_ensemble_candidate_genome_origin_trace() -> None:
    class SeedModel:
        def __init__(self, candidate: CandidateGenome) -> None:
            self.candidate = candidate

        def seed_population(self, **_kwargs):
            return [self.candidate]

    a = CandidateGenome(id="a", concise_claim="a")
    b = CandidateGenome(id="b", concise_claim="b")

    result = SeedModelEnsembleAdapter([SeedModel(a), SeedModel(b)]).seed_population(contract=None, world=None, policy=None)

    assert [candidate.id for candidate in result] == ["a", "b"]
    assert [candidate.metadata["origin_model_index"] for candidate in result] == [0, 1]


def test_nextgen_soft_signals_do_not_appear_in_hard_gate_consumers() -> None:
    root = Path(__file__).resolve().parents[1]
    consumers = [
        root / "cognitive_evolve_runtime/archives/manager.py",
        root / "cognitive_evolve_runtime/ranking/parent_selection.py",
        root / "cognitive_evolve_runtime/ranking/relative_rater.py",
        root / "cognitive_evolve_runtime/nexus/population_control.py",
        root / "cognitive_evolve_runtime/nexus/reproduction.py",
        root / "cognitive_evolve_runtime/nexus/project_verification.py",
        root / "cognitive_evolve_runtime/nexus/repair_reactivation.py",
        root / "cognitive_evolve_runtime/nexus/synthesis.py",
        root / "cognitive_evolve_runtime/nexus/display_selection.py",
        root / "cognitive_evolve_runtime/nexus/final_projection.py",
        root / "cognitive_evolve_runtime/nexus/final_gate.py",
        root / "cognitive_evolve_runtime/tools/verification_stack.py",
    ]
    forbidden = (
        "productive_child_observation",
        "must_not_block",
        "near_verbatim_reskin",
        "engineering_noise",
        "would_throttle",
    )

    for path in consumers:
        text = path.read_text(encoding="utf-8")
        assert all(token not in text for token in forbidden), path
