from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.display_selection import build_display_context, select_displayed_candidate
from cognitive_evolve_runtime.ranking.relative_rater import RelativeRankingResult


def _candidate(candidate_id: str, *, binding_class: str = "", artifact: str = "artifact") -> CandidateGenome:
    metadata = {}
    if binding_class:
        metadata["source_binding_manifest"] = {"binding_class": binding_class}
        metadata["source_binding_class"] = binding_class
    return CandidateGenome(id=candidate_id, artifact=artifact, concise_claim=f"claim {candidate_id}", core_mechanism=f"mechanism {candidate_id}", metadata=metadata)


def test_display_selector_keeps_ranking_order_even_when_source_required() -> None:
    contract = NexusObjectiveContract(
        original_user_goal="patch source",
        normalized_goal="patch source",
        verification_preferences=["source_binding", "local_tests"],
    )
    unbound_best = _candidate("A")
    resolved_reference = _candidate("B", binding_class="resolved")
    context = build_display_context(
        candidates=[resolved_reference, unbound_best],
        ranking=RelativeRankingResult(best_final_answer_id="A", strongest_mechanism_id="B"),
        contract=contract,
    )

    selection = select_displayed_candidate(
        context,
        candidates=[resolved_reference, unbound_best],
        final_eligible=lambda candidate: True,
    )

    assert [item.candidate_id for item in context.ordered_candidates[:2]] == ["A", "B"]
    assert selection.candidate_id == "A"
    assert selection.route == "final"
    assert unbound_best.metadata["display_source_binding_advisory"]["effect"] == "advisory_only_nonblocking"


def test_display_selector_does_not_block_invented_binding_in_answer_first_mode() -> None:
    invented = _candidate("invented", binding_class="invented")
    no_binding = _candidate("plain")
    context = build_display_context(
        candidates=[invented, no_binding],
        ranking={"best_final_answer_id": "invented", "strongest_mechanism_id": "plain"},
        contract=NexusObjectiveContract(original_user_goal="write a summary", normalized_goal="write a summary"),
    )

    selection = select_displayed_candidate(
        context,
        candidates=[invented, no_binding],
        final_eligible=lambda candidate: True,
    )

    assert selection.candidate_id == "invented"
    assert selection.route == "final"
    assert selection.blocked_reason == ""
    assert invented.metadata["display_source_binding_advisory"]["binding_class"] == "invented"
