from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateFate
from cognitive_evolve_runtime.nexus.fate_semantics import fate_semantics, fate_semantics_table


def test_dormant_is_parked_reactivation_pool_not_live_parent_or_terminal() -> None:
    dormant = fate_semantics(CandidateFate.DORMANT.value)

    assert dormant.role == "parked_reactivation_pool"
    assert dormant.may_be_parent is False
    assert dormant.may_reactivate is True
    assert dormant.terminal is False
    assert dormant.final_answer_candidate is False


def test_incubating_is_bounded_repair_lane_and_failed_is_terminal() -> None:
    incubating = fate_semantics(CandidateFate.INCUBATING.value)
    failed = fate_semantics(CandidateFate.FAILED.value)

    assert incubating.may_be_parent is True
    assert incubating.terminal is False
    assert incubating.final_answer_candidate is False
    assert failed.terminal is True
    assert failed.may_be_parent is False


def test_all_candidate_fates_have_documented_semantics() -> None:
    table = fate_semantics_table()

    assert set(table) == CandidateFate.ALL
    assert all(item["notes"] for item in table.values())
