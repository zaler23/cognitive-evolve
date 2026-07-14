from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus.loop.seeding import _policy_for_seed_batch, _uncovered_seed_slots
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy
from cognitive_evolve_runtime.nexus.search_kernel.harvesting import CandidateHarvester, HarvestPolicy
from cognitive_evolve_runtime.ranking.parent_selection import reproductive_value


def _low(candidate_id: str) -> CandidateGenome:
    return CandidateGenome(id=candidate_id, concise_claim=f"low {candidate_id}")


def _qualified(candidate_id: str) -> CandidateGenome:
    return CandidateGenome(
        id=candidate_id,
        artifact={"candidate": candidate_id},
        concise_claim=f"qualified {candidate_id}",
        core_mechanism=f"mechanism {candidate_id}",
        niche_memberships=[f"niche {candidate_id}"],
    )


def test_low_relevance_candidates_are_carried_without_consuming_target_quota(monkeypatch) -> None:
    monkeypatch.setenv("COGEV_MODEL_FANOUT_CONCURRENCY", "1")
    batches = [[_low("low-1"), _low("low-2")], [_qualified("high-1"), _qualified("high-2"), _qualified("high-3")]]
    harvester = CandidateHarvester(
        policy=HarvestPolicy(target_size=3, max_batches=2, relevance_floor=0.5, low_gain_patience=2),
    )

    result = harvester.harvest(request_batch=lambda index, *_args: batches[index])

    assert [candidate.id for candidate in result.accepted] == ["low-1", "low-2", "high-1", "high-2", "high-3"]
    assert result.target_qualified_count == 3
    assert result.carried_low_relevance_count == 2
    assert result.stopped_reason == "target_reached"
    assert not any(item.get("reason") == "low_relevance" for item in result.rejected)
    assert result.reservoir == []
    assert all(candidate.metadata["candidate_budget_decision"]["action"] == "advisory_deprioritize" for candidate in result.accepted[:2])


def test_target_quota_does_not_truncate_candidates_already_returned_by_provider() -> None:
    harvester = CandidateHarvester(
        policy=HarvestPolicy(target_size=1, max_batches=1, relevance_floor=0.5),
    )

    result = harvester.harvest(request_batch=lambda *_args: [_qualified("high"), _low("carried-after-target")])

    assert [candidate.id for candidate in result.accepted] == ["high", "carried-after-target"]
    assert result.target_qualified_count == 1
    assert result.carried_low_relevance_count == 1


def test_low_relevance_only_batch_is_zero_gain_for_unbounded_seed_harvest() -> None:
    calls: list[int] = []
    harvester = CandidateHarvester(
        policy=HarvestPolicy(
            target_size=1,
            max_batches=None,
            relevance_floor=0.5,
            stage="seed",
            fanout_workers=1,
            exhaust_on_no_new=True,
            low_gain_patience=1,
        ),
    )

    result = harvester.harvest(request_batch=lambda index, *_args: calls.append(index) or [_low(f"low-{index}")])

    assert calls == [0]
    assert [candidate.id for candidate in result.accepted] == ["low-0"]
    assert result.target_qualified_count == 0
    assert result.stopped_reason == "low_gain_patience"


def test_seed_quota_and_slot_receipts_ignore_carried_low_relevance() -> None:
    slot = {
        "slot_id": "family::direct_mainstream::1",
        "family_id": "family",
        "seed_axis": "direct_mainstream",
    }
    carried = _low("carried")
    carried.metadata.update(
        {
            "search_kernel_target_qualified": False,
            "seed_type": slot["slot_id"],
            "search_space": {
                "family_id": slot["family_id"],
                "seed_axis": slot["seed_axis"],
                "seed_axis_claim": "declared but below runtime relevance floor",
            },
        }
    )
    qualified = _qualified("qualified")
    qualified.metadata["search_kernel_target_qualified"] = True

    assert _uncovered_seed_slots([slot], [carried]) == [slot]

    batch_policy = _policy_for_seed_batch(
        EvolutionPolicy(),
        batch_index=1,
        accepted=[carried, qualified],
        rejected=[],
        target_size=3,
        seed_portfolio=[],
    )
    assert batch_policy.metadata["requested_candidate_count"] == 2
    assert batch_policy.metadata["accepted_seed_count"] == 1
    assert batch_policy.metadata["carried_low_relevance_count"] == 1


def test_runtime_relevance_is_a_nonnegative_reproductive_soft_signal() -> None:
    high = _qualified("high")
    low = _qualified("low")
    high.metadata["search_kernel_relevance"] = 1.0
    low.metadata["search_kernel_relevance"] = 0.0

    assert reproductive_value(high, [high, low]) > reproductive_value(low, [high, low])
