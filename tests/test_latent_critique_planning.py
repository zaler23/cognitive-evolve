from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusObjectiveContract
from cognitive_evolve_runtime.nexus.critique import CandidateCritique, CritiqueEngine


def _latent_state_payload() -> dict[str, object]:
    return {
        "version": "latent-problem-state/v1",
        "intents": [
            {
                "id": "clarity",
                "statement": "make it clearer",
                "posterior": 0.5,
                "utility_dimensions": ["clarity"],
            },
            {
                "id": "impact",
                "statement": "make it more impactful",
                "posterior": 0.5,
                "utility_dimensions": ["impact"],
            },
        ],
        "actions": [
            {
                "action_id": "probe_impact",
                "kind": "intent_disambiguation",
                "target_intent_ids": ["impact"],
                "information_gain": 0.7,
                "expected_improvement": 0.1,
            }
        ],
    }


def _latent_contract() -> NexusObjectiveContract:
    return NexusObjectiveContract(
        original_user_goal="evolve the best answer",
        normalized_goal="evolve the best answer",
        metadata={"latent_problem_state": _latent_state_payload()},
    )


def test_latent_action_attaches_to_deterministic_critique_as_weak_sidecar() -> None:
    critique = CritiqueEngine().critique(
        candidates=[CandidateGenome(id="C1")],
        round_index=1,
        contract=_latent_contract(),
    )[0]

    sidecar = critique.metadata["latent_exploration_directive"]
    assert sidecar["role"] == "weak_sidecar"
    assert sidecar["action"]["action_id"] == "probe_impact"
    assert sidecar["mutation_action"] == "case_split"
    assert "latent_exploration:case_split" in critique.proposed_mutations
    assert any("latent exploration sidecar probe_impact" in item for item in critique.missing_evidence)


def test_no_latent_state_keeps_critique_payload_shape_unchanged() -> None:
    critique = CritiqueEngine().critique(
        candidates=[CandidateGenome(id="C1")],
        round_index=1,
        contract=NexusObjectiveContract(original_user_goal="plain", normalized_goal="plain"),
    )[0]

    assert critique.metadata == {}
    assert "metadata" not in critique.to_dict()
    assert critique.proposed_mutations == ["deepen"]
    assert critique.missing_evidence == []


class _ModelCritique:
    def critique_candidates(self, **_: object) -> list[dict[str, object]]:
        return [
            {
                "candidate_id": "C1",
                "round": 2,
                "strengths": ["model says route is promising"],
                "flaws": ["needs sharper proof"],
                "missing_evidence": [],
                "proposed_mutations": ["deepen"],
                "severity": 0.4,
            }
        ]


def test_model_critique_gets_latent_sidecar_without_promoting_narrative() -> None:
    critique = CritiqueEngine(model=_ModelCritique()).critique(
        candidates=[CandidateGenome(id="C1")],
        round_index=2,
        contract=_latent_contract(),
    )[0]

    sidecar = critique.metadata["latent_exploration_directive"]
    assert critique.strengths == ["model says route is promising"]
    assert "model says route is promising" not in str(sidecar)
    assert sidecar["source"] == "runtime_bridge.latent_exploration_plan_for_contract"
    assert sidecar["role"] == "weak_sidecar"
    assert "deepen" in critique.proposed_mutations
    assert "latent_exploration:case_split" in critique.proposed_mutations
    assert any("posterior-updating evidence" in item for item in critique.missing_evidence)


def test_candidate_critique_serialization_accepts_legacy_and_sidecar_metadata() -> None:
    legacy = CandidateCritique.from_dict(
        {
            "candidate_id": "C1",
            "round": 1,
            "strengths": ["stable"],
            "missing_evidence": ["needs verifier"],
            "proposed_mutations": ["tool_ground"],
        }
    )

    assert legacy.metadata == {}
    assert "metadata" not in legacy.to_dict()

    with_sidecar = CandidateCritique.from_dict(
        {
            **legacy.to_dict(),
            "metadata": {"latent_exploration_directive": {"role": "weak_sidecar", "mutation_action": "case_split"}},
        }
    )
    payload = with_sidecar.to_dict()
    assert payload["metadata"]["latent_exploration_directive"]["role"] == "weak_sidecar"
    assert CandidateCritique.from_dict(payload).metadata == with_sidecar.metadata
