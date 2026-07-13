"""Model-driven activation contract for Nexus exploration and mutation.

Activation is a prompt-side control surface, not a finite list of privileged
fields or domains.  The runtime asks the model to decide whether persona views,
cross-domain analogy, or conceptual blending are useful for the current task and
then requires any such inspiration to become a concrete artifact delta.
"""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.nexus._serde import coerce_dict


ACTIVATION_REQUESTS = {
    "nexus_seed_population",
    "nexus_plan_mutations",
    "nexus_generate_offspring",
    "nexus_diagnose_search_state",
}


def activation_prompt_contract(*, request_type: str, contract_view: dict[str, Any] | None = None, policy_view: dict[str, Any] | None = None, policy_metadata: dict[str, Any] | None = None, semantic_control: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a compact, domain-neutral activation instruction block.

    The words persona, cross_domain_analogy, and conceptual_blending are
    deliberate model-facing affordances.  They are not hard-coded categories:
    the model must choose relevant roles, source fields, and blends from the
    objective contract, not from a runtime whitelist.
    """

    contract_view = coerce_dict(contract_view)
    policy_view = coerce_dict(policy_view)
    policy_metadata = coerce_dict(policy_metadata)
    semantic = coerce_dict(semantic_control)
    dac = contract_view.get("dynamic_artifact_contract") if isinstance(contract_view, dict) else {}
    return {
        "request_type": request_type,
        "source": "model_driven_activation_contract",
        "runtime_does_not_choose_domains": True,
        "model_decides_activation_need": True,
        "activation_controls": {
            "persona": "When useful, synthesize contrasting expert, creator, operator, or critic viewpoints relevant to this objective; when factuality/evidence dominates, keep personas quiet and evidence-first.",
            "cross_domain_analogy": "Generate any useful analogy, discipline, craft, theory, trope, mechanism, or practice from the model's own knowledge; import only transferable structure and name why it transfers.",
            "conceptual_blending": "Blend mechanisms only if the blend produces a concrete artifact delta, worked fragment, example, proof object, patch intent, outline segment, scene beat, evaluation object, or other model-defined work product.",
        },
        "anti_empty_talk_rule": "Every activated idea must either create the requested artifact/refinement/extension/materialization or a smallest concrete repair obligation; labels and discussion alone are not progress.",
        "bias_guard": "Activated routes are search material, not proof of truth, quality, originality, or final fitness; verifier/ranking evidence still decides.",
        "dynamic_artifact_contract": dac or "follow the model-defined objective contract; examples are not domain limits",
        "policy_surface": {
            "candidate_niches": policy_view.get("candidate_niches", []),
            "mutation_operators": policy_view.get("mutation_operators", []),
            "activation_preference": coerce_dict(policy_metadata.get("activation_policy")),
        },
        "semantic_control": semantic,
    }


__all__ = ["ACTIVATION_REQUESTS", "activation_prompt_contract"]
