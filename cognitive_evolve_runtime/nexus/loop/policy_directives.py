"""Policy-directive extraction and normalization for Nexus rounds."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.mutation import MutationOperator, MutationPlan
from cognitive_evolve_runtime.nexus.critique import CandidateCritique
from cognitive_evolve_runtime.nexus.obligations import candidate_obligation_delta
from cognitive_evolve_runtime.nexus.policy import EvolutionPolicy

from .repair_guidance import _failure_micro_guidance_for_parent, _repair_requirement_for_parent, _repair_seed_for_parent, _source_integration_points_for_parent

def _critique_actions(critiques: list[CandidateCritique]) -> list[str]:
    actions: list[str] = []
    for critique in critiques:
        actions.extend(critique.proposed_mutations)
    return actions or [MutationOperator.DEEPEN]


def _parent_for_plan(plan: MutationPlan, parents: list[CandidateGenome], index: int) -> CandidateGenome | None:
    by_id = {parent.id: parent for parent in parents}
    for parent_id in plan.parent_ids:
        parent = by_id.get(parent_id)
        if parent is not None:
            return parent
    if parents:
        return parents[index % len(parents)]
    return None


def _attach_policy_directives_to_plans(plans: list[MutationPlan], policy: EvolutionPolicy, *, parents: list[CandidateGenome] | None = None) -> list[MutationPlan]:
    metadata = dict(policy.metadata or {})
    mandatory_actions = [item for item in metadata.get("mandatory_actions", []) if item]
    required_evidence: list[Any] = []
    blocked = [item for item in metadata.get("blocked_or_overexplored_obligations", []) if item]
    source_required = False
    if not (mandatory_actions or required_evidence or blocked or source_required or parents):
        return plans
    out: list[MutationPlan] = []
    for index, plan in enumerate(plans):
        plan_data = plan.to_dict()
        plan_metadata = dict(plan_data.get("metadata") or {})
        parent = _parent_for_plan(plan, parents or [], index)
        integration_points = _source_integration_points_for_parent(parent) if parent is not None else []
        repair_directives = _failure_micro_guidance_for_parent(parent)
        repair_requirement = _repair_requirement_for_parent(parent)
        repair_seed = _repair_seed_for_parent(parent)
        if repair_seed:
            plan_metadata["legacy_repair_seed_advisory"] = repair_seed
            plan_metadata["targeted_repair_lane"] = False
            plan_metadata["disallowed_repeat_patterns"] = list(repair_seed.get("disallowed_repeat_patterns", []) or [])
            blocker_text = ", ".join(str(item) for item in repair_seed.get("blockers", []) if item)
            plan_data["instruction"] = (
                f"{plan_data.get('instruction') or plan.instruction} | ".strip(" |")
                + "Legacy verifier blockers are advisory only; preserve the useful mechanism, "
                + f"ignore engineering blockers {blocker_text or 'recorded verifier blockers'}, "
                + "and produce a bold direct answer rather than a patch/proof checklist."
            )
        if repair_requirement:
            if parent is not None:
                try:
                    previous_attempts = int(parent.metadata.get("repair_attempts") or 0)
                except (TypeError, ValueError):
                    previous_attempts = 0
                parent.metadata["repair_attempts"] = max(0, previous_attempts) + 1
                attempt = int(parent.metadata.get("repair_attempts") or 0)
            else:
                attempt = 0
            plan_metadata["legacy_repair_requirement_advisory"] = repair_requirement
            plan_metadata["targeted_repair_lane"] = False
            plan_metadata["repair_attempt"] = attempt
            plan_metadata["repair_attempts"] = attempt
            if not plan_data.get("instruction") or "repair" not in str(plan_data.get("instruction") or "").lower():
                blockers = ", ".join(str(item) for item in repair_requirement.get("blockers", []))
                plan_data["instruction"] = (
                    f"{plan_data.get('instruction') or plan.instruction} | ".strip(" |")
                    + f"Advisory legacy repair context: do not optimize for {blockers or 'verification blockers'}; deepen the answer mechanism instead."
                )
        if repair_directives:
            plan_metadata["legacy_repair_directives_advisory"] = repair_directives
            directive = repair_directives[index % len(repair_directives)]
            plan_data["instruction"] = (
                f"{plan_data.get('instruction') or plan.instruction} | ".strip(" |")
                + "Advisory verifier note: ignore as a blocker, but avoid repeating "
                + str(directive.get("blocker") or "")
                + "; deepen "
                + str(directive.get("next_action") or "")
                + ("; do not repeat " + str(directive.get("disallowed_repeat_pattern") or "") if directive.get("disallowed_repeat_pattern") else "")
                + "."
            )
        if mandatory_actions:
            plan_metadata["mandatory_actions"] = list(dict.fromkeys([str(item) for item in mandatory_actions]))
            action = str(mandatory_actions[index % len(mandatory_actions)])
            if not plan.instruction or "obligation" not in plan.instruction.lower():
                plan_data["instruction"] = (
                    f"{plan.instruction} | ".strip(" |")
                    + f"Exploration directive: {action}; use it only to discover a stronger direct answer, not to build proof/source scaffolding."
                )
        if required_evidence:
            plan_metadata["required_evidence_kinds"] = list(dict.fromkeys([str(item) for item in required_evidence]))
        if blocked:
            plan_metadata["target_obligation_ids"] = list(dict.fromkeys([str(item) for item in blocked]))
        elif parent is not None:
            delta = candidate_obligation_delta(parent)
            target_ids: list[str] = []
            for key in ("targeted", "blocked", "introduced"):
                value = delta.get(key)
                if isinstance(value, list):
                    target_ids.extend(str(item) for item in value if item)
                elif value:
                    target_ids.append(str(value))
            if target_ids:
                plan_metadata["target_obligation_ids"] = list(dict.fromkeys(target_ids))
        if source_required or integration_points:
            plan_metadata["legacy_source_integration_points_advisory"] = integration_points
            plan_metadata["source_grounding_required"] = False
            plan_metadata["requires_pre_fail_post_pass"] = False
            if "source" not in str(plan_data.get("instruction") or "").lower():
                plan_data["instruction"] = (
                    f"{plan_data.get('instruction') or plan.instruction} | ".strip(" |")
                    + "Source-binding context is transport/project scaffolding only; do not let it replace mathematical or conceptual mechanism search."
                )
        plan_data["metadata"] = plan_metadata
        out.append(MutationPlan.from_dict(plan_data))
    return out


__all__ = ["_attach_policy_directives_to_plans", "_critique_actions"]
