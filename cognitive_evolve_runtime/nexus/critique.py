"""Candidate critique stage for Nexus evolution.

This absorbs the useful old adaptive pattern of "generate → critique → mutate"
without bringing back the old runtime.  Critiques are structured genome feedback
that can survive checkpoints and steer mutation planning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus._serde import coerce_dict, coerce_str_list, utc_now
from cognitive_evolve_runtime.nexus._shared import MODEL_BOUNDARY_ERRORS
from cognitive_evolve_runtime.nexus.model_errors import is_quota_error
from cognitive_evolve_runtime.nexus.fallbacks import record_fallback
from cognitive_evolve_runtime.nexus.protocols import NexusModelLike

LATENT_EXPLORATION_MUTATION_PREFIX = "latent_exploration:"
LATENT_EXPLORATION_MISSING_PREFIX = "latent exploration sidecar"


@dataclass
class CandidateCritique:
    candidate_id: str
    round: int
    strengths: list[str] = field(default_factory=list)
    flaws: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    proposed_mutations: list[str] = field(default_factory=list)
    reusable_genes: list[str] = field(default_factory=list)
    severity: float = 0.0
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data.get("metadata"):
            data.pop("metadata", None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateCritique":
        return cls(
            candidate_id=str(data.get("candidate_id") or ""),
            round=int(data.get("round") or 0),
            strengths=coerce_str_list(data.get("strengths")),
            flaws=coerce_str_list(data.get("flaws")),
            missing_evidence=coerce_str_list(data.get("missing_evidence")),
            proposed_mutations=coerce_str_list(data.get("proposed_mutations")),
            reusable_genes=coerce_str_list(data.get("reusable_genes")),
            severity=float(data.get("severity", 0.0) or 0.0),
            created_at=str(data.get("created_at") or utc_now()),
            metadata=coerce_dict(data.get("metadata")),
        )


class CritiqueEngine:
    def __init__(self, model: NexusModelLike | None = None) -> None:
        self.model = model

    def critique(
        self,
        *,
        candidates: list[CandidateGenome],
        round_index: int,
        contract: Any | None = None,
        policy: Any | None = None,
        archives: Any | None = None,
    ) -> list[CandidateCritique]:
        if self.model is not None and hasattr(self.model, "critique_candidates"):
            try:
                raw = self.model.critique_candidates(candidates=candidates, round_index=round_index, contract=contract, policy=policy, archives=archives)
                critiques = [item if isinstance(item, CandidateCritique) else CandidateCritique.from_dict(item) for item in raw or [] if isinstance(item, (CandidateCritique, dict))]
                if critiques:
                    return self._attach_latent_exploration_sidecars(critiques, contract=contract)
            except MODEL_BOUNDARY_ERRORS as exc:
                if is_quota_error(exc):
                    raise
                # Critique is an exploration accelerator, not a hard dependency.
                # The model failure will still be visible in the LLM journal; the
                # loop keeps deterministic critiques so candidates are persisted.
                record_fallback(stage="critique", reason=exc.__class__.__name__, detail=str(exc))
                for candidate in candidates:
                    candidate.metadata["model_critique_degraded"] = f"{exc.__class__.__name__}: {exc}"
        critiques = [self._deterministic_critique(candidate, round_index) for candidate in candidates]
        return self._attach_latent_exploration_sidecars(critiques, contract=contract)

    def apply(self, *, candidates: list[CandidateGenome], critiques: list[CandidateCritique]) -> None:
        by_id = {candidate.id: candidate for candidate in candidates}
        for critique in critiques:
            candidate = by_id.get(critique.candidate_id)
            if candidate is None:
                continue
            candidate.add_verification_feedback({"tool_id": "nexus_critique", "status": "observed", **critique.to_dict()})
            for flaw in critique.flaws[:2]:
                if flaw and flaw not in candidate.failure_lessons:
                    candidate.failure_lessons.append(flaw)
            for gene in critique.reusable_genes[:3]:
                if gene and gene not in candidate.inherited_genes:
                    candidate.inherited_genes.append(gene)
            if critique.missing_evidence:
                candidate.missing_parts = list(dict.fromkeys(candidate.missing_parts + critique.missing_evidence[:2]))

    def _deterministic_critique(self, candidate: CandidateGenome, round_index: int) -> CandidateCritique:
        flaws: list[str] = []
        missing: list[str] = []
        proposed: list[str] = []
        if candidate.metadata.get("search_seed_not_final"):
            flaws.append("initial search seed should be sharpened into a direct answer")
            proposed.append("deepen")
        if candidate.edge_knowledge_seeds:
            proposed.append("rare_inject")
        if candidate.multihead_scores.get("auxiliary_value", 0.0) > candidate.multihead_scores.get("answer_likelihood", 0.0):
            flaws.append("candidate may be auxiliary scaffold rather than answer body")
            proposed.append("core_extraction")
            proposed.append("scaffold_removal")
        reusable = [candidate.extract_inheritable_gene_summary()] if candidate.core_mechanism or candidate.concise_claim else []
        return CandidateCritique(
            candidate_id=candidate.id,
            round=round_index,
            strengths=[candidate.core_mechanism or candidate.concise_claim or "candidate has a preserved genome"],
            flaws=flaws,
            missing_evidence=missing,
            proposed_mutations=list(dict.fromkeys(proposed or ["deepen"])),
            reusable_genes=[gene for gene in reusable if gene],
            severity=min(1.0, 0.2 + 0.15 * len(flaws) + 0.1 * len(missing)),
        )

    def _attach_latent_exploration_sidecars(
        self,
        critiques: list[CandidateCritique],
        *,
        contract: Any | None,
    ) -> list[CandidateCritique]:
        if not critiques:
            return critiques
        exploration = _latent_exploration_plan_for_contract(contract, limit=len(critiques))
        actions = [dict(item) for item in exploration.get("latent_exploration_actions", []) if isinstance(item, dict)]
        if not actions:
            return critiques
        mutation_actions = coerce_str_list(exploration.get("mutation_actions"))
        for index, critique in enumerate(critiques):
            action = actions[index % len(actions)]
            action_id = str(action.get("action_id") or action.get("kind") or "latent_exploration").strip() or "latent_exploration"
            mutation_action = mutation_actions[index % len(mutation_actions)] if mutation_actions else "latent_exploration"
            targets = coerce_str_list(action.get("target_intent_ids"))
            sidecar = {
                "source": "runtime_bridge.latent_exploration_plan_for_contract",
                "role": "weak_sidecar",
                "action": action,
                "mutation_action": mutation_action,
                "target_intent_ids": targets,
                "latent_decision_trace": coerce_dict(exploration.get("latent_decision_trace")),
                "latent_posterior_snapshot_hash": str(exploration.get("latent_posterior_snapshot_hash") or ""),
                "latent_ledger_cursor": int(exploration.get("latent_ledger_cursor") or 0),
            }
            metadata = coerce_dict(critique.metadata)
            metadata["latent_exploration_directive"] = sidecar
            critique.metadata = metadata

            mutation_directive = f"{LATENT_EXPLORATION_MUTATION_PREFIX}{mutation_action}"
            evidence_directive = (
                f"{LATENT_EXPLORATION_MISSING_PREFIX} {action_id}: collect posterior-updating evidence"
                + (f" for intents {', '.join(targets)}" if targets else "")
            )
            if mutation_directive not in critique.proposed_mutations:
                critique.proposed_mutations.append(mutation_directive)
            if evidence_directive not in critique.missing_evidence:
                critique.missing_evidence.append(evidence_directive)
        return critiques


def _latent_exploration_plan_for_contract(contract: Any | None, *, limit: int) -> dict[str, Any]:
    try:
        from cognitive_evolve_runtime.outcomes.runtime_bridge import latent_exploration_plan_for_contract

        return coerce_dict(latent_exploration_plan_for_contract(contract, limit=limit))
    except Exception:
        # Latent critique sidecars are deliberately weak accelerators.  A broken
        # bridge import or malformed latent payload must not make critique itself
        # unavailable, especially while other runtime lanes are evolving.
        return _fallback_latent_exploration_plan_for_contract(contract, limit=limit)


def _fallback_latent_exploration_plan_for_contract(contract: Any | None, *, limit: int) -> dict[str, Any]:
    metadata = coerce_dict(getattr(contract, "metadata", None))
    if not metadata and isinstance(contract, dict):
        metadata = coerce_dict(contract.get("metadata"))
    state = coerce_dict(metadata.get("latent_problem_state") or metadata.get("latent_state"))
    raw_actions = state.get("actions")
    if not isinstance(raw_actions, list):
        return {}
    actions = [
        action
        for action in (coerce_dict(item) for item in raw_actions)
        if action.get("action_id") or action.get("kind")
    ][: max(0, int(limit or 0))]
    if not actions:
        return {}
    return {
        "latent_exploration_actions": actions,
        "mutation_actions": [_mutation_action_for_raw_exploration(action) for action in actions],
        "latent_decision_trace": {},
        "latent_posterior_snapshot_hash": "",
        "latent_ledger_cursor": 0,
    }


def _mutation_action_for_raw_exploration(action: dict[str, Any]) -> str:
    kind = str(action.get("kind") or "").lower()
    if "intent" in kind or "disambigu" in kind or "probe" in kind:
        return "case_split"
    if "risk" in kind or "verify" in kind or "evidence" in kind:
        return "tool_ground"
    if "divers" in kind or "novel" in kind:
        return "rare_inject"
    if "improvement" in kind or "candidate" in kind:
        return "deepen"
    return "latent_exploration"


__all__ = ["CandidateCritique", "CritiqueEngine"]
