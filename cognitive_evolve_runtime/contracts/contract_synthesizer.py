"""AI-driven contract synthesizer with deterministic guardrails.

This module intentionally avoids domain-pack routing.  It converts a task into a
contract describing what success, progress, drift, evidence, and abstention mean.
LLM generation can be layered on top by adapters, but this package always emits a
schema-valid baseline contract for audit and revision.
"""
from __future__ import annotations

from typing import Any

from .schemas import ContractItem, EvaluationContract, MATERIAL_DELTA_TYPES


class ContractSynthesizer:
    def synthesize(
        self,
        *,
        prompt: str,
        semantic_assessment: dict[str, Any] | None = None,
        evidence_plan: dict[str, Any] | None = None,
        previous_contract: dict[str, Any] | None = None,
    ) -> EvaluationContract:
        assessment = semantic_assessment or {}
        plan = evidence_plan or {}
        objective = str(assessment.get("real_objective") or assessment.get("surface_request") or prompt or "user objective").strip()
        task_type = str(assessment.get("task_type") or "structured_decision_or_design")
        requires_evidence = bool(plan.get("required")) or bool(plan.get("required_source_types"))
        required_source_types = [str(item) for item in plan.get("required_source_types", [])]

        hard_gates = [
            ContractItem("direct_objective_bridge", "Every candidate eligible for final selection must explicitly bridge its mechanism to the user objective.", verifier="contract_gate", hard=True),
            ContractItem("material_progress_required", "A round may continue only after recording at least one material delta tied to the contract.", verifier="progress_monitor", hard=True),
            ContractItem("no_heuristic_final_authority", "Weighted scores, Elo, or LLM judge preference cannot alone certify final success.", verifier="ranking_contract", hard=True),
            ContractItem("unsupported_decisive_claims_block_final", "Decisive claims must be externally or computationally supported, labeled uncertain, or blocked.", verifier="evidence_ledger", hard=True),
        ]
        if requires_evidence:
            hard_gates.append(ContractItem("required_evidence_adapter_resolved", "When an evidence adapter is required but absent/failed, the run must return evidence_blocked or partial_result.", verifier="evidence_policy", hard=True, metadata={"required_source_types": required_source_types}))
        if _looks_like_constructive_task(prompt, objective):
            hard_gates.append(ContractItem("object_level_artifact_or_mechanism", "The answer must provide an object-level artifact, mechanism, proof path, executable change, or verification plan rather than only meta-work.", verifier="contract_gate", hard=True))

        contract = EvaluationContract(
            id=f"evaluation-contract:{task_type}",
            objective=objective,
            success_conditions=[
                ContractItem("objective_specific_success", "The output satisfies the task-defined success conditions rather than optimizing generic plausibility.", verifier="contract_gate", hard=True),
                ContractItem("validated_or_explicitly_partial", "The final state is validated, or it explicitly reports partial/not_solved/evidence_blocked with reasons.", verifier="runtime_status_gate", hard=True),
            ],
            hard_gates=hard_gates,
            soft_objectives=[
                ContractItem("mechanism_specificity", "Prefer concrete mechanisms, tests, examples, construction traces, or dependency graphs over vague summaries."),
                ContractItem("novelty_with_relevance", "Reward novelty only when it closes a contract gap or opens a verifiable search axis."),
                ContractItem("risk_control", "Expose assumptions, reversibility, and unsupported or high-risk claims."),
            ],
            evidence_requirements=[
                ContractItem("model_hypothesis_not_evidence", "LLM/model hypotheses may seed mutations but cannot increase evidence_score.", verifier="evidence_ledger", hard=True),
                *[
                    ContractItem(f"source_type_{index}", f"Resolve required evidence source type: {source_type}.", verifier="evidence_adapter", hard=True, metadata={"source_type": source_type})
                    for index, source_type in enumerate(required_source_types, start=1)
                ],
            ],
            tool_requirements=[
                ContractItem("capability_resolution", "Resolve whether a local tool/test/proof/source adapter exists before claiming verified progress.", verifier="capability_resolver", hard=requires_evidence),
            ],
            disallowed_shortcuts=[
                ContractItem("advanced_jargon_without_bridge", "Do not reward high-status terminology unless it directly satisfies a hard gate.", verifier="drift_detector", hard=True),
                ContractItem("fixture_or_mock_success", "Do not treat fixture/mock/heuristic fallback as production success.", verifier="runtime_policy", hard=True),
                ContractItem("pretty_text_no_delta", "Do not count more polished wording as progress without material delta.", verifier="progress_monitor", hard=True),
            ],
            abstention_conditions=[
                ContractItem("evidence_blocked", "Required evidence is unavailable, adapter_required, adapter_failed, or source verification cannot run.", verifier="evidence_policy", hard=True),
                ContractItem("ranking_failed", "Pairwise/ranking cannot produce an uncertainty-bounded decision.", verifier="ranking_contract", hard=True),
                ContractItem("budget_exhausted", "Budget ends before hard gates or evidence requirements are resolved.", verifier="budget_check", hard=True),
            ],
            progress_metrics=[
                ContractItem(delta, delta.replace("_", " "), verifier="progress_monitor")
                for delta in sorted(MATERIAL_DELTA_TYPES)
            ],
            drift_signals=[
                ContractItem("lineage_without_gate_progress", "A high-scoring lineage repeats without hard-gate/evidence/verifier gain.", verifier="drift_detector"),
                ContractItem("terminology_attractor", "Candidates cluster around impressive terminology with no task bridge, verifier, or construction.", verifier="drift_detector"),
                ContractItem("meta_substitution", "The search substitutes routing/classification/evaluation meta-work for the object-level task.", verifier="drift_detector"),
            ],
            restart_conditions=[
                ContractItem("stagnation_window", "Trigger targeted_resample or branch_restart after a configured no-material-delta window.", verifier="stagnation_detector"),
                ContractItem("semantic_drift", "Trigger branch_restart when drift signals dominate the active frontier.", verifier="drift_detector"),
                ContractItem("contract_underdeveloped", "Revise the contract before continuing when hard gates or progress metrics are too broad.", verifier="contract_validator"),
            ],
            metadata={
                "task_type": task_type,
                "requires_evidence": requires_evidence,
                "required_source_types": required_source_types,
                "previous_contract_version": (previous_contract or {}).get("version") if isinstance(previous_contract, dict) else None,
                "domain_pack_hardcoding": False,
            },
        )
        return contract


def _looks_like_constructive_task(*texts: str) -> bool:
    joined = " ".join(texts).lower()
    needles = ["construct", "build", "implement", "prove", "disprove", "counterexample", "design", "derive", "verify", "构造", "实现", "证明", "反例", "验证", "落地"]
    return any(needle in joined for needle in needles)


__all__ = ["ContractSynthesizer"]
