"""External evaluator and Evidence Control Plane public boundary."""
from .artifact_normalizer import artifact_policy_from_config, normalize_artifact
from .challenge_memory import ChallengeMemory, ChallengeMemoryItem, challenge_from_diagnostic, challenge_id, classify_diagnostic
from .dossier_grounding import CLAIM_STATUSES, frozen_dossier_hash, ground_frozen_dossier_claims
from .evidence_authority import EVIDENCE_AUTHORITY_ORDER, aggregate_evidence_state, artifact_identity_payload, evidence_artifact_hash, evidence_authority, evidence_authority_rank, evidence_revokes_final, stable_artifact_identity_hash
from .evidence import (
    ArtifactPolicy,
    EvidenceRecord,
    SearchPressure,
    apply_evidence_record,
    evidence_advisory_features,
    evidence_final_blocked,
    evidence_parent_blocked,
    evidence_records,
    evidence_repair_value,
    evidence_search_score,
    evidence_state,
    evidence_terminal_reject,
    has_repair_value,
    latest_evidence_record,
    select_preliminary_incumbent,
)
from .progressive import ProgressiveEvaluator
from .result import EvaluatorResult
from .runner import ExternalEvaluatorRunner, apply_evaluator_result
from .spec import EvaluatorMetricSpec, EvaluatorSpec

__all__ = [
    "ArtifactPolicy",
    "EVIDENCE_AUTHORITY_ORDER",
    "ChallengeMemory",
    "ChallengeMemoryItem",
    "CLAIM_STATUSES",
    "EvaluatorMetricSpec",
    "EvaluatorResult",
    "EvaluatorSpec",
    "EvidenceRecord",
    "ExternalEvaluatorRunner",
    "ProgressiveEvaluator",
    "SearchPressure",
    "apply_evaluator_result",
    "apply_evidence_record",
    "artifact_policy_from_config",
    "artifact_identity_payload",
    "challenge_from_diagnostic",
    "challenge_id",
    "classify_diagnostic",
    "aggregate_evidence_state",
    "evidence_advisory_features",
    "evidence_artifact_hash",
    "evidence_authority",
    "evidence_authority_rank",
    "evidence_final_blocked",
    "evidence_parent_blocked",
    "evidence_records",
    "evidence_repair_value",
    "evidence_search_score",
    "evidence_state",
    "evidence_revokes_final",
    "evidence_terminal_reject",
    "frozen_dossier_hash",
    "ground_frozen_dossier_claims",
    "has_repair_value",
    "latest_evidence_record",
    "normalize_artifact",
    "select_preliminary_incumbent",
    "stable_artifact_identity_hash",
]
