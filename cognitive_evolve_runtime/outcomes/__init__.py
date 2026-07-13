"""Verifiable outcome-improvement kernel primitives."""

from . import anytime_valid as _anytime_valid
from . import calibration as _calibration
from . import closure_bundle as _closure_bundle
from . import falsification as _falsification
from . import problem_model_compaction as _problem_model_compaction

from .latent import (
    CandidateScore,
    ConvergenceAssessment,
    ExplorationAction,
    FrontierCandidate,
    IntentHypothesis,
    LatentProblemState,
    PreferenceEvidence,
    assess_convergence,
    freeze_outcome_contract,
    pareto_frontier,
    rank_candidates,
    select_exploration_action,
    update_intent_posteriors,
)

from .latent_ledger import (
    DECISION_BOUND_TO_POSTERIOR,
    EVIDENCE_ADDED,
    EVIDENCE_DEDUPLICATED,
    EVIDENCE_REJECTED,
    EVIDENCE_RETRACTED,
    EVIDENCE_SUPERSEDED,
    LatentLedger,
    LatentLedgerEvent,
    LatentLedgerReplay,
    LatentLedgerStore,
    preference_evidence_id,
)

from .posterior_update import (
    UPDATE_MODEL_VERSION,
    LatentPosteriorSnapshot,
    PosteriorUpdateConfig,
    bounded_update_intent_posteriors,
    materialize_posterior_snapshot,
)

from .problem_model import (
    MODEL_ADDED,
    MODEL_DEDUPLICATED,
    MODEL_DECISION_BOUND,
    MODEL_PROMOTED,
    MODEL_REJECTED,
    MODEL_RETIRED,
    MODEL_SUPERSEDED,
    MODEL_VALIDATED,
    PROBLEM_MODEL_EVOLUTION_VERSION,
    ModelDiscriminationAction,
    ProblemModelHypothesis,
    ProblemModelLedger,
    ProblemModelLedgerEvent,
    ProblemModelLedgerReplay,
    ProblemModelPrediction,
    ProblemModelSnapshot,
    ProblemModelValidation,
    ProblemObjective,
    ProblemResidual,
    StructuralProposal,
    compute_problem_model_complexity,
    detect_problem_residuals,
    initial_problem_model_from_latent_state,
    materialize_problem_model_snapshot,
    problem_model_event_id,
    propose_structural_models,
    select_model_discrimination_action,
    validate_problem_model_promotion,
)

from .evidence_feedback import (
    EvidenceAdapterOutput,
    EvidenceQuarantineRecord,
    adapt_archive_observation,
    adapt_critique_result,
    adapt_improvement_certificate,
    adapt_latent_feedback,
    adapt_trial_observation,
    adapt_verifier_result,
)

from .runtime_bridge import (
    annotate_candidates_with_latent_signals,
    apply_latent_exploration_to_mutation_plans,
    audit_latent_decision_replay,
    attach_problem_model_if_needed,
    attach_latent_state_if_needed,
    best_candidate_m5_certificate,
    evaluate_m6_closure_gate,
    extract_runtime_trial_observations,
    freeze_improvement_certificate_from_trials,
    improvement_certificate_from_any,
    ingest_latent_feedback,
    ingest_runtime_trial_feedback,
    latent_completion_override,
    latent_exploration_plan_for_contract,
    latent_ledger_from_contract,
    latent_state_from_contract,
    latent_state_from_dict,
    latent_state_summary,
    latent_stop_allows_solved,
    materialize_contract_latent_posterior,
    materialize_contract_problem_model_snapshot,
    m5_certificate_summary,
    problem_model_discrimination_plan_for_contract,
    problem_model_from_contract,
    problem_model_ledger_from_contract,
    problem_model_summary,
    propose_problem_models_for_contract,
    requires_verified_improvement,
)

from .latent_audit import (
    audit_latent_replay_bundle,
    collect_latent_decision_traces,
)

from .improvement import (
    ImprovementCertificate,
    ImprovementEdge,
    OutcomeContract,
    OutcomeMetric,
    TrialObservation,
    certificate_from_dict,
    compare_outcomes,
    improvement_edge,
    verify_certificate,
)

from .anytime_valid import *  # noqa: F403
from .calibration import *  # noqa: F403
from .closure_bundle import *  # noqa: F403
from .falsification import *  # noqa: F403
from .problem_model_compaction import *  # noqa: F403

__all__ = [
    "annotate_candidates_with_latent_signals",
    "apply_latent_exploration_to_mutation_plans",
    "audit_latent_decision_replay",
    "audit_latent_replay_bundle",
    "attach_problem_model_if_needed",
    "attach_latent_state_if_needed",
    "best_candidate_m5_certificate",
    "evaluate_m6_closure_gate",
    "extract_runtime_trial_observations",
    "freeze_improvement_certificate_from_trials",
    "improvement_certificate_from_any",
    "ingest_latent_feedback",
    "ingest_runtime_trial_feedback",
    "latent_completion_override",
    "latent_exploration_plan_for_contract",
    "latent_ledger_from_contract",
    "latent_state_from_contract",
    "latent_state_from_dict",
    "latent_state_summary",
    "latent_stop_allows_solved",
    "materialize_contract_latent_posterior",
    "materialize_contract_problem_model_snapshot",
    "m5_certificate_summary",
    "problem_model_discrimination_plan_for_contract",
    "problem_model_from_contract",
    "problem_model_ledger_from_contract",
    "problem_model_summary",
    "propose_problem_models_for_contract",
    "requires_verified_improvement",
    "LatentLedger",
    "LatentLedgerEvent",
    "LatentLedgerReplay",
    "LatentLedgerStore",
    "LatentPosteriorSnapshot",
    "PosteriorUpdateConfig",
    "MODEL_ADDED",
    "MODEL_DEDUPLICATED",
    "MODEL_DECISION_BOUND",
    "MODEL_PROMOTED",
    "MODEL_REJECTED",
    "MODEL_RETIRED",
    "MODEL_SUPERSEDED",
    "MODEL_VALIDATED",
    "PROBLEM_MODEL_EVOLUTION_VERSION",
    "ModelDiscriminationAction",
    "ProblemModelHypothesis",
    "ProblemModelLedger",
    "ProblemModelLedgerEvent",
    "ProblemModelLedgerReplay",
    "ProblemModelPrediction",
    "ProblemModelSnapshot",
    "ProblemModelValidation",
    "ProblemObjective",
    "ProblemResidual",
    "StructuralProposal",
    "compute_problem_model_complexity",
    "detect_problem_residuals",
    "initial_problem_model_from_latent_state",
    "materialize_problem_model_snapshot",
    "problem_model_event_id",
    "propose_structural_models",
    "select_model_discrimination_action",
    "validate_problem_model_promotion",
    "EvidenceAdapterOutput",
    "EvidenceQuarantineRecord",
    "UPDATE_MODEL_VERSION",
    "EVIDENCE_ADDED",
    "EVIDENCE_DEDUPLICATED",
    "EVIDENCE_REJECTED",
    "EVIDENCE_RETRACTED",
    "EVIDENCE_SUPERSEDED",
    "DECISION_BOUND_TO_POSTERIOR",
    "adapt_archive_observation",
    "adapt_critique_result",
    "adapt_improvement_certificate",
    "adapt_latent_feedback",
    "adapt_trial_observation",
    "adapt_verifier_result",
    "bounded_update_intent_posteriors",
    "materialize_posterior_snapshot",
    "preference_evidence_id",
    "update_intent_posteriors",
    "select_exploration_action",
    "rank_candidates",
    "pareto_frontier",
    "freeze_outcome_contract",
    "assess_convergence",
    "PreferenceEvidence",
    "LatentProblemState",
    "IntentHypothesis",
    "FrontierCandidate",
    "ExplorationAction",
    "ConvergenceAssessment",
    "CandidateScore",
    "ImprovementCertificate",
    "ImprovementEdge",
    "OutcomeContract",
    "OutcomeMetric",
    "TrialObservation",
    "certificate_from_dict",
    "compare_outcomes",
    "collect_latent_decision_traces",
    "improvement_edge",
    "verify_certificate",
]

__all__ = list(
    dict.fromkeys(
        __all__
        + list(_anytime_valid.__all__)
        + list(_calibration.__all__)
        + list(_closure_bundle.__all__)
        + list(_falsification.__all__)
        + list(_problem_model_compaction.__all__)
    )
)
