"""Split Nexus evolution loop package.

Public imports remain compatible with the historical
``cognitive_evolve_runtime.nexus.loop`` module while implementation is divided
by lifecycle concern.
"""
from __future__ import annotations

from cognitive_evolve_runtime.nexus.reproduction import elite_gap_merge_offspring as _elite_gap_merge_offspring

from .budget import EvolutionBudget, EvolutionLoopResult
from .closure import _attach_latent_replay_audit_to_closure, _closure_certificate, _completion_status_for_budget, _is_solved_stop_reason
from .controller import EvolutionLoopController, evolve_once
from .policy_directives import _attach_policy_directives_to_plans
from .round import EvolutionRound, RoundEvaluation
from .seeding import PROJECT_SEED_TYPES, TEXT_SEED_TYPES, seed_population

__all__ = [
    "TEXT_SEED_TYPES",
    "PROJECT_SEED_TYPES",
    "EvolutionBudget",
    "EvolutionLoopResult",
    "RoundEvaluation",
    "EvolutionRound",
    "EvolutionLoopController",
    "seed_population",
    "evolve_once",
    "_attach_latent_replay_audit_to_closure",
    "_attach_policy_directives_to_plans",
    "_closure_certificate",
    "_completion_status_for_budget",
    "_elite_gap_merge_offspring",
    "_is_solved_stop_reason",
]
