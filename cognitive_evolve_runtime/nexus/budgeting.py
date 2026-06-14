"""Nexus round-budget resolution.

This is the Nexus-native replacement for the old adaptive budget hand-off.  It
keeps the project single-runtime while preserving the important legacy behavior:
API model tiers such as ``cognitive-evolve-one-shot-exhaustive`` must select an
adaptive budget instead of silently running a one-round offline fallback or
pretending a fixed profile cap is task completion.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from cognitive_evolve_runtime.nexus.request_context import get_evolution_profile, get_internal_round_cap

PROFILE_DEFAULT_ROUNDS: dict[str, int] = {
    # Deprecated compatibility export.  A value of 0 means "adaptive profile";
    # completion is decided by stop signals, not by a baked-in profile count.
    "one-shot": 0,
    "balanced": 0,
    "deep": 0,
    "ultra": 0,
    "exhaustive": 0,
    "frontier_proof": 0,
    "breakthrough": 0,
}

PROFILE_MIN_ROUNDS: dict[str, int] = {
    # Profiles do not impose domain-specific stop delays.  The model/verifier
    # stop policy decides readiness after the first completed round unless an
    # operator supplies COGEV_MIN_ROUNDS_BEFORE_STOP.
    "one-shot": 1,
    "balanced": 1,
    "deep": 1,
    "ultra": 1,
    "exhaustive": 1,
    "frontier_proof": 1,
    "breakthrough": 1,
}



PROFILE_INITIAL_CANDIDATES: dict[str, int] = {
    # 0 means "derive width from model/policy diversity"; operators can still
    # set COGEV_NEXUS_PROFILE_*_MIN_CANDIDATES for an explicit floor.
    "one-shot": 0,
    "balanced": 0,
    "deep": 0,
    "ultra": 0,
    "exhaustive": 0,
    "frontier_proof": 0,
    "breakthrough": 0,
}

PROFILE_BRANCH_FACTOR: dict[str, int] = {
    # 0 means "derive mutation width from the model-authored EvolutionPolicy".
    # Operators can still set COGEV_NEXUS_BRANCH_FACTOR or a profile-specific
    # branch factor for an explicit safety/performance shape.
    "one-shot": 0,
    "balanced": 0,
    "deep": 0,
    "ultra": 0,
    "exhaustive": 0,
    "frontier_proof": 0,
    "breakthrough": 0,
}

PROFILE_STOP_POLICY: dict[str, str] = {
    "one-shot": "llm_after_minimum",
    "balanced": "llm_after_minimum",
    "deep": "llm_after_minimum",
    "ultra": "adaptive_until_solved",
    "exhaustive": "adaptive_until_solved",
    "frontier_proof": "adaptive_until_solved",
    "breakthrough": "adaptive_until_solved",
}

PROFILE_SAFETY_ROUNDS: dict[str, int] = {
    "one-shot": 12,
    "balanced": 24,
    "deep": 96,
    "ultra": 160,
    "exhaustive": 240,
    "frontier_proof": 240,
    "breakthrough": 240,
}

ROUND_OVERRIDE_KEYS = (
    "rounds",
    "requested_rounds",
    "max_rounds",
    "cogev_rounds",
    "cognitive_evolve_rounds",
)


@dataclass(frozen=True)
class NexusRoundBudget:
    """Resolved round budget with provenance for runtime artifacts/tests."""

    max_rounds: int
    profile: str
    source: str
    scoped_cap: int | None = None
    explicit_override: int | None = None
    initial_candidate_count: int = 8
    mutation_branches_per_round: int = 0
    stop_policy: str = "llm_after_minimum"
    min_rounds_before_stop: int = 1
    adaptive: bool = False
    round_safety_limit: int = 0
    completion_requires_stop_signal: bool = False
    config_warnings: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["config_warnings"] = list(self.config_warnings or [])
        return data


def resolve_nexus_round_budget(context: dict[str, Any] | None = None, *, default_profile: str = "balanced") -> NexusRoundBudget:
    """Resolve the max round count for a Nexus run.

    Precedence:
    1. Explicit request/context round override.
    2. Request-local model cap from the OpenAI-compatible API model mapping.
       A cap of ``0`` means "adaptive profile" rather than one round.
    3. Request-local evolution profile safety limit.
    4. Environment defaults.

    The API's ``cognitive-evolve-one-shot-exhaustive`` profile maps here to a
    model/verification-driven adaptive budget even when the client does not
    pass ``rounds``.  Hitting the safety limit is a checkpoint/continuation
    condition, not a success condition.
    """

    ctx = dict(context or {})
    explicit = _explicit_round_override(ctx)
    profile = _profile(ctx, default_profile=default_profile)
    warnings = _legacy_config_warnings(profile)
    if explicit is not None:
        return _with_profile_shape(max_rounds=_clamp_rounds(explicit), profile=profile, source="explicit_request", explicit_override=explicit, warnings=warnings)

    scoped_cap = get_internal_round_cap()
    if scoped_cap is not None and int(scoped_cap) > 0:
        return _with_profile_shape(max_rounds=_clamp_rounds(scoped_cap), profile=profile, source="model_round_cap", scoped_cap=int(scoped_cap), warnings=warnings)

    # scoped_cap == 0 is an intentional adaptive hand-off from the API.
    safety_rounds = _profile_safety_limit(profile)
    return _with_profile_shape(max_rounds=_clamp_rounds(safety_rounds), profile=profile, source="adaptive_evolution_profile", scoped_cap=scoped_cap, adaptive=True, warnings=warnings)



def _with_profile_shape(
    *,
    max_rounds: int,
    profile: str,
    source: str,
    scoped_cap: int | None = None,
    explicit_override: int | None = None,
    adaptive: bool = False,
    warnings: list[str] | None = None,
) -> NexusRoundBudget:
    return NexusRoundBudget(
        max_rounds=max_rounds,
        profile=profile,
        source=source,
        scoped_cap=scoped_cap,
        explicit_override=explicit_override,
        initial_candidate_count=_profile_candidate_count(profile),
        mutation_branches_per_round=_profile_branch_factor(profile),
        stop_policy=_stop_policy(profile),
        min_rounds_before_stop=_min_rounds_before_stop(profile),
        adaptive=adaptive,
        round_safety_limit=max_rounds if adaptive else 0,
        completion_requires_stop_signal=adaptive,
        config_warnings=list(warnings or []),
    )


def _profile_candidate_count(profile: str) -> int:
    suffix = profile.replace('-', '_').upper()
    for env_name in (
        f"COGEV_NEXUS_PROFILE_{suffix}_MIN_CANDIDATES",
        f"COGEV_NEXUS_PROFILE_{suffix}_SEED_FLOOR",
        "COGEV_NEXUS_MIN_CANDIDATES",
    ):
        env_value = _parse_positive_int(os.environ.get(env_name))
        if env_value is not None:
            return env_value
    legacy_name = f"COGEV_NEXUS_PROFILE_{suffix}_CANDIDATES"
    if _legacy_env_enabled("COGEV_ACCEPT_LEGACY_PROFILE_CANDIDATES"):
        env_value = _parse_positive_int(os.environ.get(legacy_name))
        if env_value is not None:
            return env_value
    return PROFILE_INITIAL_CANDIDATES.get(profile, 0)


def _profile_branch_factor(profile: str) -> int:
    suffix = profile.replace('-', '_').upper()
    for env_name in (
        f"COGEV_NEXUS_PROFILE_{suffix}_BRANCH_FACTOR",
        "COGEV_NEXUS_BRANCH_FACTOR",
    ):
        env_value = _parse_positive_int(os.environ.get(env_name))
        if env_value is not None:
            return env_value
    if _legacy_env_enabled("COGEV_ACCEPT_LEGACY_MUTATION_BRANCH_FACTOR"):
        legacy_value = _parse_positive_int(os.environ.get("COGEV_MUTATION_BRANCH_FACTOR"))
        if legacy_value is not None:
            return legacy_value
    return PROFILE_BRANCH_FACTOR.get(profile, PROFILE_BRANCH_FACTOR["balanced"])


def _stop_policy(profile: str) -> str:
    value = os.environ.get("COGEV_STOP_POLICY") or PROFILE_STOP_POLICY.get(profile, PROFILE_STOP_POLICY["balanced"])
    normalized = str(value or "").strip().lower()
    aliases = {
        "budget_guarded": "adaptive_until_solved",
        "budget-guarded": "adaptive_until_solved",
        "adaptive": "adaptive_until_solved",
        "until_solved": "adaptive_until_solved",
        "until-solved": "adaptive_until_solved",
        "convergence": "convergence_or_max_rounds",
        "converged": "convergence_or_max_rounds",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {"llm_after_minimum", "convergence_or_max_rounds", "max_rounds", "adaptive_until_solved", "route_incomplete_single_diagnostic"}
    return normalized if normalized in allowed else PROFILE_STOP_POLICY.get(profile, PROFILE_STOP_POLICY["balanced"])


def _min_rounds_before_stop(profile: str) -> int:
    env_value = _parse_positive_int(os.environ.get("COGEV_MIN_ROUNDS_BEFORE_STOP"))
    if env_value is not None:
        return env_value
    return PROFILE_MIN_ROUNDS.get(profile, PROFILE_MIN_ROUNDS["balanced"])

def _profile(ctx: dict[str, Any], *, default_profile: str) -> str:
    value = ctx.get("evolution_profile") or ctx.get("profile") or get_evolution_profile() or os.environ.get("COGEV_EVOLUTION_PROFILE") or default_profile
    normalized = str(value or default_profile).strip().lower().replace("_", "-")
    aliases = {
        "oneshot": "one-shot",
        "one_shot": "one-shot",
        "default": "balanced",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in PROFILE_DEFAULT_ROUNDS else default_profile


def _explicit_round_override(ctx: dict[str, Any]) -> int | None:
    for value in _round_values(ctx):
        parsed = _parse_positive_int(value)
        if parsed is not None:
            return parsed
    return None


def _round_values(ctx: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    for key in ROUND_OVERRIDE_KEYS:
        values.append(ctx.get(key))
    raw_request = ctx.get("raw_request") if isinstance(ctx.get("raw_request"), dict) else {}
    if raw_request:
        for key in ROUND_OVERRIDE_KEYS:
            values.append(raw_request.get(key))
        metadata = raw_request.get("metadata") if isinstance(raw_request.get("metadata"), dict) else {}
        cogev = raw_request.get("cognitive_evolve") if isinstance(raw_request.get("cognitive_evolve"), dict) else {}
        extra = raw_request.get("extra_body") if isinstance(raw_request.get("extra_body"), dict) else {}
        for source in (metadata, cogev, extra):
            for key in ROUND_OVERRIDE_KEYS:
                values.append(source.get(key))
    return values


def _profile_default(profile: str) -> int:
    return PROFILE_DEFAULT_ROUNDS.get(profile, PROFILE_DEFAULT_ROUNDS["balanced"])


def _profile_safety_limit(profile: str) -> int:
    suffix = profile.replace('-', '_').upper()
    env_name = f"COGEV_NEXUS_PROFILE_{suffix}_SAFETY_ROUNDS"
    env_value = _parse_positive_int(os.environ.get(env_name))
    if env_value is not None:
        return env_value
    global_value = _parse_positive_int(os.environ.get("COGEV_NEXUS_SAFETY_MAX_ROUNDS"))
    if global_value is not None:
        return global_value
    legacy_name = f"COGEV_NEXUS_PROFILE_{suffix}_ROUNDS"
    if _legacy_env_enabled("COGEV_ACCEPT_LEGACY_PROFILE_ROUNDS"):
        rounds_env = _parse_positive_int(os.environ.get(legacy_name))
        if rounds_env is not None:
            return rounds_env
    return PROFILE_SAFETY_ROUNDS.get(profile, PROFILE_SAFETY_ROUNDS["balanced"])


def _clamp_rounds(value: int) -> int:
    rounds = max(1, int(value))
    ceiling = _parse_positive_int(os.environ.get("COGEV_NEXUS_MAX_ROUNDS"))
    if ceiling is not None:
        rounds = min(rounds, ceiling)
    return rounds


def _parse_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _legacy_env_enabled(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _legacy_config_warnings(profile: str) -> list[str]:
    suffix = profile.replace("-", "_").upper()
    warnings: list[str] = []
    legacy_rounds = f"COGEV_NEXUS_PROFILE_{suffix}_ROUNDS"
    if os.environ.get(legacy_rounds) and not _legacy_env_enabled("COGEV_ACCEPT_LEGACY_PROFILE_ROUNDS"):
        warnings.append(f"{legacy_rounds}_ignored_use_COGEV_NEXUS_PROFILE_{suffix}_SAFETY_ROUNDS")
    legacy_candidates = f"COGEV_NEXUS_PROFILE_{suffix}_CANDIDATES"
    if os.environ.get(legacy_candidates) and not _legacy_env_enabled("COGEV_ACCEPT_LEGACY_PROFILE_CANDIDATES"):
        warnings.append(f"{legacy_candidates}_ignored_use_COGEV_NEXUS_PROFILE_{suffix}_MIN_CANDIDATES")
    if os.environ.get("COGEV_NEXUS_DEFAULT_ROUNDS"):
        warnings.append("COGEV_NEXUS_DEFAULT_ROUNDS_ignored_use_COGEV_NEXUS_SAFETY_MAX_ROUNDS_or_explicit_rounds")
    if os.environ.get("COGEV_MUTATION_BRANCH_FACTOR") and not _legacy_env_enabled("COGEV_ACCEPT_LEGACY_MUTATION_BRANCH_FACTOR"):
        warnings.append("COGEV_MUTATION_BRANCH_FACTOR_ignored_use_COGEV_NEXUS_BRANCH_FACTOR")
    if os.environ.get("COGEV_ACTIVE_POOL_LIMIT"):
        warnings.append("COGEV_ACTIVE_POOL_LIMIT_ignored_active_pool_is_policy_internal")
    return warnings


__all__ = ["NexusRoundBudget", "PROFILE_DEFAULT_ROUNDS", "PROFILE_INITIAL_CANDIDATES", "PROFILE_BRANCH_FACTOR", "PROFILE_SAFETY_ROUNDS", "resolve_nexus_round_budget"]
