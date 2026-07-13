"""Cross-niche recombination and translation gates for M6."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.crossover import crossover
from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.project_candidate import ProjectCandidateGenome
from cognitive_evolve_runtime.core.serialization import coerce_dict, stable_hash

from .niches import SpeciesIndex, normalize_niche_id, resolve_niche_id


RESEAL_VERSION = "m6-cross-niche-reseal/v1"

_VERIFIED_PAYLOAD_MARKERS = (
    "verified_cert",
    "verified_certificate",
    "verified_certificates",
    "improvement_certificate",
    "improvement_certificate_hash",
    "anytime_valid_certificate",
    "anytime_valid_certificate_hash",
    "solve_certificate",
    "solve_certificate_hash",
    "m6_solve_certificate",
    "m6_solve_certificate_hash",
    "closure_certificate",
    "closure_certificate_hash",
    "e_wealth",
    "ewealth",
    "e_process",
    "e_process_state",
    "e_value",
    "seal_hash",
    "sealed_certificate",
)


def cross_niche_recombine(
    parent_a: CandidateGenome,
    parent_b: CandidateGenome,
    *,
    target_niche_id: str | None = None,
    instruction: str = "cross-niche recombination",
) -> CandidateGenome:
    source_niches = tuple(dict.fromkeys([resolve_niche_id(parent_a), resolve_niche_id(parent_b)]))
    target = normalize_niche_id(target_niche_id or _hybrid_niche_id(source_niches, parent_a.id, parent_b.id))
    child = crossover(parent_a, parent_b, instruction=instruction)
    return require_reseal(
        child,
        target_niche_id=target,
        source_niche_ids=source_niches,
        reason="cross_niche_recombination_requires_fresh_verified_closure",
    )


def translate_candidate_to_niche(
    candidate: CandidateGenome,
    target_niche_id: str,
    *,
    translation_note: str = "translate candidate into a different niche",
) -> CandidateGenome:
    source_niche = resolve_niche_id(candidate)
    target = normalize_niche_id(target_niche_id)
    child = _clone_as_child(candidate)
    child.mutation_history = list(dict.fromkeys([*candidate.mutation_history, "NicheTranslation"]))
    child.concise_claim = f"Translated {candidate.concise_claim or candidate.id} into niche {target}"
    metadata = coerce_dict(child.metadata)
    metadata["translation_note"] = translation_note
    child.metadata = metadata
    return require_reseal(
        child,
        target_niche_id=target,
        source_niche_ids=(source_niche,),
        reason="niche_translation_requires_fresh_verified_closure",
    )


def require_reseal(
    candidate: CandidateGenome,
    *,
    target_niche_id: str,
    source_niche_ids: tuple[str, ...] | list[str],
    reason: str,
) -> CandidateGenome:
    target = normalize_niche_id(target_niche_id)
    source_niches = tuple(dict.fromkeys(normalize_niche_id(item) for item in source_niche_ids if str(item).strip()))
    cleared_keys: list[str] = []
    candidate.metadata = _clear_verified_payloads(coerce_dict(candidate.metadata), cleared_keys=cleared_keys)
    candidate.verification_result = _clear_verified_payloads(coerce_dict(candidate.verification_result), cleared_keys=cleared_keys)
    candidate.obligation_delta = _clear_verified_payloads(coerce_dict(candidate.obligation_delta), cleared_keys=cleared_keys)
    candidate.evidence_delta = _clear_verified_payloads(coerce_dict(candidate.evidence_delta), cleared_keys=cleared_keys)
    candidate.verification_trace = []
    candidate.evidence_delta.pop("verified", None)
    candidate.niche_memberships = [target]
    candidate.metadata.update(
        {
            "niche_id": target,
            "primary_niche_id": target,
            "requires_reseal": True,
            "seal_status": "unsealed",
            "verified_closure": False,
            "niche_runtime": {
                "version": RESEAL_VERSION,
                "target_niche_id": target,
                "source_niche_ids": list(source_niches),
                "requires_reseal": True,
                "seal_status": "unsealed",
                "reseal_reason": reason,
                "cleared_verified_payload_keys": sorted(set(cleared_keys)),
            },
        }
    )
    candidate.verification_result.update(
        {
            "status": "unsealed",
            "requires_reseal": True,
            "reseal_reason": reason,
            "verified_closure": False,
        }
    )
    SpeciesIndex().assign(candidate, target, isolate=True)
    return candidate


def requires_reseal(candidate: CandidateGenome | dict[str, Any]) -> bool:
    if isinstance(candidate, CandidateGenome):
        return bool(
            coerce_dict(candidate.metadata).get("requires_reseal")
            or coerce_dict(candidate.metadata).get("seal_status") == "unsealed"
            or coerce_dict(candidate.verification_result).get("requires_reseal")
        )
    data = coerce_dict(candidate)
    return bool(data.get("requires_reseal") or data.get("seal_status") == "unsealed")


def _clone_as_child(candidate: CandidateGenome) -> CandidateGenome:
    data = candidate.to_dict()
    data.pop("id", None)
    data["parent_ids"] = [candidate.id]
    data["generation"] = int(candidate.generation or 0) + 1
    if isinstance(candidate, ProjectCandidateGenome):
        child = ProjectCandidateGenome.from_dict(data)
    else:
        child = CandidateGenome.from_dict(data)
    child.lineage = list(dict.fromkeys([*candidate.lineage, candidate.id, child.id]))
    return child


def _hybrid_niche_id(source_niches: tuple[str, ...], parent_a_id: str, parent_b_id: str) -> str:
    if len(source_niches) == 2:
        return f"hybrid:{source_niches[0]}+{source_niches[1]}"
    suffix = stable_hash({"source_niches": source_niches, "parents": [parent_a_id, parent_b_id]})[:12]
    return f"hybrid:{suffix}"


def _clear_verified_payloads(mapping: dict[str, Any], *, cleared_keys: list[str]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in mapping.items():
        normalized = _normalized_key(key)
        if _is_verified_payload_key(normalized):
            cleared_keys.append(str(key))
            continue
        if isinstance(value, dict):
            clean[key] = _clear_verified_payloads(value, cleared_keys=cleared_keys)
        else:
            clean[key] = value
    return clean


def _is_verified_payload_key(normalized_key: str) -> bool:
    if normalized_key in _VERIFIED_PAYLOAD_MARKERS:
        return True
    if "verified_cert" in normalized_key:
        return True
    if "certificate" in normalized_key and any(scope in normalized_key for scope in ("improvement", "solve", "closure", "anytime", "sealed")):
        return True
    return False


def _normalized_key(key: Any) -> str:
    return str(key or "").strip().lower().replace("-", "_")


translate_between_niches = translate_candidate_to_niche
translate_to_niche = translate_candidate_to_niche
recombine_across_niches = cross_niche_recombine
recombine = cross_niche_recombine
translate = translate_candidate_to_niche


__all__ = [
    "RESEAL_VERSION",
    "cross_niche_recombine",
    "recombine",
    "recombine_across_niches",
    "require_reseal",
    "requires_reseal",
    "translate",
    "translate_between_niches",
    "translate_candidate_to_niche",
    "translate_to_niche",
]
