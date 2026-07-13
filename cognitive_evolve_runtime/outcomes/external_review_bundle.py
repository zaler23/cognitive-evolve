"""Content-addressed export for independent external review."""
from __future__ import annotations

from typing import Any

from cognitive_evolve_runtime.candidates.genome import _producer_safe_candidate_payload
from cognitive_evolve_runtime.core.redaction import redact_text
from cognitive_evolve_runtime.core.serialization import stable_hash


EXTERNAL_REVIEW_BUNDLE_VERSION = "cogev.external-review-bundle/v1"
EXTERNAL_REVIEW_BUNDLE_PREFIX = "erb:"


def external_review_bundle_hash(bundle: dict[str, Any]) -> str:
    payload = dict(bundle)
    payload.pop("canonical_bundle_hash", None)
    return EXTERNAL_REVIEW_BUNDLE_PREFIX + stable_hash(payload)


def build_external_review_bundle(
    *,
    candidate: Any | None,
    objective_contract: Any,
    world: Any,
    mode: str,
    final_projection: dict[str, Any],
    verification_summaries: list[dict[str, Any]],
    producer_artifact_refs: dict[str, str],
    usage: dict[str, Any],
) -> dict[str, Any]:
    raw_candidate = candidate.to_dict() if candidate is not None and hasattr(candidate, "to_dict") else dict(candidate or {})
    raw_contract = objective_contract.to_dict() if hasattr(objective_contract, "to_dict") else dict(objective_contract or {})
    raw_world = world.to_dict() if hasattr(world, "to_dict") else dict(world or {})
    candidate_data = dict(_producer_safe_candidate_payload(_export_safe(raw_candidate)) or {})
    for opaque_key in ("artifact", "formal_artifacts", "patch_set"):
        if opaque_key in raw_candidate:
            candidate_data[opaque_key] = raw_candidate[opaque_key]
    preliminary = raw_candidate.get("preliminary_result") or raw_candidate.get("verification_result")
    if isinstance(preliminary, dict) and "answer" in preliminary:
        candidate_data.setdefault("preliminary_result", {})["answer"] = preliminary["answer"]
    contract = dict(_export_safe(raw_contract) or {})
    world_data = dict(_export_safe(raw_world) or {})
    final_projection_data = dict(_export_safe(final_projection) or {})
    if "artifact" in final_projection:
        final_projection_data["artifact"] = final_projection["artifact"]
    usage_data = dict(_export_safe(usage) or {})
    candidate_control_redacted = stable_hash(raw_candidate) != stable_hash(candidate_data)
    snapshot = dict(world_data.get("snapshot") or {})
    project_world = dict(world_data.get("project_world_model") or {})
    candidate_id = str(candidate_data.get("id") or final_projection_data.get("candidate_id") or "")
    candidate_hash = stable_hash(candidate_data) if candidate_data else ""
    input_identity = {
        "mode": str(mode or ""),
        "input_packet_id": str(world_data.get("input_packet_id") or ""),
        "snapshot_id": str(snapshot.get("snapshot_id") or project_world.get("snapshot_id") or world_data.get("snapshot_id") or ""),
        "source_root_hash": str(snapshot.get("root_hash") or world_data.get("root_hash") or ""),
        "world_hash": stable_hash(world_data),
    }
    preliminary_checks: list[dict[str, Any]] = [
        {
            "name": "final_projection_record",
            "signal_class": "static",
            "claim_scope": "producer_preliminary_record_only",
            "record": {
                "status": final_projection_data.get("status"),
                "candidate_id": final_projection_data.get("candidate_id"),
                "objective_solved": final_projection_data.get("objective_solved"),
                "blocking_issues": list(final_projection_data.get("blocking_issues") or []),
                "advisory_issues": list(final_projection_data.get("advisory_issues") or []),
            },
        }
    ]
    verification_result = candidate_data.get("preliminary_result") or candidate_data.get("verification_result")
    if isinstance(verification_result, dict) and verification_result:
        signal_class = str(verification_result.get("signal_class") or "")
        preliminary_checks.append(
            {
                "name": "candidate_preliminary_record",
                "signal_class": signal_class if signal_class in {"measured", "model_judged", "static"} else "static",
                "claim_scope": "producer_preliminary_record_only",
                "record": dict(verification_result),
            }
        )
    if candidate_data.get("multihead_scores"):
        preliminary_checks.append(
            {
                "name": "candidate_multihead_scores",
                "signal_class": "model_judged",
                "claim_scope": "ranking_signal_not_correctness",
                "record": dict(candidate_data.get("multihead_scores") or {}),
            }
        )
    preliminary_checks.extend(
        {
            "name": "project_verification_summary",
            "signal_class": "measured",
            "claim_scope": "local_execution_only",
            "record": dict(_export_safe(summary) or {}),
        }
        for summary in verification_summaries
        if candidate_id and isinstance(summary, dict) and str(summary.get("candidate_id") or "") == candidate_id
    )
    evidence_refs = [dict(item) for item in candidate_data.get("evidence_refs", []) if isinstance(item, dict)]
    limitations = [
        "producer_performed_preliminary_checks_only",
        "external_reviewer_must_reproduce_acceptance_checks_independently",
        "source_or_input_material_is_not_embedded; obtain material matching input_identity before replay",
        "canonical_hash_detects_content_change_but_does_not_authenticate_the_producer",
    ]
    limitations.extend(str(item) for item in candidate_data.get("missing_parts", []) if str(item))
    limitations.extend(str(item) for item in candidate_data.get("uncertainty_notes", []) if str(item))
    limitations.extend(str(item) for item in final_projection_data.get("blocking_issues", []) if str(item))
    limitations.extend(str(item) for item in final_projection_data.get("advisory_issues", []) if str(item))
    limitations.append("candidate_artifact_exported_verbatim_review_before_sharing")
    if candidate_control_redacted:
        limitations.append("candidate_control_metadata_redacted_for_export")
    if producer_artifact_refs:
        limitations.append("producer_artifact_hints_are_unbound_navigation_only")
    if any(not (item.get("sha256") or item.get("digest") or item.get("hash")) for item in evidence_refs):
        limitations.append("evidence_refs_without_digest_are_unbound_hints")
    if any(isinstance(item, dict) and item.get("provenance") == "unavailable" for item in usage_data.values()):
        limitations.append("one_or_more_usage_values_have_unavailable_provenance")
    bundle = {
        "version": EXTERNAL_REVIEW_BUNDLE_VERSION,
        "bundle_role": "export_only_external_review_input",
        "external_review_required": True,
        "producer_correctness_claim": "not_claimed",
        "candidate": {
            "id": candidate_id,
            "genome_hash": candidate_hash,
            "genome": candidate_data or None,
            "final_projection": final_projection_data,
        },
        "objective_contract": contract,
        "acceptance_contract": {
            "outcome_policy": dict(contract.get("outcome_policy") or {}),
            "dynamic_artifact_contract": dict(contract.get("dynamic_artifact_contract") or {}),
            "input_constraints": list(contract.get("input_constraints") or []),
            "expected_output_forms": list(contract.get("expected_output_forms") or []),
            "verification_preferences": list(contract.get("verification_preferences") or []),
            "success_dimensions": list(contract.get("success_dimensions") or []),
            "failure_dimensions": list(contract.get("failure_dimensions") or []),
        },
        "input_identity": input_identity,
        "preliminary_checks": preliminary_checks,
        "evidence_refs": evidence_refs,
        "known_limitations": list(dict.fromkeys(limitations)),
        "replay_recipe": {
            "requirements": [
                "independent external reviewer or verifier",
                "source or input matching input_identity",
                "tools required by the acceptance contract and evidence references",
            ],
            "steps": [
                {"order": 1, "action": "verify_bundle_hash"},
                {"order": 2, "action": "verify_source_or_input_identity"},
                {"order": 3, "action": "replay_preliminary_checks_when_relevant"},
                {"order": 4, "action": "run_independent_acceptance_review"},
                {"order": 5, "action": "record_external_result_outside_this_export"},
            ],
            "hash_recipe": {
                "algorithm": "sha256",
                "canonicalization": "UTF-8 compact JSON with sorted keys; omit canonical_bundle_hash",
                "prefix": EXTERNAL_REVIEW_BUNDLE_PREFIX,
            },
            "unbound_producer_artifact_hints": dict(_export_safe(producer_artifact_refs) or {}),
        },
        "usage": usage_data,
    }
    bundle["canonical_bundle_hash"] = external_review_bundle_hash(bundle)
    return bundle


def verify_external_review_bundle(bundle: dict[str, Any] | None) -> bool:
    if not isinstance(bundle, dict):
        return False
    expected = str(bundle.get("canonical_bundle_hash") or "")
    candidate = bundle.get("candidate")
    replay = bundle.get("replay_recipe")
    input_identity = bundle.get("input_identity")
    checks = bundle.get("preliminary_checks")
    complete = bool(
        isinstance(candidate, dict)
        and str(candidate.get("id") or "")
        and isinstance(candidate.get("genome"), dict)
        and candidate.get("genome")
        and candidate.get("genome_hash") == stable_hash(candidate.get("genome"))
        and isinstance(candidate.get("final_projection"), dict)
        and candidate["final_projection"].get("objective_solved") is False
        and isinstance(bundle.get("objective_contract"), dict)
        and bundle.get("objective_contract")
        and isinstance(input_identity, dict)
        and str(input_identity.get("world_hash") or "")
        and isinstance(checks, list)
        and checks
        and all(
            isinstance(item, dict)
            and item.get("signal_class") in {"measured", "model_judged", "static"}
            and isinstance(item.get("record"), dict)
            for item in checks
        )
        and isinstance(bundle.get("known_limitations"), list)
        and isinstance(replay, dict)
        and isinstance(replay.get("steps"), list)
        and replay.get("steps")
        and isinstance(bundle.get("usage"), dict)
    )
    return bool(
        expected
        and complete
        and bundle.get("version") == EXTERNAL_REVIEW_BUNDLE_VERSION
        and bundle.get("bundle_role") == "export_only_external_review_input"
        and bundle.get("external_review_required") is True
        and bundle.get("producer_correctness_claim") == "not_claimed"
        and expected == external_review_bundle_hash(bundle)
    )


def _export_safe(value: Any) -> Any:
    """Redact paths/secret-shaped text without treating domain field names as secrets."""

    if isinstance(value, dict):
        return {str(key): _export_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_export_safe(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


__all__ = [
    "EXTERNAL_REVIEW_BUNDLE_VERSION",
    "build_external_review_bundle",
    "external_review_bundle_hash",
    "verify_external_review_bundle",
]
