"""Versioned embedding coverage observations that never enter selection."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.core.serialization import coerce_dict, stable_hash, stable_json
from cognitive_evolve_runtime.nexus.search_kernel.fingerprints import candidate_materialized_artifact
from cognitive_evolve_runtime.ranking.novelty import novelty_distance

_LIVE_FATES = {CandidateFate.ACTIVE.value, CandidateFate.ELITE.value, CandidateFate.INCUBATING.value}


@dataclass(frozen=True)
class RepresentationSpec:
    provider: str
    model_revision: str
    dimension: int
    normalization: str

    @property
    def representation_id(self) -> str:
        return f"{self.provider}:{self.model_revision}:dim={self.dimension}:normalization={self.normalization}"


class RepresentationProvider(Protocol):
    """Optional boundary for converting a candidate summary into a fixed vector."""

    def representation_spec(self) -> RepresentationSpec: ...

    def represent(self, candidate_id: str, candidate_summary: str) -> Sequence[float]: ...


class DeterministicStubRepresentationProvider:
    """Offline fixture provider backed by an injected candidate-id vector table."""

    def __init__(
        self,
        *,
        vectors: dict[str, Sequence[float]],
        provider: str = "deterministic-stub",
        model_revision: str = "fixture-v1",
        dimension: int,
        normalization: str = "none",
    ) -> None:
        self._vectors = {str(candidate_id): tuple(float(value) for value in vector) for candidate_id, vector in vectors.items()}
        self._spec = RepresentationSpec(
            provider=str(provider),
            model_revision=str(model_revision),
            dimension=int(dimension),
            normalization=str(normalization),
        )
        self.call_count = 0

    def representation_spec(self) -> RepresentationSpec:
        return self._spec

    def represent(self, candidate_id: str, candidate_summary: str) -> Sequence[float]:
        del candidate_summary
        self.call_count += 1
        return self._vectors[candidate_id]


class RepresentationVectorStore:
    """Candidate-id lookup isolated by the frozen representation version."""

    SCHEMA_VERSION = "representation-vector-store/v1"

    def __init__(self, records: dict[str, dict[str, dict[str, Any]]] | None = None) -> None:
        self._records = records or {}

    def get(self, representation_id: str, candidate_id: str, *, input_hash: str | None = None) -> tuple[float, ...] | None:
        record = self._records.get(str(representation_id), {}).get(str(candidate_id))
        if not isinstance(record, dict):
            return None
        if input_hash is not None and str(record.get("input_hash") or "") != input_hash:
            return None
        vector = record.get("vector")
        if not isinstance(vector, list):
            return None
        return tuple(float(value) for value in vector)

    def put(self, representation_id: str, candidate_id: str, *, input_hash: str, vector: Sequence[float]) -> None:
        values = [float(value) for value in vector]
        self._records.setdefault(str(representation_id), {})[str(candidate_id)] = {
            "input_hash": str(input_hash),
            "vector": values,
            "vector_hash": stable_hash(values),
        }

    def representation_ids(self) -> list[str]:
        return sorted(self._records)

    def namespace_hash(self, representation_id: str) -> str:
        return stable_hash(self._records.get(str(representation_id), {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "representations": {
                representation_id: {
                    candidate_id: {
                        "input_hash": str(record.get("input_hash") or ""),
                        "vector": [float(value) for value in record.get("vector", [])],
                        "vector_hash": str(record.get("vector_hash") or ""),
                    }
                    for candidate_id, record in sorted(records.items())
                }
                for representation_id, records in sorted(self._records.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RepresentationVectorStore":
        payload = coerce_dict(data)
        representations = coerce_dict(payload.get("representations"))
        records: dict[str, dict[str, dict[str, Any]]] = {}
        for representation_id, raw_namespace in representations.items():
            namespace = coerce_dict(raw_namespace)
            records[str(representation_id)] = {
                str(candidate_id): {
                    "input_hash": str(record.get("input_hash") or ""),
                    "vector": [float(value) for value in record.get("vector", [])],
                    "vector_hash": str(record.get("vector_hash") or ""),
                }
                for candidate_id, raw_record in namespace.items()
                if (record := coerce_dict(raw_record))
            }
        return cls(records)


class RepresentationShadowLayer:
    """Compute audit-only embedding coverage at the generation boundary."""

    def __init__(
        self,
        provider: RepresentationProvider,
        *,
        store: RepresentationVectorStore | None = None,
        top_k: int = 5,
    ) -> None:
        self.provider = provider
        self.store = store or RepresentationVectorStore()
        self.top_k = max(1, int(top_k))

    def observe(
        self,
        *,
        round_index: int,
        candidates: list[CandidateGenome],
        fate_assignments: list[Any],
    ) -> dict[str, Any]:
        spec = self.provider.representation_spec()
        _validate_spec(spec)
        representation_id = spec.representation_id
        candidate_by_id = {candidate.id: candidate for candidate in candidates}
        live_ids = {
            str(getattr(assignment, "candidate_id", "") or (assignment.get("candidate_id") if isinstance(assignment, dict) else ""))
            for assignment in fate_assignments
            if CandidateFate.normalize(
                getattr(assignment, "fate", "") or (assignment.get("fate") if isinstance(assignment, dict) else ""),
                default="",
            )
            in _LIVE_FATES
        }
        active = [candidate_by_id[candidate_id] for candidate_id in sorted(live_ids) if candidate_id in candidate_by_id]
        vectors: dict[str, tuple[float, ...]] = {}
        vector_hashes: dict[str, str] = {}
        for candidate in active:
            summary = _candidate_summary(candidate)
            input_hash = stable_hash(summary)
            vector = self.store.get(representation_id, candidate.id, input_hash=input_hash)
            if vector is None:
                vector = _validated_vector(self.provider.represent(candidate.id, summary), dimension=spec.dimension)
                self.store.put(representation_id, candidate.id, input_hash=input_hash, vector=vector)
            vectors[candidate.id] = vector
            vector_hashes[candidate.id] = stable_hash(list(vector))

        rankings = _distance_rankings(vectors)
        features = _shadow_features(rankings, vector_hashes, top_k=self.top_k)
        measurements = {
            "leave_one_out_neighbor_rank_stability": _leave_one_out_stability(rankings, top_k=self.top_k),
            "lexical_distance_correlation": _lexical_distance_correlation(active, vectors),
            "low_density_region_recovery": _low_density_region_recovery(features),
        }
        audit = {
            "schema_version": "representation-shadow/v1",
            "mode": "shadow_only",
            "round_index": int(round_index),
            "representation_id": representation_id,
            "representation": {
                "provider": spec.provider,
                "model_revision": spec.model_revision,
                "dimension": spec.dimension,
                "normalization": spec.normalization,
                "distance_metric": "cosine_distance",
            },
            "top_k": min(self.top_k, max(0, len(active) - 1)),
            "features": features,
            "measurements": measurements,
            "vector_store_namespace_hash": self.store.namespace_hash(representation_id),
            "boundary": {
                "observed_at": "generation_epoch_boundary",
                "online_rebin_applied": False,
                "allowed_rebin_boundaries": ["checkpoint", "epoch"],
            },
            "activation_gate": {
                "activated_for_selection": False,
                "selection_consumers": [],
                "archive_reordering": False,
                "sampling_effect": False,
                "exact_dedupe_effect": False,
                "descriptor_archive_effect": False,
            },
        }
        audit["audit_hash"] = stable_hash(audit)
        return audit


def _validate_spec(spec: RepresentationSpec) -> None:
    if not spec.provider or not spec.model_revision or not spec.normalization or int(spec.dimension) <= 0:
        raise ValueError("representation provider must declare provider, model_revision, positive dimension, and normalization")


def _validated_vector(vector: Sequence[float], *, dimension: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if len(values) != dimension:
        raise ValueError(f"representation vector dimension mismatch: expected {dimension}, got {len(values)}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("representation vector must contain only finite values")
    return values


def _candidate_summary(candidate: CandidateGenome) -> str:
    return stable_json(
        {
            "artifact": candidate_materialized_artifact(candidate),
            "artifact_type": candidate.artifact_type,
            "concise_claim": candidate.concise_claim,
            "core_mechanism": candidate.core_mechanism,
            "novelty_descriptors": candidate.novelty_descriptors,
            "niche_memberships": candidate.niche_memberships,
        }
    )


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    similarity = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return max(0.0, min(2.0, 1.0 - max(-1.0, min(1.0, similarity))))


def _distance_rankings(vectors: dict[str, tuple[float, ...]]) -> dict[str, list[tuple[str, float]]]:
    return {
        candidate_id: sorted(
            (
                (other_id, _cosine_distance(vector, other_vector))
                for other_id, other_vector in vectors.items()
                if other_id != candidate_id
            ),
            key=lambda item: (item[1], item[0]),
        )
        for candidate_id, vector in vectors.items()
    }


def _shadow_features(
    rankings: dict[str, list[tuple[str, float]]],
    vector_hashes: dict[str, str],
    *,
    top_k: int,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for candidate_id, ranking in sorted(rankings.items()):
        nearest = ranking[: min(top_k, len(ranking))]
        mean_distance = sum(distance for _, distance in nearest) / len(nearest) if nearest else 0.0
        features[candidate_id] = {
            "novelty": _rounded(nearest[0][1] if nearest else 0.0),
            "local_density": _rounded(1.0 / (1.0 + mean_distance) if nearest else 0.0),
            "neighbors": [
                {"candidate_id": neighbor_id, "distance": _rounded(distance)}
                for neighbor_id, distance in nearest
            ],
            "vector_hash": vector_hashes[candidate_id],
        }
    return features


def _leave_one_out_stability(
    rankings: dict[str, list[tuple[str, float]]],
    *,
    top_k: int,
) -> dict[str, Any]:
    candidate_ids = sorted(rankings)
    scores: list[float] = []
    for held_out_id in candidate_ids:
        for query_id in candidate_ids:
            if query_id == held_out_id:
                continue
            baseline = [candidate_id for candidate_id, _ in rankings[query_id]][:top_k]
            replay = [candidate_id for candidate_id, _ in rankings[query_id] if candidate_id != held_out_id][:top_k]
            denominator = max(1, min(top_k, len(baseline)))
            agreement = sum(
                1.0 / (1.0 + abs(baseline.index(candidate_id) - replay.index(candidate_id)))
                for candidate_id in baseline
                if candidate_id in replay
            ) / denominator
            scores.append(agreement)
    return {
        "method": "held_out_candidate_top_k_rank_agreement",
        "sample_count": len(scores),
        "mean_rank_agreement": _rounded(sum(scores) / len(scores)) if scores else None,
    }


def _lexical_distance_correlation(
    candidates: list[CandidateGenome],
    vectors: dict[str, tuple[float, ...]],
) -> dict[str, Any]:
    embedding_distances: list[float] = []
    lexical_distances: list[float] = []
    for index, candidate in enumerate(candidates):
        for other in candidates[index + 1 :]:
            embedding_distances.append(_cosine_distance(vectors[candidate.id], vectors[other.id]))
            lexical_distances.append(novelty_distance(candidate, other))
    correlation = _pearson_correlation(embedding_distances, lexical_distances)
    return {
        "method": "pearson_pairwise_distance",
        "pair_count": len(embedding_distances),
        "pearson_r": _rounded(correlation) if correlation is not None else None,
    }


def _pearson_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return max(-1.0, min(1.0, numerator / (left_scale * right_scale)))


def _low_density_region_recovery(features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not features:
        return {"lowest_density_candidate_ids": [], "minimum_density": None, "density_span": None}
    densities = {candidate_id: float(feature["local_density"]) for candidate_id, feature in features.items()}
    minimum = min(densities.values())
    maximum = max(densities.values())
    return {
        "lowest_density_candidate_ids": sorted(candidate_id for candidate_id, density in densities.items() if density == minimum),
        "minimum_density": _rounded(minimum),
        "density_span": _rounded(maximum - minimum),
    }


def _rounded(value: float) -> float:
    return round(float(value), 12)


__all__ = [
    "DeterministicStubRepresentationProvider",
    "RepresentationProvider",
    "RepresentationShadowLayer",
    "RepresentationSpec",
    "RepresentationVectorStore",
]
