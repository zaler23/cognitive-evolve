"""Deterministic lexical similarity over complete materialized artifacts."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping


SparseTextVector = dict[str, float]


def artifact_vector(text: str, *, ngram_range: range = range(3, 6)) -> SparseTextVector:
    """Return a sparse, L2-normalized character n-gram term-frequency vector."""

    normalized = " ".join(str(text or "").split()).casefold()
    counts: Counter[str] = Counter()
    for size in ngram_range:
        if size <= 0:
            raise ValueError("character n-gram sizes must be positive")
        counts.update(normalized[index : index + size] for index in range(max(0, len(normalized) - size + 1)))
    norm = math.sqrt(sum(value * value for value in counts.values()))
    if norm == 0.0:
        return {}
    return {key: value / norm for key, value in counts.items()}


def cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Return sparse cosine similarity in ``[0, 1]``."""

    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return max(0.0, min(1.0, sum(value * float(right.get(key, 0.0)) for key, value in left.items())))


def lexical_similarity(left: str, right: str) -> float:
    return cosine(artifact_vector(left), artifact_vector(right))


__all__ = ["SparseTextVector", "artifact_vector", "cosine", "lexical_similarity"]
