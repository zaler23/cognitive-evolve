"""Multi-head Elo updates for relative Nexus rankings."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus.policy import DEFAULT_FITNESS_AXES


@dataclass
class MultiHeadElo:
    ratings: dict[str, dict[str, float]] = field(default_factory=dict)
    initial_rating: float = 1000.0
    k_factor: float = 24.0

    def ensure(self, candidate_id: str) -> None:
        self.ratings.setdefault(candidate_id, {axis: self.initial_rating for axis in DEFAULT_FITNESS_AXES})

    def update_pairwise(self, winner_id: str, loser_id: str, *, axis: str = "answer_likelihood", weight: float = 1.0) -> None:
        self.ensure(winner_id)
        self.ensure(loser_id)
        if axis not in self.ratings[winner_id]:
            self.ratings[winner_id][axis] = self.initial_rating
            self.ratings[loser_id][axis] = self.initial_rating
        win = self.ratings[winner_id][axis]
        lose = self.ratings[loser_id][axis]
        expected_win = 1.0 / (1.0 + 10 ** ((lose - win) / 400.0))
        delta = self.k_factor * weight * (1.0 - expected_win)
        self.ratings[winner_id][axis] = win + delta
        self.ratings[loser_id][axis] = lose - delta

    def update_from_relative(self, ranking: Any) -> None:
        preferences = [dict(pref) for pref in getattr(ranking, "pairwise_preferences", []) or [] if isinstance(pref, dict)]
        confidence = _preference_consistency(preferences)
        for pref in preferences:
            winner = str(pref.get("winner") or "")
            loser = str(pref.get("loser") or "")
            if winner and loser:
                self.update_pairwise(
                    winner,
                    loser,
                    axis=str(pref.get("axis") or "answer_likelihood"),
                    weight=_float(pref.get("weight"), default=1.0) * confidence,
                )

    def population_signal(self, candidate_ids: list[str], *, axes: list[str] | None = None) -> dict[str, float]:
        """Return a bounded 0..1 Elo signal for candidate selection.

        Elo becomes useful only when it feeds back into selection pressure.  The
        signal is relative to the current live population, not an absolute truth
        claim: equal ratings map to 0.5 and only observed pairwise preferences
        can move a candidate above or below that midpoint.
        """

        ids = [str(candidate_id) for candidate_id in candidate_ids if str(candidate_id)]
        if not ids:
            return {}
        selected_axes = list(axes or DEFAULT_FITNESS_AXES)
        raw: dict[str, float] = {}
        for candidate_id in ids:
            self.ensure(candidate_id)
            scores = self.ratings.get(candidate_id, {})
            values = [float(scores.get(axis, self.initial_rating)) for axis in selected_axes]
            raw[candidate_id] = sum(values) / len(values) if values else self.initial_rating
        lo = min(raw.values())
        hi = max(raw.values())
        if hi <= lo:
            return {candidate_id: 0.5 for candidate_id in ids}
        return {candidate_id: max(0.0, min(1.0, (value - lo) / (hi - lo))) for candidate_id, value in raw.items()}

    def apply_to_candidates(self, candidates: list[CandidateGenome], *, axes: list[str] | None = None) -> None:
        """Attach current Elo selection signal to candidate multihead scores."""

        signal = self.population_signal([candidate.id for candidate in candidates], axes=axes)
        for candidate in candidates:
            if candidate.id not in signal:
                continue
            self.ensure(candidate.id)
            axis_ratings = self.ratings.get(candidate.id, {})
            candidate.multihead_scores["elo_reproductive_signal"] = signal[candidate.id]
            candidate.multihead_scores["elo_mean_rating"] = sum(axis_ratings.values()) / len(axis_ratings) if axis_ratings else self.initial_rating

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiHeadElo":
        return cls(
            ratings={str(cid): {str(axis): float(value) for axis, value in dict(scores).items()} for cid, scores in dict(data.get("ratings") or {}).items() if isinstance(scores, dict)},
            initial_rating=float(data.get("initial_rating", 1000.0) or 1000.0),
            k_factor=float(data.get("k_factor", 24.0) or 24.0),
        )


def _float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def _preference_consistency(preferences: list[dict[str, Any]]) -> float:
    """Reduce Elo step size when the same batch contradicts itself.

    This is a local, self-observed confidence signal.  It avoids treating noisy
    position/order effects as strong evolutionary pressure without introducing a
    domain-specific threshold.
    """

    if not preferences:
        return 1.0
    seen: set[tuple[str, str, str]] = set()
    conflicts = 0
    for pref in preferences:
        winner = str(pref.get("winner") or "")
        loser = str(pref.get("loser") or "")
        axis = str(pref.get("axis") or "answer_likelihood")
        if not winner or not loser:
            continue
        if (loser, winner, axis) in seen:
            conflicts += 1
        seen.add((winner, loser, axis))
    return 1.0 / (1.0 + conflicts)


__all__ = ["MultiHeadElo"]
