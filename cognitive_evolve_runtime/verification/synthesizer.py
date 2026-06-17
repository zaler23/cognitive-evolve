"""Verification plan synthesizer."""
from __future__ import annotations

import re
from typing import Any

from cognitive_evolve_runtime.nexus._serde import stable_hash
from .ladder import VerificationStrength
from .reformulation import reformulate_for_verification
from .types import VerificationPlan

_EXEC_HINT_RE = re.compile(r"\b(code|python|pytest|function|program|script|algorithm|execute|run)\b", re.I)
_FORMAL_HINT_RE = re.compile(r"\b(prove|theorem|invariant|z3|smt|lean|formal)\b", re.I)
_EMPIRICAL_HINT_RE = re.compile(r"\b(data|simulate|experiment|measurement|empirical|statistical)\b", re.I)
_DECOMPOSE_HINT_RE = re.compile(r"\b(explain|analyze|compare|argue|why|what)\b", re.I)


class VerificationSynthesizer:
    def __init__(self, *, model: Any | None = None) -> None:
        self.model = model

    def synthesize(self, problem: Any) -> VerificationPlan:
        text = _problem_text(problem)
        modality = "adversarial"
        if _EXEC_HINT_RE.search(text):
            modality = "executable"
        elif _FORMAL_HINT_RE.search(text):
            modality = "formal"
        elif _EMPIRICAL_HINT_RE.search(text):
            modality = "empirical"
        elif _DECOMPOSE_HINT_RE.search(text):
            modality = "decomposed"
        diagnostic_strength = VerificationStrength.from_value(0)
        reformulations = [item.to_dict() for item in reformulate_for_verification(text)] if modality == "adversarial" else []
        fingerprint = "verifier-" + stable_hash({"modality": modality, "problem": text})[:16]
        return VerificationPlan(
            verifier_id=f"{modality}-verifier",
            strength=diagnostic_strength,
            modality=modality,
            verifier_fingerprint=fingerprint,
            replayable=False,
            diagnostics=[f"selected_modality:{modality}"],
            reformulations=reformulations,
            metadata={
                "replay_scope": "verifier_on_frozen_artifact_only",
                "diagnostics_only": True,
                "probe_requirements": [{"kind": f"{modality}_grounding_probe", "expected_verdict_flip": False}],
                "falsification_budget": {"count": 1 if modality in {"executable", "formal", "empirical", "decomposed"} else 0},
            },
        )


def _problem_text(problem: Any) -> str:
    if isinstance(problem, str):
        return problem
    if isinstance(problem, dict):
        for key in ("problem", "prompt", "goal", "objective"):
            if problem.get(key):
                return str(problem.get(key))
    return str(problem or "")


__all__ = ["VerificationSynthesizer"]
