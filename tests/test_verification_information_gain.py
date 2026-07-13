from __future__ import annotations

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.verification.information_gain import UNDEFINED_SIGNATURE, compute_marginal_gain, grounded_signature


def test_grounded_signature_undefined_for_unverified_candidate_yields_zero_gain() -> None:
    candidate = CandidateGenome(id="C1", artifact="This is a theory with no obligation results.")
    assert grounded_signature(candidate) == UNDEFINED_SIGNATURE
    assert compute_marginal_gain({UNDEFINED_SIGNATURE}, set(), 1.0, 5) == 0.0


def test_ast_signature_ignores_variable_rename() -> None:
    left = CandidateGenome(id="C1", artifact="def add(x):\n    y = x + 1\n    return y\n")
    right = CandidateGenome(id="C2", artifact="def add(z):\n    q = z + 1\n    return q\n")
    assert grounded_signature(left) == grounded_signature(right)


def test_structured_spec_signature_ignores_runtime_jitter_fields() -> None:
    left = CandidateGenome(id="C1", artifact={"answer": 1, "runtime": 1.23, "round": 1})
    right = CandidateGenome(id="C2", artifact={"answer": 1, "runtime": 9.99, "round": 42})
    assert grounded_signature(left) == grounded_signature(right)


def test_obligation_set_signature_for_text_candidate() -> None:
    candidate = CandidateGenome(id="C1", artifact="theory")
    candidate.verification_trace = [
        {"passed": True, "metadata": {"obligation_id": "a"}},
        {"passed": False, "metadata": {"obligation_id": "b"}},
    ]
    assert grounded_signature(candidate).startswith("obl-")
