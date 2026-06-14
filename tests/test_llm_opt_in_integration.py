from __future__ import annotations

import os

import pytest

from cognitive_evolve_runtime.nexus.runtime import NexusRuntime


@pytest.mark.skipif(os.environ.get("COGEV_RUN_LLM_TESTS", "").lower() not in {"1", "true", "yes", "on"}, reason="real LLM integration tests are opt-in")
def test_real_llm_text_runtime_smoke(tmp_path) -> None:
    """Tiny live-provider smoke test for schema drift and synthesis binding.

    This is intentionally skipped in hermetic CI.  Operators can run it with a
    configured provider and ``COGEV_RUN_LLM_TESTS=1`` to verify the actual
    model transport, JSON repair, seed/rank/mutate, and final synthesis path.
    """

    result = NexusRuntime.with_configured_llm(output_dir=tmp_path).run_text(
        "Return a one sentence explanation of why strict JSON output matters.",
        max_rounds=1,
        min_population_size=2,
    )

    assert result.final_answer.strip()
    assert result.evolution["synthesis"]["best_candidate_id"]
