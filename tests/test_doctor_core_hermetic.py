from __future__ import annotations

from cognitive_evolve_runtime.doctor import doctor


def test_doctor_core_is_hermetic_without_llm_fixture(monkeypatch) -> None:
    monkeypatch.delenv("COGEV_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("COGEV_LLM_FIXTURE", raising=False)
    monkeypatch.delenv("COGEV_LLM_MODEL", raising=False)

    assert doctor("core") == 0
