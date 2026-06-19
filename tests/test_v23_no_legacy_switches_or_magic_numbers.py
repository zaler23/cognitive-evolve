from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V23_SWITCHES = (
    "COGEV_COMPACT_MODE",
    "COGEV_DYNAMIC_ADVERSARIAL_BUDGET",
    "COGEV_HONESTY_CONTROL",
    "COGEV_CROSSOVER_MODE",
)


def test_v23_deprecated_switches_are_confined_to_typed_config_and_tests() -> None:
    allowed = {
        ROOT / "cognitive_evolve_runtime" / "nexus" / "v23_theory_config.py",
        ROOT / "tests" / "test_v23_theory_config.py",
        ROOT / "tests" / "test_v23_no_legacy_switches_or_magic_numbers.py",
    }
    offenders: list[str] = []
    for path in (ROOT / "cognitive_evolve_runtime").rglob("*.py"):
        if path in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        for switch in V23_SWITCHES:
            if switch in text:
                offenders.append(f"{path.relative_to(ROOT)}:{switch}")
    assert offenders == []


def test_v23_numeric_defaults_live_in_typed_config() -> None:
    config_text = (ROOT / "cognitive_evolve_runtime" / "nexus" / "v23_theory_config.py").read_text(encoding="utf-8")
    for constant in (
        "DEFAULT_CELL_ELITE_RESERVE",
        "DEFAULT_RARE_RESERVE_PER_CELL",
        "DEFAULT_MIN_BUDGET_PER_CANDIDATE",
        "DEFAULT_PI_WINDOW",
        "DEFAULT_CONTROL_GAIN",
        "DEFAULT_MIN_SHARED_DESCRIPTOR_TOKENS",
    ):
        assert constant in config_text

    implementation_text = "\n".join(
        (ROOT / rel).read_text(encoding="utf-8")
        for rel in (
            "cognitive_evolve_runtime/archives/quality_diversity.py",
            "cognitive_evolve_runtime/verification/minimax_budget.py",
            "cognitive_evolve_runtime/nexus/honesty_control.py",
            "cognitive_evolve_runtime/candidates/crossover.py",
        )
    )
    assert "V23TheoryRuntimeConfig" in implementation_text or "EntropyCompactionConfig" in implementation_text
    assert "CACrossoverConfig" in implementation_text
    assert "HonestyControlConfig" in implementation_text
