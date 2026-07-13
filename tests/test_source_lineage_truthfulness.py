from __future__ import annotations

from pathlib import Path

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.nexus.source_lineage import allowed_materialization_path, analyze_source_lineage


def _project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "cognitive_evolve_runtime").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (root / "cognitive_evolve_runtime" / "module.py").write_text("def existing():\n    return 1\n", encoding="utf-8")
    return root


def test_existing_symbol_extension_requires_path_scoped_patch(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    candidate = CandidateGenome(
        id="C-lineage",
        artifact_type="patch",
        artifact={"content": "def new_symbol():\n    return 2\n"},
        source_bindings=[{"path": "cognitive_evolve_runtime/module.py", "symbol": "new_symbol", "mode": "extend"}],
        concise_claim="add new symbol",
    )

    analysis = analyze_source_lineage(candidate, project_root=root)

    assert analysis.passed is False
    assert "patch_target_missing" in analysis.diagnostics
    assert analysis.facts[0].patch_touches_path is False


def test_materialization_scope_rejects_parent_escape() -> None:
    assert allowed_materialization_path("../cognitive_evolve_runtime/escape.py") is False
    assert allowed_materialization_path("./cognitive_evolve_runtime/new_module.py") is True
