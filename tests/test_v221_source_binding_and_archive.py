from __future__ import annotations

from pathlib import Path

from cognitive_evolve_runtime.archives.manager import ArchiveManager
from cognitive_evolve_runtime.archives.types import FateAssignment
from cognitive_evolve_runtime.candidates.genome import CandidateFate, CandidateGenome
from cognitive_evolve_runtime.nexus.source_binding_resolver import annotate_candidate_source_bindings, resolve_candidate_source_bindings


def _project(root: Path) -> None:
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (root / "cognitive_evolve_runtime").mkdir()
    (root / "cognitive_evolve_runtime" / "mod.py").write_text("def value():\n    return 1\n", encoding="utf-8")


def test_source_binding_resolver_resolves_existing_symbol(tmp_path: Path) -> None:
    _project(tmp_path)
    candidate = CandidateGenome(id="ok", source_bindings=[{"path": "cognitive_evolve_runtime/mod.py", "symbol": "value"}])
    manifest = resolve_candidate_source_bindings(candidate, project_root=tmp_path)
    assert manifest.binding_class == "resolved"
    assert manifest.admission_route == "normal"


def test_invented_binding_routed_to_repair_only_and_blocked_from_answer_archive(tmp_path: Path) -> None:
    _project(tmp_path)
    candidate = CandidateGenome(id="bad", current_fate=CandidateFate.ELITE, source_bindings=[{"path": "cognitive_evolve_runtime/missing.py", "symbol": "x"}])
    annotate_candidate_source_bindings(candidate, project_root=tmp_path)
    assert candidate.metadata["source_binding_class"] == "invented"
    archives = ArchiveManager()
    archives.update([FateAssignment("bad", CandidateFate.ELITE)], candidates=[candidate])
    assert "bad" not in archives.answer_archive
    assert any(str(target).startswith("AnswerArchiveBlocked") for target in archives.history[-1]["assignments"][0]["archive_targets"])


def test_materialized_patch_claim_can_resolve_missing_file(tmp_path: Path) -> None:
    _project(tmp_path)
    candidate = CandidateGenome(
        id="mat",
        artifact={"patch": "+++ cognitive_evolve_runtime/new_mod.py\n+class NewSymbol:\n+    pass\n"},
        source_bindings=[{"path": "cognitive_evolve_runtime/new_mod.py", "symbol": "NewSymbol", "binding_mode": "materialize"}],
    )
    candidate.patch_application_result = {"status": "applied", "applied_files": ["cognitive_evolve_runtime/new_mod.py"]}
    manifest = resolve_candidate_source_bindings(candidate, project_root=tmp_path)
    assert manifest.binding_class == "resolved"
