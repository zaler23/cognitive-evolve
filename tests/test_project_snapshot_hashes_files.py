from __future__ import annotations

from pathlib import Path

from cognitive_evolve_runtime.inputs.project_map import ProjectWorldModel
from cognitive_evolve_runtime.inputs.project_snapshot import ProjectSnapshot


def test_project_snapshot_hashes_files_and_detects_commands(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_mod.py").write_text("from pkg.mod import add\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")

    snapshot = ProjectSnapshot.from_path(tmp_path)
    world = ProjectWorldModel.from_snapshot(snapshot, objective="fix add")

    assert snapshot.snapshot_id.startswith("snapshot-")
    assert "pkg/mod.py" in snapshot.file_hashes
    assert snapshot.file_manifest[0]["sha256"]
    assert "python" in snapshot.package_managers
    assert "python -m pytest -q" in snapshot.detected_commands
    assert world.file_roles["tests/test_mod.py"] == "test"
    assert "add" in world.symbol_graph["pkg/mod.py"]
