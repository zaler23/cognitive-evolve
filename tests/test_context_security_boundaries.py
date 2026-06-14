from __future__ import annotations

import os
from pathlib import Path

from cognitive_evolve_runtime.contracts.objective_contract import NexusProjectObjectiveContract
from cognitive_evolve_runtime.inputs.context_selector import ContextRequest, ContextSelector
from cognitive_evolve_runtime.inputs.project_map import ProjectWorldModel
from cognitive_evolve_runtime.inputs.project_snapshot import ProjectSnapshot


def _build_world(root: Path) -> tuple[ProjectSnapshot, ProjectWorldModel]:
    snapshot = ProjectSnapshot.from_path(root)
    world = ProjectWorldModel.from_snapshot(snapshot, objective="security boundary")
    return snapshot, world


def test_project_snapshot_excludes_env_and_symlink_targets(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "safe.py").write_text("SAFE = True\n", encoding="utf-8")
    (tmp_path / ".env").write_text("COGEV_LLM_API_KEY=secret\n", encoding="utf-8")
    outside = tmp_path.parent / f"outside-secret-{tmp_path.name}.txt"
    outside.write_text("outside-secret", encoding="utf-8")
    try:
        os.symlink(outside, tmp_path / "pkg" / "linked_secret.txt")
    except (OSError, NotImplementedError):
        pass

    snapshot = ProjectSnapshot.from_path(tmp_path)
    paths = {item["path"] for item in snapshot.file_manifest}

    assert "pkg/safe.py" in paths
    assert ".env" not in paths
    assert "pkg/linked_secret.txt" not in paths
    assert all("secret" not in path.lower() for path in paths)


def test_context_selector_reads_only_snapshot_manifest_paths(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "safe.py").write_text("SAFE = True\n", encoding="utf-8")
    (tmp_path / ".env").write_text("COGEV_LLM_API_KEY=secret\n", encoding="utf-8")
    outside = tmp_path.parent / f"outside-{tmp_path.name}.txt"
    outside.write_text("outside-secret", encoding="utf-8")
    snapshot, world = _build_world(tmp_path)

    packet = ContextSelector().build_context_packet(
        contract=NexusProjectObjectiveContract(original_user_goal="repair", normalized_goal="repair"),
        snapshot=snapshot,
        world=world,
        request=ContextRequest(need_files=["pkg/safe.py", "pkg/../../outside.txt", "sub/../.env", str(outside)]),
    )

    assert "pkg/safe.py" in packet.raw_file_slices
    assert all("secret" not in text for text in packet.raw_file_slices.values())
    assert ".env" not in packet.coverage["selected_files"]
    assert "pkg/../../outside.txt" not in packet.coverage["selected_files"]


def test_context_selector_does_not_fall_back_to_unmatched_raw_path(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "safe.py").write_text("SAFE = True\n", encoding="utf-8")
    snapshot, world = _build_world(tmp_path)

    packet = ContextSelector().build_context_packet(
        contract=NexusProjectObjectiveContract(original_user_goal="repair", normalized_goal="repair"),
        snapshot=snapshot,
        world=world,
        request=ContextRequest(need_files=["missing.py"]),
    )

    assert "missing.py" not in packet.coverage["selected_files"]
    assert "missing.py" not in packet.raw_file_slices
