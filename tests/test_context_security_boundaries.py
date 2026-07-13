from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from cognitive_evolve_runtime.candidates.project_candidate import PatchOperation, ProjectCandidateGenome
from cognitive_evolve_runtime.contracts.objective_contract import NexusProjectObjectiveContract
from cognitive_evolve_runtime.inputs.context_selector import ContextRequest, ContextSelector
from cognitive_evolve_runtime.inputs.project_map import ProjectWorldModel
from cognitive_evolve_runtime.inputs.project_snapshot import ProjectSnapshot
from cognitive_evolve_runtime.tools.patch_sandbox import PatchSandbox


def _build_world(root: Path) -> tuple[ProjectSnapshot, ProjectWorldModel]:
    snapshot = ProjectSnapshot.from_path(root)
    world = ProjectWorldModel.from_snapshot(snapshot, objective="security boundary")
    return snapshot, world


def test_project_snapshot_excludes_env_and_symlink_targets(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "safe.py").write_text("SAFE = True\n", encoding="utf-8")
    (tmp_path / ".env").write_text("COGEV_LLM_API_KEY=secret\n", encoding="utf-8")
    (tmp_path / "id_ed25519").write_text("private-key\n", encoding="utf-8")
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
    assert "id_ed25519" not in paths
    assert "pkg/linked_secret.txt" not in paths
    assert all("secret" not in path.lower() for path in paths)


def test_patch_sandbox_copy_excludes_sensitive_generated_and_symlink_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "safe.py").write_text("SAFE = True\n", encoding="utf-8")
    for name in (".env", ".npmrc", ".pypirc", "id_rsa", "deploy.pem", "service-secret.json", "access-token.txt"):
        (source / name).write_text("sensitive\n", encoding="utf-8")
    for name in ("tokenizer.py", "token_budget.py", "secret_sharing.py", "tokenizer.json", "token_budget.yaml", "secret_sharing.txt"):
        (source / name).write_text("DOMAIN_SOURCE = True\n", encoding="utf-8")
    (source / "dist").mkdir()
    (source / "dist" / "generated.js").write_text("generated\n", encoding="utf-8")
    outside = tmp_path / "outside-private-key"
    outside.write_text("private-key\n", encoding="utf-8")
    (source / "linked-key").symlink_to(outside)

    sandbox = PatchSandbox(source, tmp_path / "sandboxes").prepare("candidate")

    assert (sandbox / "safe.py").is_file()
    assert not (sandbox / "dist").exists()
    assert not (sandbox / "linked-key").exists()
    assert all(not (sandbox / name).exists() for name in (".env", ".npmrc", ".pypirc", "id_rsa", "deploy.pem", "service-secret.json", "access-token.txt"))
    assert all(
        (sandbox / name).is_file()
        for name in ("tokenizer.py", "token_budget.py", "secret_sharing.py", "tokenizer.json", "token_budget.yaml", "secret_sharing.txt")
    )


def test_patch_sandbox_hashes_token_named_domain_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "tokenizer.py").write_text("VALUE = 1\n", encoding="utf-8")
    candidate = ProjectCandidateGenome(
        id="candidate",
        patch_set=[
            PatchOperation(
                path="tokenizer.py",
                operation="replace",
                old_text="VALUE = 1",
                new_text="VALUE = 2",
            )
        ],
    )

    result = PatchSandbox(source, tmp_path / "sandboxes").apply(candidate)

    assert result.status == "applied"
    assert result.pre_hash != result.post_hash
    assert (Path(result.sandbox_path) / "tokenizer.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def test_patch_sandbox_makes_read_only_source_copy_mutable_without_changing_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    package = source / "pkg"
    package.mkdir(parents=True)
    script = package / "script.sh"
    notes = package / "notes.txt"
    removed = package / "removed.txt"
    script.write_text("echo before\n", encoding="utf-8")
    notes.write_text("before\n", encoding="utf-8")
    removed.write_text("remove me\n", encoding="utf-8")
    script.chmod(0o555)
    notes.chmod(0o444)
    removed.chmod(0o444)
    package.chmod(0o555)
    source.chmod(0o555)
    candidate = ProjectCandidateGenome(
        id="candidate",
        patch_set=[
            PatchOperation(path="pkg/script.sh", operation="replace", old_text="before", new_text="after"),
            PatchOperation(path="pkg/notes.txt", operation="append", content="after\n"),
            PatchOperation(path="pkg/created.txt", operation="write", content="created\n"),
            PatchOperation(path="pkg/removed.txt", operation="delete"),
        ],
    )

    try:
        patch_sandbox = PatchSandbox(source, tmp_path / "sandboxes")
        result = patch_sandbox.apply(candidate)
        sandbox = Path(result.sandbox_path)

        assert result.status == "applied"
        assert (sandbox / "pkg" / "script.sh").read_text(encoding="utf-8") == "echo after\n"
        assert (sandbox / "pkg" / "notes.txt").read_text(encoding="utf-8") == "before\nafter\n"
        assert (sandbox / "pkg" / "created.txt").read_text(encoding="utf-8") == "created\n"
        assert not (sandbox / "pkg" / "removed.txt").exists()
        assert stat.S_IMODE((sandbox / "pkg").stat().st_mode) == 0o755
        assert stat.S_IMODE((sandbox / "pkg" / "script.sh").stat().st_mode) == 0o755
        assert stat.S_IMODE((sandbox / "pkg" / "notes.txt").stat().st_mode) == 0o644

        rebuilt = patch_sandbox.prepare(candidate.id)
        assert (rebuilt / "pkg" / "script.sh").read_text(encoding="utf-8") == "echo before\n"
        assert not (rebuilt / "pkg" / "created.txt").exists()
        assert (rebuilt / "pkg" / "removed.txt").read_text(encoding="utf-8") == "remove me\n"

        assert script.read_text(encoding="utf-8") == "echo before\n"
        assert notes.read_text(encoding="utf-8") == "before\n"
        assert removed.read_text(encoding="utf-8") == "remove me\n"
        assert stat.S_IMODE(source.stat().st_mode) == 0o555
        assert stat.S_IMODE(package.stat().st_mode) == 0o555
        assert stat.S_IMODE(script.stat().st_mode) == 0o555
        assert stat.S_IMODE(notes.stat().st_mode) == 0o444
    finally:
        source.chmod(0o755)
        package.chmod(0o755)
        for path in package.iterdir():
            path.chmod(0o644)


def test_patch_sandbox_applies_unified_diff_to_read_only_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    package = source / "pkg"
    package.mkdir(parents=True)
    target = package / "value.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    target.chmod(0o444)
    package.chmod(0o555)
    source.chmod(0o555)
    candidate = ProjectCandidateGenome(
        id="unified",
        artifact={
            "unified_diff": (
                "diff --git a/pkg/value.py b/pkg/value.py\n"
                "--- a/pkg/value.py\n"
                "+++ b/pkg/value.py\n"
                "@@ -1 +1 @@\n"
                "-VALUE = 1\n"
                "+VALUE = 2\n"
            )
        },
    )

    try:
        result = PatchSandbox(source, tmp_path / "sandboxes").apply(candidate)

        assert result.status == "applied"
        assert (Path(result.sandbox_path) / "pkg" / "value.py").read_text(encoding="utf-8") == "VALUE = 2\n"
        assert target.read_text(encoding="utf-8") == "VALUE = 1\n"
        assert stat.S_IMODE(target.stat().st_mode) == 0o444
    finally:
        source.chmod(0o755)
        package.chmod(0o755)
        target.chmod(0o644)


def test_patch_sandbox_prepare_supports_in_tree_sandbox_root_without_recursion(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "safe.py").write_text("SAFE = True\n", encoding="utf-8")
    sandbox_root = source / "nexus-runtime" / "patch-sandboxes"
    (sandbox_root.parent / "keep.txt").parent.mkdir(parents=True)
    (sandbox_root.parent / "keep.txt").write_text("keep\n", encoding="utf-8")

    sandbox = PatchSandbox(source, sandbox_root).prepare("candidate")

    assert (sandbox / "safe.py").is_file()
    assert (sandbox / "nexus-runtime" / "keep.txt").read_text(encoding="utf-8") == "keep\n"
    assert not (sandbox / "nexus-runtime" / "patch-sandboxes").exists()


def test_patch_sandbox_rejects_unsafe_candidate_ids_before_touching_sandbox(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "safe.py").write_text("SAFE = True\n", encoding="utf-8")
    sandbox_root = tmp_path / "sandboxes"
    victim = tmp_path / "victim"
    victim.mkdir()
    sentinel = victim / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    patch_sandbox = PatchSandbox(source, sandbox_root)

    for candidate_id in ("../victim", "", ".", "..", "nested/id", "nested\\id", str(victim)):
        with pytest.raises(ValueError, match="unsafe candidate id"):
            patch_sandbox.prepare(candidate_id)
    with pytest.raises(ValueError, match="unsafe candidate id"):
        patch_sandbox.apply(ProjectCandidateGenome(id="../victim"))
    with pytest.raises(ValueError, match="unsafe candidate id"):
        PatchSandbox(source, tmp_path).prepare(source.name)
    alias_root = tmp_path / "alias-sandboxes"
    alias_root.mkdir()
    alias_victim = alias_root / "victim"
    alias_victim.mkdir()
    alias_sentinel = alias_victim / "sentinel.txt"
    alias_sentinel.write_text("keep\n", encoding="utf-8")
    (alias_root / "candidate").symlink_to(alias_victim, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe candidate id"):
        PatchSandbox(source, alias_root).prepare("candidate")

    assert (source / "safe.py").read_text(encoding="utf-8") == "SAFE = True\n"
    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert alias_sentinel.read_text(encoding="utf-8") == "keep\n"
    assert not sandbox_root.exists()


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
