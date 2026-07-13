from __future__ import annotations

from cognitive_evolve_runtime.artifacts.task_files import _task_seed_prompt, ensure_task_skeleton


def _task_with_user_input(tmp_path, text: str):
    task_dir = tmp_path / "task"
    ensure_task_skeleton(task_dir, task_type="nexus", slug="prompt")
    (task_dir / "intake").mkdir()
    (task_dir / "intake" / "user-input.md").write_text(text, encoding="utf-8")
    return task_dir


def test_task_seed_prompt_prefers_plain_intake_user_input_over_slug(tmp_path) -> None:
    original = "Prompt.\n\nLet P ⊂ R^2 be finite. Resolve the Erdős unit-distance problem completely."
    assert _task_seed_prompt(_task_with_user_input(tmp_path, original)) == original


def test_task_seed_prompt_keeps_fenced_text_compatibility(tmp_path) -> None:
    assert _task_seed_prompt(_task_with_user_input(tmp_path, "```text\nreal prompt\n```")) == "real prompt"


def test_task_seed_prompt_preserves_real_prompt_with_embedded_fence(tmp_path) -> None:
    original = "Solve the actual task.\n```text\nsupporting block\n```\nDo not replace the goal with the block."
    assert _task_seed_prompt(_task_with_user_input(tmp_path, original)) == original
