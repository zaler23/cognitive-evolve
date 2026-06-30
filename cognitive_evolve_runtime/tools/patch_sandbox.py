"""Apply project candidate patches inside isolated local sandboxes."""
from __future__ import annotations

import re
import shutil
import subprocess
import fnmatch
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.candidates.genome import CandidateGenome
from cognitive_evolve_runtime.candidates.project_candidate import PatchApplicationResult, PatchOperation, ProjectCandidateGenome
from cognitive_evolve_runtime.inputs.project_snapshot import ProjectSnapshot
from cognitive_evolve_runtime.durable.file_lock import file_lock


class PatchSandbox:
    def __init__(self, source_root: str | Path, sandbox_root: str | Path, *, allowed_patch_scope: list[str] | None = None) -> None:
        self.source_root = Path(source_root).resolve()
        self.sandbox_root = Path(sandbox_root).resolve()
        self.allowed_patch_scope = [str(item) for item in allowed_patch_scope or [] if str(item).strip()]

    def prepare(self, candidate_id: str) -> Path:
        target = self.sandbox_root / candidate_id
        lock_path = self.sandbox_root / f".{candidate_id}.sandbox.lock"
        with file_lock(lock_path):
            if target.exists():
                shutil.rmtree(target)
            ignore = _sandbox_copy_ignore
            shutil.copytree(self.source_root, target, ignore=ignore)
        return target

    def apply(self, candidate: CandidateGenome) -> PatchApplicationResult:
        lock_path = self.sandbox_root / f".{candidate.id}.apply.lock"
        with file_lock(lock_path):
            return self._apply_locked(candidate)

    def _apply_locked(self, candidate: CandidateGenome) -> PatchApplicationResult:
        sandbox = self.prepare(candidate.id)
        pre_hash = ProjectSnapshot.from_path(sandbox).root_hash
        applied: list[str] = []
        failed: list[str] = []
        diagnostics: list[str] = []
        patch_set = list(getattr(candidate, "patch_set", []) or [])
        for op in patch_set:
            ok, message = self._apply_operation(sandbox, op)
            if ok:
                applied.append(op.path)
            else:
                failed.append(op.path)
                diagnostics.append(message)
        if not patch_set:
            generic_patch = _generic_unified_patch_text(candidate)
            if generic_patch:
                ok, message, patch_files = self._apply_unified_patch(sandbox, generic_patch)
                if ok:
                    applied.extend(patch_files)
                else:
                    failed.extend(patch_files or _paths_from_unified_patch(generic_patch, strip=_strip_level_for_unified_patch(generic_patch)))
                    diagnostics.append(message)
        post_hash = ProjectSnapshot.from_path(sandbox).root_hash
        if failed:
            status = "failed"
        elif not applied:
            status = "no_op"
            diagnostics.append("patch_no_effect:no_files_applied")
        elif pre_hash == post_hash:
            status = "no_op"
            diagnostics.append("patch_no_effect:pre_hash_equals_post_hash")
        else:
            status = "applied"
        result = PatchApplicationResult(
            status=status,
            diagnostics=diagnostics,
            applied_files=applied,
            failed_files=failed,
            pre_hash=pre_hash,
            post_hash=post_hash,
            sandbox_path=str(sandbox),
        )
        setattr(candidate, "patch_application_result", result.to_dict())
        return result

    def _apply_operation(self, sandbox: Path, op: PatchOperation) -> tuple[bool, str]:
        scope_error = _patch_scope_error(op.path, self.allowed_patch_scope)
        if scope_error:
            return False, scope_error
        if _source_path_uses_symlink(self.source_root, op.path):
            return False, f"unsafe source symlink in patch path: {op.path}"
        target, guard_error = _safe_patch_target(sandbox, op.path)
        if guard_error:
            return False, guard_error
        try:
            if op.operation == "delete":
                target, guard_error = _safe_patch_target(sandbox, op.path, allow_missing=True)
                if guard_error:
                    return False, guard_error
                if target.exists():
                    if target.is_dir():
                        return False, f"delete target is a directory: {op.path}"
                    target.unlink()
                return True, "deleted"
            if op.operation == "write":
                target, guard_error = _safe_patch_target(sandbox, op.path, create_parent=True)
                if guard_error:
                    return False, guard_error
                target.write_text(op.content, encoding="utf-8")
                return True, "written"
            if op.operation == "append":
                target, guard_error = _safe_patch_target(sandbox, op.path, create_parent=True)
                if guard_error:
                    return False, guard_error
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(op.content)
                return True, "appended"
            if op.operation == "replace":
                target, guard_error = _safe_patch_target(sandbox, op.path)
                if guard_error:
                    return False, guard_error
                if not target.exists():
                    return False, f"replace target missing: {op.path}"
                text = target.read_text(encoding="utf-8")
                if op.old_text not in text:
                    return False, f"old_text not found in {op.path}"
                target.write_text(text.replace(op.old_text, op.new_text), encoding="utf-8")
                return True, "replaced"
            return False, f"unsupported patch operation: {op.operation}"
        except OSError as exc:
            return False, str(exc)

    def _apply_unified_patch(self, sandbox: Path, patch_text: str) -> tuple[bool, str, list[str]]:
        patch_text, repair_notes = _repair_unified_patch_text(patch_text)
        strip = _strip_level_for_unified_patch(patch_text)
        patch_files = _paths_from_unified_patch(patch_text, strip=strip)
        preflight = preflight_unified_patch(patch_text)
        if not preflight["ok"]:
            return False, "patch_preflight_failed:" + ";".join(preflight["diagnostics"]), patch_files
        if not patch_files:
            return False, "patch_no_effect:no_files_declared", []
        for path in patch_files:
            scope_error = _patch_scope_error(path, self.allowed_patch_scope)
            if scope_error:
                return False, scope_error, patch_files
            if _source_path_uses_symlink(self.source_root, path):
                return False, f"unsafe source symlink in patch path: {path}", patch_files
            _target, guard_error = _safe_patch_target(sandbox, path, allow_missing=True)
            if guard_error:
                return False, guard_error, patch_files
        command = ["patch", f"-p{strip}", "-N", "-t"]
        try:
            completed = subprocess.run(
                command,
                cwd=sandbox,
                input=patch_text,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=20,
            )
        except FileNotFoundError:
            return False, "patch_tool_unavailable", patch_files
        except subprocess.TimeoutExpired:
            return False, "patch_tool_timeout", patch_files
        output = completed.stdout.strip()
        if completed.returncode != 0:
            detail = output.splitlines()[-1] if output else f"patch exited {completed.returncode}"
            return False, f"unified_patch_failed:{detail}", patch_files
        message = output or "unified_patch_applied"
        if repair_notes:
            message = message + "; " + ";".join(repair_notes)
        return True, message, patch_files


_COPY_EXCLUDES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv", "node_modules"}


def _sandbox_copy_ignore(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    root = Path(directory)
    for name in names:
        if name in _COPY_EXCLUDES or (root / name).is_symlink():
            ignored.add(name)
    return ignored


def _safe_patch_target(sandbox: Path, raw_path: str, *, create_parent: bool = False, allow_missing: bool = False) -> tuple[Path, str]:
    relative = Path(raw_path)
    if not raw_path or relative.is_absolute() or ".." in relative.parts:
        return sandbox, f"unsafe patch path: {raw_path}"
    sandbox_root = sandbox.resolve()
    target = sandbox / relative
    parent = target.parent
    if not _path_within(parent, sandbox_root):
        return target, f"unsafe patch parent outside sandbox: {raw_path}"
    for ancestor in _ancestors_between(sandbox_root, parent):
        if ancestor.exists() and ancestor.is_symlink():
            return target, f"unsafe symlink ancestor in patch path: {raw_path}"
        if ancestor.exists() and not _path_within(ancestor, sandbox_root):
            return target, f"unsafe patch ancestor outside sandbox: {raw_path}"
    if create_parent:
        parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink():
            return target, f"unsafe symlink patch target: {raw_path}"
        if not _path_within(target, sandbox_root):
            return target, f"unsafe patch target outside sandbox: {raw_path}"
    elif not allow_missing and not create_parent:
        # Existing-target operations call this before opening; missing is not a
        # safety violation and is handled by the operation-specific branch.
        pass
    return target, ""


def _patch_scope_error(raw_path: str, allowed_patch_scope: list[str]) -> str:
    relative = Path(raw_path)
    if not raw_path or relative.is_absolute() or ".." in relative.parts:
        return f"unsafe patch path: {raw_path}"
    normalized = relative.as_posix()
    if allowed_patch_scope and not any(fnmatch.fnmatchcase(normalized, pattern) for pattern in allowed_patch_scope):
        return f"patch path outside allowed_patch_scope: {raw_path}"
    return ""


def _generic_unified_patch_text(candidate: CandidateGenome) -> str:
    artifact = getattr(candidate, "artifact", None)
    if not isinstance(artifact, dict):
        return ""
    for key in ("patch", "patch_content", "diff", "unified_diff"):
        value = artifact.get(key)
        if isinstance(value, str) and value.strip():
            return value
    content = artifact.get("content")
    if isinstance(content, str) and _looks_like_unified_patch_text(content):
        return content
    return ""


_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
_DIFF_PATH_RE = re.compile(r"^(?:---|\+\+\+)\s+(.+?)\s*$")
_HUNK_RE = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@")


def preflight_unified_patch(patch_text: str) -> dict[str, Any]:
    """Return deterministic diagnostics before invoking the external patch tool."""

    text = str(patch_text or "")
    diagnostics: list[str] = []
    if not text.strip():
        diagnostics.append("patch_empty")
        return {"ok": False, "diagnostics": diagnostics}
    lowered = text.lower()
    if "*** begin patch" in lowered and "*** end patch" not in lowered:
        diagnostics.append("patch_truncated:missing_end_patch_marker")
    if any(marker in lowered for marker in ("[truncated]", "<truncated>", "… truncated", "content omitted")):
        diagnostics.append("patch_truncated:truncation_marker_present")
    lines = text.splitlines()
    for line in lines:
        if line.startswith("@@") and not _HUNK_RE.match(line):
            diagnostics.append("malformed_hunk_header:" + line[:120])
    if any(line.startswith("@@") for line in lines):
        hunk_indices = [index for index, line in enumerate(lines) if line.startswith("@@")]
        for index in hunk_indices:
            body = lines[index + 1 : index + 4]
            if not any(item.startswith(("+", "-", " ")) and not item.startswith(("+++", "---")) for item in body):
                diagnostics.append("malformed_hunk_body:empty_or_header_only")
                break
    raw_paths = _raw_paths_from_unified_patch(text)
    if not raw_paths:
        diagnostics.append("patch_no_effect:no_files_declared")
    for raw in raw_paths:
        path = _strip_patch_path(raw, strip=1 if raw.startswith(("a/", "b/")) else 0)
        if path and (Path(path).is_absolute() or ".." in Path(path).parts):
            diagnostics.append("unsafe_patch_path:" + path[:120])
    diagnostics.extend(_obvious_python_patch_syntax_diagnostics(text))
    return {"ok": not diagnostics, "diagnostics": list(dict.fromkeys(diagnostics))}


def _repair_unified_patch_text(patch_text: str) -> tuple[str, list[str]]:
    text = str(patch_text or "")
    notes: list[str] = []
    if text and not text.endswith("\n"):
        text += "\n"
        notes.append("patch_preflight_repaired:added_trailing_newline")
    return text, notes


def _obvious_python_patch_syntax_diagnostics(patch_text: str) -> list[str]:
    diagnostics: list[str] = []
    current_path = ""
    current_new_file = False
    added_new_file: dict[str, list[str]] = {}
    for line in str(patch_text or "").splitlines():
        git_match = _DIFF_GIT_RE.match(line.strip())
        if git_match:
            current_path = git_match.group(2)
            current_new_file = False
            continue
        if line.startswith("--- "):
            raw = line[4:].strip().split("\t", 1)[0]
            current_new_file = raw == "/dev/null"
            continue
        if line.startswith("+++ "):
            raw = line[4:].strip().split("\t", 1)[0]
            current_path = _strip_patch_path(raw, strip=1 if raw.startswith(("a/", "b/")) else 0)
            continue
        if current_path.endswith(".py") and line.startswith("+") and not line.startswith("+++"):
            if current_new_file:
                added_new_file.setdefault(current_path, []).append(line[1:])
            if "SyntaxError" in line or "IndentationError" in line:
                diagnostics.append(f"python_patch_obvious_syntax:error_marker_present:{current_path}")
    for path, added_lines in added_new_file.items():
        added_text = "\n".join(added_lines)
        if added_text.count("(") < added_text.count(")") or added_text.count("[") < added_text.count("]") or added_text.count("{") < added_text.count("}"):
            diagnostics.append(f"python_patch_obvious_syntax:unbalanced_closing_bracket:{path}")
    return diagnostics




def _looks_like_unified_patch_text(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    raw_paths = _raw_paths_from_unified_patch(text)
    if not raw_paths:
        return False
    has_hunk = any(line.startswith("@@") for line in text.splitlines())
    has_old = any(line.startswith("--- ") for line in text.splitlines())
    has_new = any(line.startswith("+++ ") for line in text.splitlines())
    has_git = any(_DIFF_GIT_RE.match(line.strip()) for line in text.splitlines())
    return bool(has_hunk and (has_git or (has_old and has_new)))

def _strip_level_for_unified_patch(patch_text: str) -> int:
    paths = _raw_paths_from_unified_patch(patch_text)
    meaningful = [path for path in paths if path != "/dev/null"]
    return 1 if meaningful and all(path.startswith(("a/", "b/")) for path in meaningful) else 0


def _paths_from_unified_patch(patch_text: str, *, strip: int) -> list[str]:
    out: list[str] = []
    for raw in _raw_paths_from_unified_patch(patch_text):
        if raw == "/dev/null":
            continue
        path = _strip_patch_path(raw, strip=strip)
        if path:
            out.append(path)
    return list(dict.fromkeys(out))


def _raw_paths_from_unified_patch(patch_text: str) -> list[str]:
    paths: list[str] = []
    for line in str(patch_text or "").splitlines():
        stripped = line.strip()
        git_match = _DIFF_GIT_RE.match(stripped)
        if git_match:
            paths.extend([f"a/{git_match.group(1)}", f"b/{git_match.group(2)}"])
            continue
        path_match = _DIFF_PATH_RE.match(stripped)
        if path_match:
            raw = path_match.group(1).strip()
            if "\t" in raw:
                raw = raw.split("\t", 1)[0]
            paths.append(raw)
    return paths


def _strip_patch_path(raw_path: str, *, strip: int) -> str:
    text = str(raw_path or "").strip().replace("\\", "/")
    if not text or text == "/dev/null":
        return ""
    parts = [part for part in text.split("/") if part]
    if strip:
        parts = parts[strip:]
    if not parts or any(part in {"..", "."} for part in parts):
        return ""
    return "/".join(parts)


def _ancestors_between(root: Path, leaf: Path) -> list[Path]:
    ancestors: list[Path] = []
    current = leaf
    while True:
        ancestors.append(current)
        if current == root or current.parent == current:
            break
        current = current.parent
    return list(reversed(ancestors))


def _path_within(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve()
    except FileNotFoundError:
        resolved = path.absolute()
    try:
        resolved.relative_to(root)
        return True
    except ValueError:
        return False


def _source_path_uses_symlink(source_root: Path, raw_path: str) -> bool:
    relative = Path(raw_path)
    if not raw_path or relative.is_absolute() or ".." in relative.parts:
        return False
    current = source_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
        if not current.exists():
            return False
    return False


__all__ = ["PatchSandbox", "preflight_unified_patch"]
