#!/usr/bin/env python3
"""Build a reproducible self-evolution goal file from a preamble and input zip."""
from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path


DEFAULT_MARKDOWN_PATTERNS = ("*.md", "*.markdown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose bootstrap-goal.md from a goal preamble plus markdown input package.")
    parser.add_argument("--input-zip", required=True, help="Zip containing the self-evolve input package.")
    parser.add_argument("--goal-preamble", required=True, help="Exploration goal preamble markdown file.")
    parser.add_argument("--output", required=True, help="Output bootstrap-goal.md path.")
    parser.add_argument(
        "--include-glob",
        action="append",
        default=[],
        help="Glob inside the extracted input package. May be repeated. Defaults to markdown files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_zip = Path(args.input_zip).expanduser().resolve()
    preamble = Path(args.goal_preamble).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not input_zip.is_file():
        raise SystemExit(f"input zip not found: {input_zip}")
    if not preamble.is_file():
        raise SystemExit(f"goal preamble not found: {preamble}")

    with tempfile.TemporaryDirectory(prefix="cogev-goal-build-") as tmp:
        root = Path(tmp)
        _safe_extract(input_zip, root)
        body_files = _selected_body_files(root, tuple(args.include_glob or DEFAULT_MARKDOWN_PATTERNS))
        content = _compose_goal(preamble, body_files)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    print(str(output))
    return 0


def _safe_extract(input_zip: Path, root: Path) -> None:
    with zipfile.ZipFile(input_zip) as archive:
        for info in archive.infolist():
            name = info.filename
            if not name or name.endswith("/"):
                continue
            target = (root / name).resolve()
            try:
                target.relative_to(root.resolve())
            except ValueError as exc:
                raise SystemExit(f"unsafe zip member: {name}") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as destination:
                destination.write(source.read())


def _selected_body_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted(set(files), key=lambda path: str(path.relative_to(root)))


def _compose_goal(preamble: Path, body_files: list[Path]) -> str:
    parts = [preamble.read_text(encoding="utf-8").strip(), "\n\n---\n\n# Input package body\n"]
    for path in body_files:
        parts.append(f"\n\n## {path.name}\n\n")
        parts.append(path.read_text(encoding="utf-8", errors="replace").strip())
    parts.append("\n")
    return "".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
