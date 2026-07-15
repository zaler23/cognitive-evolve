"""Crash-atomic transactional snapshots for multi-file Nexus artifacts."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from cognitive_evolve_runtime.durable.file_lock import _fsync_dir, atomic_write_json, atomic_write_text, file_lock
from cognitive_evolve_runtime.core.serialization import utc_now


_CURRENT = "CURRENT"
_GENERATIONS = "generations"
_MANIFEST = "snapshot-transaction.json"
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotWrite:
    relative_path: str
    kind: str
    payload: Any
    sort_keys: bool = True


@dataclass(frozen=True)
class SnapshotTransactionResult:
    transaction_id: str
    files: dict[str, str] = field(default_factory=dict)
    manifest_path: str = ""


class NexusSnapshotTransaction:
    """Publish a coherent snapshot by atomically switching one pointer.

    Files are completed under ``generations/<transaction-id>`` before ``CURRENT``
    is replaced. Readers that span multiple files use :func:`snapshot_reader`;
    a crash exposes either the previous complete generation or the new one.
    Only those two generations are retained. Event JSONL remains append-only.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def commit(self, writes: list[SnapshotWrite]) -> SnapshotTransactionResult:
        self.root.mkdir(parents=True, exist_ok=True)
        transaction_id = "txn-" + uuid.uuid4().hex
        generations = self.root / _GENERATIONS
        staging = generations / f".{transaction_id}.staging"
        generation = generations / transaction_id
        lock_path = self.root / ".snapshot-transaction.lock"
        with file_lock(lock_path):
            generations.mkdir(parents=True, exist_ok=True)
            previous_root = resolve_snapshot_root(self.root)
            previous_transaction_id = previous_root.name if previous_root.parent == generations else ""
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            manifest_files: dict[str, dict[str, Any]] = {}
            try:
                for item in writes:
                    target = _safe_relative_path(item.relative_path)
                    staged_path = staging / target
                    staged_path.parent.mkdir(parents=True, exist_ok=True)
                    if item.kind == "json":
                        atomic_write_json(staged_path, item.payload, sort_keys=item.sort_keys)
                    elif item.kind == "text":
                        atomic_write_text(staged_path, str(item.payload))
                    else:
                        raise ValueError(f"unsupported snapshot write kind: {item.kind}")
                    manifest_files[target] = {
                        "kind": item.kind,
                        "sha256": _sha256(staged_path),
                        "bytes": staged_path.stat().st_size,
                    }
                manifest = {
                    "schema": "cogev.nexus_snapshot_transaction.v1",
                    "transaction_id": transaction_id,
                    "created_at": utc_now(),
                    "files": manifest_files,
                }
                atomic_write_json(staging / _MANIFEST, manifest, sort_keys=True)
                for relative_path in [*manifest_files, _MANIFEST]:
                    path = staging / relative_path
                    path.with_name(path.name + ".lock").unlink(missing_ok=True)
                _fsync_tree(staging)

                os.replace(staging, generation)
                _fsync_dir(generations)
                _publish_current(self.root, transaction_id)
                _ensure_legacy_links(self.root, [*manifest_files, _MANIFEST])
                _prune_generations(generations, keep={transaction_id, previous_transaction_id})

                published = {rel: str(generation / rel) for rel in manifest_files}
                manifest_path = generation / _MANIFEST
                return SnapshotTransactionResult(transaction_id=transaction_id, files=published, manifest_path=str(manifest_path))
            finally:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)


def _safe_relative_path(value: str) -> str:
    rel = str(value or "").strip().replace("\\", "/")
    if not rel or rel.startswith("/") or ".." in Path(rel).parts:
        raise ValueError(f"snapshot path must be project-relative and safe: {value!r}")
    return rel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_snapshot_json(snapshot_root: str | Path, relative_path: str) -> Any:
    """Read one JSON snapshot file after verifying its declared SHA-256."""

    root = Path(snapshot_root)
    relative = _safe_relative_path(relative_path)
    payload = (root / relative).read_bytes()
    manifest_path = root / _MANIFEST
    if not manifest_path.exists():
        if root.parent.name == _GENERATIONS and root.name.startswith("txn-"):
            raise ValueError(f"snapshot manifest missing for published generation: {manifest_path}")
        _LOG.info("snapshot manifest missing; skipping hash verification for legacy snapshot file %s", root / relative)
        return json.loads(payload)
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"snapshot manifest is unreadable: {manifest_path}") from exc
    files = manifest.get("files") if isinstance(manifest, dict) else None
    entry = files.get(relative) if isinstance(files, dict) else None
    actual_hash = hashlib.sha256(payload).hexdigest()
    expected_hash = str(entry.get("sha256") or "") if isinstance(entry, dict) else ""
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError(
            f"snapshot hash mismatch for {relative}: "
            f"expected sha256={expected_hash or '<missing>'}, actual sha256={actual_hash}"
        )
    return json.loads(payload)


def resolve_snapshot_root(root: str | Path) -> Path:
    """Resolve the current generation, or a legacy flat root.

    Use :func:`snapshot_reader` when the returned path must remain readable
    across more than one filesystem operation while writers may be active.
    """

    base = Path(root)
    current = base / _CURRENT
    try:
        target = Path(os.readlink(current))
    except FileNotFoundError:
        return base
    except OSError:
        if not current.exists():
            return base
        transaction_id = current.read_text(encoding="utf-8").strip()
        target = Path(_GENERATIONS) / transaction_id
    if target.is_absolute() or len(target.parts) != 2 or target.parts[0] != _GENERATIONS:
        raise ValueError(f"invalid snapshot CURRENT pointer: {target}")
    transaction_id = target.parts[1]
    if not transaction_id.startswith("txn-") or Path(transaction_id).name != transaction_id:
        raise ValueError(f"invalid snapshot generation id: {transaction_id!r}")
    return base / _GENERATIONS / transaction_id


def resolve_snapshot_path(root: str | Path, relative_path: str) -> Path:
    """Resolve one current path; use ``snapshot_reader`` for a read lease."""

    return resolve_snapshot_root(root) / _safe_relative_path(relative_path)


@contextmanager
def snapshot_reader(root: str | Path) -> Iterator[Path]:
    """Hold the snapshot lock while a resolved generation is being read."""

    base = Path(root)
    with file_lock(base / ".snapshot-transaction.lock"):
        yield resolve_snapshot_root(base)


def _fsync_tree(root: Path) -> None:
    for directory, _, _ in os.walk(root, topdown=False):
        _fsync_dir(Path(directory))


def _publish_current(root: Path, transaction_id: str) -> None:
    current = root / _CURRENT
    tmp = root / f".{_CURRENT}.{uuid.uuid4().hex}.tmp"
    try:
        try:
            os.symlink(str(Path(_GENERATIONS) / transaction_id), tmp)
        except OSError:  # pragma: no cover - exercised on platforms without symlink support.
            atomic_write_text(current, transaction_id + "\n")
            return
        os.replace(tmp, current)
        _fsync_dir(root)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _ensure_legacy_links(root: Path, relative_paths: list[str]) -> None:
    """Keep old flat artifact paths as non-authoritative CURRENT mappings."""

    if not (root / _CURRENT).is_symlink():
        current_root = resolve_snapshot_root(root)
        for relative_path in relative_paths:
            source = current_root / relative_path
            target = root / relative_path
            if not source.is_file():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.copy")
            try:
                shutil.copyfile(source, tmp)
                with tmp.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(tmp, target)
                _fsync_dir(target.parent)
            finally:
                tmp.unlink(missing_ok=True)
        return
    for relative_path in relative_paths:
        link = root / relative_path
        link.parent.mkdir(parents=True, exist_ok=True)
        target = Path(os.path.relpath(root / _CURRENT / relative_path, link.parent))
        tmp = link.with_name(f".{link.name}.{uuid.uuid4().hex}.link")
        try:
            os.symlink(str(target), tmp)
            os.replace(tmp, link)
        except OSError:
            # CURRENT-aware readers remain authoritative on platforms without symlinks.
            pass
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def _prune_generations(generations: Path, *, keep: set[str]) -> None:
    try:
        for path in generations.iterdir():
            if path.is_dir() and path.name.startswith("txn-") and path.name not in keep:
                shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


__all__ = [
    "NexusSnapshotTransaction",
    "SnapshotTransactionResult",
    "SnapshotWrite",
    "read_snapshot_json",
    "resolve_snapshot_path",
    "resolve_snapshot_root",
    "snapshot_reader",
]
