"""Best-effort transactional snapshots for multi-file Nexus artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cognitive_evolve_runtime.durable.file_lock import atomic_write_json, atomic_write_text, file_lock
from cognitive_evolve_runtime.core.serialization import utc_now


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
    """Stage a coherent set of snapshot files before publishing them.

    This keeps the existing JSON/text storage model but removes the most common
    inconsistency mode: failed serialization or staging no longer overwrites a
    subset of ``population.json``, ``archives.json``, ``checkpoint.json`` and
    ``run-result.json``.  A manifest records the generation and file hashes for
    readers and diagnostics.  Event JSONL remains append-only and is intentionally
    outside this snapshot transaction.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def commit(self, writes: list[SnapshotWrite]) -> SnapshotTransactionResult:
        self.root.mkdir(parents=True, exist_ok=True)
        transaction_id = "txn-" + uuid.uuid4().hex
        staging = self.root / f".{transaction_id}.staging"
        lock_path = self.root / ".snapshot-transaction.lock"
        with file_lock(lock_path):
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
                atomic_write_json(staging / "snapshot-transaction.json", manifest, sort_keys=True)

                published: dict[str, str] = {}
                for rel in manifest_files:
                    src = staging / rel
                    dst = self.root / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(src, dst)
                    published[rel] = str(dst)
                manifest_path = self.root / "snapshot-transaction.json"
                os.replace(staging / "snapshot-transaction.json", manifest_path)
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


__all__ = ["NexusSnapshotTransaction", "SnapshotTransactionResult", "SnapshotWrite"]
