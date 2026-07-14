from __future__ import annotations

import errno
import multiprocessing
from pathlib import Path
import warnings

import pytest

from cognitive_evolve_runtime.durable import file_lock as file_lock_module
from cognitive_evolve_runtime.durable.file_lock import file_lock


def _competing_lock_worker(
    path: str,
    attempting: multiprocessing.synchronize.Event,
    entered: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
    active: multiprocessing.sharedctypes.Synchronized,
    overlap: multiprocessing.sharedctypes.Synchronized,
) -> None:
    attempting.set()
    with file_lock(path):
        with active.get_lock():
            if active.value:
                overlap.value = 1
            active.value += 1
        entered.set()
        release.wait(15)
        with active.get_lock():
            active.value -= 1


def test_file_lock_serializes_two_real_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    active = context.Value("i", 0)
    overlap = context.Value("i", 0)
    first_attempting, first_entered, first_release = context.Event(), context.Event(), context.Event()
    second_attempting, second_entered, second_release = context.Event(), context.Event(), context.Event()
    first = context.Process(
        target=_competing_lock_worker,
        args=(str(tmp_path / "shared.lock"), first_attempting, first_entered, first_release, active, overlap),
    )
    second = context.Process(
        target=_competing_lock_worker,
        args=(str(tmp_path / "shared.lock"), second_attempting, second_entered, second_release, active, overlap),
    )
    try:
        first.start()
        assert first_attempting.wait(10)
        assert first_entered.wait(10)
        second.start()
        assert second_attempting.wait(10)
        first_release.set()
        assert second_entered.wait(15)
        second_release.set()
        first.join(10)
        second.join(10)
        assert first.exitcode == 0
        assert second.exitcode == 0
        assert overlap.value == 0
    finally:
        first_release.set()
        second_release.set()
        for process in (first, second):
            if process.is_alive():
                process.terminate()
            process.join(5)


class _FakeMsvcrt:
    LK_LOCK = 1
    LK_UNLCK = 2

    def __init__(self, lock_errors: list[int], *, unlock_error: int | None = None) -> None:
        self.lock_errors = list(lock_errors)
        self.unlock_error = unlock_error
        self.calls: list[int] = []

    def locking(self, _fd: int, mode: int, _size: int) -> None:
        self.calls.append(mode)
        if mode == self.LK_LOCK and self.lock_errors:
            error_number = self.lock_errors.pop(0)
            raise OSError(error_number, "fake locking error")
        if mode == self.LK_UNLCK and self.unlock_error is not None:
            raise OSError(self.unlock_error, "fake unlock error")


def test_windows_lock_retries_only_explicit_contention(monkeypatch, tmp_path: Path) -> None:
    backend = _FakeMsvcrt([errno.EACCES, errno.EAGAIN])
    monkeypatch.setattr(file_lock_module, "fcntl", None)
    monkeypatch.setattr(file_lock_module, "msvcrt", backend)

    with file_lock(tmp_path / "windows.lock"):
        pass

    assert backend.calls == [backend.LK_LOCK, backend.LK_LOCK, backend.LK_LOCK, backend.LK_UNLCK]


def test_windows_lock_propagates_non_contention_error(monkeypatch, tmp_path: Path) -> None:
    backend = _FakeMsvcrt([errno.EIO])
    monkeypatch.setattr(file_lock_module, "fcntl", None)
    monkeypatch.setattr(file_lock_module, "msvcrt", backend)

    with pytest.raises(OSError) as error:
        with file_lock(tmp_path / "windows.lock"):
            pass

    assert error.value.errno == errno.EIO


def test_windows_lock_propagates_unlock_error(monkeypatch, tmp_path: Path) -> None:
    backend = _FakeMsvcrt([], unlock_error=errno.EIO)
    monkeypatch.setattr(file_lock_module, "fcntl", None)
    monkeypatch.setattr(file_lock_module, "msvcrt", backend)

    with pytest.raises(OSError) as error:
        with file_lock(tmp_path / "windows.lock"):
            pass

    assert error.value.errno == errno.EIO


def test_missing_cross_process_backend_warns_once(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(file_lock_module, "fcntl", None)
    monkeypatch.setattr(file_lock_module, "msvcrt", None)
    monkeypatch.setattr(file_lock_module, "_cross_process_lock_warning_emitted", False)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with file_lock(tmp_path / "first.lock"):
            pass
        with file_lock(tmp_path / "second.lock"):
            pass

    assert [str(item.message) for item in captured] == ["cross_process_lock_unavailable"]
