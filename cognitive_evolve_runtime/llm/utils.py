from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.redaction import redact


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redact(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class DynamicStdoutHandler(logging.Handler):
    """Logging handler that resolves stdout at emit time."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - CLI behavior
        try:
            sys.stdout.write(self.format(record) + "\n")
            sys.stdout.flush()
        except Exception:
            self.handleError(record)


_CLI_LOGGER = logging.getLogger("cognitive_evolve.cli")
_CLI_LOGGER_LOCK = threading.Lock()
_CLI_LOGGER_READY = False


def cli_logger() -> logging.Logger:
    global _CLI_LOGGER_READY
    if _CLI_LOGGER_READY:
        return _CLI_LOGGER
    with _CLI_LOGGER_LOCK:
        if not _CLI_LOGGER_READY:
            if not any(getattr(handler, "_cogev_cli_handler", False) for handler in _CLI_LOGGER.handlers):
                handler = DynamicStdoutHandler()
                handler.setFormatter(logging.Formatter("%(message)s"))
                handler._cogev_cli_handler = True  # type: ignore[attr-defined]
                _CLI_LOGGER.addHandler(handler)
            _CLI_LOGGER.setLevel(logging.INFO)
            _CLI_LOGGER.propagate = False
            _CLI_LOGGER_READY = True
    return _CLI_LOGGER
