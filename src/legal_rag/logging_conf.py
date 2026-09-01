"""Structured logging configuration."""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_HANDLER_NAME = "legal_rag_stdout"


class _JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(log_entry, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging to emit structured JSON to stdout."""

    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    handler = next(
        (
            existing_handler
            for existing_handler in root_logger.handlers
            if existing_handler.get_name() == _HANDLER_NAME
        ),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
        handler.set_name(_HANDLER_NAME)
        root_logger.addHandler(handler)

    handler.setLevel(level.upper())
    handler.setFormatter(_JsonFormatter())
