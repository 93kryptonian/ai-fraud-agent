"""
CENTRAL LOGGING UTILITY
----------------------

Provides a consistent, idempotent logger configuration
across the entire Fraud AI system.

Design goals:
- No duplicate handlers
- Human-readable logs
- Safe to import anywhere
- Easy to extend (levels, JSON logs, sinks)
"""

import logging
import os
from typing import Optional

from src.observability.context import get_request_id

# Default log level (configurable via env)
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


class _RequestIdFilter(logging.Filter):
    """
    Attach the current request_id (M3 observability context) to every log
    record emitted through this logger, so log lines are correlatable back
    to a single request even though the logger itself knows nothing about
    HTTP requests.

    "-" is used when there's no request in scope (e.g. a script, a test,
    or module-import-time logging) so the format string never breaks.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a configured logger instance.

    - Idempotent: handlers are added only once
    - Stream-based: logs to stdout
    - Safe for libraries & applications
    - Every record carries request_id (see _RequestIdFilter)
    """
    logger = logging.getLogger(name)

    log_level = level or DEFAULT_LOG_LEVEL
    logger.setLevel(log_level)

    # Prevent duplicate handlers on repeated imports
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - request_id=%(request_id)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Filter is attached to the logger itself (not just the handler),
        # so record.request_id is set before any handler — including a
        # handler attached later, e.g. by a test — processes the record.
        logger.addFilter(_RequestIdFilter())

        # Prevent propagation to root logger (avoids double logging)
        logger.propagate = False

    return logger
