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

# Default log level (configurable via env)
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a configured logger instance.

    - Idempotent: handlers are added only once
    - Stream-based: logs to stdout
    - Safe for libraries & applications
    """
    logger = logging.getLogger(name)

    log_level = level or DEFAULT_LOG_LEVEL
    logger.setLevel(log_level)

    # Prevent duplicate handlers on repeated imports
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Prevent propagation to root logger (avoids double logging)
        logger.propagate = False

    return logger
