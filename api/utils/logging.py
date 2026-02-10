"""
Structured logging configuration using structlog.
Provides request-scoped logging with automatic context injection.
"""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.types import EventDict, Processor

from api.config import get_settings


def add_request_id(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add request ID to log context if available."""
    # Request ID will be injected by middleware
    return event_dict


def add_severity(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add severity level for structured logging."""
    event_dict["severity"] = method_name.upper()
    return event_dict


def configure_logging() -> None:
    """
    Configure structured logging for the application.
    Uses JSON format in production, console format in development.
    """
    settings = get_settings()

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )

    # Choose renderer based on environment
    if settings.log_format == "json" and not settings.dev_mode:
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            add_request_id,
            add_severity,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def log_event(
    logger: structlog.BoundLogger,
    level: str,
    event: str,
    **kwargs: Any,
) -> None:
    """
    Log a structured event.

    Args:
        logger: Structlog logger instance
        level: Log level (info, warning, error, etc.)
        event: Event name/message
        **kwargs: Additional context fields
    """
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(event, **kwargs)
