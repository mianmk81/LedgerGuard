"""API routers for all endpoints."""

from api.routers import (
    analysis,
    auth,
    cascades,
    comparison,
    connection,
    incidents,
    ingestion,
    metrics,
    monitors,
    simulation,
    system,
)

__all__ = [
    "auth",
    "connection",
    "ingestion",
    "analysis",
    "incidents",
    "cascades",
    "monitors",
    "comparison",
    "simulation",
    "metrics",
    "system",
]
