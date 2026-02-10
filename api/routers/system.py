"""
System health and diagnostics router.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from api.config import get_settings
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class SystemHealth(BaseModel):
    """System health status."""

    status: str
    version: str
    uptime_seconds: float
    database: str
    redis: str
    celery_workers: int


class DiagnosticsInfo(BaseModel):
    """System diagnostics information."""

    environment: str
    database_size_mb: float
    redis_memory_mb: float
    active_tasks: int
    cache_hit_rate: float


@router.get("/health", response_model=SystemHealth)
async def system_health():
    """
    Get system health status.
    """
    settings = get_settings()

    # TODO: Check actual service health
    return SystemHealth(
        status="healthy",
        version="0.1.0",
        uptime_seconds=3600.0,
        database="healthy",
        redis="healthy",
        celery_workers=4,
    )


@router.get("/diagnostics", response_model=DiagnosticsInfo)
async def system_diagnostics():
    """
    Get detailed system diagnostics.
    """
    settings = get_settings()

    logger.info("diagnostics_request")

    # TODO: Gather actual diagnostics
    return DiagnosticsInfo(
        environment=settings.intuit_env,
        database_size_mb=150.5,
        redis_memory_mb=45.2,
        active_tasks=3,
        cache_hit_rate=0.85,
    )


@router.get("/config")
async def get_system_config():
    """
    Get system configuration (non-sensitive).
    """
    settings = get_settings()

    return {
        "environment": settings.intuit_env,
        "log_level": settings.log_level,
        "db_type": settings.db_type,
        "anomaly_detection_sensitivity": settings.anomaly_detection_sensitivity,
        "min_confidence_threshold": settings.min_confidence_threshold,
        "blast_radius_max_depth": settings.blast_radius_max_depth,
        "feature_flags": {
            "ml_training": settings.enable_ml_training,
            "auto_remediation": settings.enable_auto_remediation,
            "realtime_analysis": settings.enable_realtime_analysis,
            "supplemental_data": settings.enable_supplemental_data,
        },
    }
