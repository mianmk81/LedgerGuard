"""
Health monitors and SLO management router.
"""

from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth.dependencies import get_current_realm_id
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class MonitorType(str, Enum):
    """Monitor types."""

    SLO = "slo"
    THRESHOLD = "threshold"
    ANOMALY = "anomaly"
    COMPOSITE = "composite"


class MonitorStatus(str, Enum):
    """Monitor health status."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class Monitor(BaseModel):
    """Monitor configuration."""

    monitor_id: str
    name: str
    type: MonitorType
    status: MonitorStatus
    metric: str
    threshold: Optional[float] = None
    current_value: float
    compliance: float
    enabled: bool


class CreateMonitorRequest(BaseModel):
    """Create monitor request."""

    name: str
    type: MonitorType
    metric: str
    threshold: Optional[float] = None
    enabled: bool = True


@router.get("/", response_model=List[Monitor])
async def list_monitors(
    realm_id: str = Depends(get_current_realm_id),
):
    """
    List all monitors for realm.
    """
    logger.info("monitors_list", realm_id=realm_id)

    # TODO: Query from database
    return [
        Monitor(
            monitor_id="MON-001",
            name="Invoice Processing SLO",
            type=MonitorType.SLO,
            status=MonitorStatus.HEALTHY,
            metric="invoice_processing_time",
            threshold=86400.0,
            current_value=72000.0,
            compliance=0.98,
            enabled=True,
        )
    ]


@router.post("/", response_model=Monitor)
async def create_monitor(
    request: CreateMonitorRequest,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Create new monitor.
    """
    import uuid

    monitor_id = str(uuid.uuid4())

    logger.info(
        "monitor_create",
        realm_id=realm_id,
        monitor_id=monitor_id,
        name=request.name,
        type=request.type.value,
    )

    # TODO: Save to database
    return Monitor(
        monitor_id=monitor_id,
        name=request.name,
        type=request.type,
        status=MonitorStatus.UNKNOWN,
        metric=request.metric,
        threshold=request.threshold,
        current_value=0.0,
        compliance=1.0,
        enabled=request.enabled,
    )


@router.get("/{monitor_id}", response_model=Monitor)
async def get_monitor(
    monitor_id: str,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Get monitor details.
    """
    logger.info("monitor_get", realm_id=realm_id, monitor_id=monitor_id)

    # TODO: Query from database
    return Monitor(
        monitor_id=monitor_id,
        name="Invoice Processing SLO",
        type=MonitorType.SLO,
        status=MonitorStatus.HEALTHY,
        metric="invoice_processing_time",
        threshold=86400.0,
        current_value=72000.0,
        compliance=0.98,
        enabled=True,
    )


@router.delete("/{monitor_id}")
async def delete_monitor(
    monitor_id: str,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Delete monitor.
    """
    logger.info("monitor_delete", realm_id=realm_id, monitor_id=monitor_id)

    # TODO: Delete from database
    return {"success": True, "message": f"Monitor {monitor_id} deleted"}
