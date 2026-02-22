"""
Health monitors and SLO management router.

Wired to:
- StorageBackend for monitor CRUD
- SLOEvaluator for condition evaluation
- AlertRouter for alert management
- HealthScorer for health assessment
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth.dependencies import get_current_realm_id
from api.models.enums import Severity
from api.storage import get_storage
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class CreateMonitorRequest(BaseModel):
    """Create monitor request."""

    name: str = Field(..., min_length=3, description="Monitor name (min 3 chars)")
    description: str = Field(..., min_length=10, description="Description (min 10 chars)")
    metric_name: str = Field(..., min_length=2, description="Metric name (min 2 chars)")
    condition: str = Field(..., min_length=3, description="Condition expression (min 3 chars)")
    baseline_window_days: int = 30
    check_frequency: str = "daily"
    severity_if_triggered: str = "medium"
    alert_message_template: str = Field(..., min_length=10, description="Alert template (min 10 chars)")


@router.get("/")
async def list_monitors(
    realm_id: str = Depends(get_current_realm_id),
    enabled: Optional[bool] = None,
    page: int = 1,
    page_size: int = 25,
):
    """
    List all monitor rules with optional enabled filter and pagination.

    API-designer agent: proper pagination support.
    """
    page = max(1, page)
    page_size = max(1, min(100, page_size))

    logger.info("monitors_list", realm_id=realm_id, enabled=enabled, page=page)

    storage = get_storage()
    monitors = storage.read_monitors(enabled=enabled)

    total_count = len(monitors)
    total_pages = max(1, (total_count + page_size - 1) // page_size)
    offset = (page - 1) * page_size
    page_monitors = monitors[offset : offset + page_size]

    results = [m.model_dump(mode="json") for m in page_monitors]
    return {
        "success": True,
        "data": results,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        },
    }


@router.post("/")
async def create_monitor(
    request: CreateMonitorRequest,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Create a new monitor rule.
    """
    logger.info(
        "monitor_create",
        realm_id=realm_id,
        name=request.name,
        metric_name=request.metric_name,
    )

    from api.models.monitors import MonitorRule

    severity_map = {
        "low": Severity.LOW,
        "medium": Severity.MEDIUM,
        "high": Severity.HIGH,
        "critical": Severity.CRITICAL,
    }

    monitor = MonitorRule(
        name=request.name,
        description=request.description,
        source_incident_id="manual",
        metric_name=request.metric_name,
        condition=request.condition,
        baseline_window_days=request.baseline_window_days,
        check_frequency=request.check_frequency,
        severity_if_triggered=severity_map.get(
            request.severity_if_triggered.lower(), Severity.MEDIUM
        ),
        enabled=True,
        alert_message_template=request.alert_message_template,
    )

    storage = get_storage()
    monitor_id = storage.write_monitor(monitor)

    return {
        "success": True,
        "data": monitor.model_dump(mode="json"),
    }


@router.get("/evaluate")
async def evaluate_monitors(
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Trigger on-demand evaluation of all enabled monitors.
    Returns any triggered alerts.
    """
    logger.info("monitors_evaluate", realm_id=realm_id)

    from api.engine.monitors import AlertRouter, SLOEvaluator

    storage = get_storage()
    evaluator = SLOEvaluator(storage=storage)
    alerts = evaluator.evaluate_all_monitors()

    router_instance = AlertRouter(storage=storage)
    routed = router_instance.route_alerts(alerts)

    results = []
    for alert, routing in routed:
        results.append({
            "alert": alert.model_dump(mode="json"),
            "routing": routing,
        })

    return {
        "success": True,
        "data": results,
        "alerts_triggered": len([r for r in results if not r["routing"].get("suppressed")]),
    }


@router.get("/alerts")
async def list_alerts(
    realm_id: str = Depends(get_current_realm_id),
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 25,
):
    """
    List monitor alerts with optional status filter and pagination.

    API-designer agent: proper pagination support.
    """
    page = max(1, page)
    page_size = max(1, min(100, page_size))

    logger.info("alerts_list", realm_id=realm_id, status=status, page=page)

    storage = get_storage()
    alerts = storage.read_monitor_alerts(status=status)

    total_count = len(alerts)
    total_pages = max(1, (total_count + page_size - 1) // page_size)
    offset = (page - 1) * page_size
    page_alerts = alerts[offset : offset + page_size]

    results = [a.model_dump(mode="json") for a in page_alerts]
    return {
        "success": True,
        "data": results,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        },
    }


@router.get("/error-budget")
async def get_error_budget(
    realm_id: str = Depends(get_current_realm_id),
    slo_target: float = 99.5,
    window_days: int = 30,
):
    """
    Get SLO error budget status.

    SRE-engineer agent: error budget tracking, burn rate monitoring,
    and budget exhaustion projection to enable data-driven reliability
    decisions.

    Args:
        slo_target: SLO target percentage (default 99.5)
        window_days: Measurement window in days (default 30)
    """
    logger.info(
        "error_budget_fetch",
        realm_id=realm_id,
        slo_target=slo_target,
        window_days=window_days,
    )

    # Input validation per api-designer agent
    if not (90.0 <= slo_target <= 100.0):
        raise HTTPException(
            status_code=400,
            detail="slo_target must be between 90.0 and 100.0",
        )
    if not (1 <= window_days <= 365):
        raise HTTPException(
            status_code=400,
            detail="window_days must be between 1 and 365",
        )

    from api.engine.monitors import HealthScorer

    storage = get_storage()
    scorer = HealthScorer(storage=storage)
    error_budget = scorer.compute_error_budget(
        slo_target=slo_target,
        window_days=window_days,
    )

    return {"success": True, "data": error_budget}


@router.get("/{monitor_id}")
async def get_monitor(
    monitor_id: str,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Get monitor rule details.
    """
    logger.info("monitor_get", realm_id=realm_id, monitor_id=monitor_id)

    storage = get_storage()
    monitors = storage.read_monitors()
    monitor = next(
        (m for m in monitors if m.monitor_id == monitor_id), None
    )

    if monitor is None:
        raise HTTPException(status_code=404, detail=f"Monitor {monitor_id} not found")

    return {"success": True, "data": monitor.model_dump(mode="json")}


@router.put("/{monitor_id}/toggle")
async def toggle_monitor(
    monitor_id: str,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Toggle monitor enabled/disabled status.
    """
    logger.info("monitor_toggle", realm_id=realm_id, monitor_id=monitor_id)

    storage = get_storage()
    monitors = storage.read_monitors()
    monitor = next(
        (m for m in monitors if m.monitor_id == monitor_id), None
    )

    if monitor is None:
        raise HTTPException(status_code=404, detail=f"Monitor {monitor_id} not found")

    new_enabled = not monitor.enabled
    updated = storage.update_monitor(monitor_id, enabled=new_enabled)

    return {
        "success": True,
        "data": {"monitor_id": monitor_id, "enabled": new_enabled},
    }


@router.delete("/{monitor_id}")
async def delete_monitor(
    monitor_id: str,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Disable (soft-delete) a monitor rule.
    """
    logger.info("monitor_delete", realm_id=realm_id, monitor_id=monitor_id)

    storage = get_storage()
    updated = storage.update_monitor(monitor_id, enabled=False)

    if not updated:
        raise HTTPException(status_code=404, detail=f"Monitor {monitor_id} not found")

    return {"success": True, "message": f"Monitor {monitor_id} disabled"}
