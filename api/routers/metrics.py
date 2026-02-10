"""
Business metrics aggregation router.
"""

from typing import Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth.dependencies import get_current_realm_id
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class MetricValue(BaseModel):
    """Metric value at point in time."""

    timestamp: str
    value: float


class Metric(BaseModel):
    """Business metric."""

    metric_name: str
    current_value: float
    previous_value: float
    change_percent: float
    trend: str  # up, down, stable
    sparkline: List[float]


class MetricsResponse(BaseModel):
    """Metrics response."""

    realm_id: str
    period: str
    metrics: List[Metric]


@router.get("/dashboard", response_model=MetricsResponse)
async def get_dashboard_metrics(
    realm_id: str = Depends(get_current_realm_id),
    period: str = "30d",
):
    """
    Get dashboard metrics for overview.
    """
    logger.info("metrics_dashboard", realm_id=realm_id, period=period)

    # TODO: Aggregate from database
    return MetricsResponse(
        realm_id=realm_id,
        period=period,
        metrics=[
            Metric(
                metric_name="total_revenue",
                current_value=150000.0,
                previous_value=140000.0,
                change_percent=7.14,
                trend="up",
                sparkline=[130000, 135000, 140000, 145000, 150000],
            ),
            Metric(
                metric_name="invoice_count",
                current_value=450,
                previous_value=420,
                change_percent=7.14,
                trend="up",
                sparkline=[400, 410, 420, 435, 450],
            ),
        ],
    )


@router.get("/timeseries/{metric_name}")
async def get_metric_timeseries(
    metric_name: str,
    realm_id: str = Depends(get_current_realm_id),
    period: str = "30d",
    granularity: str = "daily",
):
    """
    Get time series data for specific metric.
    """
    logger.info(
        "metrics_timeseries",
        realm_id=realm_id,
        metric_name=metric_name,
        period=period,
        granularity=granularity,
    )

    # TODO: Query from database
    from datetime import datetime, timedelta

    now = datetime.utcnow()
    return {
        "metric_name": metric_name,
        "period": period,
        "granularity": granularity,
        "data": [
            {"timestamp": (now - timedelta(days=i)).isoformat(), "value": 100000 + i * 1000}
            for i in range(30, 0, -1)
        ],
    }


@router.get("/health-score")
async def get_health_score(
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Get composite health score.
    """
    logger.info("health_score", realm_id=realm_id)

    return {
        "overall_score": 85.5,
        "components": {
            "data_quality": 90.0,
            "anomaly_rate": 95.0,
            "slo_compliance": 98.0,
            "financial_health": 60.0,
        },
        "trend": "stable",
        "last_updated": "2026-02-10T12:00:00Z",
    }
