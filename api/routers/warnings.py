"""
Risk Outlook / Early Warnings router.

Agents: backend-developer, api-designer
GET /api/v1/warnings — Returns active early warnings from trend + forward chain.

api-designer agent: Input validation on optional lookback_days, projection_days.
"""

from fastapi import APIRouter, Depends, Query

from api.auth.dependencies import get_current_realm_id
from api.engine.prediction import WarningService
from api.storage import get_storage
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


INCIDENT_LABELS = {
    "REFUND_SPIKE": "Refund Spike",
    "FULFILLMENT_SLA_DEGRADATION": "Fulfillment Delay",
    "SUPPORT_LOAD_SURGE": "Support Overload",
    "CHURN_ACCELERATION": "Customer Churn Acceleration",
    "MARGIN_COMPRESSION": "Margin Compression",
    "LIQUIDITY_CRUNCH_RISK": "Liquidity Crunch",
    "SUPPLIER_DEPENDENCY_FAILURE": "Supplier Failure",
    "CUSTOMER_SATISFACTION_REGRESSION": "Customer Satisfaction Drop",
}

METRIC_LABELS = {
    "refund_rate": "refund rate",
    "delivery_delay_rate": "delivery delays",
    "ticket_backlog": "support ticket backlog",
    "review_score_avg": "customer review scores",
    "margin_proxy": "profit margin",
    "churn_proxy": "customer churn",
    "fulfillment_backlog": "fulfillment backlog",
    "supplier_delay_rate": "supplier delays",
    "net_cash_proxy": "net cash position",
}


def _build_prediction_summary(w) -> str:
    """Build a plain-English prediction sentence for a warning."""
    incident_label = INCIDENT_LABELS.get(
        (w.incident_type or "").upper(),
        (w.incident_type or "issue").replace("_", " ").title(),
    )
    metric_label = METRIC_LABELS.get(w.metric, w.metric.replace("_", " "))

    days_text = ""
    if w.days_to_threshold is not None:
        days_int = int(round(w.days_to_threshold))
        if days_int <= 1:
            days_text = "within the next day"
        else:
            days_text = f"in about {days_int} days"
    else:
        days_text = f"within the next {w.projection_days} days"

    direction = "dropping" if w.threshold_below else "rising"

    chain_parts = [s.replace("_", " ") for s in (w.forward_chain or []) if s != w.incident_type]
    chain_text = ""
    if len(chain_parts) > 1:
        chain_text = f" The chain is: {' → '.join(chain_parts)}."

    return (
        f"We predict a {incident_label} incident {days_text}. "
        f"The root cause is {metric_label} {direction} toward a critical threshold "
        f"(currently {w.current_value:.2f}, threshold {w.threshold}).{chain_text}"
    )


@router.get("")
async def get_warnings(
    realm_id: str = Depends(get_current_realm_id),
    lookback_days: int = Query(default=14, ge=5, le=90, description="Days of metrics for trend (5-90)"),
    projection_days: int = Query(default=5, ge=1, le=30, description="Days ahead for projection (1-30)"),
):
    """
    Get active early warnings (Risk Outlook).

    Returns list of metrics trending toward incident thresholds, with
    forward causal chains, prediction summaries, and prevention recommendations.
    """
    logger.info("warnings_request", realm_id=realm_id, lookback_days=lookback_days, projection_days=projection_days)

    storage = get_storage()
    service = WarningService(storage=storage, lookback_days=lookback_days, projection_days=projection_days)
    warnings = service.get_active_warnings()

    data = [
        {
            "warning_id": w.warning_id,
            "metric": w.metric,
            "current_value": w.current_value,
            "baseline": w.baseline,
            "slope": w.slope,
            "projected_value": w.projected_value,
            "projected_ci_lower": w.projected_ci_lower,
            "projected_ci_upper": w.projected_ci_upper,
            "p_value": w.p_value,
            "projection_days": w.projection_days,
            "threshold": w.threshold,
            "threshold_below": w.threshold_below,
            "forward_chain": w.forward_chain,
            "incident_type": w.incident_type,
            "incident_label": INCIDENT_LABELS.get(
                (w.incident_type or "").upper(),
                (w.incident_type or "").replace("_", " ").title(),
            ),
            "root_cause_metric": w.metric,
            "root_cause_label": METRIC_LABELS.get(w.metric, w.metric.replace("_", " ")),
            "prediction_summary": _build_prediction_summary(w),
            "prevention_steps": w.prevention_steps,
            "severity": w.severity,
            "days_to_threshold": w.days_to_threshold,
            "created_at": w.created_at.isoformat(),
        }
        for w in warnings
    ]

    return {
        "success": True,
        "data": {
            "warnings": data,
            "count": len(data),
        },
    }
