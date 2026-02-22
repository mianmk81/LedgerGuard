"""
SMB Credit Pulse — Financial Health Score + Explainability.

Agents: backend-developer, api-designer
RESTful API for Credit Karma-style SMB financial health: composite score,
letter grade, contributing factors, and "why this score" narrative.

Reuses HealthScorer for scoring; derives explainability from domain
breakdowns and lowest-scoring metrics.

api-designer agent: Input validation on lookback_days (1–90).
"""

from fastapi import APIRouter, Depends, Query

from api.auth.dependencies import get_current_realm_id
from api.engine.monitors import HealthScorer
from api.engine.rca.causal_graph_builder import build_full_causal_graph
from api.storage import get_storage
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# User-facing metric labels
METRIC_LABELS = {
    "margin_proxy": "Profit margin",
    "refund_rate": "Refund rate",
    "net_cash_proxy": "Net cash position",
    "expense_ratio": "Expense ratio",
    "dso_proxy": "Days sales outstanding",
    "delivery_delay_rate": "Delivery delay rate",
    "fulfillment_backlog": "Fulfillment backlog",
    "supplier_delay_rate": "Supplier delay rate",
    "order_volume": "Order volume",
    "avg_delivery_delay_days": "Avg delivery delay (days)",
    "review_score_avg": "Review score",
    "ticket_backlog": "Support ticket backlog",
    "churn_proxy": "Churn risk",
    "avg_resolution_time": "Avg resolution time (hrs)",
    "review_score_trend": "Review score trend",
}


def _build_contributing_factors(health: dict) -> list[dict]:
    """
    Derive contributing factors from domain metric scores.
    Sorts by impact (lowest score = highest contribution to poor health).
    """
    factors = []
    for domain_name, domain_data in health.get("domains", {}).items():
        metrics = domain_data.get("metrics", {})
        for metric_name, metric_data in metrics.items():
            if metric_data.get("value") is None:
                continue
            score = metric_data.get("score")
            if score is None:
                continue
            status = metric_data.get("status", "no_data")
            weight = metric_data.get("weight", 0)
            value = metric_data.get("value")

            # Contribution: lower score = higher negative impact
            impact = 1.0 - (score if score is not None else 0.5)
            contribution_pct = round(impact * weight * 100, 1)
            direction = "negative" if score < 0.6 else ("neutral" if score < 0.8 else "positive")

            factors.append({
                "metric": metric_name,
                "label": METRIC_LABELS.get(metric_name, metric_name.replace("_", " ").title()),
                "value": round(value, 4) if value is not None else None,
                "score": round(score * 100, 1) if score is not None else None,
                "status": status,
                "domain": domain_name,
                "contribution_pct": contribution_pct,
                "direction": direction,
            })

    # Sort by impact (highest negative contribution first)
    factors.sort(key=lambda x: (-x["contribution_pct"] if x["direction"] == "negative" else 0, x["score"] or 100))
    return factors[:10]  # Top 10 factors




def _build_why_summary(health: dict, factors: list[dict]) -> str:
    """Generate natural-language 'why' summary from top factors."""
    negatives = [f for f in factors if f["direction"] == "negative" and f["contribution_pct"] > 5]
    if not negatives:
        return "Your financial health is stable. Key metrics are within healthy ranges."
    parts = []
    for f in negatives[:3]:
        label = f["label"]
        status = f["status"]
        if status == "critical":
            parts.append(f"{label} is critical")
        elif status == "warning":
            parts.append(f"{label} is concerning")
        elif status == "degraded":
            parts.append(f"{label} is elevated")
    if len(parts) == 1:
        return f"{parts[0]}."
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}."
    return f"{parts[0]}, {parts[1]}, and {parts[2]} are pressuring your score."


@router.get("")
async def get_credit_pulse(
    realm_id: str = Depends(get_current_realm_id),
    lookback_days: int = Query(default=7, ge=1, le=90, description="Days of data for score (1-90)"),
):
    """
    Get SMB Credit Pulse — financial health score with explainability.

    Returns composite score (0-100), letter grade (A-F), contributing factors,
    "why this score" narrative, and domain breakdowns. Credit Karma for SMBs.
    """
    logger.info("credit_pulse_request", realm_id=realm_id, lookback_days=lookback_days)

    storage = get_storage()
    scorer = HealthScorer(storage=storage, lookback_days=lookback_days)
    health = scorer.compute_health()

    contributing_factors = _build_contributing_factors(health)
    why_summary = _build_why_summary(health, contributing_factors)

    # Optional: count recent incidents for context
    recent_incidents = storage.read_incidents()
    recent_count = len(recent_incidents) if recent_incidents else 0

    metric_statuses = {}
    for domain_data in health.get("domains", {}).values():
        for metric_name, metric_data in domain_data.get("metrics", {}).items():
            s = metric_data.get("status")
            if s:
                metric_statuses[metric_name] = s

    causal_graph = build_full_causal_graph(metric_statuses=metric_statuses)

    return {
        "success": True,
        "data": {
            "score": health.get("overall_score", 0),
            "grade": health.get("overall_grade", "C"),
            "contributing_factors": contributing_factors,
            "why_summary": why_summary,
            "domains": health.get("domains", {}),
            "evaluated_at": health.get("evaluated_at"),
            "lookback_days": lookback_days,
            "recent_incidents_count": recent_count,
            "trend": health.get("trend"),
            "causal_graph": causal_graph,
        },
    }
