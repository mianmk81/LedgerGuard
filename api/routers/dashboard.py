"""
Dashboard router — Reports and Future Score for Health Dashboard.

Agents: backend-developer, api-designer, data-scientist (FutureScorePredictor)
"""

from fastapi import APIRouter, Depends, Query

from api.auth.dependencies import get_current_realm_id
from api.engine.monitors import HealthScorer
from api.engine.prediction.future_score import FutureScorePredictor
from api.engine.rca.causal_graph_builder import (
    build_full_causal_graph,
    build_incident_causal_graph,
)
from api.storage import get_storage
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/reports")
async def get_dashboard_reports(
    realm_id: str = Depends(get_current_realm_id),
    limit: int = Query(5, ge=1, le=20),
):
    """
    Reports card data: recent incidents, current score, historical causal graph.

    Returns incidents with score, and 3D causal graph (what caused, how, where, when).
    """
    logger.info("dashboard_reports", realm_id=realm_id)

    storage = get_storage()
    incidents = storage.read_incidents()
    incidents = sorted(
        incidents,
        key=lambda i: i.detected_at.isoformat() if i.detected_at else "",
        reverse=True,
    )[:limit]

    scorer = HealthScorer(storage=storage, lookback_days=7)
    health = scorer.compute_health()
    score = {
        "overall_score": health.get("overall_score"),
        "overall_grade": health.get("overall_grade"),
        "trend": health.get("trend"),
    }

    metric_statuses = {}
    for domain_data in health.get("domains", {}).values():
        for metric_name, metric_data in domain_data.get("metrics", {}).items():
            s = metric_data.get("status")
            if s:
                metric_statuses[metric_name] = s

    causal_graph = build_full_causal_graph(metric_statuses=metric_statuses)
    focal_incident = None
    if incidents:
        focal_incident = incidents[0]
        ig = build_incident_causal_graph(focal_incident, metric_statuses=metric_statuses)
        if ig and ig.get("elements", {}).get("nodes"):
            causal_graph = ig

    incidents_data = []
    for inc in incidents:
        incidents_data.append({
            "incident_id": inc.incident_id,
            "incident_type": inc.incident_type.value,
            "severity": inc.severity.value,
            "detected_at": inc.detected_at.isoformat() if inc.detected_at else None,
            "primary_metric": inc.primary_metric,
            "primary_metric_zscore": inc.primary_metric_zscore,
        })

    return {
        "success": True,
        "data": {
            "incidents": incidents_data,
            "score": score,
            "causal_graph": causal_graph,
            "focal_incident_id": focal_incident.incident_id if focal_incident else None,
        },
    }


@router.get("/future-score")
async def get_dashboard_future_score(
    realm_id: str = Depends(get_current_realm_id),
    projection_days: int = Query(30, ge=5, le=60),
):
    """
    Future Score card: predicted rating, causal graph, when, path, actions.

    data-scientist: Uses trend significance, projects with uncertainty.
    """
    logger.info("dashboard_future_score", realm_id=realm_id, projection_days=projection_days)

    storage = get_storage()
    predictor = FutureScorePredictor(
        storage=storage,
        lookback_days=14,
        projection_days=projection_days,
    )
    result = predictor.predict()

    return {"success": True, "data": result}
