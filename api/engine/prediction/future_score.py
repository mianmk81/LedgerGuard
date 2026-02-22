"""
Future Score Predictor — Data-scientist role.

Projects future health score from trend detection and warnings.
Returns causal chain, when it will hit, path, and prevention actions.
"""

from datetime import datetime, timedelta

import structlog

from api.engine.monitors import HealthScorer
from api.engine.prediction import WarningService
from api.engine.rca.causal_graph_builder import build_forward_chain_graph, build_full_causal_graph
from api.storage.base import StorageBackend

logger = structlog.get_logger()

# Grade order for projection (drop by severity)
GRADE_ORDER = ["A", "B+", "B", "B-", "C+", "C", "C-", "D", "F"]


def _grade_to_index(g: str) -> int:
    for i, gr in enumerate(GRADE_ORDER):
        if g and g.upper().startswith(gr[0]):
            return i
    return 2  # B default


def _index_to_grade(i: int) -> str:
    i = max(0, min(i, len(GRADE_ORDER) - 1))
    return GRADE_ORDER[i]


class FutureScorePredictor:
    """
    Predicts future Business Reliability Score from trends and warnings.

    data-scientist: Uses trend significance, projects with uncertainty.
    """

    def __init__(
        self,
        storage: StorageBackend,
        lookback_days: int = 14,
        projection_days: int = 30,
    ):
        self.storage = storage
        self.warning_service = WarningService(
            storage=storage,
            lookback_days=lookback_days,
            projection_days=min(projection_days, 30),
        )
        self.health_scorer = HealthScorer(storage=storage, lookback_days=7)

    def predict(self) -> dict:
        """
        Predict future score and return full explainability.

        Returns:
            {
                "current_score": float,
                "current_grade": str,
                "projected_score": float,
                "projected_grade": str,
                "projection_days": int,
                "why_summary": str,
                "causal_graph": dict,  # Cytoscape format
                "when_hit": str,       # e.g. "in 5 days"
                "path": list[str],     # metric chain
                "prevention_steps": str,
                "top_warning": dict | None,
                "warnings_count": int,
            }
        """
        health = self.health_scorer.compute_health()
        current_score = health.get("overall_score") or 0
        current_grade = health.get("overall_grade") or "B"
        warnings = self.warning_service.get_active_warnings()

        projected_score = current_score
        projected_grade = current_grade
        causal_graph = {"elements": {"nodes": [], "edges": []}}
        when_hit = None
        path = []
        prevention_steps = None
        why_summary = "No degrading trends detected. Your score is projected to remain stable."
        top_warning = None

        if not warnings:
            causal_graph = build_full_causal_graph()

        if warnings:
            w = warnings[0]
            top_warning = {
                "metric": w.metric,
                "current_value": w.current_value,
                "projected_value": w.projected_value,
                "threshold": w.threshold,
                "days_to_threshold": getattr(w, "days_to_threshold", None),
                "forward_chain": w.forward_chain,
                "incident_type": w.incident_type,
                "prevention_steps": w.prevention_steps,
                "severity": w.severity,
            }
            path = w.forward_chain or [w.metric]
            prevention_steps = w.prevention_steps

            days = getattr(w, "days_to_threshold", None)
            if days is not None:
                when_hit = f"in ~{int(days)} days" if days > 0 else "imminently"
            else:
                when_hit = f"within {self.warning_service.trend_detector.projection_days} days"

            grade_idx = _grade_to_index(current_grade)
            if w.severity == "critical":
                grade_idx = min(len(GRADE_ORDER) - 1, grade_idx + 2)
            elif w.severity == "high":
                grade_idx = min(len(GRADE_ORDER) - 1, grade_idx + 1)
            elif w.severity == "medium":
                grade_idx = min(len(GRADE_ORDER) - 1, grade_idx + 1)
            projected_grade = _index_to_grade(grade_idx)
            projected_score = max(0, current_score - (15 if w.severity == "critical" else 10))

            why_summary = (
                f"{w.metric.replace('_', ' ').title()} is trending toward threshold "
                f"(current: {w.current_value:.2f}, projected: {w.projected_value:.2f}). "
                f"This will propagate to {w.incident_type or 'incident'} if not addressed."
            )

            causal_graph = build_forward_chain_graph(path, w.incident_type)

        logger.info(
            "future_score_predicted",
            current_grade=current_grade,
            projected_grade=projected_grade,
            warnings_count=len(warnings),
        )

        return {
            "current_score": round(current_score, 1),
            "current_grade": current_grade,
            "projected_score": round(projected_score, 1),
            "projected_grade": projected_grade,
            "projection_days": self.warning_service.trend_detector.projection_days,
            "why_summary": why_summary,
            "causal_graph": causal_graph,
            "when_hit": when_hit,
            "path": path,
            "prevention_steps": prevention_steps,
            "top_warning": top_warning,
            "warnings_count": len(warnings),
        }
