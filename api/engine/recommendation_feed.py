"""
Recommendation Feed Engine — Top Actions to Improve Score.

Agents: backend-developer, react-specialist (consumption)
Derives actionable recommendations from:
1. Credit Pulse contributing factors (weak metrics)
2. Recent postmortem remediation actions
3. Domain-specific action templates

Returns prioritized checklist: "Review AR aging", "Contact top 5 at-risk customers", etc.
"""

from typing import Optional

import structlog

from api.storage.base import StorageBackend

logger = structlog.get_logger()

# Metric → actionable recommendation templates
METRIC_RECOMMENDATIONS = {
    "margin_proxy": "Review profit margin drivers — identify high-cost products or expenses",
    "refund_rate": "Investigate refund root causes — review recent refund incidents",
    "net_cash_proxy": "Improve cash flow — prioritize collections and expense timing",
    "expense_ratio": "Audit expense categories — find areas to trim without impact",
    "dso_proxy": "Review accounts receivable aging report — follow up on overdue invoices",
    "delivery_delay_rate": "Address fulfillment bottlenecks — check supplier and logistics",
    "fulfillment_backlog": "Reduce order backlog — increase capacity or prioritize orders",
    "supplier_delay_rate": "Engage with delayed suppliers — consider backup vendors",
    "order_volume": "Review order trends — investigate volume decline if unexpected",
    "avg_delivery_delay_days": "Improve delivery SLA — trace delay sources",
    "review_score_avg": "Address customer satisfaction — review negative feedback",
    "ticket_backlog": "Tackle support ticket backlog — increase resolution capacity",
    "churn_proxy": "Contact at-risk customers — review churn patterns",
    "avg_resolution_time": "Improve support resolution time — streamline processes",
    "review_score_trend": "Stabilize review scores — identify declining drivers",
}

# Blast radius → recommendation
BLAST_RADIUS_RECOMMENDATIONS = {
    "revenue_exposure": "Contact customers with high revenue at risk — prioritize collections",
    "refund_exposure": "Review invoices at refund risk — address quality issues",
    "churn_exposure": "Reach out to churn-risk customers — offer retention incentives",
}


class RecommendationFeedService:
    """
    Builds a prioritized recommendation feed from health factors and incidents.

    Combines weak metrics from Credit Pulse, postmortem actions, and blast radius
    signals into a short actionable checklist.
    """

    def __init__(
        self,
        storage: StorageBackend,
        lookback_days: int = 30,
        top_n: int = 5,
    ):
        self.storage = storage
        self.lookback_days = lookback_days
        self.top_n = top_n
        self.logger = structlog.get_logger()

    def get_recommendations(
        self,
        realm_id: Optional[str] = None,
    ) -> dict:
        """
        Get top N actionable recommendations to improve business health.

        Returns:
            {
                "recommendations": [{
                    "action": str,
                    "source": "metric" | "postmortem" | "blast_radius",
                    "priority": int,
                    "metric"?: str,
                    "incident_id"?: str,
                }],
                "summary": {...}
            }
        """
        from api.engine.monitors import HealthScorer

        recommendations = []

        # 1. From Credit Pulse contributing factors
        scorer = HealthScorer(storage=self.storage, lookback_days=7)
        health = scorer.compute_health()

        for domain_name, domain_data in health.get("domains", {}).items():
            metrics = domain_data.get("metrics", {})
            for metric_name, metric_data in metrics.items():
                if metric_data.get("status") in ("critical", "warning") and metric_data.get("value") is not None:
                    action = METRIC_RECOMMENDATIONS.get(
                        metric_name,
                        f"Address {metric_name.replace('_', ' ').title()} — metric is {metric_data.get('status')}",
                    )
                    recommendations.append({
                        "action": action,
                        "source": "metric",
                        "priority": 1 if metric_data.get("status") == "critical" else 2,
                        "metric": metric_name,
                        "status": metric_data.get("status"),
                    })

        # 2. From recent postmortems
        incidents = self.storage.read_incidents() or []
        for inc in incidents[:10]:
            pm = self.storage.read_postmortem(inc.incident_id)
            if pm and pm.recommendations:
                for action in pm.recommendations[:2]:
                    rec = action if isinstance(action, str) else str(action)
                    recommendations.append({
                        "action": rec,
                        "source": "postmortem",
                        "priority": 2,
                        "incident_id": inc.incident_id,
                        "incident_type": inc.incident_type.value if hasattr(inc.incident_type, "value") else str(inc.incident_type),
                    })

        # 3. From blast radius (high exposure)
        for inc in incidents[:5]:
            blast = self.storage.read_blast_radius(inc.incident_id)
            if blast:
                rev = getattr(blast, "estimated_revenue_exposure", 0) or 0
                churn = getattr(blast, "estimated_churn_exposure", 0) or 0
                if rev > 5000:
                    recommendations.append({
                        "action": "Contact customers with high revenue at risk — prioritize collections",
                        "source": "blast_radius",
                        "priority": 1,
                        "incident_id": inc.incident_id,
                        "revenue_at_risk": rev,
                    })
                elif churn > 3:
                    recommendations.append({
                        "action": "Reach out to churn-risk customers — offer retention incentives",
                        "source": "blast_radius",
                        "priority": 2,
                        "incident_id": inc.incident_id,
                    })

        # Deduplicate by action (keep highest priority)
        seen = set()
        unique = []
        for r in sorted(recommendations, key=lambda x: (x["priority"], x.get("action", ""))):
            key = (r["action"][:50], r["source"])
            if key not in seen:
                seen.add(key)
                unique.append(r)

        top = unique[: self.top_n]

        self.logger.info(
            "recommendation_feed_computed",
            total=len(recommendations),
            unique=len(unique),
            returned=len(top),
        )

        return {
            "recommendations": top,
            "summary": {
                "total_suggestions": len(recommendations),
                "returned": len(top),
                "sources": {
                    "metric": sum(1 for r in top if r["source"] == "metric"),
                    "postmortem": sum(1 for r in top if r["source"] == "postmortem"),
                    "blast_radius": sum(1 for r in top if r["source"] == "blast_radius"),
                },
            },
        }
