"""
Recurring Incident Patterns — Seasonality Detection.

Agents: data-scientist
Surfaces patterns like "Refund spikes tend to recur every quarter",
"Delivery delays often occur in these weeks".
Uses incident history and simple seasonal decomposition.
"""

from calendar import month_abbr
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Optional

import structlog

from api.storage.base import StorageBackend

logger = structlog.get_logger()


class RecurringPatternsService:
    """
    Detects recurring incident patterns from historical data.

    Identifies seasonality by month, quarter, and day-of-week.
    """

    def __init__(self, storage: StorageBackend, lookback_days: int = 365):
        self.storage = storage
        self.lookback_days = lookback_days
        self.logger = structlog.get_logger()

    def get_patterns(
        self,
        target_date: Optional[date] = None,
    ) -> dict:
        """
        Analyze incident history for recurring patterns.

        Returns:
            {
                "patterns": [{
                    "incident_type": str,
                    "pattern_description": str,
                    "confidence": str,
                    "examples": [{"month": str, "count": int}, ...],
                    "recommendation": str,
                }],
                "summary": str,
            }
        """
        target = target_date or date.today()
        start = target - timedelta(days=self.lookback_days)

        incidents = self.storage.read_incidents() or []
        in_range = []
        for inc in incidents:
            det = getattr(inc, "detected_at", None)
            if not det:
                continue
            d = det.date() if hasattr(det, "date") else det
            if start <= d <= target:
                in_range.append(inc)

        if len(in_range) < 2:
            return {
                "patterns": [],
                "summary": "Not enough incident history to detect patterns. Run more analyses.",
            }

        # Group by incident type
        by_type: dict[str, list[datetime]] = {}
        for inc in in_range:
            t = inc.incident_type.value if hasattr(inc.incident_type, "value") else str(inc.incident_type)
            det = inc.detected_at
            by_type.setdefault(t, []).append(det)

        patterns = []
        for inc_type, dates in by_type.items():
            if len(dates) < 2:
                continue

            # By month
            by_month = defaultdict(int)
            by_quarter = defaultdict(int)
            for d in dates:
                dt = d if hasattr(d, "month") else datetime.fromisoformat(str(d))
                by_month[dt.month] += 1
                q = (dt.month - 1) // 3 + 1
                by_quarter[f"Q{q}"] += 1

            # Find dominant month/quarter
            sorted_months = sorted(by_month.items(), key=lambda x: -x[1])
            sorted_quarters = sorted(by_quarter.items(), key=lambda x: -x[1])

            pattern_desc = None
            confidence = "low"
            examples = []

            if sorted_months and sorted_months[0][1] >= 2:
                m1, c1 = sorted_months[0]
                total = sum(by_month.values())
                pct = c1 / total * 100
                if pct >= 40:
                    pattern_desc = f"{inc_type.replace('_', ' ').title()} tends to occur in {month_abbr[m1]}"
                    confidence = "high" if pct >= 60 else "medium"
                    examples = [{"period": month_abbr[m], "count": c} for m, c in sorted_months[:4]]

            if not pattern_desc and sorted_quarters and sorted_quarters[0][1] >= 2:
                q1, c1 = sorted_quarters[0]
                total = sum(by_quarter.values())
                pct = c1 / total * 100
                if pct >= 40:
                    pattern_desc = f"{inc_type.replace('_', ' ').title()} tends to recur in {q1}"
                    confidence = "high" if pct >= 60 else "medium"
                    examples = [{"period": q, "count": c} for q, c in sorted_quarters]

            if pattern_desc:
                rec = _get_recommendation(inc_type)
                patterns.append({
                    "incident_type": inc_type,
                    "pattern_description": pattern_desc,
                    "confidence": confidence,
                    "examples": examples,
                    "recommendation": rec,
                    "occurrence_count": len(dates),
                })

        summary = (
            f"Found {len(patterns)} recurring pattern(s) across {len(in_range)} incidents. "
            + (f"Most common: {patterns[0]['pattern_description']}." if patterns else "")
        )

        self.logger.info(
            "recurring_patterns_computed",
            incidents_analyzed=len(in_range),
            patterns_found=len(patterns),
        )

        return {
            "patterns": patterns,
            "summary": summary,
            "incidents_analyzed": len(in_range),
        }


def _get_recommendation(incident_type: str) -> str:
    """Domain-specific recommendation for recurring incident type."""
    recs = {
        "refund_spike": "Review refund drivers before peak months. Set up early-warning monitors.",
        "fulfillment_sla_degradation": "Pre-staff or pre-inventory ahead of high-volume periods.",
        "support_load_surge": "Scale support capacity seasonally. Add chatbots or FAQ.",
        "churn_acceleration": "Launch retention campaigns before typical churn windows.",
        "margin_compression": "Review pricing and costs ahead of seasonal pressure.",
        "liquidity_crunch_risk": "Secure credit line or accelerate collections before tight periods.",
        "supplier_dependency_failure": "Diversify suppliers or build buffer stock before peak demand.",
        "customer_satisfaction_regression": "Proactively survey customers before known weak periods.",
    }
    return recs.get(incident_type, "Monitor this incident type and prepare mitigation before expected recurrence.")
