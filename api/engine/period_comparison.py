"""
Period Comparison Engine — This Month vs Last, This Quarter vs Last.

Agents: data-engineer, backend-developer
Compares health score, key metrics, and incident count across periods.
Simple improvement/regression story for SMB dashboards.
"""

from datetime import date, datetime, timedelta
from typing import Optional

import structlog

from api.storage.base import StorageBackend

logger = structlog.get_logger()

KEY_METRICS = [
    "net_cash_proxy",
    "daily_revenue",
    "daily_expenses",
    "refund_rate",
    "margin_proxy",
    "expense_ratio",
    "dso_proxy",
    "order_volume",
    "delivery_delay_rate",
    "ticket_backlog",
    "review_score_avg",
    "churn_proxy",
]


class PeriodComparisonService:
    """
    Compares business metrics and health across time periods.

    Supports month-over-month and quarter-over-quarter comparisons.
    """

    def __init__(self, storage: StorageBackend):
        self.storage = storage
        self.logger = structlog.get_logger()

    def compare(
        self,
        period: str = "month",
        target_date: Optional[date] = None,
    ) -> dict:
        """
        Compare current period to previous period.

        Args:
            period: "month" or "quarter"
            target_date: Reference date (default: today)

        Returns:
            {
                "current_period": {"label": str, "start": str, "end": str},
                "previous_period": {"label": str, "start": str, "end": str},
                "health_score": {"current": float, "previous": float, "change_pct": float, "trend": str},
                "incidents": {"current": int, "previous": int, "change": int},
                "metrics": [{metric, current, previous, change_pct, direction}, ...],
                "summary": str,
            }
        """
        target = target_date or date.today()

        if period == "quarter":
            curr_start, curr_end = _quarter_range(target)
            q = (curr_start.month - 1) // 3
            prev_q = (q - 1) % 4
            prev_month = prev_q * 3 + 1
            prev_year = curr_start.year if q > 0 else curr_start.year - 1
            prev_start = date(prev_year, prev_month, 1)
            prev_end = curr_start - timedelta(days=1)
            curr_label = f"Q{(target.month - 1) // 3 + 1} {target.year}"
            prev_label = f"Q{(prev_start.month - 1) // 3 + 1} {prev_start.year}"
        else:
            curr_start = date(target.year, target.month, 1)
            curr_end = target
            prev_end = curr_start - timedelta(days=1)
            prev_start = date(prev_end.year, prev_end.month, 1)
            curr_label = target.strftime("%B %Y")
            prev_label = prev_end.strftime("%B %Y")

        # Fetch Gold metrics for both periods
        curr_metrics = self._fetch_metrics(curr_start, curr_end)
        prev_metrics = self._fetch_metrics(prev_start, prev_end)

        # Health score (simplified: average of normalized key metrics)
        curr_health = self._compute_simple_health(curr_metrics)
        prev_health = self._compute_simple_health(prev_metrics)
        health_change = (curr_health - prev_health) / prev_health * 100 if prev_health else 0
        health_trend = "up" if health_change > 1 else ("down" if health_change < -1 else "flat")

        # Incidents
        curr_incidents = self._count_incidents(curr_start, curr_end)
        prev_incidents = self._count_incidents(prev_start, prev_end)
        incident_change = curr_incidents - prev_incidents

        # Per-metric comparison
        metrics_comparison = []
        for m in KEY_METRICS:
            c = curr_metrics.get(m)
            p = prev_metrics.get(m)
            if c is not None and p is not None and p != 0:
                change_pct = (c - p) / abs(p) * 100
                direction = "higher_better" if m in ("daily_revenue", "margin_proxy", "order_volume", "review_score_avg", "net_cash_proxy") else "lower_better"
                improvement = (change_pct > 0 and direction == "higher_better") or (change_pct < 0 and direction == "lower_better")
                metrics_comparison.append({
                    "metric": m,
                    "current": round(c, 4),
                    "previous": round(p, 4),
                    "change_pct": round(change_pct, 1),
                    "improved": improvement,
                })
            elif c is not None or p is not None:
                metrics_comparison.append({
                    "metric": m,
                    "current": round(c, 4) if c is not None else None,
                    "previous": round(p, 4) if p is not None else None,
                    "change_pct": None,
                    "improved": None,
                })

        # Summary
        summary_parts = []
        if health_change > 1:
            summary_parts.append(f"Health score improved {health_change:.1f}% vs {prev_label}.")
        elif health_change < -1:
            summary_parts.append(f"Health score declined {abs(health_change):.1f}% vs {prev_label}.")
        if incident_change > 0:
            summary_parts.append(f"{incident_change} more incident(s) this period.")
        elif incident_change < 0:
            summary_parts.append(f"{abs(incident_change)} fewer incident(s) this period.")
        if not summary_parts:
            summary_parts.append("Metrics stable compared to previous period.")
        summary = " ".join(summary_parts)

        self.logger.info(
            "period_comparison_computed",
            period=period,
            curr_label=curr_label,
            health_change=health_change,
            incident_change=incident_change,
        )

        return {
            "current_period": {"label": curr_label, "start": curr_start.isoformat(), "end": curr_end.isoformat()},
            "previous_period": {"label": prev_label, "start": prev_start.isoformat(), "end": prev_end.isoformat()},
            "health_score": {
                "current": round(curr_health, 1),
                "previous": round(prev_health, 1),
                "change_pct": round(health_change, 1),
                "trend": health_trend,
            },
            "incidents": {
                "current": curr_incidents,
                "previous": prev_incidents,
                "change": incident_change,
            },
            "metrics": metrics_comparison,
            "summary": summary,
        }

    def _fetch_metrics(self, start: date, end: date) -> dict:
        rows = self.storage.read_gold_metrics(
            metric_names=KEY_METRICS,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )
        by_metric: dict[str, list[float]] = {}
        for r in rows:
            name = r.get("metric_name")
            val = r.get("metric_value")
            if name and val is not None:
                by_metric.setdefault(name, []).append(float(val))
        return {k: sum(v) / len(v) if v else None for k, v in by_metric.items()}

    def _compute_simple_health(self, metrics: dict) -> float:
        """Simple 0-100 health from key metrics (normalize each, average)."""
        if not metrics:
            return 50.0
        scores = []
        for m, v in metrics.items():
            if v is None:
                continue
            if m == "refund_rate":
                scores.append(max(0, 100 - v * 500))
            elif m in ("expense_ratio", "churn_proxy"):
                scores.append(max(0, 100 - v * 200))
            elif m == "dso_proxy":
                scores.append(max(0, 100 - v))
            elif m in ("delivery_delay_rate",):
                scores.append(max(0, 100 - v * 250))
            elif m == "ticket_backlog":
                scores.append(max(0, 100 - v / 2))
            elif m in ("margin_proxy", "review_score_avg"):
                scores.append(min(100, v * 25))
            elif m in ("daily_revenue", "net_cash_proxy"):
                scores.append(min(100, max(0, v / 500)))
            else:
                scores.append(50)
        return sum(scores) / len(scores) if scores else 50.0

    def _count_incidents(self, start: date, end: date) -> int:
        incidents = self.storage.read_incidents() or []
        count = 0
        for inc in incidents:
            det = getattr(inc, "detected_at", None)
            if det:
                d = det.date() if hasattr(det, "date") else det
                if start <= d <= end:
                    count += 1
        return count


def _quarter_range(d: date) -> tuple[date, date]:
    q = (d.month - 1) // 3
    start = date(d.year, q * 3 + 1, 1)
    if d.month in (1, 2, 3):
        end = date(d.year, 3, 31)
    elif d.month in (4, 5, 6):
        end = date(d.year, 6, 30)
    elif d.month in (7, 8, 9):
        end = date(d.year, 9, 30)
    else:
        end = date(d.year, 12, 31)
    return start, min(end, d)
