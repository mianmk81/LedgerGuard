"""
Tax Liability Estimate (H7) - Fintech Engineer Role.

From P&L (Gold metrics) → Estimated tax liability + audit risk flag.
Implements standard effective rate assumptions and audit risk heuristics.
"""

from datetime import date, datetime, timedelta
from typing import Optional

import structlog

from api.storage.base import StorageBackend

logger = structlog.get_logger()


# Effective federal + state rate assumption (simplified SMB)
DEFAULT_EFFECTIVE_TAX_RATE = 0.25
# Thresholds for audit risk heuristics (fintech: regulatory/compliance awareness)
EXPENSE_RATIO_AUDIT_THRESHOLD = 0.95  # expenses > 95% of revenue
MARGIN_AUDIT_LOW = 0.05  # margin < 5% = higher scrutiny
REVENUE_SPIKE_THRESHOLD = 2.0  # current/prior > 2x = unusual


class TaxLiabilityEstimator:
    """
    Estimates tax liability from P&L and flags audit risk.

    Uses Gold metrics: daily_revenue, daily_expenses, margin_proxy, etc.
    Output: Estimated tax liability, effective rate, audit risk level.
    """

    def __init__(self, storage: StorageBackend, lookback_days: int = 365):
        self.storage = storage
        self.lookback_days = lookback_days

    def estimate(
        self,
        target_date: Optional[date] = None,
        effective_rate: Optional[float] = None,
    ) -> dict:
        """
        Estimate tax liability and audit risk.

        Args:
            target_date: Reference date
            effective_rate: Override effective tax rate (0-1)

        Returns:
            {
                "estimated_tax_liability": float,
                "taxable_income_proxy": float,
                "effective_rate": float,
                "audit_risk": "low"|"elevated"|"high",
                "audit_factors": [str],
                "period": {...},
            }
        """
        target = target_date or date.today()
        rate = effective_rate if effective_rate is not None else DEFAULT_EFFECTIVE_TAX_RATE
        rate = max(0, min(1, rate))
        end_str = target.isoformat()
        start_str = (target - timedelta(days=self.lookback_days)).isoformat()

        metrics = self.storage.read_gold_metrics(
            metric_names=[
                "daily_revenue",
                "daily_expenses",
                "daily_refunds",
                "margin_proxy",
                "expense_ratio",
            ],
            start_date=start_str,
            end_date=end_str,
        )

        # Aggregate by date
        by_date: dict[str, dict[str, float]] = {}
        for m in metrics:
            d = m.get("metric_date", "")
            if hasattr(d, "isoformat"):
                d = d.isoformat()[:10] if hasattr(d, "strftime") else str(d)[:10]
            else:
                d = str(d)[:10]
            if d not in by_date:
                by_date[d] = {}
            by_date[d][m["metric_name"]] = float(m.get("metric_value", 0))

        total_revenue = sum(r.get("daily_revenue", 0) for r in by_date.values())
        total_expenses = sum(r.get("daily_expenses", 0) for r in by_date.values())
        total_refunds = sum(r.get("daily_refunds", 0) for r in by_date.values())

        # Taxable income proxy = revenue - expenses - refunds
        taxable_income = total_revenue - total_expenses - total_refunds
        estimated_tax = taxable_income * rate if taxable_income > 0 else 0.0

        # Audit risk heuristics
        audit_factors = []
        days_with_data = len(by_date)
        if days_with_data < 7:
            audit_factors.append("Limited historical data")
        revenue_7d = 0.0
        expense_7d = 0.0
        sorted_dates = sorted(by_date.keys(), reverse=True)[:7]
        for d in sorted_dates:
            revenue_7d += by_date[d].get("daily_revenue", 0)
            expense_7d += by_date[d].get("daily_expenses", 0)
        if revenue_7d > 0:
            recent_exp_ratio = expense_7d / revenue_7d
            if recent_exp_ratio >= EXPENSE_RATIO_AUDIT_THRESHOLD:
                audit_factors.append("Very high expense ratio (expenses ≥ 95% of revenue)")
        # Margin
        margins = [r.get("margin_proxy", 0) for r in by_date.values() if r.get("margin_proxy") is not None]
        if margins:
            avg_margin = sum(margins) / len(margins)
            if avg_margin < MARGIN_AUDIT_LOW and total_revenue > 10000:
                audit_factors.append("Very low margin (< 5%) with material revenue")
        # Revenue spike
        if len(sorted_dates) >= 14:
            recent_rev = sum(by_date[d].get("daily_revenue", 0) for d in sorted_dates[:7])
            prior_rev = sum(by_date[d].get("daily_revenue", 0) for d in sorted_dates[7:14])
            if prior_rev > 0 and recent_rev / prior_rev >= REVENUE_SPIKE_THRESHOLD:
                audit_factors.append("Significant revenue increase vs prior period")

        if len(audit_factors) >= 2:
            audit_risk = "high"
        elif len(audit_factors) >= 1:
            audit_risk = "elevated"
        else:
            audit_risk = "low"

        logger.info(
            "tax_liability_estimated",
            estimated_tax=estimated_tax,
            taxable_income=taxable_income,
            audit_risk=audit_risk,
        )

        return {
            "estimated_tax_liability": round(estimated_tax, 2),
            "taxable_income_proxy": round(taxable_income, 2),
            "effective_rate": rate,
            "period_revenue": round(total_revenue, 2),
            "period_expenses": round(total_expenses, 2),
            "period_refunds": round(total_refunds, 2),
            "audit_risk": audit_risk,
            "audit_factors": audit_factors,
            "period": {
                "start_date": start_str,
                "end_date": end_str,
                "days_with_data": days_with_data,
            },
            "target_date": target.isoformat(),
        }
