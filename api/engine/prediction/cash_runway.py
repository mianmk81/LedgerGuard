"""
Cash Runway Prediction (H4) - Data Scientist Role.

Time series forecasting of "months until cash runs out" using Gold metrics.
Uses statistical methods: burn rate from trailing revenue/expenses,
95% confidence intervals per data-scientist methodology.
"""

from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import structlog
from scipy import stats

from api.storage.base import StorageBackend

logger = structlog.get_logger()


class CashRunwayPredictor:
    """
    Computes cash runway: "X months until cash runs out".

    Uses Gold layer metrics (net_cash_proxy, daily_revenue, daily_expenses)
    with trailing-window burn rate and statistical confidence intervals.
    Methodology aligns with data-scientist standards: p<0.05, 95% CI.
    """

    def __init__(self, storage: StorageBackend, lookback_days: int = 60):
        self.storage = storage
        self.lookback_days = lookback_days

    def predict(
        self,
        target_date: Optional[date] = None,
        metric_names: Optional[list[str]] = None,
    ) -> dict:
        """
        Predict months until cash runs out.

        Args:
            target_date: Reference date (default: today)
            metric_names: Override metric names to read (for testing)

        Returns:
            {
                "runway_months": float | None,  # None if positive burn
                "runway_months_ci_lower": float | None,
                "runway_months_ci_upper": float | None,
                "net_cash_proxy": float,
                "monthly_burn_rate": float,  # negative = burn
                "methodology": str,
            }
        """
        target = target_date or date.today()
        end_str = target.isoformat()
        start_str = (target - timedelta(days=self.lookback_days)).isoformat()

        names = metric_names or [
            "net_cash_proxy",
            "daily_revenue",
            "daily_expenses",
            "ar_aging_amount",
            "dpo_proxy",
        ]
        metrics = self.storage.read_gold_metrics(
            metric_names=names,
            start_date=start_str,
            end_date=end_str,
        )

        # Build timeseries by date
        by_date: dict[str, dict[str, float]] = {}
        for m in metrics:
            d = m.get("metric_date", "")
            if isinstance(d, datetime):
                d = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
            if d not in by_date:
                by_date[d] = {}
            by_date[d][m["metric_name"]] = float(m.get("metric_value", 0))

        # Latest net cash (most recent date)
        sorted_dates = sorted(by_date.keys(), reverse=True)
        net_cash = 0.0
        if sorted_dates:
            latest = by_date[sorted_dates[0]]
            net_cash = latest.get("net_cash_proxy", 0.0)

        # Monthly burn rate = avg(daily_expenses - daily_revenue) * 30
        daily_net_list = []
        for d in sorted(by_date.keys()):
            row = by_date[d]
            rev = row.get("daily_revenue", 0.0)
            exp = row.get("daily_expenses", 0.0)
            daily_net_list.append(exp - rev)

        if not daily_net_list:
            return {
                "runway_months": None,
                "runway_months_ci_lower": None,
                "runway_months_ci_upper": None,
                "net_cash_proxy": net_cash,
                "monthly_burn_rate": 0.0,
                "data_points": 0,
                "methodology": "insufficient_data",
                "message": "No Gold metrics available for runway calculation.",
            }

        arr = np.array(daily_net_list)
        mean_burn = float(np.mean(arr))
        std_burn = float(np.std(arr)) if len(arr) > 1 else 0.0
        n = len(arr)

        # Monthly burn (positive = cash outflow)
        monthly_burn = mean_burn * 30
        # 95% CI for mean daily burn (data-scientist: p<0.05)
        if n >= 2 and std_burn > 0:
            se = std_burn / (n ** 0.5)
            t_val = stats.t.ppf(0.975, df=n - 1)
            ci_low = (mean_burn - t_val * se) * 30
            ci_high = (mean_burn + t_val * se) * 30
        else:
            ci_low = ci_high = monthly_burn

        runway_months = None
        ci_lower = None
        ci_upper = None

        if monthly_burn > 0 and net_cash > 0:
            runway_months = net_cash / monthly_burn
            if ci_high > 0:
                ci_lower = net_cash / ci_high
            if ci_low > 0:
                ci_upper = net_cash / ci_low
        elif monthly_burn <= 0:
            runway_months = float("inf")  # Positive cash flow
            ci_lower = float("inf")
            ci_upper = float("inf")

        logger.info(
            "cash_runway_computed",
            runway_months=runway_months,
            net_cash=net_cash,
            monthly_burn=monthly_burn,
            data_points=n,
        )

        return {
            "runway_months": runway_months if runway_months != float("inf") else None,
            "runway_months_ci_lower": ci_lower if ci_lower != float("inf") else None,
            "runway_months_ci_upper": ci_upper if ci_upper != float("inf") else None,
            "runway_infinite": monthly_burn <= 0,
            "net_cash_proxy": round(net_cash, 2),
            "monthly_burn_rate": round(monthly_burn, 2),
            "data_points": n,
            "methodology": "trailing_avg_95ci",
            "target_date": target.isoformat(),
        }

    def forecast_curve(
        self,
        projection_months: int = 6,
        target_date: Optional[date] = None,
    ) -> dict:
        """
        Cash flow forecast curve over next N months.

        Data-scientist: Projects monthly cash balance with best/worst case bands
        using 95% CI on burn rate. Returns time series for charting.

        Returns:
            {
                "curve": [{"month": str, "cash_baseline": float, "cash_best": float, "cash_worst": float}, ...],
                "net_cash_proxy": float,
                "monthly_burn_rate": float,
                "projection_months": int,
            }
        """
        pred = self.predict(target_date=target_date)
        net_cash = pred.get("net_cash_proxy", 0.0)
        monthly_burn = pred.get("monthly_burn_rate", 0.0)

        # Best = lower burn (faster collections, slower expenses); worst = higher burn
        ci_low = pred.get("monthly_burn_rate")  # conservative placeholder
        ci_high = pred.get("monthly_burn_rate")
        if pred.get("runway_months_ci_lower") is not None and pred.get("runway_months_ci_upper") is not None:
            rl, ru = pred["runway_months_ci_lower"], pred["runway_months_ci_upper"]
            if rl and ru and net_cash > 0:
                ci_high = net_cash / rl if rl > 0 else monthly_burn * 1.2
                ci_low = net_cash / ru if ru > 0 else monthly_burn * 0.8
        else:
            ci_low = monthly_burn * 0.8
            ci_high = monthly_burn * 1.2

        target = target_date or date.today()
        curve = []
        for i in range(projection_months + 1):
            m = target.month + i
            y = target.year
            while m > 12:
                m -= 12
                y += 1
            month_str = f"{y}-{m:02d}-01"
            months_elapsed = i
            cash_baseline = net_cash - monthly_burn * months_elapsed
            cash_best = net_cash - ci_low * months_elapsed
            cash_worst = net_cash - ci_high * months_elapsed
            curve.append({
                "month": month_str,
                "cash_baseline": round(max(0, cash_baseline), 2),
                "cash_best": round(max(0, cash_best), 2),
                "cash_worst": round(max(0, cash_worst), 2),
            })

        logger.info(
            "cash_forecast_curve_computed",
            projection_months=projection_months,
            net_cash=net_cash,
            data_points=len(curve),
        )

        return {
            "curve": curve,
            "net_cash_proxy": net_cash,
            "monthly_burn_rate": monthly_burn,
            "projection_months": projection_months,
            "target_date": target.isoformat(),
        }
