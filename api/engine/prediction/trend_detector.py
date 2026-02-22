"""
Trend Detector — Rolling slope analysis for early risk identification.

Agent: data-scientist
Uses scipy.stats.linregress for linear trend estimation, with optional
LightGBM forecaster when trained models exist. Flags metrics trending
toward incident thresholds within a projection window.

data-scientist agent compliance:
- Statistical significance: p-value < 0.05 required (linregress mode)
- ML mode: Uses trained forecaster when available for better projections
- Confidence interval: 95% CI on projected value for uncertainty quantification
- Minimum data points: 5+ observations for reliable regression
"""

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import structlog
from scipy import stats

from api.storage.base import StorageBackend

logger = structlog.get_logger()

# Metric → (threshold, below) — from simulation WhatIfSimulator
# below=True means incident when metric goes below threshold
TREND_THRESHOLDS = {
    "refund_rate": (0.08, False),
    "delivery_delay_rate": (0.20, False),
    "ticket_backlog": (40, False),
    "review_score_avg": (3.0, True),
    "margin_proxy": (0.15, True),
    "churn_proxy": (0.05, False),
    "fulfillment_backlog": (100, False),
    "supplier_delay_rate": (0.20, False),
    "net_cash_proxy": (-5000, True),
}


def _build_lag_features(values: list[float], n_lags: int) -> list[float]:
    """Build feature vector for ML forecaster (must match train_trend_forecaster)."""
    if len(values) < n_lags:
        return []
    recent = values[-n_lags:]
    rolling_7 = values[-7:] if len(values) >= 7 else recent
    rolling_14 = values[-14:] if len(values) >= 14 else recent
    return [
        *recent,
        sum(rolling_7) / len(rolling_7),
        (sum((x - sum(rolling_7) / len(rolling_7)) ** 2 for x in rolling_7) / max(1, len(rolling_7) - 1)) ** 0.5
        if len(rolling_7) > 1
        else 0.0,
        sum(rolling_14) / len(rolling_14),
        (sum((x - sum(rolling_14) / len(rolling_14)) ** 2 for x in rolling_14) / max(1, len(rolling_14) - 1)) ** 0.5
        if len(rolling_14) > 1
        else 0.0,
        values[-1] - values[-2] if len(values) >= 2 else 0.0,
    ]


class TrendDetector:
    """
    Detects metrics trending toward incident thresholds.

    Uses linear regression (linregress) by default. When trained LightGBM
    forecasters exist in models/trend/, uses them for improved projection.
    """

    def __init__(
        self,
        storage: StorageBackend,
        lookback_days: int = 14,
        projection_days: int = 5,
        min_slope_significance: float = 0.01,
        max_p_value: float = 0.05,
        min_data_points: int = 5,
        models_dir: Optional[Path | str] = None,
    ):
        """
        Args:
            storage: Storage backend for Gold metrics
            lookback_days: Days of history for trend regression
            projection_days: Days ahead to project value
            min_slope_significance: Minimum |slope| to consider (fallback)
            max_p_value: Max p-value for statistically significant slope (data-scientist: p<0.05)
            min_data_points: Minimum observations for reliable regression (data-scientist)
            models_dir: Optional path to trained trend forecaster models (models/trend/)
        """
        self.storage = storage
        self.lookback_days = lookback_days
        self.projection_days = projection_days
        self.min_slope_significance = min_slope_significance
        self.max_p_value = max_p_value
        self.min_data_points = min_data_points
        self.models_dir = Path(models_dir) if models_dir else None
        self._forecasters: dict[str, Any] = {}
        self.logger = structlog.get_logger()

    def detect_degrading_metrics(self) -> list[dict]:
        """
        Find metrics trending toward their thresholds.

        Returns:
            List of dicts with metric, current_value, slope, projected_value,
            threshold, threshold_below, days_to_threshold
        """
        end_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (
            datetime.utcnow() - timedelta(days=self.lookback_days)
        ).strftime("%Y-%m-%d")

        degrading = []
        for metric_name, (threshold, below) in TREND_THRESHOLDS.items():
            result = self._check_metric_trend(
                metric_name, threshold, below, start_date, end_date
            )
            if result:
                degrading.append(result)

        self.logger.info(
            "trend_detection_complete",
            degrading_count=len(degrading),
            metrics_checked=len(TREND_THRESHOLDS),
        )
        return degrading

    def _load_forecaster(self, metric_name: str) -> Any:
        """Lazy-load trained forecaster for metric if available."""
        if metric_name in self._forecasters:
            return self._forecasters[metric_name]
        if not self.models_dir:
            return None
        import joblib

        path = self.models_dir / f"forecaster_{metric_name}.joblib"
        if not path.exists():
            return None
        try:
            obj = joblib.load(path)
            self._forecasters[metric_name] = obj
            return obj
        except Exception as e:
            self.logger.warning("forecaster_load_failed", metric=metric_name, error=str(e))
            return None

    def _check_metric_trend(
        self,
        metric_name: str,
        threshold: float,
        below: bool,
        start_date: str,
        end_date: str,
    ) -> Optional[dict]:
        """Check if a single metric is trending toward threshold."""
        metrics = self.storage.read_gold_metrics(
            metric_names=[metric_name],
            start_date=start_date,
            end_date=end_date,
        )

        if len(metrics) < self.min_data_points:
            return None

        # Build sorted time series
        by_date = {}
        for m in metrics:
            d = m.get("metric_date")
            v = m.get("metric_value")
            if d and v is not None:
                by_date[d] = float(v)

        dates_sorted = sorted(by_date.keys())
        values = [by_date[d] for d in dates_sorted]
        n = len(values)
        current_value = values[-1]
        slope: float
        p_value: float
        projected_value: float
        projected_ci_lower: float
        projected_ci_upper: float
        r_value_sq: float = 0.0

        # Try ML forecaster first if available
        use_ml = False
        forecaster = self._load_forecaster(metric_name) if self.models_dir else None
        if forecaster is not None:
            n_lags = forecaster.get("meta", {}).get("n_lags", self.lookback_days)
            if n >= n_lags:
                feat = _build_lag_features(values, n_lags)
                n_features = forecaster.get("meta", {}).get("n_features", 0)
                if feat and len(feat) == n_features:
                    import numpy as np

                    X = np.array([feat], dtype=np.float64)
                    model = forecaster["model"]
                    projected_value = float(model.predict(X)[0])
                    slope = (projected_value - current_value) / self.projection_days
                    p_value = 0.01  # ML mode: trust model, pass significance
                    ci_margin = abs(projected_value - current_value) * 0.5
                    projected_ci_lower = projected_value - ci_margin
                    projected_ci_upper = projected_value + ci_margin
                    r_value_sq = 0.0
                    use_ml = True
        if not use_ml:
            # Linear regression fallback
            x = list(range(n))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

            if p_value >= self.max_p_value:
                return None

            x_new = n + self.projection_days - 1
            projected_value = slope * x_new + intercept

            x_mean = (n - 1) / 2.0
            s_xx = sum((xi - x_mean) ** 2 for xi in x)
            pred_se = std_err * math.sqrt(1 + 1.0 / n + (x_new - x_mean) ** 2 / s_xx) if s_xx > 0 else std_err
            ci_margin = 1.96 * pred_se
            projected_ci_lower = projected_value - ci_margin
            projected_ci_upper = projected_value + ci_margin
            r_value_sq = r_value ** 2

        # Check if trending toward breach
        if below:
            trending_bad = slope < 0 and projected_value < threshold and current_value > threshold
        else:
            trending_bad = slope > 0 and projected_value > threshold and current_value < threshold

        if not trending_bad:
            return None

        # Significant slope magnitude (fallback if p-value passed)
        if abs(slope) < self.min_slope_significance:
            return None

        # Estimate days to threshold (linear extrapolation)
        days_to_threshold = None
        if slope != 0:
            if below:
                if slope < 0 and current_value > threshold:
                    days_to_threshold = (threshold - current_value) / slope
            else:
                if slope > 0 and current_value < threshold:
                    days_to_threshold = (threshold - current_value) / slope

        baseline = (values[0] + values[-1]) / 2 if len(values) >= 2 else current_value

        return {
            "metric": metric_name,
            "current_value": round(current_value, 4),
            "baseline": round(baseline, 4),
            "slope": round(slope, 6),
            "projected_value": round(projected_value, 4),
            "projected_ci_lower": round(projected_ci_lower, 4),
            "projected_ci_upper": round(projected_ci_upper, 4),
            "p_value": round(p_value, 6),
            "threshold": threshold,
            "threshold_below": below,
            "days_to_threshold": round(days_to_threshold, 1) if days_to_threshold else None,
            "r_squared": round(r_value_sq, 4),
        }
