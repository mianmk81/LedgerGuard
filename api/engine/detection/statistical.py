"""
Statistical Anomaly Detection using MAD-based Z-score.

This module implements Layer 1 detection: robust statistical anomaly detection
using Median Absolute Deviation (MAD) for outlier-resistant threshold computation.
MAD provides stability against single spikes that would inflate standard deviation.

Detection Algorithm:
    1. Compute baseline statistics from trailing N days (default 30)
    2. Calculate Median and MAD of baseline values
    3. Compute modified z-score: z = 0.6745 * (current - median) / MAD
    4. Flag anomaly if |z| > threshold (default 3.0)

Why MAD over StdDev:
    - Robust to outliers (single spike won't inflate threshold)
    - More stable for business metrics with occasional spikes
    - 0.6745 constant normalizes MAD to match StdDev for normal distributions

This detector ALWAYS runs as the foundational detection layer.
"""

from datetime import datetime
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger()


class StatisticalDetector:
    """
    MAD-based z-score anomaly detector for business metrics.

    Implements robust statistical anomaly detection using Median Absolute Deviation
    to identify significant deviations from baseline behavior. Designed to handle
    business metrics with occasional spikes without inflating detection thresholds.

    Attributes:
        baseline_days: Number of trailing days to use for baseline computation
        zscore_threshold: Z-score threshold for anomaly detection (default: 3.0)
        mad_normalization_constant: Constant to normalize MAD to StdDev scale

    Example:
        >>> detector = StatisticalDetector(baseline_days=30, zscore_threshold=3.0)
        >>> result = detector.detect("refund_rate", 0.15, [0.03, 0.02, 0.04, ...])
        >>> if result["is_anomaly"]:
        ...     print(f"Anomaly detected with z-score: {result['zscore']}")
    """

    # MAD normalization constant: 1 / (75th percentile of standard normal dist)
    # Makes MAD comparable to standard deviation for Gaussian distributions
    MAD_NORMALIZATION = 0.6745

    def __init__(
        self,
        baseline_days: int = 30,
        zscore_threshold: float = 3.0,
    ):
        """
        Initialize the statistical detector.

        Args:
            baseline_days: Number of trailing days for baseline (default: 30)
            zscore_threshold: Absolute z-score threshold for anomaly (default: 3.0)
        """
        self.baseline_days = baseline_days
        self.zscore_threshold = zscore_threshold
        self.logger = structlog.get_logger()

    def detect(
        self,
        metric_name: str,
        current_value: float,
        baseline_values: list[float],
    ) -> dict:
        """
        Detect anomalies in a single metric using MAD-based z-score.

        Computes baseline statistics from historical values and determines if
        the current value represents a statistically significant deviation.

        Args:
            metric_name: Name of the metric being evaluated
            current_value: Current value to test for anomaly
            baseline_values: Historical values for baseline computation

        Returns:
            Dictionary containing detection results:
            {
                "metric_name": str,
                "current_value": float,
                "is_anomaly": bool,
                "zscore": float,
                "median": float,
                "mad": float,
                "threshold": float,
                "baseline_count": int
            }

        Example:
            >>> result = detector.detect("refund_rate", 0.15, [0.02, 0.03, 0.02])
            >>> print(f"Z-score: {result['zscore']:.2f}")
        """
        self.logger.debug(
            "running_statistical_detection",
            metric_name=metric_name,
            current_value=current_value,
            baseline_count=len(baseline_values),
        )

        # Handle edge cases
        if not baseline_values:
            self.logger.warning(
                "no_baseline_data",
                metric_name=metric_name,
            )
            return {
                "metric_name": metric_name,
                "current_value": current_value,
                "is_anomaly": False,
                "zscore": 0.0,
                "median": current_value,
                "mad": 0.0,
                "threshold": self.zscore_threshold,
                "baseline_count": 0,
            }

        # Convert to numpy array for efficient computation
        baseline_array = np.array(baseline_values, dtype=float)

        # Remove NaN values
        baseline_array = baseline_array[~np.isnan(baseline_array)]

        if len(baseline_array) == 0:
            self.logger.warning(
                "all_baseline_values_nan",
                metric_name=metric_name,
            )
            return {
                "metric_name": metric_name,
                "current_value": current_value,
                "is_anomaly": False,
                "zscore": 0.0,
                "median": current_value,
                "mad": 0.0,
                "threshold": self.zscore_threshold,
                "baseline_count": 0,
            }

        # Compute robust statistics
        median = float(np.median(baseline_array))

        # Median Absolute Deviation
        absolute_deviations = np.abs(baseline_array - median)
        mad = float(np.median(absolute_deviations))

        # Compute modified z-score
        if mad == 0.0:
            # All baseline values are identical - any deviation is anomalous
            if abs(current_value - median) > 1e-10:
                zscore = 10.0 if current_value > median else -10.0
            else:
                zscore = 0.0
        else:
            zscore = (self.MAD_NORMALIZATION * (current_value - median)) / mad

        # Determine if anomaly
        is_anomaly = abs(zscore) > self.zscore_threshold

        result = {
            "metric_name": metric_name,
            "current_value": round(current_value, 4),
            "is_anomaly": is_anomaly,
            "zscore": round(zscore, 4),
            "median": round(median, 4),
            "mad": round(mad, 4),
            "threshold": self.zscore_threshold,
            "baseline_count": len(baseline_array),
        }

        if is_anomaly:
            self.logger.info(
                "anomaly_detected",
                metric_name=metric_name,
                zscore=result["zscore"],
                current_value=result["current_value"],
                baseline_median=result["median"],
            )

        return result

    def detect_all(
        self,
        current_metrics: dict,
        historical_metrics: list[dict],
        baseline_days: Optional[int] = None,
    ) -> list[dict]:
        """
        Run detection across all metrics in current state.

        Processes multiple metrics simultaneously, extracting baseline values
        from historical data and running MAD-based detection on each.

        Args:
            current_metrics: Dictionary of current metric values by name
            historical_metrics: List of historical metric dictionaries
            baseline_days: Override default baseline window (optional)

        Returns:
            List of detection result dictionaries, one per metric

        Example:
            >>> current = {"refund_rate": 0.15, "churn_proxy": 0.08}
            >>> historical = [
            ...     {"refund_rate": 0.03, "churn_proxy": 0.02},
            ...     {"refund_rate": 0.02, "churn_proxy": 0.03},
            ... ]
            >>> results = detector.detect_all(current, historical)
            >>> anomalies = [r for r in results if r["is_anomaly"]]
        """
        baseline_window = baseline_days or self.baseline_days

        self.logger.info(
            "running_detection_on_all_metrics",
            metrics_count=len(current_metrics),
            historical_count=len(historical_metrics),
            baseline_days=baseline_window,
        )

        results = []

        # Limit historical data to baseline window
        historical_baseline = historical_metrics[-baseline_window:] if historical_metrics else []

        for metric_name, current_value in current_metrics.items():
            # Skip None values
            if current_value is None:
                continue

            # Extract baseline values for this metric
            baseline_values = []
            for hist_metrics in historical_baseline:
                if metric_name in hist_metrics and hist_metrics[metric_name] is not None:
                    baseline_values.append(hist_metrics[metric_name])

            # Run detection
            result = self.detect(metric_name, current_value, baseline_values)
            results.append(result)

        anomaly_count = sum(1 for r in results if r["is_anomaly"])

        self.logger.info(
            "detection_complete",
            total_metrics=len(results),
            anomalies_detected=anomaly_count,
        )

        return results

    def get_baseline_stats(self, baseline_values: list[float]) -> dict:
        """
        Compute baseline statistics for a metric.

        Utility method to extract baseline statistics without running
        anomaly detection. Useful for debugging and visualization.

        Args:
            baseline_values: List of historical metric values

        Returns:
            Dictionary containing baseline statistics:
            {
                "median": float,
                "mad": float,
                "mean": float,
                "std": float,
                "min": float,
                "max": float,
                "count": int
            }

        Example:
            >>> stats = detector.get_baseline_stats([0.02, 0.03, 0.04, 0.02])
            >>> print(f"Median: {stats['median']}, MAD: {stats['mad']}")
        """
        if not baseline_values:
            return {
                "median": 0.0,
                "mad": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }

        baseline_array = np.array(baseline_values, dtype=float)
        baseline_array = baseline_array[~np.isnan(baseline_array)]

        if len(baseline_array) == 0:
            return {
                "median": 0.0,
                "mad": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }

        median = float(np.median(baseline_array))
        absolute_deviations = np.abs(baseline_array - median)
        mad = float(np.median(absolute_deviations))

        return {
            "median": round(median, 4),
            "mad": round(mad, 4),
            "mean": round(float(np.mean(baseline_array)), 4),
            "std": round(float(np.std(baseline_array)), 4),
            "min": round(float(np.min(baseline_array)), 4),
            "max": round(float(np.max(baseline_array)), 4),
            "count": len(baseline_array),
        }
