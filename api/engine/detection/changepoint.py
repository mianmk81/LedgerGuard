"""
Changepoint Detection using Ruptures Library (PELT Algorithm).

This module implements Layer 3 detection: changepoint detection for regime changes
in time series metrics. Detects structural breaks and mean shifts rather than
isolated spikes.

PELT Algorithm (Pruned Exact Linear Time):
    - Detects multiple changepoints in time series
    - Optimizes cost function with dynamic programming
    - Pruning strategy achieves linear time complexity
    - BIC (Bayesian Information Criterion) penalty prevents overfitting

Use Cases:
    - Detect regime changes (e.g., sustained increase in refund rate)
    - Identify structural breaks (e.g., change in average margin)
    - Distinguish persistent changes from temporary spikes

This detector runs conditionally (if enabled) as Layer 3.
"""

from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import structlog

try:
    import ruptures as rpt

    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False

logger = structlog.get_logger()


class ChangepointDetector:
    """
    Changepoint detection using ruptures library with PELT algorithm.

    Identifies regime changes and structural breaks in business metrics using
    Pruned Exact Linear Time (PELT) algorithm with Bayesian Information Criterion
    penalty for changepoint selection.

    Attributes:
        penalty: BIC penalty parameter (higher = fewer changepoints)
        min_size: Minimum segment size between changepoints
        model: Cost function model ("l2" = mean shift detection)

    Example:
        >>> detector = ChangepointDetector(penalty="bic", min_size=5)
        >>> result = detector.detect([0.02, 0.03, 0.02, 0.15, 0.16, 0.14])
        >>> if result["has_changepoint"]:
        ...     print(f"Regime change at index: {result['changepoint_indices']}")
    """

    def __init__(
        self,
        penalty: str = "bic",
        min_size: int = 5,
        model: str = "l2",
    ):
        """
        Initialize the changepoint detector.

        Args:
            penalty: Penalty type for PELT ("bic", "aic", or numeric value)
            min_size: Minimum segment size between changepoints (default: 5)
            model: Cost function model ("l2" = mean shift, "rbf" = general)
        """
        if not RUPTURES_AVAILABLE:
            raise ImportError(
                "ruptures library not available. Install with: pip install ruptures"
            )

        self.penalty = penalty
        self.min_size = min_size
        self.model = model
        self.logger = structlog.get_logger()

    def detect(
        self,
        metric_values: list[float],
        dates: Optional[list[date]] = None,
    ) -> dict:
        """
        Detect changepoints in a time series.

        Runs PELT algorithm to identify indices where the time series exhibits
        regime changes or structural breaks.

        Args:
            metric_values: List of metric values in chronological order
            dates: Optional list of dates corresponding to values

        Returns:
            Dictionary containing detection results:
            {
                "has_changepoint": bool,
                "changepoint_indices": list[int],
                "changepoint_dates": list[str],  # If dates provided
                "segment_count": int,
                "segment_means": list[float],
                "cost": float
            }

        Example:
            >>> values = [0.02, 0.03, 0.02, 0.15, 0.16, 0.14]
            >>> result = detector.detect(values)
        """
        self.logger.debug(
            "running_changepoint_detection",
            values_count=len(metric_values),
            min_size=self.min_size,
        )

        # Handle edge cases
        if len(metric_values) < self.min_size * 2:
            self.logger.warning(
                "insufficient_data_for_changepoint",
                values_count=len(metric_values),
                min_required=self.min_size * 2,
            )
            return {
                "has_changepoint": False,
                "changepoint_indices": [],
                "changepoint_dates": [],
                "segment_count": 1,
                "segment_means": [float(np.mean(metric_values))] if metric_values else [0.0],
                "cost": 0.0,
            }

        # Convert to numpy array
        signal = np.array(metric_values, dtype=float)

        # Handle NaN values
        if np.any(np.isnan(signal)):
            self.logger.warning("nan_values_in_signal")
            signal = np.nan_to_num(signal, nan=np.nanmean(signal))

        # Reshape for ruptures (requires 2D array)
        signal_2d = signal.reshape(-1, 1)

        try:
            # Initialize PELT algorithm
            algo = rpt.Pelt(model=self.model, min_size=self.min_size)
            algo.fit(signal_2d)

            # Detect changepoints
            changepoint_indices = algo.predict(pen=self._get_penalty_value(len(signal)))

            # Remove the last index (always n by design)
            if changepoint_indices and changepoint_indices[-1] == len(signal):
                changepoint_indices = changepoint_indices[:-1]

            has_changepoint = len(changepoint_indices) > 0

            # Compute segment means
            segment_means = self._compute_segment_means(signal, changepoint_indices)

            # Map to dates if provided
            changepoint_dates = []
            if dates and len(dates) == len(metric_values):
                for idx in changepoint_indices:
                    if idx < len(dates):
                        changepoint_dates.append(str(dates[idx]))

            result = {
                "has_changepoint": has_changepoint,
                "changepoint_indices": changepoint_indices,
                "changepoint_dates": changepoint_dates,
                "segment_count": len(changepoint_indices) + 1,
                "segment_means": [round(m, 4) for m in segment_means],
                "cost": 0.0,  # Could compute with algo.cost if needed
            }

            if has_changepoint:
                self.logger.info(
                    "changepoints_detected",
                    changepoint_count=len(changepoint_indices),
                    indices=changepoint_indices,
                )

            return result

        except Exception as e:
            self.logger.error(
                "changepoint_detection_failed",
                error=str(e),
                values_count=len(metric_values),
            )
            return {
                "has_changepoint": False,
                "changepoint_indices": [],
                "changepoint_dates": [],
                "segment_count": 1,
                "segment_means": [float(np.mean(signal))],
                "cost": 0.0,
            }

    def detect_regime_change(
        self,
        metric_name: str,
        values: list[float],
        dates: list[date],
    ) -> dict:
        """
        Detect regime change in a metric with detailed analysis.

        Runs changepoint detection and analyzes the magnitude and direction
        of regime changes, specifically focusing on the most recent change.

        Args:
            metric_name: Name of the metric being analyzed
            values: List of metric values in chronological order
            dates: List of dates corresponding to values

        Returns:
            Dictionary containing regime change analysis:
            {
                "metric_name": str,
                "has_change": bool,
                "change_date": Optional[str],
                "change_index": Optional[int],
                "before_mean": float,
                "after_mean": float,
                "mean_shift": float,
                "percent_change": float,
                "direction": "increase|decrease|stable"
            }

        Example:
            >>> values = [0.02, 0.03, 0.02, 0.15, 0.16, 0.14]
            >>> dates = [date(2026, 2, i) for i in range(1, 7)]
            >>> result = detector.detect_regime_change("refund_rate", values, dates)
        """
        self.logger.info(
            "analyzing_regime_change",
            metric_name=metric_name,
            values_count=len(values),
        )

        # Run changepoint detection
        detection_result = self.detect(values, dates)

        if not detection_result["has_changepoint"]:
            return {
                "metric_name": metric_name,
                "has_change": False,
                "change_date": None,
                "change_index": None,
                "before_mean": float(np.mean(values)) if values else 0.0,
                "after_mean": float(np.mean(values)) if values else 0.0,
                "mean_shift": 0.0,
                "percent_change": 0.0,
                "direction": "stable",
            }

        # Analyze most recent changepoint
        last_changepoint_idx = detection_result["changepoint_indices"][-1]
        segment_means = detection_result["segment_means"]

        # Before and after means
        before_mean = segment_means[-2] if len(segment_means) > 1 else segment_means[0]
        after_mean = segment_means[-1]

        # Compute shift
        mean_shift = after_mean - before_mean
        percent_change = (mean_shift / before_mean * 100) if before_mean != 0 else 0.0

        # Determine direction
        if abs(mean_shift) < 0.01:
            direction = "stable"
        elif mean_shift > 0:
            direction = "increase"
        else:
            direction = "decrease"

        # Change date
        change_date = None
        if detection_result["changepoint_dates"]:
            change_date = detection_result["changepoint_dates"][-1]

        result = {
            "metric_name": metric_name,
            "has_change": True,
            "change_date": change_date,
            "change_index": last_changepoint_idx,
            "before_mean": round(before_mean, 4),
            "after_mean": round(after_mean, 4),
            "mean_shift": round(mean_shift, 4),
            "percent_change": round(percent_change, 2),
            "direction": direction,
        }

        self.logger.info(
            "regime_change_analyzed",
            metric_name=metric_name,
            direction=direction,
            percent_change=result["percent_change"],
        )

        return result

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _get_penalty_value(self, n_samples: int) -> float:
        """
        Get penalty value for PELT algorithm.

        Converts penalty type to numeric value. BIC penalty scales with
        sample size to prevent overfitting.

        Args:
            n_samples: Number of samples in time series

        Returns:
            Numeric penalty value
        """
        if self.penalty == "bic":
            # BIC penalty: log(n) * dimension
            # For univariate signal, dimension = 1
            return np.log(n_samples)
        elif self.penalty == "aic":
            # AIC penalty: 2 * dimension
            return 2.0
        else:
            # Numeric penalty
            try:
                return float(self.penalty)
            except (ValueError, TypeError):
                self.logger.warning(
                    "invalid_penalty_defaulting_to_bic",
                    penalty=self.penalty,
                )
                return np.log(n_samples)

    def _compute_segment_means(
        self, signal: np.ndarray, changepoint_indices: list[int]
    ) -> list[float]:
        """
        Compute mean value for each segment.

        Splits signal at changepoint indices and computes mean of each segment.

        Args:
            signal: 1D array of metric values
            changepoint_indices: List of changepoint indices

        Returns:
            List of segment means
        """
        if not changepoint_indices:
            return [float(np.mean(signal))]

        segment_means = []
        start_idx = 0

        for cp_idx in changepoint_indices:
            segment = signal[start_idx:cp_idx]
            if len(segment) > 0:
                segment_means.append(float(np.mean(segment)))
            start_idx = cp_idx

        # Last segment
        last_segment = signal[start_idx:]
        if len(last_segment) > 0:
            segment_means.append(float(np.mean(last_segment)))

        return segment_means


# Graceful fallback if ruptures not installed
if not RUPTURES_AVAILABLE:
    logger.warning(
        "ruptures_library_not_available",
        message="Changepoint detection disabled. Install with: pip install ruptures",
    )
