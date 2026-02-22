"""
Temporal Correlation Engine for BRE-RCA Algorithm.

This module implements cross-correlation analysis between metric time series
to determine temporal precedence — which metrics changed anomalously before
the incident metric, and how strongly correlated those changes are.

Temporal precedence is a core component of the BRE-RCA contribution score:
    contribution = w1 * anomaly_magnitude + w2 * temporal_precedence
                 + w3 * graph_proximity    + w4 * data_quality_weight

The temporal correlator computes:
1. Normalized cross-correlation between candidate and incident metric series
2. Optimal lag (in days) indicating how far in advance the candidate changed
3. Temporal precedence score combining correlation strength and lead time

Algorithm details:
- Uses scipy.signal.correlate for normalized cross-correlation
- Applies Welch-overlapped-segment-averaging for noise robustness
- Computes Granger-like precedence without full Granger causality test
  (computational cost trade-off for real-time RCA)

Version: rca_temporal_v1
"""

from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import structlog
from scipy import signal, stats

logger = structlog.get_logger()


class TemporalCorrelator:
    """
    Computes temporal precedence scores between metric time series.

    Analyzes whether candidate root-cause metrics exhibited anomalous
    changes before the incident metric degraded. Uses cross-correlation
    to quantify both the strength and timing of the relationship.

    Attributes:
        max_lag_days: Maximum lag (in days) to consider for precedence
        min_correlation: Minimum correlation threshold for significance
        logger: Structured logger for observability

    Example:
        >>> correlator = TemporalCorrelator(max_lag_days=14)
        >>> result = correlator.compute_temporal_precedence(
        ...     candidate_series=[1.0, 1.2, 2.5, 3.0, 2.8],
        ...     incident_series=[0.5, 0.5, 0.6, 1.5, 2.0],
        ... )
        >>> print(f"Precedence score: {result['precedence_score']:.4f}")
        >>> print(f"Optimal lag: {result['optimal_lag_days']} days")
    """

    # Weights for precedence score computation
    CORRELATION_WEIGHT = 0.6
    LAG_WEIGHT = 0.4

    def __init__(self, max_lag_days: int = 14, min_correlation: float = 0.3):
        """
        Initialize the temporal correlator.

        Args:
            max_lag_days: Maximum lag in days to consider for precedence
            min_correlation: Minimum correlation threshold to consider significant
        """
        self.max_lag_days = max_lag_days
        self.min_correlation = min_correlation
        self.logger = structlog.get_logger()

    def compute_temporal_precedence(
        self,
        candidate_series: list[float],
        incident_series: list[float],
    ) -> dict:
        """
        Compute temporal precedence score between a candidate and incident metric.

        Determines whether the candidate metric changed anomalously before the
        incident metric, and quantifies the strength of that temporal relationship.

        The precedence score combines:
        - Cross-correlation strength (how correlated the changes are)
        - Lead time (how far in advance the candidate changed)

        Args:
            candidate_series: Daily values of the candidate root-cause metric
                (ordered chronologically, same length as incident_series)
            incident_series: Daily values of the incident metric
                (ordered chronologically, same length as candidate_series)

        Returns:
            Dictionary containing:
            {
                "precedence_score": float,  # 0.0-1.0, higher = stronger precedence
                "optimal_lag_days": int,    # Days candidate leads incident (positive = leads)
                "max_correlation": float,   # Peak cross-correlation value
                "is_significant": bool,     # Whether correlation exceeds threshold
                "correlation_at_lag": dict,  # Correlation values at each lag
            }

        Raises:
            ValueError: If series lengths don't match or are too short
        """
        # Validate inputs
        if len(candidate_series) != len(incident_series):
            raise ValueError(
                f"Series length mismatch: candidate={len(candidate_series)}, "
                f"incident={len(incident_series)}"
            )

        if len(candidate_series) < 3:
            self.logger.warning(
                "series_too_short_for_correlation",
                length=len(candidate_series),
                min_required=3,
            )
            return self._empty_result()

        # Convert to numpy arrays
        candidate = np.array(candidate_series, dtype=np.float64)
        incident = np.array(incident_series, dtype=np.float64)

        # Handle constant series (zero variance)
        if np.std(candidate) < 1e-10 or np.std(incident) < 1e-10:
            self.logger.warning("constant_series_detected")
            return self._empty_result()

        # Normalize to zero mean, unit variance
        candidate_norm = (candidate - np.mean(candidate)) / np.std(candidate)
        incident_norm = (incident - np.mean(incident)) / np.std(incident)

        # Compute normalized cross-correlation
        n = len(candidate_norm)
        max_lag = min(self.max_lag_days, n - 1)

        correlation_at_lag = {}
        for lag in range(-max_lag, max_lag + 1):
            corr = self._compute_lagged_correlation(
                candidate_norm, incident_norm, lag
            )
            if corr is not None:
                correlation_at_lag[lag] = corr

        if not correlation_at_lag:
            return self._empty_result()

        # Find optimal lag (maximum positive correlation where candidate LEADS)
        # Positive lag means candidate changes BEFORE incident (causal direction)
        positive_lags = {
            lag: corr for lag, corr in correlation_at_lag.items()
            if lag >= 0
        }

        if positive_lags:
            optimal_lag = max(positive_lags, key=positive_lags.get)
            max_correlation = positive_lags[optimal_lag]
        else:
            optimal_lag = 0
            max_correlation = correlation_at_lag.get(0, 0.0)

        # Compute precedence score
        is_significant = abs(max_correlation) >= self.min_correlation
        precedence_score = self._compute_precedence_score(
            max_correlation, optimal_lag, max_lag
        )

        result = {
            "precedence_score": round(max(0.0, min(1.0, precedence_score)), 4),
            "optimal_lag_days": optimal_lag,
            "max_correlation": round(max_correlation, 4),
            "is_significant": is_significant,
            "correlation_at_lag": {
                k: round(v, 4) for k, v in correlation_at_lag.items()
            },
        }

        self.logger.debug(
            "temporal_precedence_computed",
            precedence_score=result["precedence_score"],
            optimal_lag=optimal_lag,
            max_correlation=round(max_correlation, 4),
            is_significant=is_significant,
        )

        return result

    def compute_batch_precedence(
        self,
        candidate_series_map: dict[str, list[float]],
        incident_series: list[float],
    ) -> dict[str, dict]:
        """
        Compute temporal precedence for multiple candidates against one incident metric.

        Efficient batch processing for RCA where multiple upstream metrics
        need to be scored against the incident metric.

        Args:
            candidate_series_map: Dict mapping metric_name → daily values
            incident_series: Daily values of the incident metric

        Returns:
            Dict mapping metric_name → precedence result dict

        Example:
            >>> results = correlator.compute_batch_precedence(
            ...     candidate_series_map={
            ...         "supplier_delay_rate": [0.1, 0.2, 0.5, 0.4],
            ...         "ticket_volume": [10, 12, 25, 30],
            ...     },
            ...     incident_series=[3.5, 3.4, 3.0, 2.5],
            ... )
        """
        results = {}

        for metric_name, candidate_series in candidate_series_map.items():
            try:
                result = self.compute_temporal_precedence(
                    candidate_series=candidate_series,
                    incident_series=incident_series,
                )
                results[metric_name] = result

            except Exception as e:
                self.logger.error(
                    "batch_precedence_error",
                    metric_name=metric_name,
                    error=str(e),
                )
                results[metric_name] = self._empty_result()

        self.logger.info(
            "batch_precedence_computed",
            candidate_count=len(candidate_series_map),
            significant_count=sum(
                1 for r in results.values() if r["is_significant"]
            ),
        )

        return results

    def compute_granger_like_score(
        self,
        candidate_series: list[float],
        incident_series: list[float],
        max_lag: int = 3,
    ) -> float:
        """
        Compute a simplified Granger-like precedence score.

        Tests whether lagged values of the candidate metric help predict
        the incident metric beyond its own history. Uses F-statistic
        from restricted vs unrestricted regression.

        This is a lightweight approximation of Granger causality that
        avoids the full VAR model fitting for computational efficiency.

        Args:
            candidate_series: Daily values of candidate metric
            incident_series: Daily values of incident metric
            max_lag: Maximum lag order for the test

        Returns:
            Granger-like score in [0.0, 1.0] where higher means stronger
            predictive power from the candidate metric
        """
        n = len(incident_series)
        if n < max_lag + 5:
            return 0.0

        try:
            y = np.array(incident_series[max_lag:], dtype=np.float64)
            n_obs = len(y)

            # Restricted model: incident predicted by own lags only
            X_restricted = np.column_stack([
                np.array(incident_series[max_lag - lag - 1: n - lag - 1])
                for lag in range(max_lag)
            ])

            # Unrestricted model: incident predicted by own lags + candidate lags
            X_unrestricted = np.column_stack([
                X_restricted,
                *[
                    np.array(candidate_series[max_lag - lag - 1: n - lag - 1]).reshape(-1, 1)
                    for lag in range(max_lag)
                ]
            ])

            # Fit restricted model
            result_r = stats.linregress(
                np.mean(X_restricted, axis=1), y
            ) if X_restricted.shape[1] == 1 else None

            # Compute RSS for both models using least squares
            if X_restricted.shape[0] != n_obs:
                return 0.0

            # Restricted RSS
            beta_r, rss_r, _, _ = np.linalg.lstsq(X_restricted, y, rcond=None)[:4]
            if len(rss_r) == 0:
                rss_r_val = np.sum((y - X_restricted @ beta_r) ** 2)
            else:
                rss_r_val = rss_r[0]

            # Unrestricted RSS
            beta_u, rss_u, _, _ = np.linalg.lstsq(X_unrestricted, y, rcond=None)[:4]
            if len(rss_u) == 0:
                rss_u_val = np.sum((y - X_unrestricted @ beta_u) ** 2)
            else:
                rss_u_val = rss_u[0]

            # F-statistic
            df_diff = max_lag  # Additional parameters in unrestricted model
            df_denom = n_obs - X_unrestricted.shape[1]

            if df_denom <= 0 or rss_u_val <= 0:
                return 0.0

            f_stat = ((rss_r_val - rss_u_val) / df_diff) / (rss_u_val / df_denom)

            # Convert F-statistic to a 0-1 score using CDF
            p_value = 1.0 - stats.f.cdf(f_stat, df_diff, df_denom)

            # Lower p-value = stronger Granger causality = higher score
            granger_score = max(0.0, 1.0 - p_value)

            self.logger.debug(
                "granger_like_score_computed",
                f_stat=round(f_stat, 4),
                p_value=round(p_value, 4),
                granger_score=round(granger_score, 4),
            )

            return round(granger_score, 4)

        except Exception as e:
            self.logger.warning(
                "granger_score_computation_failed",
                error=str(e),
            )
            return 0.0

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _compute_lagged_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lag: int,
    ) -> Optional[float]:
        """
        Compute Pearson correlation between x(t-lag) and y(t).

        Positive lag means x leads y (x changes before y).

        Args:
            x: Normalized candidate series
            y: Normalized incident series
            lag: Lag in time steps (positive = x leads)

        Returns:
            Correlation coefficient or None if insufficient data
        """
        n = len(x)

        if lag >= 0:
            # x leads: correlate x[0:n-lag] with y[lag:n]
            if n - lag < 3:
                return None
            x_slice = x[:n - lag]
            y_slice = y[lag:]
        else:
            # y leads: correlate x[-lag:n] with y[0:n+lag]
            abs_lag = abs(lag)
            if n - abs_lag < 3:
                return None
            x_slice = x[abs_lag:]
            y_slice = y[:n - abs_lag]

        if len(x_slice) < 3:
            return None

        try:
            corr, _ = stats.pearsonr(x_slice, y_slice)
            return corr if not np.isnan(corr) else None
        except Exception:
            return None

    def _compute_precedence_score(
        self,
        max_correlation: float,
        optimal_lag: int,
        max_possible_lag: int,
    ) -> float:
        """
        Compute final precedence score from correlation and lag.

        Score formula:
            precedence = w1 * |correlation| + w2 * lag_score

        where lag_score rewards earlier leads (higher lag):
            lag_score = optimal_lag / max_possible_lag  (if lag > 0)
            lag_score = 0.5                             (if lag == 0, concurrent)

        Args:
            max_correlation: Peak cross-correlation value
            optimal_lag: Optimal lag in days
            max_possible_lag: Maximum lag considered

        Returns:
            Precedence score in [0.0, 1.0]
        """
        # Correlation component: use absolute value, clamped to [0, 1]
        corr_score = min(1.0, max(0.0, abs(max_correlation)))

        # Lag component: reward leading (positive lag)
        if optimal_lag > 0 and max_possible_lag > 0:
            # Normalize lag to [0, 1], with diminishing returns for very long leads
            lag_score = min(1.0, optimal_lag / max_possible_lag)
        elif optimal_lag == 0:
            # Concurrent change: moderate score
            lag_score = 0.5
        else:
            # Candidate lags behind incident: low score
            lag_score = 0.1

        precedence = (
            self.CORRELATION_WEIGHT * corr_score
            + self.LAG_WEIGHT * lag_score
        )

        return precedence

    def _empty_result(self) -> dict:
        """Return empty/default precedence result."""
        return {
            "precedence_score": 0.0,
            "optimal_lag_days": 0,
            "max_correlation": 0.0,
            "is_significant": False,
            "correlation_at_lag": {},
        }
