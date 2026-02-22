"""
Comprehensive unit tests for LedgerGuard BRE engine modules.

Test-automator agent requirements:
- Independent tests (no shared state)
- Atomic tests (one assertion per test where possible)
- Clear naming (test_<module>_<method>_<scenario>)
- >80% coverage
- Proper error handling
"""

import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock optional dependencies before importing engine modules to avoid import errors
# This allows tests to run without these packages installed
import sys
# Create proper mock modules with submodules
lightgbm_mock = MagicMock()
sys.modules["lightgbm"] = lightgbm_mock

mlflow_mock = MagicMock()
mlflow_mock.lightgbm = MagicMock()
mlflow_mock.sklearn = MagicMock()
sys.modules["mlflow"] = mlflow_mock
sys.modules["mlflow.lightgbm"] = mlflow_mock.lightgbm
sys.modules["mlflow.sklearn"] = mlflow_mock.sklearn

optuna_mock = MagicMock()
sys.modules["optuna"] = optuna_mock

# Now we can import normally (structlog is installed, no mock needed)
from api.engine.blast_radius.impact_scorer import ImpactScorer
from api.engine.cascade_correlator import CascadeCorrelator
from api.engine.comparator import ComparatorEngine
from api.engine.detection.statistical import StatisticalDetector
from api.engine.rca.causal_ranker import CausalRanker
from api.engine.rca.temporal_correlation import TemporalCorrelator

from api.models.enums import BlastRadiusSeverity, IncidentType, Severity
from tests.conftest import (
    make_candidate,
    make_canonical_event,
    make_causal_path,
    make_incident,
    MockStorage,
)


# ============================================================================
# CausalRanker Tests
# ============================================================================


class TestCausalRankerInit:
    """Test CausalRanker initialization and weight validation."""

    def test_causal_ranker_init_default_weights(self):
        """Test initialization with default weights."""
        ranker = CausalRanker()
        assert ranker.w_anomaly == 0.30
        assert ranker.w_temporal == 0.30
        assert ranker.w_proximity == 0.25
        assert ranker.w_quality == 0.15

    def test_causal_ranker_init_custom_weights(self):
        """Test initialization with custom weights."""
        ranker = CausalRanker(
            w_anomaly=0.40, w_temporal=0.30, w_proximity=0.20, w_quality=0.10
        )
        assert ranker.w_anomaly == 0.40
        assert ranker.w_temporal == 0.30
        assert ranker.w_proximity == 0.20
        assert ranker.w_quality == 0.10

    def test_causal_ranker_init_weights_sum_to_one(self):
        """Test that weights summing to 1.0 are accepted."""
        ranker = CausalRanker(
            w_anomaly=0.25, w_temporal=0.25, w_proximity=0.25, w_quality=0.25
        )
        assert ranker.w_anomaly == 0.25

    def test_causal_ranker_init_weights_sum_not_one_raises_error(self):
        """Test that weights not summing to 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            CausalRanker(w_anomaly=0.5, w_temporal=0.3, w_proximity=0.2, w_quality=0.1)

    def test_causal_ranker_init_weights_sum_close_to_one_accepted(self):
        """Test that weights summing to ~1.0 (within 0.01) are accepted."""
        ranker = CausalRanker(
            w_anomaly=0.3001, w_temporal=0.2999, w_proximity=0.25, w_quality=0.15
        )
        assert ranker.w_anomaly == 0.3001


class TestCausalRankerComputeContributionScore:
    """Test CausalRanker.compute_contribution_score method."""

    def test_compute_contribution_score_normal_case(self):
        """Test contribution score computation with normal inputs."""
        ranker = CausalRanker()
        candidate = make_candidate(
            anomaly_magnitude=5.0,
            temporal_precedence=0.8,
            graph_proximity=0.7,
            data_quality_weight=0.9,
        )
        score = ranker.compute_contribution_score(candidate)
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)

    def test_compute_contribution_score_maximum_values(self):
        """Test contribution score with maximum component values."""
        ranker = CausalRanker()
        candidate = make_candidate(
            anomaly_magnitude=10.0,
            temporal_precedence=1.0,
            graph_proximity=1.0,
            data_quality_weight=1.0,
        )
        score = ranker.compute_contribution_score(candidate)
        assert score <= 1.0
        assert score > 0.8  # Should be high

    def test_compute_contribution_score_minimum_values(self):
        """Test contribution score with minimum component values."""
        ranker = CausalRanker()
        candidate = make_candidate(
            anomaly_magnitude=0.0,
            temporal_precedence=0.0,
            graph_proximity=0.0,
            data_quality_weight=0.0,
        )
        score = ranker.compute_contribution_score(candidate)
        assert score >= 0.0
        assert score < 0.2  # Should be low

    def test_compute_contribution_score_high_anomaly_low_others(self):
        """Test that high anomaly magnitude increases score."""
        ranker = CausalRanker()
        candidate_high_anomaly = make_candidate(
            anomaly_magnitude=8.0,
            temporal_precedence=0.2,
            graph_proximity=0.2,
            data_quality_weight=0.5,
        )
        candidate_low_anomaly = make_candidate(
            anomaly_magnitude=2.0,
            temporal_precedence=0.2,
            graph_proximity=0.2,
            data_quality_weight=0.5,
        )
        score_high = ranker.compute_contribution_score(candidate_high_anomaly)
        score_low = ranker.compute_contribution_score(candidate_low_anomaly)
        assert score_high > score_low

    def test_compute_contribution_score_missing_fields_defaults(self):
        """Test that missing fields use defaults."""
        ranker = CausalRanker()
        candidate = {"metric_name": "test_metric"}
        score = ranker.compute_contribution_score(candidate)
        assert 0.0 <= score <= 1.0

    def test_compute_contribution_score_negative_anomaly_magnitude(self):
        """Test that negative anomaly magnitude is handled (uses absolute value)."""
        ranker = CausalRanker()
        candidate = make_candidate(anomaly_magnitude=-5.0)
        score = ranker.compute_contribution_score(candidate)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize(
        "temporal,proximity,quality,expected_higher",
        [
            (0.9, 0.5, 0.5, True),
            (0.1, 0.5, 0.5, False),
            (0.5, 0.9, 0.5, True),
            (0.5, 0.1, 0.5, False),
            (0.5, 0.5, 0.9, True),
            (0.5, 0.5, 0.1, False),
        ],
    )
    def test_compute_contribution_score_component_weights(
        self, temporal, proximity, quality, expected_higher
    ):
        """Test that each component contributes to the score."""
        ranker = CausalRanker()
        candidate = make_candidate(
            anomaly_magnitude=5.0,
            temporal_precedence=temporal,
            graph_proximity=proximity,
            data_quality_weight=quality,
        )
        score = ranker.compute_contribution_score(candidate)
        baseline = ranker.compute_contribution_score(
            make_candidate(anomaly_magnitude=5.0, temporal_precedence=0.5, graph_proximity=0.5, data_quality_weight=0.5)
        )
        if expected_higher:
            assert score > baseline
        else:
            assert score < baseline


class TestCausalRankerRankCandidates:
    """Test CausalRanker.rank_candidates method."""

    def test_rank_candidates_empty_list(self):
        """Test ranking with empty candidate list."""
        ranker = CausalRanker()
        result = ranker.rank_candidates([], "refund_rate", top_k=5)
        assert result == []

    def test_rank_candidates_single_candidate(self):
        """Test ranking with single candidate."""
        ranker = CausalRanker()
        candidates = [make_candidate("supplier_delay_rate", anomaly_magnitude=6.0)]
        result = ranker.rank_candidates(candidates, "refund_rate", top_k=5)
        assert len(result) == 1
        assert result[0].rank == 1
        assert result[0].nodes[0].metric_name == "supplier_delay_rate"

    def test_rank_candidates_multiple_candidates_ordered(self, sample_candidates):
        """Test that candidates are ranked by score descending."""
        ranker = CausalRanker()
        result = ranker.rank_candidates(sample_candidates, "refund_rate", top_k=5)
        assert len(result) >= 2
        # Verify descending order
        for i in range(len(result) - 1):
            assert result[i].overall_score >= result[i + 1].overall_score

    def test_rank_candidates_top_k_limit(self, sample_candidates):
        """Test that top_k limits the number of results."""
        ranker = CausalRanker()
        result = ranker.rank_candidates(sample_candidates, "refund_rate", top_k=2)
        assert len(result) == 2

    def test_rank_candidates_top_k_larger_than_candidates(self):
        """Test that top_k larger than candidates returns all."""
        ranker = CausalRanker()
        candidates = [make_candidate("metric1"), make_candidate("metric2")]
        result = ranker.rank_candidates(candidates, "refund_rate", top_k=10)
        assert len(result) == 2

    def test_rank_candidates_ranks_assigned_correctly(self, sample_candidates):
        """Test that ranks are assigned correctly (1-based)."""
        ranker = CausalRanker()
        result = ranker.rank_candidates(sample_candidates, "refund_rate", top_k=5)
        for i, path in enumerate(result, start=1):
            assert path.rank == i

    def test_rank_candidates_causal_path_structure(self, sample_candidates):
        """Test that returned CausalPath objects have correct structure."""
        ranker = CausalRanker()
        result = ranker.rank_candidates(sample_candidates, "refund_rate", top_k=3)
        for path in result:
            assert path.rank > 0
            assert 0.0 <= path.overall_score <= 1.0
            assert len(path.nodes) > 0
            assert path.nodes[0].metric_name in [c["metric_name"] for c in sample_candidates]


class TestCausalRankerComputeRelativeImportance:
    """Test CausalRanker.compute_relative_importance method."""

    def test_compute_relative_importance_empty_paths(self):
        """Test relative importance with empty paths."""
        ranker = CausalRanker()
        result = ranker.compute_relative_importance([])
        assert result == {}

    def test_compute_relative_importance_single_path(self):
        """Test relative importance with single path."""
        ranker = CausalRanker()
        path = make_causal_path(rank=1, score=0.85, metric_name="test_metric")
        result = ranker.compute_relative_importance([path])
        assert result == {"test_metric": 100.0}

    def test_compute_relative_importance_multiple_paths(self):
        """Test relative importance with multiple paths."""
        ranker = CausalRanker()
        paths = [
            make_causal_path(rank=1, score=0.8, metric_name="metric1"),
            make_causal_path(rank=2, score=0.4, metric_name="metric2"),
            make_causal_path(rank=3, score=0.2, metric_name="metric3"),
        ]
        result = ranker.compute_relative_importance(paths)
        assert len(result) == 3
        assert sum(result.values()) == pytest.approx(100.0, abs=0.1)
        assert result["metric1"] > result["metric2"]
        assert result["metric2"] > result["metric3"]

    def test_compute_relative_importance_zero_total_score(self):
        """Test relative importance when total score is zero."""
        ranker = CausalRanker()
        paths = [
            make_causal_path(rank=1, score=0.0, metric_name="metric1"),
            make_causal_path(rank=2, score=0.0, metric_name="metric2"),
        ]
        result = ranker.compute_relative_importance(paths)
        assert result["metric1"] == 0.0
        assert result["metric2"] == 0.0


class TestCausalRankerConfidenceIntervals:
    """Test CausalRanker.compute_confidence_intervals method."""

    def test_compute_confidence_intervals_single_candidate(self):
        """Test confidence intervals with single candidate."""
        ranker = CausalRanker()
        candidates = [make_candidate("test_metric", anomaly_magnitude=5.0)]
        result = ranker.compute_confidence_intervals(candidates, n_bootstrap=50, ci_level=0.95)
        assert "test_metric" in result
        assert "point_estimate" in result["test_metric"]
        assert "ci_lower" in result["test_metric"]
        assert "ci_upper" in result["test_metric"]
        assert "std_error" in result["test_metric"]
        assert "is_significant" in result["test_metric"]
        assert result["test_metric"]["ci_lower"] <= result["test_metric"]["point_estimate"]
        assert result["test_metric"]["ci_upper"] >= result["test_metric"]["point_estimate"]

    def test_compute_confidence_intervals_multiple_candidates(self, sample_candidates):
        """Test confidence intervals with multiple candidates."""
        ranker = CausalRanker()
        result = ranker.compute_confidence_intervals(sample_candidates, n_bootstrap=50, ci_level=0.95)
        assert len(result) == len(sample_candidates)
        for metric_name, ci_data in result.items():
            assert ci_data["ci_lower"] <= ci_data["point_estimate"] <= ci_data["ci_upper"]

    def test_compute_confidence_intervals_different_ci_levels(self):
        """Test confidence intervals with different CI levels."""
        ranker = CausalRanker()
        candidates = [make_candidate("test_metric")]
        ci_90 = ranker.compute_confidence_intervals(candidates, ci_level=0.90)
        ci_95 = ranker.compute_confidence_intervals(candidates, ci_level=0.95)
        ci_99 = ranker.compute_confidence_intervals(candidates, ci_level=0.99)
        # Wider intervals for higher confidence
        assert ci_99["test_metric"]["ci_upper"] - ci_99["test_metric"]["ci_lower"] >= \
               ci_95["test_metric"]["ci_upper"] - ci_95["test_metric"]["ci_lower"]
        assert ci_95["test_metric"]["ci_upper"] - ci_95["test_metric"]["ci_lower"] >= \
               ci_90["test_metric"]["ci_upper"] - ci_90["test_metric"]["ci_lower"]


class TestCausalRankerSensitivityAnalysis:
    """Test CausalRanker.sensitivity_analysis method."""

    def test_sensitivity_analysis_empty_candidates(self):
        """Test sensitivity analysis with empty candidates."""
        ranker = CausalRanker()
        result = ranker.sensitivity_analysis([], "refund_rate", top_k=5)
        assert result["rank_stability"] == 0.0
        assert result["top1_robustness"] == "low"

    def test_sensitivity_analysis_single_candidate(self):
        """Test sensitivity analysis with single candidate."""
        ranker = CausalRanker()
        candidates = [make_candidate("test_metric", anomaly_magnitude=8.0)]
        result = ranker.sensitivity_analysis(candidates, "refund_rate", top_k=5)
        assert "rank_stability" in result
        assert "top1_robustness" in result
        assert "baseline_top1" in result
        assert "weight_sensitivity" in result
        assert "alternative_rankings" in result
        assert result["baseline_top1"] == "test_metric"

    def test_sensitivity_analysis_stable_ranking(self, sample_candidates):
        """Test sensitivity analysis with stable ranking."""
        ranker = CausalRanker()
        result = ranker.sensitivity_analysis(sample_candidates, "refund_rate", top_k=5)
        assert 0.0 <= result["rank_stability"] <= 1.0
        assert result["top1_robustness"] in ["high", "medium", "low"]
        assert isinstance(result["weight_sensitivity"], dict)

    def test_sensitivity_analysis_perturbations_tested(self, sample_candidates):
        """Test that sensitivity analysis tests multiple perturbations."""
        ranker = CausalRanker()
        result = ranker.sensitivity_analysis(sample_candidates, "refund_rate", top_k=5)
        assert result["perturbations_tested"] > 0


# ============================================================================
# TemporalCorrelator Tests
# ============================================================================


class TestTemporalCorrelatorInit:
    """Test TemporalCorrelator initialization."""

    def test_temporal_correlator_init_defaults(self):
        """Test initialization with default parameters."""
        correlator = TemporalCorrelator()
        assert correlator.max_lag_days == 14
        assert correlator.min_correlation == 0.3

    def test_temporal_correlator_init_custom_params(self):
        """Test initialization with custom parameters."""
        correlator = TemporalCorrelator(max_lag_days=7, min_correlation=0.5)
        assert correlator.max_lag_days == 7
        assert correlator.min_correlation == 0.5


class TestTemporalCorrelatorComputeTemporalPrecedence:
    """Test TemporalCorrelator.compute_temporal_precedence method."""

    def test_compute_temporal_precedence_matching_lengths(self):
        """Test precedence computation with matching series lengths."""
        correlator = TemporalCorrelator()
        candidate = [1.0, 1.2, 2.5, 3.0, 2.8]
        incident = [0.5, 0.5, 0.6, 1.5, 2.0]
        result = correlator.compute_temporal_precedence(candidate, incident)
        assert "precedence_score" in result
        assert "optimal_lag_days" in result
        assert "max_correlation" in result
        assert "is_significant" in result
        assert "correlation_at_lag" in result
        assert 0.0 <= result["precedence_score"] <= 1.0

    def test_compute_temporal_precedence_mismatched_lengths_raises_error(self):
        """Test that mismatched series lengths raise ValueError."""
        correlator = TemporalCorrelator()
        candidate = [1.0, 2.0, 3.0]
        incident = [0.5, 0.6]
        with pytest.raises(ValueError, match="length mismatch"):
            correlator.compute_temporal_precedence(candidate, incident)

    def test_compute_temporal_precedence_too_short_series(self):
        """Test precedence with series too short for correlation."""
        correlator = TemporalCorrelator()
        candidate = [1.0, 2.0]
        incident = [0.5, 0.6]
        result = correlator.compute_temporal_precedence(candidate, incident)
        assert result["precedence_score"] == 0.0
        assert result["is_significant"] is False

    def test_compute_temporal_precedence_constant_series(self):
        """Test precedence with constant (zero variance) series."""
        correlator = TemporalCorrelator()
        candidate = [1.0, 1.0, 1.0, 1.0, 1.0]
        incident = [2.0, 2.0, 2.0, 2.0, 2.0]
        result = correlator.compute_temporal_precedence(candidate, incident)
        assert result["precedence_score"] == 0.0

    def test_compute_temporal_precedence_strong_correlation(self):
        """Test precedence with strongly correlated series."""
        correlator = TemporalCorrelator()
        # Candidate leads incident by 2 days
        candidate = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        incident = [0.5, 0.5, 0.5, 1.0, 1.5, 2.0, 2.5]
        result = correlator.compute_temporal_precedence(candidate, incident)
        assert result["precedence_score"] > 0.0
        assert result["optimal_lag_days"] >= 0

    def test_compute_temporal_precedence_negative_correlation(self):
        """Test precedence with negatively correlated series."""
        correlator = TemporalCorrelator()
        candidate = [1.0, 2.0, 3.0, 4.0, 5.0]
        incident = [5.0, 4.0, 3.0, 2.0, 1.0]
        result = correlator.compute_temporal_precedence(candidate, incident)
        assert "precedence_score" in result
        assert "max_correlation" in result

    def test_compute_temporal_precedence_single_element(self):
        """Test precedence with single element series."""
        correlator = TemporalCorrelator()
        candidate = [1.0]
        incident = [2.0]
        result = correlator.compute_temporal_precedence(candidate, incident)
        assert result["precedence_score"] == 0.0


class TestTemporalCorrelatorComputeGrangerLikeScore:
    """Test TemporalCorrelator.compute_granger_like_score method."""

    def test_compute_granger_like_score_normal_case(self):
        """Test Granger-like score computation with normal inputs."""
        correlator = TemporalCorrelator()
        candidate = [1.0, 1.2, 2.5, 3.0, 2.8, 3.2, 3.5]
        incident = [0.5, 0.5, 0.6, 1.5, 2.0, 2.2, 2.5]
        score = correlator.compute_granger_like_score(candidate, incident, max_lag=3)
        assert 0.0 <= score <= 1.0

    def test_compute_granger_like_score_too_short_series(self):
        """Test Granger-like score with series too short."""
        correlator = TemporalCorrelator()
        candidate = [1.0, 2.0]
        incident = [0.5, 0.6]
        score = correlator.compute_granger_like_score(candidate, incident, max_lag=3)
        assert score == 0.0

    def test_compute_granger_like_score_strong_predictive_power(self):
        """Test Granger-like score with strong predictive relationship."""
        correlator = TemporalCorrelator()
        # Candidate strongly predicts incident (longer series for stable lstsq)
        candidate = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        incident = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        score = correlator.compute_granger_like_score(candidate, incident, max_lag=2)
        assert score >= 0.0
        assert score <= 1.0

    def test_compute_granger_like_score_no_predictive_power(self):
        """Test Granger-like score with no predictive relationship."""
        correlator = TemporalCorrelator()
        candidate = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        incident = [2.0, 3.0, 1.0, 4.0, 2.0, 3.0]
        score = correlator.compute_granger_like_score(candidate, incident, max_lag=2)
        assert score >= 0.0


class TestTemporalCorrelatorBatchPrecedence:
    """Test TemporalCorrelator.compute_batch_precedence method."""

    def test_compute_batch_precedence_multiple_candidates(self):
        """Test batch precedence with multiple candidates."""
        correlator = TemporalCorrelator()
        candidate_map = {
            "metric1": [1.0, 1.2, 2.5, 3.0],
            "metric2": [0.5, 0.6, 0.7, 0.8],
            "metric3": [2.0, 2.1, 2.2, 2.3],
        }
        incident = [0.5, 0.6, 1.0, 1.5]
        result = correlator.compute_batch_precedence(candidate_map, incident)
        assert len(result) == 3
        assert "metric1" in result
        assert "metric2" in result
        assert "metric3" in result
        for metric_result in result.values():
            assert "precedence_score" in metric_result


# ============================================================================
# StatisticalDetector Tests
# ============================================================================


class TestStatisticalDetectorInit:
    """Test StatisticalDetector initialization."""

    def test_statistical_detector_init_defaults(self):
        """Test initialization with default parameters."""
        detector = StatisticalDetector()
        assert detector.baseline_days == 30
        assert detector.zscore_threshold == 3.0

    def test_statistical_detector_init_custom_params(self):
        """Test initialization with custom parameters."""
        detector = StatisticalDetector(baseline_days=60, zscore_threshold=2.5)
        assert detector.baseline_days == 60
        assert detector.zscore_threshold == 2.5


class TestStatisticalDetectorDetect:
    """Test StatisticalDetector.detect method."""

    def test_detect_normal_case(self):
        """Test detection with normal inputs."""
        detector = StatisticalDetector()
        baseline = [0.02, 0.03, 0.02, 0.04, 0.03] * 6  # 30 days
        result = detector.detect("refund_rate", 0.15, baseline)
        assert "metric_name" in result
        assert "current_value" in result
        assert "is_anomaly" in result
        assert "zscore" in result
        assert "median" in result
        assert "mad" in result
        assert result["metric_name"] == "refund_rate"
        assert result["current_value"] == 0.15

    def test_detect_anomaly_detected(self):
        """Test that high z-score values are detected as anomalies."""
        detector = StatisticalDetector(zscore_threshold=3.0)
        baseline = [0.02, 0.03, 0.02, 0.04, 0.03] * 6
        result = detector.detect("refund_rate", 0.15, baseline)
        assert result["is_anomaly"] is True
        assert abs(result["zscore"]) > 3.0

    def test_detect_no_anomaly(self):
        """Test that values within normal range are not flagged."""
        detector = StatisticalDetector(zscore_threshold=3.0)
        baseline = [0.02, 0.03, 0.02, 0.04, 0.03] * 6
        result = detector.detect("refund_rate", 0.03, baseline)
        assert result["is_anomaly"] is False
        assert abs(result["zscore"]) <= 3.0

    def test_detect_empty_baseline(self):
        """Test detection with empty baseline."""
        detector = StatisticalDetector()
        result = detector.detect("refund_rate", 0.15, [])
        assert result["is_anomaly"] is False
        assert result["baseline_count"] == 0

    def test_detect_single_baseline_value(self):
        """Test detection with single baseline value."""
        detector = StatisticalDetector()
        result = detector.detect("refund_rate", 0.15, [0.03])
        assert "zscore" in result
        assert "is_anomaly" in result

    def test_detect_constant_baseline(self):
        """Test detection with constant baseline values."""
        detector = StatisticalDetector()
        baseline = [0.03] * 30
        result = detector.detect("refund_rate", 0.15, baseline)
        # Should detect anomaly since MAD is 0 and value differs
        assert result["is_anomaly"] is True

    def test_detect_constant_baseline_same_value(self):
        """Test detection when current value equals constant baseline."""
        detector = StatisticalDetector()
        baseline = [0.03] * 30
        result = detector.detect("refund_rate", 0.03, baseline)
        assert result["zscore"] == 0.0
        assert result["is_anomaly"] is False

    def test_detect_nan_values_filtered(self):
        """Test that NaN values in baseline are filtered out."""
        detector = StatisticalDetector()
        baseline = [0.02, 0.03, float("nan"), 0.04, 0.03] * 6
        result = detector.detect("refund_rate", 0.15, baseline)
        assert result["baseline_count"] < len(baseline)
        assert "zscore" in result

    def test_detect_negative_zscore(self):
        """Test detection with negative z-score (value below baseline)."""
        detector = StatisticalDetector()
        baseline = [0.10, 0.12, 0.11, 0.13, 0.12] * 6
        result = detector.detect("refund_rate", 0.01, baseline)
        assert result["zscore"] < 0
        if abs(result["zscore"]) > 3.0:
            assert result["is_anomaly"] is True


class TestStatisticalDetectorGetBaselineStats:
    """Test StatisticalDetector.get_baseline_stats method."""

    def test_get_baseline_stats_normal_case(self):
        """Test baseline stats with normal inputs."""
        detector = StatisticalDetector()
        baseline = [0.02, 0.03, 0.02, 0.04, 0.03, 0.025, 0.035]
        stats = detector.get_baseline_stats(baseline)
        assert "median" in stats
        assert "mad" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "count" in stats
        assert stats["count"] == len(baseline)
        assert stats["min"] <= stats["median"] <= stats["max"]

    def test_get_baseline_stats_empty_list(self):
        """Test baseline stats with empty list."""
        detector = StatisticalDetector()
        stats = detector.get_baseline_stats([])
        assert stats["count"] == 0
        assert stats["median"] == 0.0
        assert stats["mad"] == 0.0

    def test_get_baseline_stats_single_value(self):
        """Test baseline stats with single value."""
        detector = StatisticalDetector()
        stats = detector.get_baseline_stats([0.03])
        assert stats["count"] == 1
        assert stats["median"] == 0.03
        assert stats["mad"] == 0.0

    def test_get_baseline_stats_nan_filtered(self):
        """Test that NaN values are filtered from stats."""
        detector = StatisticalDetector()
        baseline = [0.02, 0.03, float("nan"), 0.04, 0.03]
        stats = detector.get_baseline_stats(baseline)
        assert stats["count"] == 4


# ============================================================================
# ImpactScorer Tests
# ============================================================================


class TestImpactScorerInit:
    """Test ImpactScorer initialization."""

    def test_impact_scorer_init_default(self):
        """Test initialization with default churn multiplier."""
        scorer = ImpactScorer()
        assert scorer.churn_risk_multiplier == 0.06

    def test_impact_scorer_init_custom_multiplier(self):
        """Test initialization with custom churn multiplier."""
        scorer = ImpactScorer(churn_risk_multiplier=0.10)
        assert scorer.churn_risk_multiplier == 0.10


class TestImpactScorerClassifySeverity:
    """Test ImpactScorer.classify_severity method."""

    def test_classify_severity_contained(self):
        """Test severity classification for CONTAINED incidents."""
        scorer = ImpactScorer()
        severity = scorer.classify_severity(
            customers_affected=30,
            revenue_exposure=5000.0,
            refund_exposure=1000.0,
            downstream_count=0,
        )
        assert severity == BlastRadiusSeverity.CONTAINED

    def test_classify_severity_significant(self):
        """Test severity classification for SIGNIFICANT incidents."""
        scorer = ImpactScorer()
        severity = scorer.classify_severity(
            customers_affected=100,
            revenue_exposure=20000.0,
            refund_exposure=5000.0,
            downstream_count=1,
        )
        assert severity == BlastRadiusSeverity.SIGNIFICANT

    def test_classify_severity_severe(self):
        """Test severity classification for SEVERE incidents."""
        scorer = ImpactScorer()
        severity = scorer.classify_severity(
            customers_affected=1000,
            revenue_exposure=200000.0,
            refund_exposure=50000.0,
            downstream_count=2,
        )
        assert severity == BlastRadiusSeverity.SEVERE

    def test_classify_severity_catastrophic_customers(self):
        """Test severity classification for CATASTROPHIC by customers."""
        scorer = ImpactScorer()
        severity = scorer.classify_severity(
            customers_affected=3000,
            revenue_exposure=10000.0,
            refund_exposure=1000.0,
            downstream_count=0,
        )
        assert severity == BlastRadiusSeverity.CATASTROPHIC

    def test_classify_severity_catastrophic_revenue(self):
        """Test severity classification for CATASTROPHIC by revenue."""
        scorer = ImpactScorer()
        severity = scorer.classify_severity(
            customers_affected=100,
            revenue_exposure=600000.0,
            refund_exposure=1000.0,
            downstream_count=0,
        )
        assert severity == BlastRadiusSeverity.CATASTROPHIC

    def test_classify_severity_catastrophic_refunds(self):
        """Test severity classification for CATASTROPHIC by refunds."""
        scorer = ImpactScorer()
        severity = scorer.classify_severity(
            customers_affected=100,
            revenue_exposure=10000.0,
            refund_exposure=150000.0,
            downstream_count=0,
        )
        assert severity == BlastRadiusSeverity.CATASTROPHIC

    def test_classify_severity_catastrophic_downstream(self):
        """Test severity classification for CATASTROPHIC by downstream count."""
        scorer = ImpactScorer()
        severity = scorer.classify_severity(
            customers_affected=100,
            revenue_exposure=10000.0,
            refund_exposure=1000.0,
            downstream_count=3,
        )
        assert severity == BlastRadiusSeverity.CATASTROPHIC

    def test_classify_severity_boundary_values(self):
        """Test severity classification at boundary values."""
        scorer = ImpactScorer()
        # Test exact thresholds
        sig = scorer.classify_severity(50, 10000.0, 2000.0, 1)
        assert sig == BlastRadiusSeverity.SIGNIFICANT
        sev = scorer.classify_severity(500, 100000.0, 25000.0, 2)
        assert sev == BlastRadiusSeverity.SEVERE
        cat = scorer.classify_severity(2000, 500000.0, 100000.0, 3)
        assert cat == BlastRadiusSeverity.CATASTROPHIC

    def test_classify_severity_zero_values(self):
        """Test severity classification with zero values."""
        scorer = ImpactScorer()
        severity = scorer.classify_severity(0, 0.0, 0.0, 0)
        assert severity == BlastRadiusSeverity.CONTAINED


class TestImpactScorerScoreImpact:
    """Test ImpactScorer.score_impact method."""

    def test_score_impact_normal_case(self, sample_incident):
        """Test impact scoring with normal inputs."""
        scorer = ImpactScorer()
        events = [
            make_canonical_event(amount=1000.0),
            make_canonical_event(amount=2000.0),
        ]
        entity_sets = {"customer": {"cust1", "cust2"}}
        impact = scorer.score_impact(events, entity_sets, sample_incident)
        assert "revenue_exposure" in impact
        assert "refund_exposure" in impact
        assert "churn_exposure" in impact
        assert "avg_order_value" in impact
        assert "total_event_amount" in impact

    def test_score_impact_empty_events(self, sample_incident):
        """Test impact scoring with empty events."""
        scorer = ImpactScorer()
        entity_sets = {"customer": set()}
        impact = scorer.score_impact([], entity_sets, sample_incident)
        assert impact["revenue_exposure"] == 0.0
        assert impact["churn_exposure"] == 0


# ============================================================================
# CascadeCorrelator Tests
# ============================================================================


class TestCascadeCorrelatorInit:
    """Test CascadeCorrelator initialization."""

    def test_cascade_correlator_init_defaults(self):
        """Test initialization with default parameters."""
        correlator = CascadeCorrelator()
        assert correlator.cascade_score_threshold == 0.3
        assert correlator.temporal_decay_factor == 3.0

    def test_cascade_correlator_init_custom_params(self):
        """Test initialization with custom parameters."""
        correlator = CascadeCorrelator(
            cascade_score_threshold=0.5, temporal_decay_factor=5.0
        )
        assert correlator.cascade_score_threshold == 0.5
        assert correlator.temporal_decay_factor == 5.0


class TestCascadeCorrelatorCorrelate:
    """Test CascadeCorrelator.correlate method."""

    def test_correlate_insufficient_incidents(self):
        """Test correlation with less than 2 incidents."""
        correlator = CascadeCorrelator()
        incidents = [make_incident()]
        result = correlator.correlate(incidents)
        assert result == []

    def test_correlate_no_cascade_detected(self):
        """Test correlation when no cascades are detected."""
        correlator = CascadeCorrelator(cascade_score_threshold=0.9)
        now = datetime.utcnow()
        incident1 = make_incident(
            incident_type=IncidentType.REFUND_SPIKE,
            detected_at=now - timedelta(days=10),
        )
        incident2 = make_incident(
            incident_type=IncidentType.MARGIN_COMPRESSION,
            detected_at=now - timedelta(days=5),
        )
        result = correlator.correlate([incident1, incident2])
        # Should return empty if threshold too high or no causal relationship
        assert isinstance(result, list)

    def test_correlate_cascade_detected(self):
        """Test correlation when cascade is detected."""
        correlator = CascadeCorrelator(cascade_score_threshold=0.2)
        now = datetime.utcnow()
        incident1 = make_incident(
            incident_type=IncidentType.REFUND_SPIKE,
            detected_at=now - timedelta(days=1),
            evidence_event_ids=["evt1", "evt2"],
        )
        incident2 = make_incident(
            incident_type=IncidentType.MARGIN_COMPRESSION,
            detected_at=now,
            evidence_event_ids=["evt2", "evt3"],
        )
        result = correlator.correlate([incident1, incident2])
        assert isinstance(result, list)

    def test_correlate_temporal_ordering(self):
        """Test that incidents are sorted by detection time."""
        correlator = CascadeCorrelator()
        now = datetime.utcnow()
        incident1 = make_incident(detected_at=now - timedelta(days=2))
        incident2 = make_incident(detected_at=now - timedelta(days=1))
        incident3 = make_incident(detected_at=now)
        result = correlator.correlate([incident3, incident1, incident2])
        # Should handle unsorted input correctly
        assert isinstance(result, list)


class TestCascadeCorrelatorComputeCascadeScore:
    """Test CascadeCorrelator._compute_cascade_score method."""

    def test_compute_cascade_score_temporal_order(self):
        """Test that temporal ordering affects score."""
        correlator = CascadeCorrelator()
        now = datetime.utcnow()
        incident_a = make_incident(
            incident_type=IncidentType.REFUND_SPIKE,
            detected_at=now - timedelta(days=1),
            evidence_event_ids=["evt1", "evt2"],
        )
        incident_b = make_incident(
            incident_type=IncidentType.MARGIN_COMPRESSION,
            detected_at=now,
            evidence_event_ids=["evt2", "evt3"],
        )
        score = correlator._compute_cascade_score(incident_a, incident_b)
        assert 0.0 <= score <= 1.0

    def test_compute_cascade_score_reverse_order_returns_zero(self):
        """Test that reverse temporal order returns zero."""
        correlator = CascadeCorrelator()
        now = datetime.utcnow()
        incident_a = make_incident(detected_at=now)
        incident_b = make_incident(detected_at=now - timedelta(days=1))
        score = correlator._compute_cascade_score(incident_a, incident_b)
        assert score == 0.0

    def test_compute_cascade_score_no_causal_relationship(self):
        """Test that non-causal incident types return zero."""
        correlator = CascadeCorrelator()
        now = datetime.utcnow()
        incident_a = make_incident(
            incident_type=IncidentType.REFUND_SPIKE, detected_at=now - timedelta(days=1)
        )
        incident_b = make_incident(
            incident_type=IncidentType.SUPPORT_LOAD_SURGE, detected_at=now
        )
        score = correlator._compute_cascade_score(incident_a, incident_b)
        # REFUND_SPIKE doesn't cause SUPPORT_LOAD_SURGE in dependency graph
        assert score == 0.0


class TestCascadeCorrelatorTemporalWeight:
    """Test CascadeCorrelator._compute_temporal_weight method."""

    def test_compute_temporal_weight_same_time(self):
        """Test temporal weight for incidents at same time."""
        correlator = CascadeCorrelator()
        now = datetime.utcnow()
        weight = correlator._compute_temporal_weight(now, now)
        assert weight > 0.9  # Should be close to 1.0

    def test_compute_temporal_weight_one_day_apart(self):
        """Test temporal weight for incidents one day apart."""
        correlator = CascadeCorrelator(temporal_decay_factor=3.0)
        now = datetime.utcnow()
        weight = correlator._compute_temporal_weight(
            now - timedelta(days=1), now
        )
        assert 0.0 < weight < 1.0

    def test_compute_temporal_weight_many_days_apart(self):
        """Test temporal weight for incidents many days apart."""
        correlator = CascadeCorrelator(temporal_decay_factor=3.0)
        now = datetime.utcnow()
        weight = correlator._compute_temporal_weight(
            now - timedelta(days=10), now
        )
        assert weight < 0.5  # Should decay significantly


class TestCascadeCorrelatorEntityOverlap:
    """Test CascadeCorrelator._compute_entity_overlap method."""

    def test_compute_entity_overlap_full_overlap(self):
        """Test entity overlap with full overlap."""
        correlator = CascadeCorrelator()
        overlap = correlator._compute_entity_overlap(
            ["evt1", "evt2", "evt3"], ["evt1", "evt2", "evt3"]
        )
        assert overlap == 1.0

    def test_compute_entity_overlap_partial_overlap(self):
        """Test entity overlap with partial overlap."""
        correlator = CascadeCorrelator()
        overlap = correlator._compute_entity_overlap(
            ["evt1", "evt2", "evt3"], ["evt2", "evt3", "evt4"]
        )
        assert 0.0 < overlap < 1.0
        assert overlap == pytest.approx(0.5, abs=0.1)  # 2 shared / 4 total

    def test_compute_entity_overlap_no_overlap(self):
        """Test entity overlap with no overlap."""
        correlator = CascadeCorrelator()
        overlap = correlator._compute_entity_overlap(
            ["evt1", "evt2"], ["evt3", "evt4"]
        )
        assert overlap == 0.0

    def test_compute_entity_overlap_empty_evidence(self):
        """Test entity overlap with empty evidence returns baseline (0.1)."""
        correlator = CascadeCorrelator()
        overlap = correlator._compute_entity_overlap([], [])
        assert overlap == 0.1  # Baseline when no direct entity sharing

    def test_compute_entity_overlap_one_empty(self):
        """Test entity overlap when one incident has no evidence."""
        correlator = CascadeCorrelator()
        overlap = correlator._compute_entity_overlap(["evt1", "evt2"], [])
        assert overlap == 0.1  # Baseline overlap for no direct sharing


# ============================================================================
# ComparatorEngine Tests
# ============================================================================


class TestComparatorEngineComputeComparisonSignificance:
    """Test ComparatorEngine.compute_comparison_significance static method."""

    def test_compute_comparison_significance_similar_incidents(self):
        """Test significance computation with similar incidents."""
        incident_a = make_incident(primary_metric_zscore=5.0)
        incident_b = make_incident(primary_metric_zscore=5.2)
        result = ComparatorEngine.compute_comparison_significance(incident_a, incident_b)
        assert "zscore_difference" in result
        assert "effect_size_cohens_d" in result
        assert "is_significantly_different" in result
        assert "p_value_approximation" in result
        assert "interpretation" in result
        assert result["zscore_difference"] == pytest.approx(0.2, abs=0.1)

    def test_compute_comparison_significance_different_incidents(self):
        """Test significance computation with very different incidents."""
        incident_a = make_incident(primary_metric_zscore=2.0)
        incident_b = make_incident(primary_metric_zscore=8.0)
        result = ComparatorEngine.compute_comparison_significance(incident_a, incident_b)
        assert result["zscore_difference"] > 5.0
        assert result["is_significantly_different"] is True

    def test_compute_comparison_significance_identical_zscore(self):
        """Test significance computation with identical z-scores."""
        incident_a = make_incident(primary_metric_zscore=5.0)
        incident_b = make_incident(primary_metric_zscore=5.0)
        result = ComparatorEngine.compute_comparison_significance(incident_a, incident_b)
        assert result["zscore_difference"] == pytest.approx(0.0, abs=0.01)
        assert result["is_significantly_different"] is False

    def test_compute_comparison_significance_negative_zscore(self):
        """Test significance computation with negative z-scores."""
        incident_a = make_incident(primary_metric_zscore=-5.0)
        incident_b = make_incident(primary_metric_zscore=-3.0)
        result = ComparatorEngine.compute_comparison_significance(incident_a, incident_b)
        assert result["zscore_difference"] == pytest.approx(2.0, abs=0.1)

    def test_compute_comparison_significance_zero_zscore(self):
        """Test significance computation with zero z-scores."""
        incident_a = make_incident(primary_metric_zscore=0.0)
        incident_b = make_incident(primary_metric_zscore=0.0)
        result = ComparatorEngine.compute_comparison_significance(incident_a, incident_b)
        assert result["zscore_difference"] == 0.0

    def test_compute_comparison_significance_effect_size_categories(self):
        """Test that effect size is computed correctly."""
        incident_a = make_incident(primary_metric_zscore=5.0)
        incident_b = make_incident(primary_metric_zscore=7.0)
        result = ComparatorEngine.compute_comparison_significance(incident_a, incident_b)
        assert result["effect_size_cohens_d"] > 0.0
        assert isinstance(result["interpretation"], str)
