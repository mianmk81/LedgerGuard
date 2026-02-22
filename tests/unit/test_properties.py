"""
Property-based tests using Hypothesis for LedgerGuard BRE project.

Test-automator agent requirements:
- Invariant checking
- Comprehensive edge case discovery
- Reproducible failures

These tests verify mathematical invariants and bounds across the BRE engine
components using property-based testing with Hypothesis strategies.
"""

from datetime import datetime, timedelta
from uuid import uuid4

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, assume, given, settings
from pydantic import ValidationError

from api.engine.blast_radius.impact_scorer import ImpactScorer
from api.engine.comparator import ComparatorEngine
from api.engine.rca.causal_ranker import CausalRanker
from api.models.enums import BlastRadiusSeverity, Confidence, DetectionMethod, IncidentStatus, IncidentType, Severity
from api.models.incidents import Incident
from api.models.rca import CausalNode, CausalPath
from tests.conftest import make_candidate, make_causal_path, make_incident


# =============================================================================
# CausalRanker Property Tests
# =============================================================================


@given(
    anomaly_magnitude=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    temporal_precedence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    graph_proximity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    data_quality_weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_prop_causal_ranker_score_bounds(
    anomaly_magnitude: float,
    temporal_precedence: float,
    graph_proximity: float,
    data_quality_weight: float,
):
    """
    Invariant 1: CausalRanker contribution_score must be in [0.0, 1.0] for ANY valid candidate.
    
    Property: For any valid candidate dict, contribution_score ∈ [0.0, 1.0]
    """
    ranker = CausalRanker()
    candidate = {
        "metric_name": "test_metric",
        "anomaly_magnitude": anomaly_magnitude,
        "temporal_precedence": temporal_precedence,
        "graph_proximity": graph_proximity,
        "data_quality_weight": data_quality_weight,
    }
    
    score = ranker.compute_contribution_score(candidate)
    
    assert 0.0 <= score <= 1.0, f"Contribution score {score} out of bounds for candidate {candidate}"


@given(
    anomaly_magnitude_low=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    anomaly_magnitude_high=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    temporal_precedence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    graph_proximity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    data_quality_weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_prop_causal_ranker_monotonicity(
    anomaly_magnitude_low: float,
    anomaly_magnitude_high: float,
    temporal_precedence: float,
    graph_proximity: float,
    data_quality_weight: float,
):
    """
    Invariant 2: Higher anomaly_magnitude should produce higher contribution scores
    (when other factors are equal).
    
    Property: If anomaly_magnitude_high > anomaly_magnitude_low and all other factors equal,
              then score_high >= score_low
    """
    assume(anomaly_magnitude_high > anomaly_magnitude_low)
    
    ranker = CausalRanker()
    
    candidate_low = {
        "metric_name": "test_metric",
        "anomaly_magnitude": anomaly_magnitude_low,
        "temporal_precedence": temporal_precedence,
        "graph_proximity": graph_proximity,
        "data_quality_weight": data_quality_weight,
    }
    
    candidate_high = {
        "metric_name": "test_metric",
        "anomaly_magnitude": anomaly_magnitude_high,
        "temporal_precedence": temporal_precedence,
        "graph_proximity": graph_proximity,
        "data_quality_weight": data_quality_weight,
    }
    
    score_low = ranker.compute_contribution_score(candidate_low)
    score_high = ranker.compute_contribution_score(candidate_high)
    
    assert score_high >= score_low, (
        f"Monotonicity violated: anomaly_magnitude {anomaly_magnitude_high} > {anomaly_magnitude_low} "
        f"but score {score_high} < {score_low}"
    )


@given(
    w_anomaly=st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
    w_temporal=st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
    w_proximity=st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
    w_quality=st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_prop_causal_ranker_weight_validation(
    w_anomaly: float,
    w_temporal: float,
    w_proximity: float,
    w_quality: float,
):
    """
    Invariant 3: CausalRanker weights that don't sum to ~1.0 must raise ValueError.
    
    Property: If |sum(weights) - 1.0| > 0.01, then CausalRanker.__init__ raises ValueError
    """
    total = w_anomaly + w_temporal + w_proximity + w_quality
    
    if abs(total - 1.0) > 0.01:
        with pytest.raises(ValueError, match="weights must sum to"):
            CausalRanker(
                w_anomaly=w_anomaly,
                w_temporal=w_temporal,
                w_proximity=w_proximity,
                w_quality=w_quality,
            )
    else:
        # Valid weights should not raise
        ranker = CausalRanker(
            w_anomaly=w_anomaly,
            w_temporal=w_temporal,
            w_proximity=w_proximity,
            w_quality=w_quality,
        )
        assert ranker is not None


@given(
    scores=st.lists(
        st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10,
    ),
)
@settings(max_examples=100)
def test_prop_relative_importance_sums_to_100(scores: list[float]):
    """
    Invariant 4: compute_relative_importance percentages should sum to approximately 100%.
    
    Property: For any list of CausalPaths with positive scores, sum ≈ 100.0
    """
    
    ranker = CausalRanker()
    paths = [make_causal_path(rank=i + 1, score=s, metric_name=f"metric_{i}") for i, s in enumerate(scores)]
    importance = ranker.compute_relative_importance(paths)
    
    if importance:
        total_pct = sum(importance.values())
        # Allow rounding error from round(pct, 1) - can sum to 99.9 or 100.1
        assert abs(total_pct - 100.0) <= 1.0, (
            f"Relative importance percentages sum to {total_pct}, expected ~100.0"
        )


@given(
    value=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    midpoint=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    steepness=st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_prop_sigmoid_normalization_bounds(
    value: float,
    midpoint: float,
    steepness: float,
):
    """
    Invariant 5: _sigmoid_normalize always returns values in [0.0, 1.0].
    
    Property: For any non-negative value, midpoint, steepness, sigmoid_normalize ∈ [0.0, 1.0]
    """
    ranker = CausalRanker()
    normalized = ranker._sigmoid_normalize(value, midpoint=midpoint, steepness=steepness)
    
    assert 0.0 <= normalized <= 1.0, (
        f"Sigmoid normalization out of bounds: {normalized} for value={value}, "
        f"midpoint={midpoint}, steepness={steepness}"
    )


# =============================================================================
# BlastRadiusSeverity Property Tests
# =============================================================================


@given(
    customers_low=st.integers(min_value=0, max_value=10000),
    customers_high=st.integers(min_value=0, max_value=10000),
    revenue_low=st.floats(min_value=0.0, max_value=1000000.0, allow_nan=False, allow_infinity=False),
    revenue_high=st.floats(min_value=0.0, max_value=1000000.0, allow_nan=False, allow_infinity=False),
    refunds_low=st.floats(min_value=0.0, max_value=500000.0, allow_nan=False, allow_infinity=False),
    refunds_high=st.floats(min_value=0.0, max_value=500000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
def test_prop_blast_radius_severity_ordering(
    customers_low: int,
    customers_high: int,
    revenue_low: float,
    revenue_high: float,
    refunds_low: float,
    refunds_high: float,
):
    """
    Invariant 6: Higher impact metrics should never produce lower severity.
    
    Property: If all impact metrics are >= for high vs low, then severity_high >= severity_low
    """
    assume(
        customers_high >= customers_low
        and revenue_high >= revenue_low
        and refunds_high >= refunds_low
    )
    
    scorer = ImpactScorer()
    
    severity_low = scorer.classify_severity(
        customers_affected=customers_low,
        revenue_exposure=revenue_low,
        refund_exposure=refunds_low,
        downstream_count=0,
    )
    
    severity_high = scorer.classify_severity(
        customers_affected=customers_high,
        revenue_exposure=revenue_high,
        refund_exposure=refunds_high,
        downstream_count=0,
    )
    
    # Define severity ordering
    severity_order = {
        BlastRadiusSeverity.CONTAINED: 1,
        BlastRadiusSeverity.SIGNIFICANT: 2,
        BlastRadiusSeverity.SEVERE: 3,
        BlastRadiusSeverity.CATASTROPHIC: 4,
    }
    
    assert severity_order[severity_high] >= severity_order[severity_low], (
        f"Severity ordering violated: high impact ({customers_high}, {revenue_high}, {refunds_high}) "
        f"→ {severity_high} but low impact ({customers_low}, {revenue_low}, {refunds_low}) → {severity_low}"
    )


# =============================================================================
# Incident Model Property Tests
# =============================================================================


@given(
    data_quality_score=st.floats(min_value=-1.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    zscore=st.floats(min_value=-150.0, max_value=150.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_prop_incident_model_validation(
    data_quality_score: float,
    zscore: float,
):
    """
    Invariant 7: Incident model validation - data_quality_score must be in [0.0, 1.0],
    z-score within [-100, 100].
    
    Property: Incident creation validates bounds and raises ValidationError if violated
    """
    # Test data_quality_score validation (Pydantic Field ge/le or custom validator)
    if not (0.0 <= data_quality_score <= 1.0):
        with pytest.raises(ValidationError, match="data_quality|quality|greater than or equal|0.0"):
            make_incident(
                data_quality_score=data_quality_score,
                primary_metric_zscore=5.0,
            )
    else:
        incident = make_incident(
            data_quality_score=data_quality_score,
            primary_metric_zscore=5.0,
        )
        assert incident.data_quality_score == round(data_quality_score, 4)

    # Test z-score validation
    if abs(zscore) > 100:
        with pytest.raises(ValidationError, match="zscore|Z-score|exceeds|reasonable|100"):
            make_incident(
                data_quality_score=0.95,
                primary_metric_zscore=zscore,
            )
    else:
        incident = make_incident(
            data_quality_score=0.95,
            primary_metric_zscore=zscore,
        )
        assert abs(incident.primary_metric_zscore) <= 100


# =============================================================================
# CausalNode Property Tests
# =============================================================================


@given(
    contribution_score=st.floats(min_value=-0.5, max_value=1.5, allow_nan=False, allow_infinity=False),
    temporal_precedence=st.floats(min_value=-0.5, max_value=1.5, allow_nan=False, allow_infinity=False),
    graph_proximity=st.floats(min_value=-0.5, max_value=1.5, allow_nan=False, allow_infinity=False),
    data_quality_weight=st.floats(min_value=-0.5, max_value=1.5, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_prop_causal_node_score_validation(
    contribution_score: float,
    temporal_precedence: float,
    graph_proximity: float,
    data_quality_weight: float,
):
    """
    Invariant 8: CausalNode score validation - all scores must be in [0.0, 1.0].
    
    Property: CausalNode creation validates bounds and raises ValueError if violated
    """
    now = datetime.utcnow()
    
    # Check each score field
    scores_to_check = [
        ("contribution_score", contribution_score),
        ("temporal_precedence", temporal_precedence),
        ("graph_proximity", graph_proximity),
        ("data_quality_weight", data_quality_weight),
    ]
    
    for field_name, score_value in scores_to_check:
        if not (0.0 <= score_value <= 1.0):
            with pytest.raises(ValidationError, match="Score must be between|0.0 and 1.0|greater than or equal|less than or equal"):
                CausalNode(
                    metric_name="test_metric",
                    contribution_score=contribution_score if field_name == "contribution_score" else 0.5,
                    anomaly_magnitude=5.0,
                    temporal_precedence=temporal_precedence if field_name == "temporal_precedence" else 0.5,
                    graph_proximity=graph_proximity if field_name == "graph_proximity" else 0.5,
                    data_quality_weight=data_quality_weight if field_name == "data_quality_weight" else 0.5,
                    metric_value=0.25,
                    metric_baseline=0.05,
                    metric_zscore=5.0,
                    anomaly_window=(now - timedelta(hours=6), now),
                    evidence_clusters=[],
                )
        else:
            # Valid score should work
            node = CausalNode(
                metric_name="test_metric",
                contribution_score=contribution_score if field_name == "contribution_score" else 0.5,
                anomaly_magnitude=5.0,
                temporal_precedence=temporal_precedence if field_name == "temporal_precedence" else 0.5,
                graph_proximity=graph_proximity if field_name == "graph_proximity" else 0.5,
                data_quality_weight=data_quality_weight if field_name == "data_quality_weight" else 0.5,
                metric_value=0.25,
                metric_baseline=0.05,
                metric_zscore=5.0,
                anomaly_window=(now - timedelta(hours=6), now),
                evidence_clusters=[],
            )
            assert node is not None


# =============================================================================
# TemporalCorrelator Property Tests
# =============================================================================


@given(
    series_length=st.integers(min_value=3, max_value=100),
    values=st.lists(
        st.tuples(
            st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=3,
        max_size=100,
    ),
)
@settings(max_examples=100)
def test_prop_temporal_correlator_range(series_length: int, values: list[tuple[float, float]]):
    """
    Invariant 9: compute_temporal_precedence results always in valid range.
    
    Property: precedence_score ∈ [0.0, 1.0], optimal_lag_days is integer,
              max_correlation ∈ [-1.0, 1.0]
    """
    from api.engine.rca.temporal_correlation import TemporalCorrelator

    candidate_values = [v[0] for v in values]
    incident_values = [v[1] for v in values]
    correlator = TemporalCorrelator(max_lag_days=14, min_correlation=0.3)

    try:
        result = correlator.compute_temporal_precedence(
            candidate_series=candidate_values,
            incident_series=incident_values,
        )
        assert 0.0 <= result["precedence_score"] <= 1.0, (
            f"Precedence score {result['precedence_score']} out of bounds"
        )
        assert isinstance(result["optimal_lag_days"], int), (
            f"Optimal lag {result['optimal_lag_days']} is not an integer"
        )
        assert -1.0 <= result["max_correlation"] <= 1.0, (
            f"Max correlation {result['max_correlation']} out of bounds"
        )
        assert result["is_significant"] in (True, False), "is_significant must be boolean"
    except ValueError:
        pass


# =============================================================================
# ComparatorEngine Property Tests
# =============================================================================


@given(
    zscore_a=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    zscore_b=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_prop_comparison_significance(
    zscore_a: float,
    zscore_b: float,
):
    """
    Invariant 10: Comparison significance - p_value always in [0.0, 1.0],
    Cohen's d always >= 0.
    
    Property: compute_comparison_significance returns valid statistical measures
    """
    assume(abs(zscore_a) <= 100 and abs(zscore_b) <= 100)
    
    incident_a = make_incident(
        primary_metric_zscore=zscore_a,
        data_quality_score=0.95,
    )
    incident_b = make_incident(
        primary_metric_zscore=zscore_b,
        data_quality_score=0.95,
    )
    
    result = ComparatorEngine.compute_comparison_significance(incident_a, incident_b)
    
    assert 0.0 <= result["p_value_approximation"] <= 1.0, (
        f"P-value {result['p_value_approximation']} out of bounds"
    )
    assert result["effect_size_cohens_d"] >= 0.0, (
        f"Cohen's d {result['effect_size_cohens_d']} is negative"
    )
    assert isinstance(result["is_significantly_different"], bool)
    assert isinstance(result["zscore_difference"], float)
    assert isinstance(result["interpretation"], str)


# =============================================================================
# Additional Property Tests
# =============================================================================


@given(
    n_candidates=st.integers(min_value=1, max_value=20),
    top_k=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=50)
def test_prop_causal_ranker_ranking_order(
    n_candidates: int,
    top_k: int,
):
    """
    Additional invariant: Ranked candidates should be in descending order by contribution score.
    
    Property: For any list of candidates, rank_candidates returns paths sorted descending
    """
    assume(top_k <= n_candidates)
    
    ranker = CausalRanker()
    candidates = []
    
    for i in range(n_candidates):
        candidate = make_candidate(
            metric_name=f"metric_{i}",
            anomaly_magnitude=float(i + 1),  # Vary anomaly magnitude
            temporal_precedence=0.5,
            graph_proximity=0.5,
            data_quality_weight=0.9,
        )
        candidates.append(candidate)
    
    paths = ranker.rank_candidates(
        candidates=candidates,
        incident_metric="test_incident",
        top_k=top_k,
    )
    
    # Check descending order
    if len(paths) > 1:
        for i in range(len(paths) - 1):
            assert paths[i].overall_score >= paths[i + 1].overall_score, (
                f"Ranking order violated: path {i} score {paths[i].overall_score} < "
                f"path {i+1} score {paths[i+1].overall_score}"
            )


@given(
    score=st.floats(min_value=-0.5, max_value=1.5, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_prop_causal_path_score_bounds(score: float):
    """
    Additional invariant: CausalPath overall_score must be in [0.0, 1.0].
    
    Property: CausalPath creation validates overall_score bounds
    """
    if not (0.0 <= score <= 1.0):
        with pytest.raises(ValidationError, match="Overall score must be between|0.0 and 1.0|greater than or equal|less than or equal"):
            CausalPath(
                rank=1,
                overall_score=score,
                nodes=[make_causal_path(rank=1, score=0.5).nodes[0]],
            )
    else:
        path = CausalPath(
            rank=1,
            overall_score=score,
            nodes=[make_causal_path(rank=1, score=0.5).nodes[0]],
        )
        assert path.overall_score == round(score, 4)


@given(
    customers=st.integers(min_value=0, max_value=10000),
    revenue=st.floats(min_value=0.0, max_value=1000000.0, allow_nan=False, allow_infinity=False),
    refunds=st.floats(min_value=0.0, max_value=500000.0, allow_nan=False, allow_infinity=False),
    downstream=st.integers(min_value=0, max_value=10),
)
@settings(max_examples=100)
def test_prop_blast_radius_severity_bounds(
    customers: int,
    revenue: float,
    refunds: float,
    downstream: int,
):
    """
    Additional invariant: BlastRadiusSeverity classification always returns valid enum value.
    
    Property: classify_severity always returns one of the four severity levels
    """
    scorer = ImpactScorer()
    severity = scorer.classify_severity(
        customers_affected=customers,
        revenue_exposure=revenue,
        refund_exposure=refunds,
        downstream_count=downstream,
    )
    
    assert severity in [
        BlastRadiusSeverity.CONTAINED,
        BlastRadiusSeverity.SIGNIFICANT,
        BlastRadiusSeverity.SEVERE,
        BlastRadiusSeverity.CATASTROPHIC,
    ], f"Invalid severity value: {severity}"


@given(
    w1=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    w2=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    w3=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    w4=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_prop_causal_ranker_weight_sum_normalization(
    w1: float,
    w2: float,
    w3: float,
    w4: float,
):
    """
    Additional invariant: CausalRanker weights are validated and normalized correctly.
    
    Property: Weights summing to 1.0 ± 0.01 are accepted, others rejected
    """
    total = w1 + w2 + w3 + w4
    
    if abs(total - 1.0) <= 0.01:
        # Should succeed
        ranker = CausalRanker(w_anomaly=w1, w_temporal=w2, w_proximity=w3, w_quality=w4)
        assert ranker is not None
    else:
        # Should raise ValueError
        with pytest.raises(ValueError, match="weights must sum"):
            CausalRanker(w_anomaly=w1, w_temporal=w2, w_proximity=w3, w_quality=w4)


@given(
    values=st.lists(
        st.tuples(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=3,
        max_size=50,
    ),
)
@settings(max_examples=50)
def test_prop_temporal_correlator_result_structure(values: list[tuple[float, float]]):
    """
    Additional invariant: Temporal correlator result structure is always valid.
    
    Property: compute_temporal_precedence always returns dict with required keys
    """
    from api.engine.rca.temporal_correlation import TemporalCorrelator

    candidate_series = [v[0] for v in values]
    incident_series = [v[1] for v in values]
    correlator = TemporalCorrelator(max_lag_days=14, min_correlation=0.3)
    
    try:
        result = correlator.compute_temporal_precedence(
            candidate_series=candidate_series,
            incident_series=incident_series,
        )
        required_keys = [
            "precedence_score",
            "optimal_lag_days",
            "max_correlation",
            "is_significant",
            "correlation_at_lag",
        ]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
        assert isinstance(result["precedence_score"], float)
        assert isinstance(result["optimal_lag_days"], int)
        assert isinstance(result["max_correlation"], float)
        assert result["is_significant"] in (True, False)
        assert isinstance(result["correlation_at_lag"], dict)
    except ValueError:
        pass
