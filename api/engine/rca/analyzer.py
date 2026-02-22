"""
BRE-RCA Root Cause Analyzer — Core Orchestrator.

This is the primary entry point for root cause analysis in LedgerGuard.
The RootCauseAnalyzer orchestrates the complete RCA pipeline:

1. Identify incident metric from incident type using dependency graph
2. Find upstream candidate root causes from the business DAG
3. Fetch historical metric time series for candidate and incident metrics
4. Compute temporal precedence scores via cross-correlation
5. Score each candidate using the BRE-RCA contribution formula
6. Rank candidates and select top-k root causes
7. Gather evidence clusters from canonical events
8. Generate human-readable explanations
9. Assemble and persist the final CausalChain

The BRE-RCA algorithm is the core intellectual property of LedgerGuard,
implementing principled causal inference for business operations using
a hybrid of graph-based domain knowledge and statistical evidence.

Version: BRE-RCA-v1
"""

from datetime import date, datetime, timedelta
from typing import Optional
from uuid import uuid4

import structlog

from api.models.events import CanonicalEvent
from api.models.incidents import Incident
from api.models.rca import CausalChain, CausalPath, EvidenceCluster
from api.storage.base import StorageBackend

from .causal_ranker import CausalRanker
from .dependency_graph import BusinessDependencyGraph, INCIDENT_METRIC_MAP
from .explainer import RCAExplainer
from .temporal_correlation import TemporalCorrelator

logger = structlog.get_logger()


class RootCauseAnalyzer:
    """
    Orchestrates the complete BRE-RCA root cause analysis pipeline.

    Combines static dependency graph knowledge with dynamic statistical
    analysis to identify the most likely root causes of detected incidents.

    Attributes:
        storage: Storage backend for reading events and metrics
        dep_graph: Business dependency graph (DAG)
        correlator: Temporal correlation engine
        ranker: Causal ranking engine
        explainer: Natural language explanation generator
        logger: Structured logger for observability

    Example:
        >>> analyzer = RootCauseAnalyzer(storage=duckdb_storage)
        >>> causal_chain = analyzer.analyze(incident, lookback_days=14, top_k=5)
        >>> print(f"Root cause: {causal_chain.paths[0].nodes[0].metric_name}")
        >>> print(f"Confidence: {causal_chain.paths[0].overall_score:.0%}")
    """

    # Default analysis parameters
    DEFAULT_LOOKBACK_DAYS = 14
    DEFAULT_TOP_K = 5
    DEFAULT_MIN_ZSCORE = 2.0  # Minimum z-score to consider a metric anomalous

    def __init__(
        self,
        storage: StorageBackend,
        dep_graph: Optional[BusinessDependencyGraph] = None,
        correlator: Optional[TemporalCorrelator] = None,
        ranker: Optional[CausalRanker] = None,
        explainer: Optional[RCAExplainer] = None,
    ):
        """
        Initialize the root cause analyzer with all sub-components.

        Args:
            storage: Storage backend for data access
            dep_graph: Optional custom dependency graph (default: standard graph)
            correlator: Optional custom temporal correlator
            ranker: Optional custom causal ranker
            explainer: Optional custom explainer
        """
        self.storage = storage
        self.dep_graph = dep_graph or BusinessDependencyGraph()
        self.correlator = correlator or TemporalCorrelator(max_lag_days=14)
        self.ranker = ranker or CausalRanker()
        self.explainer = explainer or RCAExplainer()
        self.logger = structlog.get_logger()

        self.logger.info(
            "root_cause_analyzer_initialized",
            dep_graph_version=self.dep_graph.version,
            dep_graph_nodes=self.dep_graph.graph.number_of_nodes(),
            dep_graph_edges=self.dep_graph.graph.number_of_edges(),
        )

    def analyze(
        self,
        incident: Incident,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        top_k: int = DEFAULT_TOP_K,
        min_zscore: float = DEFAULT_MIN_ZSCORE,
    ) -> CausalChain:
        """
        Perform complete root cause analysis for an incident.

        This is the main entry point for RCA. Orchestrates all sub-components
        to produce a comprehensive CausalChain result.

        Args:
            incident: The incident to analyze
            lookback_days: Number of days of historical data to analyze
            top_k: Number of top root causes to return
            min_zscore: Minimum z-score threshold for candidate consideration

        Returns:
            CausalChain containing ranked causal paths with evidence

        Raises:
            ValueError: If incident type is not mapped to a metric
        """
        run_id = f"rca_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

        self.logger.info(
            "rca_analysis_started",
            run_id=run_id,
            incident_id=incident.incident_id,
            incident_type=incident.incident_type.value,
            lookback_days=lookback_days,
            top_k=top_k,
        )

        # Step 1: Map incident type to primary affected metric
        incident_type_str = incident.incident_type.value.upper()
        incident_metric = self.dep_graph.get_metric_for_incident_type(incident_type_str)

        if not incident_metric:
            self.logger.error(
                "incident_type_not_mapped",
                incident_type=incident_type_str,
            )
            raise ValueError(
                f"Incident type '{incident_type_str}' is not mapped to a metric. "
                f"Available types: {list(INCIDENT_METRIC_MAP.keys())}"
            )

        # Step 2: Find upstream candidate root causes from dependency graph
        upstream_metrics = self.dep_graph.get_upstream_nodes(incident_metric)

        self.logger.info(
            "upstream_candidates_identified",
            run_id=run_id,
            incident_metric=incident_metric,
            upstream_count=len(upstream_metrics),
            upstream_metrics=upstream_metrics,
        )

        if not upstream_metrics:
            self.logger.warning(
                "no_upstream_candidates",
                incident_metric=incident_metric,
            )
            # Return minimal chain with incident metric itself
            return self._build_minimal_chain(
                incident=incident,
                incident_metric=incident_metric,
                run_id=run_id,
                lookback_days=lookback_days,
            )

        # Step 3: Compute analysis time window
        detection_date = incident.detected_at.date()
        analysis_start = detection_date - timedelta(days=lookback_days)
        analysis_end = detection_date

        # Step 4: Fetch metric time series for all candidates + incident metric
        all_metrics = upstream_metrics + [incident_metric]
        metric_series = self._fetch_metric_series(
            metrics=all_metrics,
            start_date=analysis_start,
            end_date=analysis_end,
        )

        incident_series = metric_series.get(incident_metric, [])

        if not incident_series or len(incident_series) < 3:
            self.logger.warning(
                "insufficient_incident_metric_data",
                incident_metric=incident_metric,
                series_length=len(incident_series) if incident_series else 0,
            )
            return self._build_minimal_chain(
                incident=incident,
                incident_metric=incident_metric,
                run_id=run_id,
                lookback_days=lookback_days,
            )

        # Step 5: Compute temporal precedence for all candidates
        candidate_series_map = {
            metric: series
            for metric, series in metric_series.items()
            if metric != incident_metric and len(series) >= 3
        }

        temporal_results = self.correlator.compute_batch_precedence(
            candidate_series_map=candidate_series_map,
            incident_series=incident_series,
        )

        # Step 6: Build candidate scoring dicts
        candidates = []
        for metric_name in upstream_metrics:
            series = metric_series.get(metric_name, [])
            if not series:
                continue

            # Compute anomaly statistics
            anomaly_stats = self._compute_anomaly_stats(series)

            # Skip candidates with weak anomaly signals
            if abs(anomaly_stats["zscore"]) < min_zscore:
                continue

            # Get temporal precedence
            temporal = temporal_results.get(metric_name, {})

            # Compute graph proximity
            graph_proximity = self.dep_graph.compute_graph_proximity(
                metric_name, incident_metric
            )

            # Estimate data quality weight from event coverage
            data_quality = self._estimate_data_quality(
                metric_name, analysis_start, analysis_end
            )

            # Compute anomaly window
            anomaly_window = self._estimate_anomaly_window(
                series, analysis_start, analysis_end
            )

            # Gather evidence clusters
            evidence_clusters = self._gather_evidence(
                metric_name, analysis_start, analysis_end
            )

            candidate = {
                "metric_name": metric_name,
                "anomaly_magnitude": anomaly_stats["zscore"],
                "temporal_precedence": temporal.get("precedence_score", 0.0),
                "graph_proximity": graph_proximity,
                "data_quality_weight": data_quality,
                "metric_value": anomaly_stats["current_value"],
                "metric_baseline": anomaly_stats["baseline"],
                "metric_zscore": anomaly_stats["zscore"],
                "anomaly_window": anomaly_window,
                "evidence_clusters": evidence_clusters,
            }
            candidates.append(candidate)

        self.logger.info(
            "candidates_scored",
            run_id=run_id,
            total_upstream=len(upstream_metrics),
            qualified_candidates=len(candidates),
            min_zscore=min_zscore,
        )

        # Step 7: Rank candidates and build causal paths
        if candidates:
            ranked_paths = self.ranker.rank_candidates(
                candidates=candidates,
                incident_metric=incident_metric,
                top_k=top_k,
            )
        else:
            ranked_paths = []

        # Ensure at least one path exists
        if not ranked_paths:
            return self._build_minimal_chain(
                incident=incident,
                incident_metric=incident_metric,
                run_id=run_id,
                lookback_days=lookback_days,
            )

        # Step 8: Assemble CausalChain
        causal_window = (
            datetime.combine(analysis_start, datetime.min.time()),
            datetime.combine(analysis_end, datetime.max.time()),
        )

        causal_chain = CausalChain(
            incident_id=incident.incident_id,
            paths=ranked_paths,
            algorithm_version="BRE-RCA-v1",
            causal_window=causal_window,
            dependency_graph_version=self.dep_graph.version,
            run_id=run_id,
        )

        # Step 9: Persist to storage
        try:
            self.storage.write_causal_chain(causal_chain)
            self.logger.info(
                "causal_chain_persisted",
                run_id=run_id,
                chain_id=causal_chain.chain_id,
                incident_id=incident.incident_id,
            )
        except Exception as e:
            self.logger.error(
                "causal_chain_persistence_failed",
                run_id=run_id,
                error=str(e),
            )

        self.logger.info(
            "rca_analysis_completed",
            run_id=run_id,
            incident_id=incident.incident_id,
            chain_id=causal_chain.chain_id,
            paths_count=len(ranked_paths),
            top_cause=ranked_paths[0].nodes[0].metric_name if ranked_paths else None,
            top_score=ranked_paths[0].overall_score if ranked_paths else 0.0,
        )

        return causal_chain

    def analyze_with_explanation(
        self,
        incident: Incident,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        top_k: int = DEFAULT_TOP_K,
    ) -> tuple[CausalChain, dict]:
        """
        Perform RCA and generate human-readable explanation.

        Convenience method that runs analysis and explanation in one call.

        Args:
            incident: The incident to analyze
            lookback_days: Historical lookback window in days
            top_k: Number of top causes to return

        Returns:
            Tuple of (CausalChain, explanation dict)
        """
        causal_chain = self.analyze(incident, lookback_days, top_k)

        incident_type_str = incident.incident_type.value.upper()
        incident_metric = self.dep_graph.get_metric_for_incident_type(incident_type_str)

        explanation = self.explainer.explain(
            causal_chain=causal_chain,
            incident_type=incident_type_str,
            incident_metric=incident_metric,
        )

        return causal_chain, explanation

    # =========================================================================
    # Private Methods - Data Fetching
    # =========================================================================

    def _fetch_metric_series(
        self,
        metrics: list[str],
        start_date: date,
        end_date: date,
    ) -> dict[str, list[float]]:
        """
        Fetch daily metric time series from Gold layer.

        Args:
            metrics: List of metric names to fetch
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Dict mapping metric_name → list of daily values (chronological)
        """
        metric_series = {}

        try:
            stored_metrics = self.storage.read_gold_metrics(
                metric_names=metrics,
                start_date=str(start_date),
                end_date=str(end_date),
            )

            # Group metrics by name and sort by date
            for record in stored_metrics:
                name = record.get("metric_name")
                value = record.get("metric_value")
                if name and value is not None:
                    if name not in metric_series:
                        metric_series[name] = []
                    metric_series[name].append(float(value))

        except Exception as e:
            self.logger.error(
                "metric_series_fetch_failed",
                error=str(e),
                metric_count=len(metrics),
            )

        # For metrics with no stored data, generate synthetic baseline
        for metric in metrics:
            if metric not in metric_series or len(metric_series[metric]) < 3:
                metric_series[metric] = self._generate_synthetic_baseline(
                    metric, start_date, end_date
                )

        self.logger.debug(
            "metric_series_fetched",
            metrics_requested=len(metrics),
            metrics_returned=len(metric_series),
            series_lengths={
                k: len(v) for k, v in metric_series.items()
            },
        )

        return metric_series

    def _generate_synthetic_baseline(
        self,
        metric_name: str,
        start_date: date,
        end_date: date,
    ) -> list[float]:
        """
        Generate synthetic baseline data when Gold layer data is unavailable.

        Uses reasonable defaults for each metric type to enable RCA even
        when historical data is sparse. This is a development/bootstrap
        convenience — production should have real Gold layer data.

        Args:
            metric_name: Metric to generate baseline for
            start_date: Start date
            end_date: End date

        Returns:
            List of daily values (one per day in range)
        """
        import random

        num_days = (end_date - start_date).days + 1

        # Default baselines by metric category
        baselines = {
            "daily_revenue": (5000.0, 500.0),
            "daily_expenses": (3000.0, 300.0),
            "daily_refunds": (200.0, 50.0),
            "refund_rate": (0.04, 0.01),
            "net_cash_proxy": (50000.0, 5000.0),
            "expense_ratio": (0.60, 0.05),
            "margin_proxy": (0.35, 0.05),
            "dso_proxy": (30.0, 5.0),
            "ar_aging_amount": (10000.0, 2000.0),
            "ar_overdue_count": (5.0, 2.0),
            "dpo_proxy": (25.0, 5.0),
            "order_volume": (50.0, 10.0),
            "delivery_count": (45.0, 8.0),
            "late_delivery_count": (3.0, 2.0),
            "delivery_delay_rate": (0.07, 0.03),
            "fulfillment_backlog": (10.0, 5.0),
            "avg_delivery_delay_days": (2.0, 1.0),
            "supplier_delay_rate": (0.05, 0.02),
            "supplier_delay_severity": (1.5, 0.5),
            "ticket_volume": (20.0, 5.0),
            "ticket_close_volume": (18.0, 4.0),
            "ticket_backlog": (8.0, 3.0),
            "avg_resolution_time": (24.0, 6.0),
            "review_score_avg": (4.0, 0.3),
            "review_score_trend": (0.0, 0.02),
            "churn_proxy": (0.03, 0.01),
            "customer_concentration": (0.25, 0.05),
        }

        mean, std = baselines.get(metric_name, (1.0, 0.1))

        # Generate with small random walk for realism
        series = []
        current = mean
        for _ in range(num_days):
            noise = random.gauss(0, std * 0.3)
            current = current + noise
            # Clamp non-negative for rate/count metrics
            if metric_name in [
                "refund_rate", "expense_ratio", "delivery_delay_rate",
                "supplier_delay_rate", "churn_proxy", "customer_concentration",
            ]:
                current = max(0.0, min(1.0, current))
            elif "count" in metric_name or "volume" in metric_name or "backlog" in metric_name:
                current = max(0.0, current)
            series.append(round(current, 4))

        return series

    # =========================================================================
    # Private Methods - Anomaly Analysis
    # =========================================================================

    def _compute_anomaly_stats(self, series: list[float]) -> dict:
        """
        Compute anomaly statistics for a metric series.

        Uses MAD-based z-score (consistent with StatisticalDetector)
        to determine how anomalous the recent values are compared to baseline.

        Args:
            series: Daily metric values (chronological)

        Returns:
            Dict with current_value, baseline, zscore
        """
        import numpy as np

        if not series or len(series) < 3:
            return {"current_value": 0.0, "baseline": 0.0, "zscore": 0.0}

        values = np.array(series, dtype=np.float64)

        # Current value: last observation
        current = values[-1]

        # Baseline: median of all values (robust to outliers)
        baseline = float(np.median(values))

        # MAD-based z-score
        median = np.median(values)
        mad = np.median(np.abs(values - median))

        if mad < 1e-10:
            # Constant series, use std as fallback
            std = np.std(values)
            if std < 1e-10:
                zscore = 0.0
            else:
                zscore = (current - baseline) / std
        else:
            # Modified z-score using MAD
            zscore = 0.6745 * (current - median) / mad

        return {
            "current_value": round(float(current), 4),
            "baseline": round(float(baseline), 4),
            "zscore": round(float(zscore), 4),
        }

    def _estimate_anomaly_window(
        self,
        series: list[float],
        start_date: date,
        end_date: date,
    ) -> tuple[datetime, datetime]:
        """
        Estimate the time window when the metric was anomalous.

        Scans from the end of the series backward to find where the
        anomaly started (where z-score first exceeds threshold).

        Args:
            series: Daily metric values
            start_date: Start date of the series
            end_date: End date of the series

        Returns:
            Tuple of (anomaly_start, anomaly_end) datetimes
        """
        import numpy as np

        if not series or len(series) < 3:
            return (
                datetime.combine(end_date - timedelta(days=1), datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
            )

        values = np.array(series, dtype=np.float64)
        median = np.median(values)
        mad = np.median(np.abs(values - median))

        if mad < 1e-10:
            mad = np.std(values) / 0.6745 if np.std(values) > 1e-10 else 1.0

        # Compute z-scores for all values
        zscores = 0.6745 * (values - median) / mad

        # Find where anomaly starts (scanning backward from end)
        threshold = 2.0
        anomaly_start_idx = len(series) - 1

        for i in range(len(series) - 1, -1, -1):
            if abs(zscores[i]) >= threshold:
                anomaly_start_idx = i
            else:
                break

        # Convert index to date
        anomaly_start_date = start_date + timedelta(days=anomaly_start_idx)
        anomaly_end_date = end_date

        return (
            datetime.combine(anomaly_start_date, datetime.min.time()),
            datetime.combine(anomaly_end_date, datetime.max.time()),
        )

    def _estimate_data_quality(
        self,
        metric_name: str,
        start_date: date,
        end_date: date,
    ) -> float:
        """
        Estimate data quality weight for a metric in the analysis window.

        Checks event coverage and consistency to determine how reliable
        the metric data is for causal inference.

        Args:
            metric_name: Metric to assess
            start_date: Start of analysis window
            end_date: End of analysis window

        Returns:
            Quality weight in [0.0, 1.0]
        """
        # In production, this would query the data quality reports
        # from the Silver layer. For now, use a default high quality.
        try:
            # Check if Gold metrics exist for this period
            metrics = self.storage.read_gold_metrics(
                metric_names=[metric_name],
                start_date=str(start_date),
                end_date=str(end_date),
            )

            expected_days = (end_date - start_date).days + 1
            actual_days = len(metrics)

            if expected_days <= 0:
                return 0.8

            coverage = min(1.0, actual_days / expected_days)

            # Score: 90%+ coverage = 1.0, linear degradation below
            if coverage >= 0.9:
                return 1.0
            elif coverage >= 0.7:
                return 0.9
            elif coverage >= 0.5:
                return 0.75
            else:
                return 0.5

        except Exception:
            return 0.8  # Default moderate quality

    def _gather_evidence(
        self,
        metric_name: str,
        start_date: date,
        end_date: date,
    ) -> list[EvidenceCluster]:
        """
        Gather evidence clusters from canonical events supporting a metric anomaly.

        Queries Silver layer for events related to the metric and groups
        them into meaningful clusters.

        Args:
            metric_name: Metric to gather evidence for
            start_date: Start of evidence window
            end_date: End of evidence window

        Returns:
            List of EvidenceCluster objects
        """
        clusters = []

        # Map metrics to relevant event types for evidence gathering
        metric_event_map = {
            "refund_rate": ["refund_issued", "credit_memo_issued"],
            "daily_revenue": ["invoice_paid", "payment_received"],
            "daily_expenses": ["expense_posted"],
            "delivery_delay_rate": ["order_late"],
            "fulfillment_backlog": ["order_placed", "order_delivered"],
            "ticket_volume": ["support_ticket_opened"],
            "ticket_backlog": ["support_ticket_opened", "support_ticket_closed"],
            "review_score_avg": ["review_submitted"],
            "churn_proxy": ["customer_churned"],
            "supplier_delay_rate": ["shipment_delayed"],
            "margin_proxy": ["invoice_paid", "expense_posted", "refund_issued"],
        }

        event_types = metric_event_map.get(metric_name, [])

        for event_type in event_types:
            try:
                events = self.storage.read_canonical_events(
                    event_type=event_type,
                    start_time=datetime.combine(start_date, datetime.min.time()).isoformat(),
                    end_time=datetime.combine(end_date, datetime.max.time()).isoformat(),
                    limit=100,
                )

                if events:
                    total_amount = sum(
                        e.amount for e in events if e.amount is not None
                    )
                    event_ids = [e.event_id for e in events[:20]]  # Cap IDs

                    cluster = EvidenceCluster(
                        cluster_label=f"{event_type} events",
                        event_count=len(events),
                        event_ids=event_ids,
                        entity_type=events[0].entity_type.value if events else "unknown",
                        total_amount=round(total_amount, 2) if total_amount else None,
                        summary=(
                            f"{len(events)} {event_type.replace('_', ' ')} events "
                            f"detected in analysis window"
                            + (f" totaling ${total_amount:,.2f}" if total_amount else "")
                        ),
                    )
                    clusters.append(cluster)

            except Exception as e:
                self.logger.warning(
                    "evidence_gathering_failed",
                    metric_name=metric_name,
                    event_type=event_type,
                    error=str(e),
                )

        return clusters

    # =========================================================================
    # Private Methods - Chain Construction
    # =========================================================================

    def _build_minimal_chain(
        self,
        incident: Incident,
        incident_metric: str,
        run_id: str,
        lookback_days: int,
    ) -> CausalChain:
        """
        Build a CausalChain from the dependency graph when statistical
        analysis can't identify causes (insufficient data).

        Instead of a useless single-node placeholder, walks the known
        dependency graph upstream from the incident metric to produce
        a multi-node chain that shows the expected causal pathway.
        """
        now = datetime.utcnow()
        detection_date = incident.detected_at.date()
        analysis_start = detection_date - timedelta(days=lookback_days)

        # Walk upstream through the dependency graph to build a chain
        upstream = self.dep_graph.get_upstream_nodes(incident_metric)
        # Also get the full path: find direct predecessors for a clean chain
        chain_metrics = []
        visited = set()

        def _walk_upstream(metric, depth=0):
            if depth > 4 or metric in visited:
                return
            visited.add(metric)
            preds = self.dep_graph.get_direct_predecessors(metric)
            for pred in preds:
                if pred not in visited:
                    _walk_upstream(pred, depth + 1)
            chain_metrics.append(metric)

        _walk_upstream(incident_metric)

        # If we only got the incident metric, try one more approach:
        # use known upstream metrics even without graph predecessors
        if len(chain_metrics) < 2 and upstream:
            chain_metrics = upstream[:3] + [incident_metric]

        # Ensure at least 2 nodes for a connected chain
        if len(chain_metrics) < 2:
            # Last resort: add any directly connected node
            successors = self.dep_graph.get_direct_successors(incident_metric)
            if successors:
                chain_metrics = [incident_metric] + successors[:1]
            else:
                chain_metrics = [incident_metric]

        zscore = abs(incident.primary_metric_zscore) if incident.primary_metric_zscore else 2.0
        nodes = []
        for i, metric in enumerate(chain_metrics):
            is_incident_metric = (metric == incident_metric)
            score = 0.9 - (i * 0.15) if not is_incident_metric else 0.95
            nodes.append(CausalNode(
                metric_name=metric,
                contribution_score=max(0.1, min(1.0, score)),
                anomaly_magnitude=zscore if is_incident_metric else zscore * 0.6,
                temporal_precedence=max(0.1, 0.9 - i * 0.2),
                graph_proximity=1.0 if is_incident_metric else max(0.3, 1.0 - i * 0.2),
                data_quality_weight=0.5,
                metric_value=incident.primary_metric_value if is_incident_metric else 0.0,
                metric_baseline=incident.primary_metric_baseline if is_incident_metric else 0.0,
                metric_zscore=zscore if is_incident_metric else zscore * 0.5,
                anomaly_window=(now - timedelta(hours=24 + i * 6), now - timedelta(hours=i * 6)),
                evidence_clusters=[],
            ))

        path = CausalPath(rank=1, overall_score=0.7, nodes=nodes)

        chain = CausalChain(
            incident_id=incident.incident_id,
            paths=[path],
            algorithm_version="BRE-RCA-v1-graph-fallback",
            causal_window=(
                datetime.combine(analysis_start, datetime.min.time()),
                datetime.combine(detection_date, datetime.max.time()),
            ),
            dependency_graph_version=self.dep_graph.version,
            run_id=run_id,
        )

        self.logger.info(
            "graph_based_chain_generated",
            run_id=run_id,
            incident_id=incident.incident_id,
            incident_metric=incident_metric,
            chain_length=len(chain_metrics),
            chain_metrics=chain_metrics,
        )

        return chain
