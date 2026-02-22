"""
Ensemble Detection Engine and Incident Creation.

This module fuses results from all detection layers (statistical, ML, changepoint)
and creates typed incidents across 8 incident types with precise severity classification.

Ensemble Fusion Rules:
    - Layer 1 alone → MEDIUM confidence
    - Layer 1 + Layer 2 agree → HIGH confidence
    - Layer 1 + Layer 2 + Layer 3 agree → VERY_HIGH confidence
    - Only Layer 2 or Layer 3 without Layer 1 → LOW confidence

Incident Types (8 total):
    1. REFUND_SPIKE: Abnormal refund rate
    2. FULFILLMENT_SLA_DEGRADATION: Delivery delays and backlog
    3. SUPPORT_LOAD_SURGE: Support ticket volume spike
    4. CHURN_ACCELERATION: Customer churn increase
    5. MARGIN_COMPRESSION: Profit margin decline
    6. LIQUIDITY_CRUNCH_RISK: Cash flow issues
    7. SUPPLIER_DEPENDENCY_FAILURE: Supplier delays
    8. CUSTOMER_SATISFACTION_REGRESSION: Review score decline

Each incident is fully populated with detection metadata, evidence, and severity.
"""

from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

import structlog

from api.config import get_settings
from api.engine.detection.changepoint import ChangepointDetector
from api.engine.detection.ml_detector import MLDetector
from api.engine.detection.statistical import StatisticalDetector
from api.models.enums import (
    Confidence,
    DetectionMethod,
    EventType,
    IncidentType,
    Severity,
)
from api.models.events import CanonicalEvent
from api.models.incidents import Incident

logger = structlog.get_logger()


class EnsembleDetector:
    """
    Ensemble detector that fuses multiple detection layers and creates incidents.

    Coordinates statistical, ML, and changepoint detection layers, applies fusion
    rules to determine confidence, and creates fully-populated typed incidents
    with appropriate severity classification.

    Attributes:
        statistical_detector: Layer 1 statistical detector
        ml_detector: Layer 2 ML detector (optional)
        changepoint_detector: Layer 3 changepoint detector (optional)
        enable_ml: Whether to run ML detection
        enable_changepoint: Whether to run changepoint detection

    Example:
        >>> detector = EnsembleDetector(enable_ml=True, enable_changepoint=True)
        >>> incidents = detector.run_full_detection(current, historical, events)
        >>> critical = [i for i in incidents if i.severity == Severity.CRITICAL]
    """

    # Revenue threshold for CRITICAL severity (10% of daily revenue)
    CRITICAL_REVENUE_THRESHOLD_PCT = 0.10

    def __init__(
        self,
        enable_ml: bool = True,
        enable_changepoint: bool = True,
        baseline_days: int = 30,
    ):
        """
        Initialize the ensemble detector.

        Args:
            enable_ml: Enable ML detection layer (default: True)
            enable_changepoint: Enable changepoint detection layer (default: True)
            baseline_days: Baseline window for statistical detection (default: 30)
        """
        self.statistical_detector = StatisticalDetector(baseline_days=baseline_days)
        use_pretrained = get_settings().anomaly_use_pretrained
        self.ml_detector = (
            MLDetector(use_pretrained=use_pretrained) if enable_ml else None
        )
        self.changepoint_detector = ChangepointDetector() if enable_changepoint else None
        self.enable_ml = enable_ml
        self.enable_changepoint = enable_changepoint
        self.logger = structlog.get_logger()

    def run_full_detection(
        self,
        current_metrics: dict,
        historical_metrics: list[dict],
        events: list[CanonicalEvent],
        run_id: Optional[str] = None,
    ) -> list[Incident]:
        """
        Run complete multi-layered detection and create incidents.

        Orchestrates all detection layers, applies fusion rules, and creates
        typed incidents for all detected anomalies.

        Args:
            current_metrics: Current day's metrics across all domains
            historical_metrics: Historical metrics for baseline computation
            events: Canonical events for evidence gathering
            run_id: Optional run ID for tracking (generated if not provided)

        Returns:
            List of detected Incident objects

        Example:
            >>> current = {
            ...     "financial": {"refund_rate": 0.15, "margin_proxy": 0.10},
            ...     "operational": {"delivery_delay_rate": 0.25},
            ...     "customer": {"churn_proxy": 0.08}
            ... }
            >>> incidents = detector.run_full_detection(current, historical, events)
        """
        run_id = run_id or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        self.logger.info(
            "starting_ensemble_detection",
            run_id=run_id,
            enable_ml=self.enable_ml,
            enable_changepoint=self.enable_changepoint,
        )

        incidents = []

        # Flatten current metrics for detection
        flat_current = self._flatten_metrics(current_metrics)
        flat_historical = [self._flatten_metrics(m) for m in historical_metrics]

        # Train ML detector on historical data so it can contribute to detection
        if self.enable_ml and self.ml_detector and len(flat_historical) >= 30:
            try:
                # Use domain that doesn't exist so train() uses full flat dict per sample
                result = self.ml_detector.train(
                    flat_historical,
                    domain="__all__",
                    min_samples=30,
                )
                if result.get("success"):
                    self.logger.info(
                        "ml_detector_trained",
                        domain=result.get("domain"),
                        samples=result.get("samples_count"),
                        features=result.get("features_count"),
                    )
            except Exception as e:
                self.logger.warning("ml_detector_train_failed", error=str(e))

        # Layer 1: Statistical detection (always runs)
        statistical_results = self.statistical_detector.detect_all(
            flat_current, flat_historical
        )

        # Create lookup for quick access
        stat_results_map = {r["metric_name"]: r for r in statistical_results}

        # Layer 2: ML detection (if enabled and trained)
        ml_results_map = {}
        if self.enable_ml and self.ml_detector and self.ml_detector.model:
            try:
                ml_result = self.ml_detector.detect(flat_current)
                # Store global ML result
                ml_results_map["_global"] = ml_result
            except Exception as e:
                self.logger.warning("ml_detection_failed", error=str(e))

        # Layer 3: Changepoint detection (if enabled)
        changepoint_results_map = {}
        if self.enable_changepoint and self.changepoint_detector:
            changepoint_results_map = self._run_changepoint_detection(
                flat_current, flat_historical
            )

        # Compute incident window (assume current day)
        window_end = datetime.utcnow()
        window_start = window_end - timedelta(hours=24)

        # Detect each incident type
        incidents.extend(
            self._detect_refund_spike(
                stat_results_map,
                ml_results_map,
                changepoint_results_map,
                events,
                window_start,
                window_end,
                run_id,
            )
        )

        incidents.extend(
            self._detect_fulfillment_sla_degradation(
                stat_results_map,
                ml_results_map,
                changepoint_results_map,
                events,
                window_start,
                window_end,
                run_id,
            )
        )

        incidents.extend(
            self._detect_support_load_surge(
                stat_results_map,
                ml_results_map,
                changepoint_results_map,
                events,
                window_start,
                window_end,
                run_id,
            )
        )

        incidents.extend(
            self._detect_churn_acceleration(
                stat_results_map,
                ml_results_map,
                changepoint_results_map,
                events,
                window_start,
                window_end,
                run_id,
            )
        )

        incidents.extend(
            self._detect_margin_compression(
                stat_results_map,
                ml_results_map,
                changepoint_results_map,
                events,
                window_start,
                window_end,
                run_id,
            )
        )

        incidents.extend(
            self._detect_liquidity_crunch(
                stat_results_map,
                ml_results_map,
                changepoint_results_map,
                events,
                window_start,
                window_end,
                run_id,
            )
        )

        incidents.extend(
            self._detect_supplier_dependency_failure(
                stat_results_map,
                ml_results_map,
                changepoint_results_map,
                events,
                window_start,
                window_end,
                run_id,
            )
        )

        incidents.extend(
            self._detect_customer_satisfaction_regression(
                stat_results_map,
                ml_results_map,
                changepoint_results_map,
                events,
                window_start,
                window_end,
                run_id,
            )
        )

        self.logger.info(
            "ensemble_detection_complete",
            run_id=run_id,
            incidents_detected=len(incidents),
        )

        return incidents

    # =========================================================================
    # Incident Type Detectors
    # =========================================================================

    def _detect_refund_spike(
        self,
        stat_map: dict,
        ml_map: dict,
        cp_map: dict,
        events: list[CanonicalEvent],
        window_start: datetime,
        window_end: datetime,
        run_id: str,
    ) -> list[Incident]:
        """
        Detect REFUND_SPIKE incidents.

        Trigger: refund_rate z-score > 3.0
        Severity: Based on z-score range and revenue impact
        """
        incidents = []

        refund_result = stat_map.get("refund_rate")
        if not refund_result or not refund_result["is_anomaly"]:
            return incidents

        zscore = refund_result["zscore"]
        if zscore <= 3.0:
            return incidents

        # Classify severity
        severity = self._classify_refund_spike_severity(refund_result, stat_map)

        # Determine confidence
        confidence = self._compute_confidence(
            refund_result, ml_map, cp_map, "refund_rate"
        )

        # Gather evidence
        evidence_event_ids = self._gather_refund_evidence(
            events, window_start, window_end
        )

        # Supporting metrics
        supporting_metrics = []
        if "daily_refunds" in stat_map:
            supporting_metrics.append({
                "metric": "daily_refunds",
                "value": stat_map["daily_refunds"]["current_value"],
                "baseline": stat_map["daily_refunds"]["median"],
                "zscore": stat_map["daily_refunds"].get("zscore", 0.0),
            })

        # Compute data quality
        data_quality = self._compute_data_quality(events, window_start, window_end)

        # Detection methods
        detection_methods = [DetectionMethod.MAD_ZSCORE]
        if ml_map.get("_global", {}).get("is_anomaly"):
            detection_methods.append(DetectionMethod.ISOLATION_FOREST)
        if cp_map.get("refund_rate", {}).get("has_change"):
            detection_methods.append(DetectionMethod.CHANGEPOINT)

        incident = Incident(
            incident_type=IncidentType.REFUND_SPIKE,
            incident_window_start=window_start,
            incident_window_end=window_end,
            severity=severity,
            confidence=confidence,
            detection_methods=detection_methods,
            primary_metric="refund_rate",
            primary_metric_value=refund_result["current_value"],
            primary_metric_baseline=refund_result["median"],
            primary_metric_zscore=refund_result["zscore"],
            supporting_metrics=supporting_metrics,
            evidence_event_ids=evidence_event_ids,
            evidence_event_count=len(evidence_event_ids),
            data_quality_score=data_quality,
            run_id=run_id,
        )

        incidents.append(incident)
        return incidents

    def _detect_fulfillment_sla_degradation(
        self,
        stat_map: dict,
        ml_map: dict,
        cp_map: dict,
        events: list[CanonicalEvent],
        window_start: datetime,
        window_end: datetime,
        run_id: str,
    ) -> list[Incident]:
        """
        Detect FULFILLMENT_SLA_DEGRADATION incidents.

        Trigger: delivery_delay_rate OR fulfillment_backlog z-score > 3.0
        """
        incidents = []

        delay_result = stat_map.get("delivery_delay_rate")
        backlog_result = stat_map.get("fulfillment_backlog")

        triggered = False
        primary_metric = None
        primary_result = None

        if delay_result and delay_result["is_anomaly"] and delay_result["zscore"] > 3.0:
            triggered = True
            primary_metric = "delivery_delay_rate"
            primary_result = delay_result

        if backlog_result and backlog_result["is_anomaly"] and backlog_result["zscore"] > 3.0:
            if not triggered or (backlog_result["zscore"] > delay_result.get("zscore", 0)):
                triggered = True
                primary_metric = "fulfillment_backlog"
                primary_result = backlog_result

        if not triggered:
            return incidents

        # Classify severity
        severity = self._classify_fulfillment_severity(primary_result)

        # Confidence
        confidence = self._compute_confidence(primary_result, ml_map, cp_map, primary_metric)

        # Evidence
        evidence_event_ids = self._gather_fulfillment_evidence(events, window_start, window_end)

        # Supporting metrics
        supporting_metrics = []
        if delay_result and primary_metric != "delivery_delay_rate":
            supporting_metrics.append({
                "metric": "delivery_delay_rate",
                "value": delay_result["current_value"],
                "baseline": delay_result["median"],
                "zscore": delay_result.get("zscore", 0.0),
            })
        if backlog_result and primary_metric != "fulfillment_backlog":
            supporting_metrics.append({
                "metric": "fulfillment_backlog",
                "value": backlog_result["current_value"],
                "baseline": backlog_result["median"],
                "zscore": backlog_result.get("zscore", 0.0),
            })

        data_quality = self._compute_data_quality(events, window_start, window_end)

        detection_methods = [DetectionMethod.MAD_ZSCORE]
        if ml_map.get("_global", {}).get("is_anomaly"):
            detection_methods.append(DetectionMethod.ISOLATION_FOREST)
        if cp_map.get(primary_metric, {}).get("has_change"):
            detection_methods.append(DetectionMethod.CHANGEPOINT)

        incident = Incident(
            incident_type=IncidentType.FULFILLMENT_SLA_DEGRADATION,
            incident_window_start=window_start,
            incident_window_end=window_end,
            severity=severity,
            confidence=confidence,
            detection_methods=detection_methods,
            primary_metric=primary_metric,
            primary_metric_value=primary_result["current_value"],
            primary_metric_baseline=primary_result["median"],
            primary_metric_zscore=primary_result["zscore"],
            supporting_metrics=supporting_metrics,
            evidence_event_ids=evidence_event_ids,
            evidence_event_count=len(evidence_event_ids),
            data_quality_score=data_quality,
            run_id=run_id,
        )

        incidents.append(incident)
        return incidents

    def _detect_support_load_surge(
        self,
        stat_map: dict,
        ml_map: dict,
        cp_map: dict,
        events: list[CanonicalEvent],
        window_start: datetime,
        window_end: datetime,
        run_id: str,
    ) -> list[Incident]:
        """
        Detect SUPPORT_LOAD_SURGE incidents.

        Trigger: ticket_volume OR ticket_backlog z-score > 3.0
        Upgrade severity if avg_resolution_time also anomalous
        """
        incidents = []

        volume_result = stat_map.get("ticket_volume")
        backlog_result = stat_map.get("ticket_backlog")

        triggered = False
        primary_metric = None
        primary_result = None

        if volume_result and volume_result["is_anomaly"] and volume_result["zscore"] > 3.0:
            triggered = True
            primary_metric = "ticket_volume"
            primary_result = volume_result

        if backlog_result and backlog_result["is_anomaly"] and backlog_result["zscore"] > 3.0:
            if not triggered or (backlog_result["zscore"] > volume_result.get("zscore", 0)):
                triggered = True
                primary_metric = "ticket_backlog"
                primary_result = backlog_result

        if not triggered:
            return incidents

        # Check resolution time for severity upgrade
        resolution_result = stat_map.get("avg_resolution_time")
        resolution_degraded = (
            resolution_result
            and resolution_result["is_anomaly"]
            and resolution_result["zscore"] > 2.0
        )

        severity = self._classify_support_severity(primary_result, resolution_degraded)
        confidence = self._compute_confidence(primary_result, ml_map, cp_map, primary_metric)

        evidence_event_ids = self._gather_support_evidence(events, window_start, window_end)

        supporting_metrics = []
        if volume_result and primary_metric != "ticket_volume":
            supporting_metrics.append({
                "metric": "ticket_volume",
                "value": volume_result["current_value"],
                "baseline": volume_result["median"],
                "zscore": volume_result.get("zscore", 0.0),
            })
        if backlog_result and primary_metric != "ticket_backlog":
            supporting_metrics.append({
                "metric": "ticket_backlog",
                "value": backlog_result["current_value"],
                "baseline": backlog_result["median"],
                "zscore": backlog_result.get("zscore", 0.0),
            })
        if resolution_result:
            supporting_metrics.append({
                "metric": "avg_resolution_time",
                "value": resolution_result["current_value"],
                "baseline": resolution_result["median"],
                "zscore": resolution_result.get("zscore", 0.0),
            })

        data_quality = self._compute_data_quality(events, window_start, window_end)

        detection_methods = [DetectionMethod.MAD_ZSCORE]
        if ml_map.get("_global", {}).get("is_anomaly"):
            detection_methods.append(DetectionMethod.ISOLATION_FOREST)
        if cp_map.get(primary_metric, {}).get("has_change"):
            detection_methods.append(DetectionMethod.CHANGEPOINT)

        incident = Incident(
            incident_type=IncidentType.SUPPORT_LOAD_SURGE,
            incident_window_start=window_start,
            incident_window_end=window_end,
            severity=severity,
            confidence=confidence,
            detection_methods=detection_methods,
            primary_metric=primary_metric,
            primary_metric_value=primary_result["current_value"],
            primary_metric_baseline=primary_result["median"],
            primary_metric_zscore=primary_result["zscore"],
            supporting_metrics=supporting_metrics,
            evidence_event_ids=evidence_event_ids,
            evidence_event_count=len(evidence_event_ids),
            data_quality_score=data_quality,
            run_id=run_id,
        )

        incidents.append(incident)
        return incidents

    def _detect_churn_acceleration(
        self,
        stat_map: dict,
        ml_map: dict,
        cp_map: dict,
        events: list[CanonicalEvent],
        window_start: datetime,
        window_end: datetime,
        run_id: str,
    ) -> list[Incident]:
        """
        Detect CHURN_ACCELERATION incidents.

        Trigger: churn_proxy z-score > 2.5 (lower threshold)
        Upgrade severity if customer_concentration high
        """
        incidents = []

        churn_result = stat_map.get("churn_proxy")
        if not churn_result or not churn_result["is_anomaly"]:
            return incidents

        zscore = churn_result["zscore"]
        if zscore <= 2.5:
            return incidents

        # Check customer concentration for severity upgrade
        concentration_result = stat_map.get("customer_concentration")
        high_concentration = (
            concentration_result
            and concentration_result["current_value"] > 0.5
        )

        severity = self._classify_churn_severity(churn_result, high_concentration)
        confidence = self._compute_confidence(churn_result, ml_map, cp_map, "churn_proxy")

        evidence_event_ids = self._gather_churn_evidence(events, window_start, window_end)

        supporting_metrics = []
        if concentration_result:
            supporting_metrics.append({
                "metric": "customer_concentration",
                "value": concentration_result["current_value"],
                "baseline": concentration_result["median"],
                "zscore": concentration_result.get("zscore", 0.0),
            })

        data_quality = self._compute_data_quality(events, window_start, window_end)

        detection_methods = [DetectionMethod.MAD_ZSCORE]
        if ml_map.get("_global", {}).get("is_anomaly"):
            detection_methods.append(DetectionMethod.ISOLATION_FOREST)
        if cp_map.get("churn_proxy", {}).get("has_change"):
            detection_methods.append(DetectionMethod.CHANGEPOINT)

        incident = Incident(
            incident_type=IncidentType.CHURN_ACCELERATION,
            incident_window_start=window_start,
            incident_window_end=window_end,
            severity=severity,
            confidence=confidence,
            detection_methods=detection_methods,
            primary_metric="churn_proxy",
            primary_metric_value=churn_result["current_value"],
            primary_metric_baseline=churn_result["median"],
            primary_metric_zscore=churn_result["zscore"],
            supporting_metrics=supporting_metrics,
            evidence_event_ids=evidence_event_ids,
            evidence_event_count=len(evidence_event_ids),
            data_quality_score=data_quality,
            run_id=run_id,
        )

        incidents.append(incident)
        return incidents

    def _detect_margin_compression(
        self,
        stat_map: dict,
        ml_map: dict,
        cp_map: dict,
        events: list[CanonicalEvent],
        window_start: datetime,
        window_end: datetime,
        run_id: str,
    ) -> list[Incident]:
        """
        Detect MARGIN_COMPRESSION incidents.

        Trigger: margin_proxy z < -3 OR expense_ratio z > 3
        """
        incidents = []

        margin_result = stat_map.get("margin_proxy")
        expense_result = stat_map.get("expense_ratio")

        triggered = False
        primary_metric = None
        primary_result = None

        if margin_result and margin_result["is_anomaly"] and margin_result["zscore"] < -3.0:
            triggered = True
            primary_metric = "margin_proxy"
            primary_result = margin_result

        if expense_result and expense_result["is_anomaly"] and expense_result["zscore"] > 3.0:
            if not triggered or (expense_result["zscore"] > abs(margin_result.get("zscore", 0))):
                triggered = True
                primary_metric = "expense_ratio"
                primary_result = expense_result

        if not triggered:
            return incidents

        severity = self._classify_margin_severity(margin_result, expense_result)
        confidence = self._compute_confidence(primary_result, ml_map, cp_map, primary_metric)

        evidence_event_ids = self._gather_financial_evidence(events, window_start, window_end)

        supporting_metrics = []
        if margin_result and primary_metric != "margin_proxy":
            supporting_metrics.append({
                "metric": "margin_proxy",
                "value": margin_result["current_value"],
                "baseline": margin_result["median"],
                "zscore": margin_result.get("zscore", 0.0),
            })
        if expense_result and primary_metric != "expense_ratio":
            supporting_metrics.append({
                "metric": "expense_ratio",
                "value": expense_result["current_value"],
                "baseline": expense_result["median"],
                "zscore": expense_result.get("zscore", 0.0),
            })

        data_quality = self._compute_data_quality(events, window_start, window_end)

        detection_methods = [DetectionMethod.MAD_ZSCORE]
        if ml_map.get("_global", {}).get("is_anomaly"):
            detection_methods.append(DetectionMethod.ISOLATION_FOREST)
        if cp_map.get(primary_metric, {}).get("has_change"):
            detection_methods.append(DetectionMethod.CHANGEPOINT)

        incident = Incident(
            incident_type=IncidentType.MARGIN_COMPRESSION,
            incident_window_start=window_start,
            incident_window_end=window_end,
            severity=severity,
            confidence=confidence,
            detection_methods=detection_methods,
            primary_metric=primary_metric,
            primary_metric_value=primary_result["current_value"],
            primary_metric_baseline=primary_result["median"],
            primary_metric_zscore=primary_result["zscore"],
            supporting_metrics=supporting_metrics,
            evidence_event_ids=evidence_event_ids,
            evidence_event_count=len(evidence_event_ids),
            data_quality_score=data_quality,
            run_id=run_id,
        )

        incidents.append(incident)
        return incidents

    def _detect_liquidity_crunch(
        self,
        stat_map: dict,
        ml_map: dict,
        cp_map: dict,
        events: list[CanonicalEvent],
        window_start: datetime,
        window_end: datetime,
        run_id: str,
    ) -> list[Incident]:
        """
        Detect LIQUIDITY_CRUNCH_RISK incidents.

        Trigger: net_cash_proxy OR ar_aging_amount anomalous
        CRITICAL if both fire
        """
        incidents = []

        cash_result = stat_map.get("net_cash_proxy")
        ar_result = stat_map.get("ar_aging_amount")

        cash_anomaly = cash_result and cash_result["is_anomaly"] and cash_result["zscore"] < -3.0
        ar_anomaly = ar_result and ar_result["is_anomaly"] and ar_result["zscore"] > 3.0

        if not (cash_anomaly or ar_anomaly):
            return incidents

        # Determine primary metric
        primary_metric = "net_cash_proxy" if cash_anomaly else "ar_aging_amount"
        primary_result = cash_result if cash_anomaly else ar_result

        # CRITICAL if both fire
        both_fire = cash_anomaly and ar_anomaly
        severity = Severity.CRITICAL if both_fire else self._classify_liquidity_severity(primary_result)

        confidence = self._compute_confidence(primary_result, ml_map, cp_map, primary_metric)

        evidence_event_ids = self._gather_financial_evidence(events, window_start, window_end)

        supporting_metrics = []
        if cash_result and primary_metric != "net_cash_proxy":
            supporting_metrics.append({
                "metric": "net_cash_proxy",
                "value": cash_result["current_value"],
                "baseline": cash_result["median"],
                "zscore": cash_result.get("zscore", 0.0),
            })
        if ar_result and primary_metric != "ar_aging_amount":
            supporting_metrics.append({
                "metric": "ar_aging_amount",
                "value": ar_result["current_value"],
                "baseline": ar_result["median"],
                "zscore": ar_result.get("zscore", 0.0),
            })

        data_quality = self._compute_data_quality(events, window_start, window_end)

        detection_methods = [DetectionMethod.MAD_ZSCORE]
        if ml_map.get("_global", {}).get("is_anomaly"):
            detection_methods.append(DetectionMethod.ISOLATION_FOREST)
        if cp_map.get(primary_metric, {}).get("has_change"):
            detection_methods.append(DetectionMethod.CHANGEPOINT)

        incident = Incident(
            incident_type=IncidentType.LIQUIDITY_CRUNCH_RISK,
            incident_window_start=window_start,
            incident_window_end=window_end,
            severity=severity,
            confidence=confidence,
            detection_methods=detection_methods,
            primary_metric=primary_metric,
            primary_metric_value=primary_result["current_value"],
            primary_metric_baseline=primary_result["median"],
            primary_metric_zscore=primary_result["zscore"],
            supporting_metrics=supporting_metrics,
            evidence_event_ids=evidence_event_ids,
            evidence_event_count=len(evidence_event_ids),
            data_quality_score=data_quality,
            run_id=run_id,
        )

        incidents.append(incident)
        return incidents

    def _detect_supplier_dependency_failure(
        self,
        stat_map: dict,
        ml_map: dict,
        cp_map: dict,
        events: list[CanonicalEvent],
        window_start: datetime,
        window_end: datetime,
        run_id: str,
    ) -> list[Incident]:
        """
        Detect SUPPLIER_DEPENDENCY_FAILURE incidents.

        Trigger: supplier_delay_rate OR supplier_delay_severity anomalous
        """
        incidents = []

        rate_result = stat_map.get("supplier_delay_rate")
        severity_metric_result = stat_map.get("supplier_delay_severity")

        triggered = False
        primary_metric = None
        primary_result = None

        if rate_result and rate_result["is_anomaly"] and rate_result["zscore"] > 3.0:
            triggered = True
            primary_metric = "supplier_delay_rate"
            primary_result = rate_result

        if severity_metric_result and severity_metric_result["is_anomaly"] and severity_metric_result["zscore"] > 3.0:
            if not triggered or (severity_metric_result["zscore"] > rate_result.get("zscore", 0)):
                triggered = True
                primary_metric = "supplier_delay_severity"
                primary_result = severity_metric_result

        if not triggered:
            return incidents

        severity = self._classify_supplier_severity(primary_result)
        confidence = self._compute_confidence(primary_result, ml_map, cp_map, primary_metric)

        evidence_event_ids = self._gather_supplier_evidence(events, window_start, window_end)

        supporting_metrics = []
        if rate_result and primary_metric != "supplier_delay_rate":
            supporting_metrics.append({
                "metric": "supplier_delay_rate",
                "value": rate_result["current_value"],
                "baseline": rate_result["median"],
                "zscore": rate_result.get("zscore", 0.0),
            })
        if severity_metric_result and primary_metric != "supplier_delay_severity":
            supporting_metrics.append({
                "metric": "supplier_delay_severity",
                "value": severity_metric_result["current_value"],
                "baseline": severity_metric_result["median"],
                "zscore": severity_metric_result.get("zscore", 0.0),
            })

        data_quality = self._compute_data_quality(events, window_start, window_end)

        detection_methods = [DetectionMethod.MAD_ZSCORE]
        if ml_map.get("_global", {}).get("is_anomaly"):
            detection_methods.append(DetectionMethod.ISOLATION_FOREST)
        if cp_map.get(primary_metric, {}).get("has_change"):
            detection_methods.append(DetectionMethod.CHANGEPOINT)

        incident = Incident(
            incident_type=IncidentType.SUPPLIER_DEPENDENCY_FAILURE,
            incident_window_start=window_start,
            incident_window_end=window_end,
            severity=severity,
            confidence=confidence,
            detection_methods=detection_methods,
            primary_metric=primary_metric,
            primary_metric_value=primary_result["current_value"],
            primary_metric_baseline=primary_result["median"],
            primary_metric_zscore=primary_result["zscore"],
            supporting_metrics=supporting_metrics,
            evidence_event_ids=evidence_event_ids,
            evidence_event_count=len(evidence_event_ids),
            data_quality_score=data_quality,
            run_id=run_id,
        )

        incidents.append(incident)
        return incidents

    def _detect_customer_satisfaction_regression(
        self,
        stat_map: dict,
        ml_map: dict,
        cp_map: dict,
        events: list[CanonicalEvent],
        window_start: datetime,
        window_end: datetime,
        run_id: str,
    ) -> list[Incident]:
        """
        Detect CUSTOMER_SATISFACTION_REGRESSION incidents.

        Trigger: review_score_avg z < -2.5 OR review_score_trend < -0.1
        """
        incidents = []

        score_result = stat_map.get("review_score_avg")
        trend_result = stat_map.get("review_score_trend")

        triggered = False
        primary_metric = None
        primary_result = None

        if score_result and score_result["is_anomaly"] and score_result["zscore"] < -2.5:
            triggered = True
            primary_metric = "review_score_avg"
            primary_result = score_result

        # Check trend (slope < -0.1)
        if trend_result and trend_result["current_value"] < -0.1:
            if not triggered or (abs(trend_result["current_value"]) > abs(score_result.get("current_value", 0))):
                triggered = True
                primary_metric = "review_score_trend"
                primary_result = trend_result

        if not triggered:
            return incidents

        severity = self._classify_satisfaction_severity(score_result, trend_result)
        confidence = self._compute_confidence(primary_result, ml_map, cp_map, primary_metric)

        evidence_event_ids = self._gather_satisfaction_evidence(events, window_start, window_end)

        supporting_metrics = []
        if score_result and primary_metric != "review_score_avg":
            supporting_metrics.append({
                "metric": "review_score_avg",
                "value": score_result["current_value"],
                "baseline": score_result["median"],
                "zscore": score_result.get("zscore", 0.0),
            })
        if trend_result and primary_metric != "review_score_trend":
            supporting_metrics.append({
                "metric": "review_score_trend",
                "value": trend_result["current_value"],
                "baseline": trend_result["median"],
                "zscore": trend_result.get("zscore", 0.0),
            })

        data_quality = self._compute_data_quality(events, window_start, window_end)

        detection_methods = [DetectionMethod.MAD_ZSCORE]
        if ml_map.get("_global", {}).get("is_anomaly"):
            detection_methods.append(DetectionMethod.ISOLATION_FOREST)
        if cp_map.get(primary_metric, {}).get("has_change"):
            detection_methods.append(DetectionMethod.CHANGEPOINT)

        incident = Incident(
            incident_type=IncidentType.CUSTOMER_SATISFACTION_REGRESSION,
            incident_window_start=window_start,
            incident_window_end=window_end,
            severity=severity,
            confidence=confidence,
            detection_methods=detection_methods,
            primary_metric=primary_metric,
            primary_metric_value=primary_result["current_value"],
            primary_metric_baseline=primary_result["median"],
            primary_metric_zscore=primary_result["zscore"],
            supporting_metrics=supporting_metrics,
            evidence_event_ids=evidence_event_ids,
            evidence_event_count=len(evidence_event_ids),
            data_quality_score=data_quality,
            run_id=run_id,
        )

        incidents.append(incident)
        return incidents

    # =========================================================================
    # Severity Classification
    # =========================================================================

    def _classify_refund_spike_severity(
        self, refund_result: dict, stat_map: dict
    ) -> Severity:
        """Classify severity for refund spike based on z-score and revenue impact."""
        zscore = refund_result["zscore"]
        refund_rate = refund_result["current_value"]

        # Get revenue for impact assessment
        revenue_result = stat_map.get("daily_revenue")
        revenue = revenue_result["current_value"] if revenue_result else 0.0

        # CRITICAL: z > 8 AND refund rate > 10% of revenue
        if zscore > 8.0 and refund_rate > self.CRITICAL_REVENUE_THRESHOLD_PCT:
            return Severity.CRITICAL

        # HIGH: z in (6, 8]
        if zscore > 6.0:
            return Severity.HIGH

        # MEDIUM: z in (4, 6]
        if zscore > 4.0:
            return Severity.MEDIUM

        # LOW: z in (3, 4]
        return Severity.LOW

    def _classify_fulfillment_severity(self, primary_result: dict) -> Severity:
        """Classify severity for fulfillment SLA degradation."""
        zscore = abs(primary_result["zscore"])

        if zscore > 6.0:
            return Severity.HIGH
        elif zscore > 4.0:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _classify_support_severity(
        self, primary_result: dict, resolution_degraded: bool
    ) -> Severity:
        """Classify severity for support load surge."""
        zscore = abs(primary_result["zscore"])

        # Upgrade severity if resolution time degraded
        if resolution_degraded and zscore > 4.0:
            return Severity.HIGH

        if zscore > 6.0:
            return Severity.HIGH
        elif zscore > 4.0:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _classify_churn_severity(
        self, churn_result: dict, high_concentration: bool
    ) -> Severity:
        """Classify severity for churn acceleration."""
        zscore = churn_result["zscore"]

        # CRITICAL if high customer concentration and high churn
        if high_concentration and zscore > 4.0:
            return Severity.CRITICAL

        if zscore > 5.0:
            return Severity.HIGH
        elif zscore > 3.5:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _classify_margin_severity(
        self, margin_result: Optional[dict], expense_result: Optional[dict]
    ) -> Severity:
        """Classify severity for margin compression."""
        max_zscore = 0.0

        if margin_result:
            max_zscore = max(max_zscore, abs(margin_result["zscore"]))

        if expense_result:
            max_zscore = max(max_zscore, abs(expense_result["zscore"]))

        if max_zscore > 6.0:
            return Severity.CRITICAL
        elif max_zscore > 4.5:
            return Severity.HIGH
        elif max_zscore > 3.5:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _classify_liquidity_severity(self, primary_result: dict) -> Severity:
        """Classify severity for liquidity crunch."""
        zscore = abs(primary_result["zscore"])

        if zscore > 6.0:
            return Severity.HIGH
        elif zscore > 4.0:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _classify_supplier_severity(self, primary_result: dict) -> Severity:
        """Classify severity for supplier dependency failure."""
        zscore = abs(primary_result["zscore"])

        if zscore > 6.0:
            return Severity.HIGH
        elif zscore > 4.0:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _classify_satisfaction_severity(
        self, score_result: Optional[dict], trend_result: Optional[dict]
    ) -> Severity:
        """Classify severity for customer satisfaction regression."""
        max_impact = 0.0

        if score_result:
            max_impact = max(max_impact, abs(score_result["zscore"]))

        if trend_result and trend_result["current_value"] < -0.1:
            # Severe downward trend
            max_impact = max(max_impact, abs(trend_result["current_value"]) * 20)

        if max_impact > 5.0:
            return Severity.HIGH
        elif max_impact > 3.5:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    # =========================================================================
    # Confidence Computation
    # =========================================================================

    def _compute_confidence(
        self,
        stat_result: dict,
        ml_map: dict,
        cp_map: dict,
        metric_name: str,
    ) -> Confidence:
        """
        Compute ensemble confidence based on fusion rules.

        Rules:
        - Layer 1 alone → MEDIUM
        - Layer 1 + Layer 2 → HIGH
        - Layer 1 + Layer 2 + Layer 3 → VERY_HIGH
        - Only Layer 2/3 without Layer 1 → LOW
        """
        layer1_agrees = stat_result.get("is_anomaly", False)
        layer2_agrees = ml_map.get("_global", {}).get("is_anomaly", False)
        layer3_agrees = cp_map.get(metric_name, {}).get("has_change", False)

        if not layer1_agrees:
            # No statistical detection
            if layer2_agrees or layer3_agrees:
                return Confidence.LOW
            else:
                return Confidence.LOW

        # Layer 1 detected
        if layer1_agrees and layer2_agrees and layer3_agrees:
            return Confidence.VERY_HIGH
        elif layer1_agrees and layer2_agrees:
            return Confidence.HIGH
        elif layer1_agrees and layer3_agrees:
            return Confidence.HIGH
        else:
            return Confidence.MEDIUM

    # =========================================================================
    # Evidence Gathering
    # =========================================================================

    def _gather_refund_evidence(
        self, events: list[CanonicalEvent], start: datetime, end: datetime
    ) -> list[str]:
        """Gather refund-related event IDs as evidence."""
        refund_event_types = [EventType.REFUND_ISSUED, EventType.CREDIT_MEMO_ISSUED]
        return [
            e.event_id
            for e in events
            if e.event_type in refund_event_types and start <= e.event_time <= end
        ]

    def _gather_fulfillment_evidence(
        self, events: list[CanonicalEvent], start: datetime, end: datetime
    ) -> list[str]:
        """Gather fulfillment-related event IDs as evidence."""
        fulfillment_event_types = [EventType.ORDER_LATE, EventType.ORDER_PLACED, EventType.ORDER_DELIVERED]
        return [
            e.event_id
            for e in events
            if e.event_type in fulfillment_event_types and start <= e.event_time <= end
        ]

    def _gather_support_evidence(
        self, events: list[CanonicalEvent], start: datetime, end: datetime
    ) -> list[str]:
        """Gather support-related event IDs as evidence."""
        support_event_types = [EventType.SUPPORT_TICKET_OPENED, EventType.SUPPORT_TICKET_CLOSED]
        return [
            e.event_id
            for e in events
            if e.event_type in support_event_types and start <= e.event_time <= end
        ]

    def _gather_churn_evidence(
        self, events: list[CanonicalEvent], start: datetime, end: datetime
    ) -> list[str]:
        """Gather churn-related event IDs as evidence."""
        churn_event_types = [EventType.CUSTOMER_CHURNED]
        return [
            e.event_id
            for e in events
            if e.event_type in churn_event_types and start <= e.event_time <= end
        ]

    def _gather_financial_evidence(
        self, events: list[CanonicalEvent], start: datetime, end: datetime
    ) -> list[str]:
        """Gather financial-related event IDs as evidence."""
        financial_event_types = [
            EventType.INVOICE_PAID,
            EventType.INVOICE_OVERDUE,
            EventType.EXPENSE_POSTED,
            EventType.PAYMENT_RECEIVED,
        ]
        return [
            e.event_id
            for e in events
            if e.event_type in financial_event_types and start <= e.event_time <= end
        ]

    def _gather_supplier_evidence(
        self, events: list[CanonicalEvent], start: datetime, end: datetime
    ) -> list[str]:
        """Gather supplier-related event IDs as evidence."""
        supplier_event_types = [EventType.SHIPMENT_DELAYED, EventType.PURCHASE_ORDER_PLACED]
        return [
            e.event_id
            for e in events
            if e.event_type in supplier_event_types and start <= e.event_time <= end
        ]

    def _gather_satisfaction_evidence(
        self, events: list[CanonicalEvent], start: datetime, end: datetime
    ) -> list[str]:
        """Gather satisfaction-related event IDs as evidence."""
        satisfaction_event_types = [EventType.REVIEW_SUBMITTED]
        return [
            e.event_id
            for e in events
            if e.event_type in satisfaction_event_types and start <= e.event_time <= end
        ]

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _flatten_metrics(self, metrics: dict) -> dict:
        """Flatten nested metrics dictionary into single level."""
        flat = {}

        for key, value in metrics.items():
            if isinstance(value, dict):
                # Nested domain metrics
                flat.update(value)
            else:
                flat[key] = value

        return flat

    def _run_changepoint_detection(
        self, current_metrics: dict, historical_metrics: list[dict]
    ) -> dict:
        """Run changepoint detection on all metrics."""
        if not self.changepoint_detector:
            return {}

        results = {}

        # Build time series for each metric
        for metric_name in current_metrics.keys():
            values = []
            for hist in historical_metrics:
                if metric_name in hist and hist[metric_name] is not None:
                    values.append(hist[metric_name])

            # Add current value
            if current_metrics[metric_name] is not None:
                values.append(current_metrics[metric_name])

            # Need sufficient data
            if len(values) < 10:
                continue

            try:
                result = self.changepoint_detector.detect(values)
                if result["has_changepoint"]:
                    results[metric_name] = result
            except Exception as e:
                self.logger.warning(
                    "changepoint_detection_failed_for_metric",
                    metric_name=metric_name,
                    error=str(e),
                )

        return results

    def _compute_data_quality(
        self, events: list[CanonicalEvent], start: datetime, end: datetime
    ) -> float:
        """Compute data quality score based on event quality flags."""
        window_events = [e for e in events if start <= e.event_time <= end]

        if not window_events:
            return 0.0

        # Count events with quality issues
        events_with_issues = sum(
            1 for e in window_events if e.data_quality_flags
        )

        # Quality score: 1.0 - (issue_rate)
        quality_score = 1.0 - (events_with_issues / len(window_events))

        return round(quality_score, 4)
