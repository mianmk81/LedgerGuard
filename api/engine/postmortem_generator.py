"""
Postmortem Generator — Comprehensive Incident Report Assembly.

This module synthesizes all BRE analysis outputs (detection, RCA, blast radius)
into a structured Postmortem report suitable for executive review, operational
handoff, and machine learning feedback loops.

The generator orchestrates:
1. Incident metadata extraction
2. Timeline reconstruction from events
3. Root cause summary from CausalChain
4. Impact narrative from BlastRadius
5. Contributing factor identification
6. Monitor rule recommendation generation
7. Remediation action recommendations
8. Confidence and quality assessment

Version: postmortem_gen_v1
"""

from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

import structlog

from api.models.blast_radius import BlastRadius
from api.models.enums import IncidentStatus, MonitorStatus, Severity
from api.models.incidents import Incident
from api.models.monitors import MonitorRule
from api.models.postmortem import Postmortem, TimelineEntry
from api.models.rca import CausalChain
from api.storage.base import StorageBackend

logger = structlog.get_logger()


# Incident type to monitoring metric recommendations
MONITOR_RECOMMENDATIONS = {
    "refund_spike": [
        {"metric": "refund_rate", "condition": "value > baseline * 2.0", "frequency": "hourly"},
        {"metric": "daily_refunds", "condition": "value > baseline * 1.5", "frequency": "daily"},
    ],
    "fulfillment_sla_degradation": [
        {"metric": "delivery_delay_rate", "condition": "value > baseline * 1.5", "frequency": "daily"},
        {"metric": "fulfillment_backlog", "condition": "value > baseline * 2.0", "frequency": "daily"},
    ],
    "support_load_surge": [
        {"metric": "ticket_volume", "condition": "value > baseline * 2.0", "frequency": "hourly"},
        {"metric": "ticket_backlog", "condition": "value > baseline * 1.5", "frequency": "daily"},
    ],
    "churn_acceleration": [
        {"metric": "churn_proxy", "condition": "value > baseline * 1.5", "frequency": "daily"},
        {"metric": "review_score_avg", "condition": "value < baseline * 0.8", "frequency": "daily"},
    ],
    "margin_compression": [
        {"metric": "margin_proxy", "condition": "value < baseline * 0.7", "frequency": "daily"},
        {"metric": "expense_ratio", "condition": "value > baseline * 1.3", "frequency": "daily"},
    ],
    "liquidity_crunch_risk": [
        {"metric": "net_cash_proxy", "condition": "value < baseline * 0.5", "frequency": "daily"},
        {"metric": "ar_aging_amount", "condition": "value > baseline * 1.5", "frequency": "daily"},
    ],
    "supplier_dependency_failure": [
        {"metric": "supplier_delay_rate", "condition": "value > baseline * 2.0", "frequency": "daily"},
        {"metric": "supplier_delay_severity", "condition": "value > baseline * 1.5", "frequency": "daily"},
    ],
    "customer_satisfaction_regression": [
        {"metric": "review_score_avg", "condition": "value < baseline * 0.85", "frequency": "daily"},
        {"metric": "review_score_trend", "condition": "value < -0.1", "frequency": "daily"},
    ],
}


class PostmortemGenerator:
    """
    Generates comprehensive incident postmortem reports.

    Assembles all BRE analysis outputs into a structured Postmortem that
    serves as both an operational document and ML training signal.

    Attributes:
        storage: Storage backend for persistence
        logger: Structured logger

    Example:
        >>> generator = PostmortemGenerator(storage=duckdb_storage)
        >>> postmortem = generator.generate(
        ...     incident=incident,
        ...     causal_chain=causal_chain,
        ...     blast_radius=blast_radius,
        ... )
        >>> print(postmortem.one_line_summary)
    """

    def __init__(self, storage: StorageBackend):
        """
        Initialize the postmortem generator.

        Args:
            storage: Storage backend for data access and persistence
        """
        self.storage = storage
        self.logger = structlog.get_logger()

    def generate(
        self,
        incident: Incident,
        causal_chain: CausalChain,
        blast_radius: BlastRadius,
        explanation: Optional[dict] = None,
    ) -> Postmortem:
        """
        Generate a complete postmortem report.

        Orchestrates all postmortem components into a comprehensive report.

        Args:
            incident: The incident being analyzed
            causal_chain: RCA result
            blast_radius: Impact assessment
            explanation: Optional pre-generated RCA explanation dict

        Returns:
            Complete Postmortem object

        Example:
            >>> pm = generator.generate(incident, chain, blast)
        """
        run_id = f"pm_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

        self.logger.info(
            "postmortem_generation_started",
            run_id=run_id,
            incident_id=incident.incident_id,
        )

        # Generate components
        title = self._generate_title(incident, causal_chain)
        duration = self._compute_duration(incident)
        one_line = self._generate_one_line_summary(incident, causal_chain, blast_radius)
        timeline = self._build_timeline(incident, causal_chain)
        root_cause_summary = self._generate_root_cause_summary(
            causal_chain, explanation
        )
        contributing_factors = self._identify_contributing_factors(
            causal_chain, incident
        )
        monitors = self._recommend_monitors(incident, causal_chain)
        recommendations = self._generate_recommendations(
            incident, causal_chain, blast_radius, explanation
        )
        confidence_note = self._generate_confidence_note(
            incident, causal_chain
        )

        postmortem = Postmortem(
            incident_id=incident.incident_id,
            cascade_id=incident.cascade_id,
            title=title,
            severity=incident.severity,
            duration=duration,
            status=incident.status,
            one_line_summary=one_line,
            timeline=timeline,
            causal_chain=causal_chain,
            root_cause_summary=root_cause_summary,
            blast_radius=blast_radius,
            contributing_factors=contributing_factors,
            monitors=monitors,
            recommendations=recommendations,
            data_quality_score=incident.data_quality_score,
            confidence_note=confidence_note,
            algorithm_version="BRE-v1.0",
            run_id=run_id,
        )

        # Persist
        try:
            self.storage.write_postmortem(postmortem)
            self.logger.info(
                "postmortem_persisted",
                postmortem_id=postmortem.postmortem_id,
                incident_id=incident.incident_id,
            )
        except Exception as e:
            self.logger.error(
                "postmortem_persistence_failed",
                error=str(e),
            )

        self.logger.info(
            "postmortem_generated",
            run_id=run_id,
            postmortem_id=postmortem.postmortem_id,
            incident_id=incident.incident_id,
            severity=incident.severity.value,
        )

        return postmortem

    # =========================================================================
    # Component Generators
    # =========================================================================

    def _generate_title(self, incident: Incident, chain: CausalChain) -> str:
        """Generate executive-level title."""
        incident_display = incident.incident_type.value.replace("_", " ").title()
        root_cause = ""
        if chain.paths and chain.paths[0].nodes:
            metric = chain.paths[0].nodes[0].metric_name.replace("_", " ").title()
            root_cause = f" — Root Cause: {metric}"

        return f"{incident_display} Incident{root_cause}"

    def _compute_duration(self, incident: Incident) -> str:
        """Compute human-readable incident duration."""
        delta = incident.incident_window_end - incident.incident_window_start
        hours = delta.total_seconds() / 3600

        if hours < 1:
            return f"{int(delta.total_seconds() / 60)} minutes"
        elif hours < 24:
            return f"{hours:.1f} hours"
        else:
            return f"{hours / 24:.1f} days"

    def _generate_one_line_summary(
        self, incident: Incident, chain: CausalChain, blast: BlastRadius
    ) -> str:
        """Generate one-sentence executive summary in plain business language."""
        incident_type = incident.incident_type.value.replace("_", " ")
        customers = blast.customers_affected
        severity = blast.blast_radius_severity.value

        severity_words = {
            "critical": "a serious",
            "high": "a significant",
            "medium": "a moderate",
            "low": "a minor",
        }
        severity_desc = severity_words.get(severity.lower(), "a")

        root_cause_text = "underlying metric changes"
        if chain.paths and chain.paths[0].nodes:
            root_cause_text = chain.paths[0].nodes[0].metric_name.replace("_", " ")

        revenue_text = ""
        if blast.estimated_revenue_exposure and blast.estimated_revenue_exposure > 0:
            revenue_text = f", putting ${blast.estimated_revenue_exposure:,.0f} in revenue at risk"

        return (
            f"We detected {severity_desc} {incident_type} issue "
            f"affecting {customers:,} customers{revenue_text}. "
            f"The root cause was traced to {root_cause_text}."
        )

    def _build_timeline(
        self, incident: Incident, chain: CausalChain
    ) -> list[TimelineEntry]:
        """Build chronological incident timeline."""
        entries = []

        if chain.paths and chain.paths[0].nodes:
            first_node = chain.paths[0].nodes[0]
            anomaly_start = first_node.anomaly_window[0]
            metric_label = first_node.metric_name.replace("_", " ")
            entries.append(TimelineEntry(
                timestamp=anomaly_start,
                event_description=(
                    f"{metric_label.capitalize()} started behaving unusually "
                    f"— it moved far outside its normal range"
                ),
                metric_name=first_node.metric_name,
                metric_value=first_node.metric_value,
            ))

        metric_label = incident.primary_metric.replace("_", " ")
        entries.append(TimelineEntry(
            timestamp=incident.incident_window_start,
            event_description=(
                f"{metric_label.capitalize()} moved significantly away from normal "
                f"(actual: {incident.primary_metric_value:.2f} vs. "
                f"typical: {incident.primary_metric_baseline:.2f})"
            ),
            metric_name=incident.primary_metric,
            metric_value=incident.primary_metric_value,
        ))

        confidence_words = {
            "very_high": "very high",
            "high": "high",
            "medium": "moderate",
            "low": "preliminary",
        }
        conf_text = confidence_words.get(incident.confidence.value.lower(), "")
        entries.append(TimelineEntry(
            timestamp=incident.detected_at,
            event_description=(
                f"Our detection system identified a {incident.incident_type.value.replace('_', ' ')} "
                f"problem with {conf_text} confidence"
            ),
            metric_name=incident.primary_metric,
            metric_value=incident.primary_metric_value,
        ))

        # Sort chronologically
        entries.sort(key=lambda e: e.timestamp)

        return entries

    def _generate_root_cause_summary(
        self, chain: CausalChain, explanation: Optional[dict]
    ) -> str:
        """Generate human-readable root cause summary."""
        if explanation and "executive_summary" in explanation:
            return explanation["executive_summary"]

        if not chain.paths or not chain.paths[0].nodes:
            return "Root cause could not be definitively determined from available data."

        node = chain.paths[0].nodes[0]
        metric_display = node.metric_name.replace("_", " ").title()
        direction = "increase" if node.metric_zscore > 0 else "decrease"
        score = node.contribution_score

        return (
            f"Primary root cause identified as anomalous {direction} in "
            f"{metric_display} (z-score: {node.metric_zscore:.1f}, "
            f"contribution score: {score:.0%}). "
            f"Metric moved from baseline {node.metric_baseline:.2f} "
            f"to {node.metric_value:.2f} within the analysis window."
        )

    def _identify_contributing_factors(
        self, chain: CausalChain, incident: Incident
    ) -> list[str]:
        """Identify secondary contributing factors."""
        factors = []

        if chain.paths:
            # Alternative causal paths as contributing factors
            for path in chain.paths[1:3]:  # Top 2 alternatives
                if path.nodes:
                    node = path.nodes[0]
                    metric = node.metric_name.replace("_", " ").title()
                    factors.append(
                        f"Anomalous {metric} (z-score: {node.metric_zscore:.1f}, "
                        f"contribution: {node.contribution_score:.0%})"
                    )

        # Data quality as potential factor
        if incident.data_quality_score < 0.85:
            factors.append(
                f"Reduced data quality ({incident.data_quality_score:.0%}) "
                f"may have delayed detection"
            )

        if not factors:
            factors.append("No significant secondary contributing factors identified")

        return factors

    def _recommend_monitors(
        self, incident: Incident, chain: CausalChain
    ) -> list[MonitorRule]:
        """Generate recommended monitor rules."""
        monitors = []
        incident_type = incident.incident_type.value

        recommendations = MONITOR_RECOMMENDATIONS.get(incident_type, [])

        for rec in recommendations:
            severity = Severity.HIGH if incident.severity in [Severity.HIGH, Severity.CRITICAL] else Severity.MEDIUM

            monitor = MonitorRule(
                name=f"{rec['metric'].replace('_', ' ').title()} Monitor",
                description=(
                    f"Monitors {rec['metric'].replace('_', ' ')} to detect "
                    f"recurrence of {incident_type.replace('_', ' ')} patterns. "
                    f"Generated from incident {incident.incident_id}."
                ),
                source_incident_id=incident.incident_id,
                metric_name=rec["metric"],
                condition=rec["condition"],
                baseline_window_days=30,
                check_frequency=rec["frequency"],
                severity_if_triggered=severity,
                enabled=True,
                alert_message_template=(
                    f"ALERT: {rec['metric'].replace('_', ' ').title()} has breached threshold. "
                    f"Current value: {{value}}, baseline: {{baseline}}. "
                    f"Condition: {rec['condition']}. "
                    f"Similar to incident {incident.incident_id[:12]}."
                ),
            )
            monitors.append(monitor)

        return monitors

    def _generate_recommendations(
        self,
        incident: Incident,
        chain: CausalChain,
        blast: BlastRadius,
        explanation: Optional[dict],
    ) -> list[str]:
        """Generate actionable remediation recommendations."""
        recommendations = []

        # From explanation if available
        if explanation and "recommended_actions" in explanation:
            recommendations.extend(explanation["recommended_actions"][:4])

        # Severity-based recommendations
        if incident.severity in [Severity.HIGH, Severity.CRITICAL]:
            recommendations.append(
                "Escalate to senior management for immediate review and resource allocation"
            )

        if blast.estimated_churn_exposure > 10:
            recommendations.append(
                f"Launch proactive retention campaign for {blast.estimated_churn_exposure} "
                f"at-risk customers before churn materializes"
            )

        if blast.estimated_refund_exposure > 10000:
            recommendations.append(
                f"Set aside ${blast.estimated_refund_exposure:,.2f} refund reserve "
                f"and implement expedited refund processing"
            )

        # Always recommend monitoring
        recommendations.append(
            "Deploy recommended monitors to detect recurrence of this incident pattern"
        )

        # Ensure at least one recommendation
        if not recommendations:
            recommendations.append(
                "Review incident details and implement preventive measures based on root cause analysis"
            )

        return recommendations

    def _generate_confidence_note(
        self, incident: Incident, chain: CausalChain
    ) -> str:
        """Generate confidence and limitations note."""
        quality = incident.data_quality_score
        confidence = incident.confidence.value

        parts = [f"{confidence.replace('_', ' ').title()} confidence detection"]

        if quality >= 0.95:
            parts.append(f"with excellent data quality ({quality:.0%})")
        elif quality >= 0.85:
            parts.append(f"with good data quality ({quality:.0%})")
        elif quality >= 0.70:
            parts.append(f"with moderate data quality ({quality:.0%}), some findings may be less reliable")
        else:
            parts.append(f"with low data quality ({quality:.0%}), manual verification recommended")

        if chain.paths:
            top_score = chain.paths[0].overall_score
            if top_score >= 0.8:
                parts.append("Strong causal signal identified")
            elif top_score >= 0.5:
                parts.append("Moderate causal signal — consider alternative explanations")
            else:
                parts.append("Weak causal signal — root cause determination is uncertain")

        return ". ".join(parts) + "."
