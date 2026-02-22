"""
RCA Explainer — Natural Language Explanation Generator.

This module generates human-readable explanations for root cause analysis
results. It transforms the algorithmic output (CausalChain with scored
nodes and paths) into clear, actionable narratives suitable for both
technical engineers and business stakeholders.

Explanation structure:
1. Executive summary: One-sentence root cause identification
2. Causal narrative: Step-by-step how the root cause propagated
3. Evidence summary: Key data points supporting the conclusion
4. Confidence assessment: How confident the algorithm is and why
5. Recommended actions: What to investigate or fix first

Version: rca_explainer_v1
"""

from datetime import datetime
from typing import Optional

import structlog

from api.models.rca import CausalChain, CausalNode, CausalPath, EvidenceCluster

logger = structlog.get_logger()

# Human-readable metric name mappings
METRIC_DISPLAY_NAMES = {
    "daily_revenue": "Daily Revenue",
    "daily_expenses": "Daily Expenses",
    "daily_refunds": "Daily Refunds",
    "refund_rate": "Refund Rate",
    "net_cash_proxy": "Net Cash Position",
    "expense_ratio": "Expense-to-Revenue Ratio",
    "margin_proxy": "Profit Margin",
    "dso_proxy": "Days Sales Outstanding (DSO)",
    "ar_aging_amount": "Accounts Receivable Aging",
    "ar_overdue_count": "Overdue Invoice Count",
    "dpo_proxy": "Days Payable Outstanding (DPO)",
    "order_volume": "Order Volume",
    "delivery_count": "Delivery Count",
    "late_delivery_count": "Late Delivery Count",
    "delivery_delay_rate": "Delivery Delay Rate",
    "fulfillment_backlog": "Fulfillment Backlog",
    "avg_delivery_delay_days": "Average Delivery Delay (Days)",
    "supplier_delay_rate": "Supplier Delay Rate",
    "supplier_delay_severity": "Supplier Delay Severity",
    "ticket_volume": "Support Ticket Volume",
    "ticket_close_volume": "Ticket Resolution Volume",
    "ticket_backlog": "Support Ticket Backlog",
    "avg_resolution_time": "Average Resolution Time",
    "review_score_avg": "Customer Review Score",
    "review_score_trend": "Review Score Trend",
    "churn_proxy": "Customer Churn Rate",
    "customer_concentration": "Revenue Concentration",
}

# Incident type display names
INCIDENT_DISPLAY_NAMES = {
    "REFUND_SPIKE": "Refund Spike",
    "FULFILLMENT_SLA_DEGRADATION": "Fulfillment SLA Degradation",
    "SUPPORT_LOAD_SURGE": "Support Load Surge",
    "CHURN_ACCELERATION": "Churn Acceleration",
    "MARGIN_COMPRESSION": "Margin Compression",
    "LIQUIDITY_CRUNCH_RISK": "Liquidity Crunch Risk",
    "SUPPLIER_DEPENDENCY_FAILURE": "Supplier Dependency Failure",
    "CUSTOMER_SATISFACTION_REGRESSION": "Customer Satisfaction Regression",
}

# Causal narrative templates
CAUSAL_TEMPLATES = {
    # (upstream_metric, downstream_metric) → narrative fragment
    ("supplier_delay_rate", "delivery_delay_rate"):
        "Supplier delays cascaded into delivery delays",
    ("supplier_delay_rate", "fulfillment_backlog"):
        "Supplier delays caused fulfillment backlog to grow",
    ("order_volume", "fulfillment_backlog"):
        "Increased order volume strained fulfillment capacity",
    ("fulfillment_backlog", "delivery_delay_rate"):
        "Growing fulfillment backlog led to delivery delays",
    ("delivery_delay_rate", "ticket_volume"):
        "Delivery delays triggered a surge in support tickets",
    ("delivery_delay_rate", "review_score_avg"):
        "Delivery delays drove down customer review scores",
    ("ticket_volume", "ticket_backlog"):
        "Support ticket surge created a growing ticket backlog",
    ("ticket_backlog", "avg_resolution_time"):
        "Ticket backlog increased average resolution times",
    ("avg_resolution_time", "review_score_avg"):
        "Longer resolution times degraded customer satisfaction",
    ("review_score_avg", "churn_proxy"):
        "Declining satisfaction accelerated customer churn",
    ("ticket_volume", "churn_proxy"):
        "Support overload contributed to customer churn",
    ("churn_proxy", "daily_revenue"):
        "Customer churn reduced daily revenue",
    ("refund_rate", "margin_proxy"):
        "Rising refund rate compressed profit margins",
    ("refund_rate", "daily_revenue"):
        "Increased refunds directly reduced net revenue",
    ("daily_revenue", "margin_proxy"):
        "Revenue decline pressured profit margins",
    ("daily_expenses", "expense_ratio"):
        "Rising expenses increased the expense-to-revenue ratio",
    ("expense_ratio", "margin_proxy"):
        "Higher expense ratio compressed margins",
    ("margin_proxy", "net_cash_proxy"):
        "Margin compression reduced available cash",
    ("dso_proxy", "ar_aging_amount"):
        "Longer payment cycles increased accounts receivable aging",
    ("ar_aging_amount", "net_cash_proxy"):
        "Growing AR aging reduced available cash",
}


class RCAExplainer:
    """
    Generates human-readable explanations for RCA results.

    Transforms CausalChain objects into structured natural language
    narratives that explain the root cause, propagation path, evidence,
    and recommended actions.

    Attributes:
        logger: Structured logger for observability

    Example:
        >>> explainer = RCAExplainer()
        >>> explanation = explainer.explain(causal_chain, incident_type="REFUND_SPIKE")
        >>> print(explanation["executive_summary"])
        >>> print(explanation["causal_narrative"])
    """

    def __init__(self):
        """Initialize the RCA explainer."""
        self.logger = structlog.get_logger()

    def explain(
        self,
        causal_chain: CausalChain,
        incident_type: Optional[str] = None,
        incident_metric: Optional[str] = None,
    ) -> dict:
        """
        Generate complete explanation for a CausalChain.

        Produces a structured explanation dictionary with multiple
        sections suitable for different audiences and display contexts.

        Args:
            causal_chain: The RCA result to explain
            incident_type: Optional incident type for context
            incident_metric: Optional incident metric name

        Returns:
            Dictionary containing:
            {
                "executive_summary": str,
                "causal_narrative": str,
                "evidence_summary": str,
                "confidence_assessment": str,
                "recommended_actions": list[str],
                "alternative_causes": list[str],
                "metadata": dict,
            }

        Example:
            >>> explanation = explainer.explain(chain, "REFUND_SPIKE", "refund_rate")
        """
        paths = causal_chain.paths

        if not paths:
            return self._empty_explanation()

        primary_path = paths[0]
        primary_cause = primary_path.nodes[0]

        # Generate each section
        executive_summary = self._generate_executive_summary(
            primary_cause, incident_type, incident_metric
        )
        causal_narrative = self._generate_causal_narrative(
            primary_path, incident_type, incident_metric
        )
        evidence_summary = self._generate_evidence_summary(primary_path)
        confidence_assessment = self._generate_confidence_assessment(
            primary_path, causal_chain
        )
        recommended_actions = self._generate_recommended_actions(
            primary_cause, incident_type
        )
        alternative_causes = self._generate_alternative_causes(paths[1:])

        explanation = {
            "executive_summary": executive_summary,
            "causal_narrative": causal_narrative,
            "evidence_summary": evidence_summary,
            "confidence_assessment": confidence_assessment,
            "recommended_actions": recommended_actions,
            "alternative_causes": alternative_causes,
            "metadata": {
                "algorithm_version": causal_chain.algorithm_version,
                "dependency_graph_version": causal_chain.dependency_graph_version,
                "paths_analyzed": len(paths),
                "primary_cause_score": primary_cause.contribution_score,
                "generated_at": datetime.utcnow().isoformat(),
            },
        }

        self.logger.info(
            "rca_explanation_generated",
            incident_type=incident_type,
            primary_cause=primary_cause.metric_name,
            primary_score=primary_cause.contribution_score,
            paths_count=len(paths),
        )

        return explanation

    def explain_path(self, path: CausalPath, incident_metric: Optional[str] = None) -> str:
        """
        Generate narrative explanation for a single causal path.

        Args:
            path: CausalPath to explain
            incident_metric: Optional incident metric for context

        Returns:
            Human-readable narrative string
        """
        if not path.nodes:
            return "No causal nodes identified."

        nodes = path.nodes
        parts = []

        for i, node in enumerate(nodes):
            metric_display = self._get_metric_display_name(node.metric_name)
            direction = "increased" if node.metric_zscore > 0 else "decreased"

            if i == 0:
                parts.append(
                    f"{metric_display} {direction} significantly "
                    f"(z-score: {node.metric_zscore:.1f}, "
                    f"from baseline {node.metric_baseline:.2f} to {node.metric_value:.2f})"
                )
            else:
                prev_node = nodes[i - 1]
                template = CAUSAL_TEMPLATES.get(
                    (prev_node.metric_name, node.metric_name)
                )
                if template:
                    parts.append(template)
                else:
                    parts.append(
                        f"which affected {metric_display} "
                        f"({direction} to {node.metric_value:.2f})"
                    )

        if incident_metric:
            incident_display = self._get_metric_display_name(incident_metric)
            parts.append(f"ultimately triggering the {incident_display} incident")

        narrative = ". ".join(parts) + "."
        return narrative

    # =========================================================================
    # Section Generators
    # =========================================================================

    def _generate_executive_summary(
        self,
        primary_cause: CausalNode,
        incident_type: Optional[str],
        incident_metric: Optional[str],
    ) -> str:
        """Generate one-sentence executive summary."""
        cause_display = self._get_metric_display_name(primary_cause.metric_name)
        incident_display = INCIDENT_DISPLAY_NAMES.get(
            incident_type, incident_type or "operational incident"
        )

        direction = "spike" if primary_cause.metric_zscore > 0 else "drop"
        confidence_word = self._score_to_confidence_word(
            primary_cause.contribution_score
        )

        return (
            f"Root cause analysis indicates with {confidence_word} confidence that "
            f"a {direction} in {cause_display} "
            f"(z-score: {primary_cause.metric_zscore:.1f}) "
            f"is the primary driver of the {incident_display} incident. "
            f"Contribution score: {primary_cause.contribution_score:.0%}."
        )

    def _generate_causal_narrative(
        self,
        primary_path: CausalPath,
        incident_type: Optional[str],
        incident_metric: Optional[str],
    ) -> str:
        """Generate step-by-step causal narrative."""
        parts = []

        # Opening
        parts.append("**Causal Chain Analysis:**\n")

        nodes = primary_path.nodes

        for i, node in enumerate(nodes):
            metric_display = self._get_metric_display_name(node.metric_name)
            direction = "increased" if node.metric_zscore > 0 else "decreased"
            abs_zscore = abs(node.metric_zscore)

            # Severity word
            if abs_zscore >= 5.0:
                severity = "dramatically"
            elif abs_zscore >= 3.0:
                severity = "significantly"
            elif abs_zscore >= 2.0:
                severity = "notably"
            else:
                severity = "moderately"

            step_num = i + 1

            if i == 0:
                parts.append(
                    f"{step_num}. **{metric_display}** {severity} {direction} "
                    f"from {node.metric_baseline:.2f} to {node.metric_value:.2f} "
                    f"(z-score: {node.metric_zscore:.1f}). "
                    f"This anomaly preceded the incident by "
                    f"{self._format_temporal_precedence(node.temporal_precedence)} "
                    f"and was detected with {node.data_quality_weight:.0%} data quality."
                )
            else:
                prev_node = nodes[i - 1]
                template = CAUSAL_TEMPLATES.get(
                    (prev_node.metric_name, node.metric_name),
                    f"Changes in {self._get_metric_display_name(prev_node.metric_name)} "
                    f"propagated to {metric_display}",
                )
                parts.append(
                    f"{step_num}. {template}, with {metric_display} moving "
                    f"from {node.metric_baseline:.2f} to {node.metric_value:.2f}."
                )

        # Closing
        if incident_type:
            incident_display = INCIDENT_DISPLAY_NAMES.get(
                incident_type, incident_type
            )
            parts.append(
                f"\nThis cascade ultimately triggered the **{incident_display}** incident."
            )

        return "\n".join(parts)

    def _generate_evidence_summary(self, primary_path: CausalPath) -> str:
        """Generate evidence summary from clusters."""
        parts = ["**Supporting Evidence:**\n"]

        has_evidence = False
        for node in primary_path.nodes:
            if node.evidence_clusters:
                has_evidence = True
                metric_display = self._get_metric_display_name(node.metric_name)
                parts.append(f"- **{metric_display}**:")
                for cluster in node.evidence_clusters:
                    parts.append(
                        f"  - {cluster.summary} "
                        f"({cluster.event_count} events"
                        + (f", ${cluster.total_amount:,.2f}" if cluster.total_amount else "")
                        + ")"
                    )

        if not has_evidence:
            parts.append(
                "- Evidence based on metric anomaly patterns and temporal correlation. "
                "Detailed event-level evidence available in incident timeline."
            )

        return "\n".join(parts)

    def _generate_confidence_assessment(
        self, primary_path: CausalPath, chain: CausalChain
    ) -> str:
        """Generate confidence assessment."""
        score = primary_path.overall_score
        confidence_word = self._score_to_confidence_word(score)

        parts = [f"**Confidence: {confidence_word.upper()}** ({score:.0%})\n"]

        # Explain confidence factors
        if primary_path.nodes:
            node = primary_path.nodes[0]

            factors = []
            if node.temporal_precedence >= 0.7:
                factors.append("strong temporal precedence (metric changed well before incident)")
            elif node.temporal_precedence >= 0.4:
                factors.append("moderate temporal precedence")
            else:
                factors.append("weak temporal precedence (concurrent changes)")

            if node.graph_proximity >= 0.5:
                factors.append("direct causal relationship in dependency graph")
            elif node.graph_proximity >= 0.2:
                factors.append("indirect causal pathway (multi-hop)")
            else:
                factors.append("distant relationship in dependency graph")

            if abs(node.anomaly_magnitude) >= 4.0:
                factors.append(f"very strong anomaly signal (z={node.metric_zscore:.1f})")
            elif abs(node.anomaly_magnitude) >= 2.5:
                factors.append(f"strong anomaly signal (z={node.metric_zscore:.1f})")
            else:
                factors.append(f"moderate anomaly signal (z={node.metric_zscore:.1f})")

            if node.data_quality_weight >= 0.9:
                factors.append("high data quality supporting evidence")
            elif node.data_quality_weight < 0.7:
                factors.append("data quality concerns may reduce confidence")

            parts.append("Contributing factors:")
            for factor in factors:
                parts.append(f"- {factor}")

        # Note alternative causes
        alt_count = len(chain.paths) - 1
        if alt_count > 0:
            parts.append(
                f"\n{alt_count} alternative causal path(s) identified with lower scores."
            )

        return "\n".join(parts)

    def _generate_recommended_actions(
        self,
        primary_cause: CausalNode,
        incident_type: Optional[str],
    ) -> list[str]:
        """Generate recommended investigation/remediation actions."""
        actions = []
        metric = primary_cause.metric_name

        # Domain-specific recommendations
        if metric in ["supplier_delay_rate", "supplier_delay_severity"]:
            actions.extend([
                "Review supplier performance metrics and SLA compliance",
                "Contact affected suppliers for root cause and ETA",
                "Evaluate alternative suppliers for critical materials",
                "Check for systemic supply chain disruptions (weather, logistics)",
            ])
        elif metric in ["delivery_delay_rate", "fulfillment_backlog", "avg_delivery_delay_days"]:
            actions.extend([
                "Review fulfillment center capacity and staffing levels",
                "Check for process bottlenecks in pick/pack/ship workflow",
                "Evaluate carrier performance and routing efficiency",
                "Consider expedited shipping for delayed orders",
            ])
        elif metric in ["ticket_volume", "ticket_backlog", "avg_resolution_time"]:
            actions.extend([
                "Analyze support ticket categories for common themes",
                "Scale support team capacity (temporary or permanent)",
                "Implement self-service solutions for frequent issues",
                "Review and optimize ticket routing and prioritization",
            ])
        elif metric in ["refund_rate", "daily_refunds"]:
            actions.extend([
                "Analyze refund reasons by product category and customer segment",
                "Review product quality and fulfillment accuracy",
                "Check for changes in refund policy or processing",
                "Investigate potential fraud patterns in refund requests",
            ])
        elif metric in ["churn_proxy"]:
            actions.extend([
                "Identify churning customer segments and patterns",
                "Launch retention campaigns for at-risk customers",
                "Review recent changes to pricing, product, or service",
                "Conduct exit interviews or surveys with churned customers",
            ])
        elif metric in ["margin_proxy", "expense_ratio"]:
            actions.extend([
                "Review cost structure for unexpected increases",
                "Analyze revenue mix for margin dilution",
                "Check for pricing errors or unauthorized discounts",
                "Evaluate vendor contracts for cost optimization",
            ])
        elif metric in ["net_cash_proxy", "ar_aging_amount", "dso_proxy"]:
            actions.extend([
                "Review accounts receivable aging report",
                "Accelerate collections on overdue invoices",
                "Evaluate payment terms and credit policies",
                "Assess cash flow forecast for upcoming obligations",
            ])
        else:
            actions.extend([
                f"Investigate anomaly in {self._get_metric_display_name(metric)}",
                "Review recent changes that may have impacted this metric",
                "Check upstream dependencies for contributing factors",
            ])

        # Always add general monitoring action
        actions.append(
            "Set up enhanced monitoring on affected metrics to detect recurrence"
        )

        return actions

    def _generate_alternative_causes(
        self, alternative_paths: list[CausalPath]
    ) -> list[str]:
        """Generate list of alternative cause summaries."""
        alternatives = []

        for path in alternative_paths[:3]:  # Top 3 alternatives
            if path.nodes:
                node = path.nodes[0]
                metric_display = self._get_metric_display_name(node.metric_name)
                alternatives.append(
                    f"{metric_display} (score: {path.overall_score:.0%}, "
                    f"z-score: {node.metric_zscore:.1f})"
                )

        return alternatives

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_metric_display_name(self, metric_name: str) -> str:
        """Get human-readable display name for a metric."""
        return METRIC_DISPLAY_NAMES.get(metric_name, metric_name.replace("_", " ").title())

    def _score_to_confidence_word(self, score: float) -> str:
        """Convert numeric score to confidence word."""
        if score >= 0.85:
            return "very high"
        elif score >= 0.70:
            return "high"
        elif score >= 0.50:
            return "moderate"
        elif score >= 0.30:
            return "low"
        else:
            return "very low"

    def _format_temporal_precedence(self, precedence_score: float) -> str:
        """Format temporal precedence score as human-readable duration."""
        if precedence_score >= 0.8:
            return "several days"
        elif precedence_score >= 0.6:
            return "1-3 days"
        elif precedence_score >= 0.4:
            return "approximately one day"
        elif precedence_score >= 0.2:
            return "hours"
        else:
            return "a short time"

    def _empty_explanation(self) -> dict:
        """Return empty explanation when no causal paths exist."""
        return {
            "executive_summary": "No root cause identified. Insufficient data for causal analysis.",
            "causal_narrative": "The RCA engine could not determine a definitive root cause for this incident.",
            "evidence_summary": "No supporting evidence available.",
            "confidence_assessment": "Confidence: VERY LOW — Insufficient data for analysis.",
            "recommended_actions": [
                "Ensure sufficient historical data is available (minimum 14 days)",
                "Verify data quality in the ingestion pipeline",
                "Review incident classification and affected metrics",
            ],
            "alternative_causes": [],
            "metadata": {
                "algorithm_version": "BRE-RCA-v1",
                "paths_analyzed": 0,
                "generated_at": datetime.utcnow().isoformat(),
            },
        }
