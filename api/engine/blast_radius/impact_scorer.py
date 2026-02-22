"""
Impact Scorer — Quantitative Business Impact Assessment.

This module quantifies the monetary and operational impact of an incident
by analyzing the affected entities and their financial exposure. The scorer
computes revenue at risk, refund liability, and churn exposure, then
classifies the overall severity.

Severity classification thresholds:
- CONTAINED: < 50 customers, < $10k revenue, < $2k refunds
- SIGNIFICANT: < 500 customers, < $100k revenue, < $25k refunds
- SEVERE: < 2000 customers, < $500k revenue, < $100k refunds
- CATASTROPHIC: >= 2000 customers or >= $500k revenue

Version: blast_impact_v1
"""

from typing import Optional

import structlog

from api.models.enums import BlastRadiusSeverity
from api.models.events import CanonicalEvent
from api.models.incidents import Incident

logger = structlog.get_logger()


# Severity thresholds
SEVERITY_THRESHOLDS = {
    "CATASTROPHIC": {
        "customers": 2000,
        "revenue": 500000.0,
        "refunds": 100000.0,
        "downstream": 3,
    },
    "SEVERE": {
        "customers": 500,
        "revenue": 100000.0,
        "refunds": 25000.0,
        "downstream": 2,
    },
    "SIGNIFICANT": {
        "customers": 50,
        "revenue": 10000.0,
        "refunds": 2000.0,
        "downstream": 1,
    },
    # Below SIGNIFICANT = CONTAINED
}


class ImpactScorer:
    """
    Quantifies monetary and operational impact of incidents.

    Analyzes affected entities and their associated financial metrics
    to compute revenue exposure, refund liability, and churn risk.

    Attributes:
        churn_risk_multiplier: Multiplier for estimating churn from ticket/review signals
        logger: Structured logger

    Example:
        >>> scorer = ImpactScorer()
        >>> impact = scorer.score_impact(events, entity_sets, incident)
        >>> print(f"Revenue at risk: ${impact['revenue_exposure']:,.2f}")
    """

    def __init__(self, churn_risk_multiplier: float = 0.06):
        """
        Initialize the impact scorer.

        Args:
            churn_risk_multiplier: Fraction of affected customers at churn risk
                (default 0.06 = 6% based on industry average)
        """
        self.churn_risk_multiplier = churn_risk_multiplier
        self.logger = structlog.get_logger()

    def score_impact(
        self,
        events: list[CanonicalEvent],
        entity_sets: dict[str, set[str]],
        incident: Incident,
    ) -> dict:
        """
        Compute full impact scores from events and entities.

        Args:
            events: Canonical events in the incident window
            entity_sets: Entity sets from graph traversal
            incident: The incident being assessed

        Returns:
            Dict with impact metrics:
            {
                "revenue_exposure": float,
                "refund_exposure": float,
                "churn_exposure": int,
                "avg_order_value": float,
                "total_event_amount": float,
            }
        """
        # Revenue exposure: sum of amounts from revenue-related events
        revenue_events = [
            e for e in events
            if e.event_type.value in [
                "invoice_paid", "payment_received", "invoice_issued"
            ]
        ]
        revenue_exposure = sum(
            e.amount for e in revenue_events if e.amount is not None
        )

        # Refund exposure: sum of amounts from refund/credit events
        refund_events = [
            e for e in events
            if e.event_type.value in [
                "refund_issued", "credit_memo_issued"
            ]
        ]
        refund_exposure = sum(
            e.amount for e in refund_events if e.amount is not None
        )

        # Total event amount
        total_amount = sum(
            e.amount for e in events if e.amount is not None
        )

        # Average order value
        order_events = [
            e for e in events
            if e.event_type.value in ["order_placed", "invoice_issued"]
            and e.amount is not None
        ]
        avg_order_value = (
            sum(e.amount for e in order_events) / len(order_events)
            if order_events
            else 0.0
        )

        # Churn exposure: estimated customers at risk
        customers_affected = len(entity_sets.get("customer", set()))
        churn_exposure = int(customers_affected * self.churn_risk_multiplier)

        # Adjust based on incident type severity signals
        incident_type = incident.incident_type.value
        if incident_type in [
            "churn_acceleration", "customer_satisfaction_regression"
        ]:
            # Higher churn risk for customer-facing incidents
            churn_exposure = int(customers_affected * self.churn_risk_multiplier * 2.0)
        elif incident_type in ["refund_spike"]:
            # Refund spike may indicate product quality issue
            churn_exposure = int(customers_affected * self.churn_risk_multiplier * 1.5)

        impact = {
            "revenue_exposure": round(revenue_exposure, 2),
            "refund_exposure": round(refund_exposure, 2),
            "churn_exposure": churn_exposure,
            "avg_order_value": round(avg_order_value, 2),
            "total_event_amount": round(total_amount, 2),
        }

        self.logger.debug(
            "impact_scored",
            revenue_exposure=impact["revenue_exposure"],
            refund_exposure=impact["refund_exposure"],
            churn_exposure=impact["churn_exposure"],
            events_analyzed=len(events),
        )

        return impact

    def classify_severity(
        self,
        customers_affected: int,
        revenue_exposure: float,
        refund_exposure: float,
        downstream_count: int = 0,
    ) -> BlastRadiusSeverity:
        """
        Classify blast radius severity based on impact thresholds.

        Uses a waterfall classification: checks CATASTROPHIC first, then
        SEVERE, then SIGNIFICANT. Anything below is CONTAINED.

        Args:
            customers_affected: Number of customers impacted
            revenue_exposure: Revenue at risk in USD
            refund_exposure: Refund liability in USD
            downstream_count: Number of downstream cascade incidents

        Returns:
            BlastRadiusSeverity classification
        """
        # Check CATASTROPHIC
        cat = SEVERITY_THRESHOLDS["CATASTROPHIC"]
        if (
            customers_affected >= cat["customers"]
            or revenue_exposure >= cat["revenue"]
            or refund_exposure >= cat["refunds"]
            or downstream_count >= cat["downstream"]
        ):
            return BlastRadiusSeverity.CATASTROPHIC

        # Check SEVERE
        sev = SEVERITY_THRESHOLDS["SEVERE"]
        if (
            customers_affected >= sev["customers"]
            or revenue_exposure >= sev["revenue"]
            or refund_exposure >= sev["refunds"]
            or downstream_count >= sev["downstream"]
        ):
            return BlastRadiusSeverity.SEVERE

        # Check SIGNIFICANT
        sig = SEVERITY_THRESHOLDS["SIGNIFICANT"]
        if (
            customers_affected >= sig["customers"]
            or revenue_exposure >= sig["revenue"]
            or refund_exposure >= sig["refunds"]
            or downstream_count >= sig["downstream"]
        ):
            return BlastRadiusSeverity.SIGNIFICANT

        return BlastRadiusSeverity.CONTAINED
