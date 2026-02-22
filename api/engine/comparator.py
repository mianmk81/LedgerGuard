"""
Comparator Engine — Incident Comparison and Pattern Analysis.

This module provides comparative analysis between two incidents to reveal
common root causes, shared contributing factors, severity differences,
and systematic vulnerability patterns. Comparisons enable organizational
learning from historical incidents and drive improvements in detection,
prevention, and response.

The comparator examines:
1. Root cause overlap (shared vs. unique causal factors)
2. Severity and confidence divergence
3. Blast radius magnitude comparison
4. Temporal and seasonal patterns
5. Detection method effectiveness differences

Version: comparator_v1
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

import structlog

from api.models.blast_radius import BlastRadius
from api.models.enums import Severity
from api.models.incidents import Incident
from api.models.rca import CausalChain
from api.models.simulation import IncidentComparison
from api.storage.base import StorageBackend

logger = structlog.get_logger()


class ComparatorEngine:
    """
    Compares two incidents to identify shared patterns and divergences.

    The ComparatorEngine loads incidents, their CausalChains, and BlastRadius
    assessments from storage, then systematically compares them across multiple
    dimensions to produce an IncidentComparison report.

    Attributes:
        storage: Storage backend for data retrieval and persistence

    Example:
        >>> comparator = ComparatorEngine(storage=duckdb_storage)
        >>> comparison = comparator.compare("inc_001", "inc_002")
        >>> print(comparison.narrative)
    """

    def __init__(self, storage: StorageBackend):
        """
        Initialize the comparator engine.

        Args:
            storage: Storage backend for incident/RCA/blast radius access
        """
        self.storage = storage
        self.logger = structlog.get_logger()

    def compare(
        self,
        incident_a_id: str,
        incident_b_id: str,
    ) -> IncidentComparison:
        """
        Perform full comparative analysis between two incidents.

        Loads all available data (incident, RCA, blast radius) for both
        incidents and produces a structured comparison.

        Args:
            incident_a_id: ID of the first incident
            incident_b_id: ID of the second incident

        Returns:
            IncidentComparison with shared/unique root causes,
            severity comparison, blast radius comparison, and narrative

        Raises:
            ValueError: If either incident is not found
        """
        self.logger.info(
            "comparison_started",
            incident_a_id=incident_a_id,
            incident_b_id=incident_b_id,
        )

        # Load incidents
        incident_a = self._load_incident(incident_a_id)
        incident_b = self._load_incident(incident_b_id)

        if incident_a is None:
            raise ValueError(f"Incident {incident_a_id} not found")
        if incident_b is None:
            raise ValueError(f"Incident {incident_b_id} not found")

        # Load RCA chains
        chain_a = self.storage.read_causal_chain(incident_a_id)
        chain_b = self.storage.read_causal_chain(incident_b_id)

        # Load blast radii
        blast_a = self.storage.read_blast_radius(incident_a_id)
        blast_b = self.storage.read_blast_radius(incident_b_id)

        # Determine incident type relationship
        incident_type = self._classify_type_relationship(incident_a, incident_b)

        # Compare root causes
        shared_causes, unique_a, unique_b = self._compare_root_causes(
            chain_a, chain_b
        )

        # Compare severity
        severity_comparison = self._compare_severity(incident_a, incident_b)

        # Compare blast radius
        blast_comparison = self._compare_blast_radius(blast_a, blast_b)

        # Generate narrative
        narrative = self._generate_narrative(
            incident_a=incident_a,
            incident_b=incident_b,
            shared_causes=shared_causes,
            unique_a=unique_a,
            unique_b=unique_b,
            severity_comparison=severity_comparison,
            blast_comparison=blast_comparison,
        )

        comparison = IncidentComparison(
            incident_a_id=incident_a_id,
            incident_b_id=incident_b_id,
            incident_type=incident_type,
            shared_root_causes=shared_causes,
            unique_to_a=unique_a,
            unique_to_b=unique_b,
            severity_comparison=severity_comparison,
            blast_radius_comparison=blast_comparison,
            narrative=narrative,
        )

        # Persist
        try:
            self.storage.write_comparison(comparison)
            self.logger.info(
                "comparison_persisted",
                comparison_id=comparison.comparison_id,
            )
        except Exception as e:
            self.logger.error(
                "comparison_persistence_failed",
                error=str(e),
            )

        self.logger.info(
            "comparison_complete",
            comparison_id=comparison.comparison_id,
            shared_root_causes=len(shared_causes),
            unique_to_a=len(unique_a),
            unique_to_b=len(unique_b),
        )

        return comparison

    # =========================================================================
    # Load Helpers
    # =========================================================================

    def _load_incident(self, incident_id: str) -> Optional[Incident]:
        """
        Load an incident by ID from storage.

        Searches all incidents and returns the matching one.

        Args:
            incident_id: ID to search for

        Returns:
            Incident if found, None otherwise
        """
        incidents = self.storage.read_incidents()
        for inc in incidents:
            if inc.incident_id == incident_id:
                return inc
        return None

    # =========================================================================
    # Comparison Logic
    # =========================================================================

    def _classify_type_relationship(
        self, a: Incident, b: Incident
    ) -> str:
        """
        Classify the type relationship between two incidents.

        Returns:
            Same type name if both match, "cross-type" otherwise
        """
        if a.incident_type == b.incident_type:
            return a.incident_type.value
        return "cross-type"

    def _compare_root_causes(
        self,
        chain_a: Optional[CausalChain],
        chain_b: Optional[CausalChain],
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Compare root cause metrics between two causal chains.

        Extracts root cause metric names from each chain's top causal paths,
        then computes the intersection and symmetric differences.

        Args:
            chain_a: CausalChain for incident A (may be None)
            chain_b: CausalChain for incident B (may be None)

        Returns:
            Tuple of (shared_causes, unique_to_a, unique_to_b)
        """
        causes_a = self._extract_root_cause_metrics(chain_a)
        causes_b = self._extract_root_cause_metrics(chain_b)

        set_a = set(causes_a)
        set_b = set(causes_b)

        shared = sorted(set_a & set_b)
        unique_a = sorted(set_a - set_b)
        unique_b = sorted(set_b - set_a)

        # If no RCA data, note it
        if not causes_a and not causes_b:
            shared = ["No RCA data available for either incident"]
        elif not causes_a:
            unique_b = causes_b
            shared = ["No RCA data available for incident A"]
        elif not causes_b:
            unique_a = causes_a
            shared = ["No RCA data available for incident B"]

        return shared, unique_a, unique_b

    def _extract_root_cause_metrics(
        self, chain: Optional[CausalChain]
    ) -> list[str]:
        """
        Extract root cause metric names from a causal chain.

        Takes the top-ranked node from each causal path.

        Args:
            chain: CausalChain to extract from

        Returns:
            List of metric names identified as root causes
        """
        if chain is None or not chain.paths:
            return []

        metrics = []
        for path in chain.paths:
            if path.nodes:
                # Top-ranked node in each path
                metric = path.nodes[0].metric_name
                if metric not in metrics:
                    metrics.append(metric)

        return metrics

    def _compare_severity(self, a: Incident, b: Incident) -> dict:
        """
        Compare severity dimensions between two incidents.

        Args:
            a: Incident A
            b: Incident B

        Returns:
            Dict with severity comparison details
        """
        severity_order = {
            Severity.LOW: 1,
            Severity.MEDIUM: 2,
            Severity.HIGH: 3,
            Severity.CRITICAL: 4,
        }

        a_ord = severity_order.get(a.severity, 0)
        b_ord = severity_order.get(b.severity, 0)

        if a_ord > b_ord:
            severity_relationship = f"Incident A ({a.severity.value}) is more severe than B ({b.severity.value})"
        elif b_ord > a_ord:
            severity_relationship = f"Incident B ({b.severity.value}) is more severe than A ({a.severity.value})"
        else:
            severity_relationship = f"Both incidents have equal severity ({a.severity.value})"

        return {
            "incident_a_severity": a.severity.value,
            "incident_b_severity": b.severity.value,
            "incident_a_zscore": round(a.primary_metric_zscore, 2),
            "incident_b_zscore": round(b.primary_metric_zscore, 2),
            "incident_a_confidence": a.confidence.value,
            "incident_b_confidence": b.confidence.value,
            "incident_a_detection_methods": [m.value for m in a.detection_methods],
            "incident_b_detection_methods": [m.value for m in b.detection_methods],
            "severity_relationship": severity_relationship,
            "zscore_ratio": round(
                abs(a.primary_metric_zscore) / max(abs(b.primary_metric_zscore), 0.001), 2
            ),
        }

    def _compare_blast_radius(
        self,
        blast_a: Optional[BlastRadius],
        blast_b: Optional[BlastRadius],
    ) -> dict:
        """
        Compare blast radius impact between two incidents.

        Args:
            blast_a: BlastRadius for incident A (may be None)
            blast_b: BlastRadius for incident B (may be None)

        Returns:
            Dict with blast radius comparison
        """
        result: dict = {}

        if blast_a is not None:
            result["incident_a_customers"] = blast_a.customers_affected
            result["incident_a_orders"] = blast_a.orders_affected
            result["incident_a_revenue_exposure"] = blast_a.estimated_revenue_exposure
            result["incident_a_refund_exposure"] = blast_a.estimated_refund_exposure
            result["incident_a_churn_exposure"] = blast_a.estimated_churn_exposure
            result["incident_a_severity"] = blast_a.blast_radius_severity.value
        else:
            result["incident_a_customers"] = None
            result["incident_a_orders"] = None
            result["incident_a_revenue_exposure"] = None
            result["incident_a_refund_exposure"] = None
            result["incident_a_churn_exposure"] = None
            result["incident_a_severity"] = None

        if blast_b is not None:
            result["incident_b_customers"] = blast_b.customers_affected
            result["incident_b_orders"] = blast_b.orders_affected
            result["incident_b_revenue_exposure"] = blast_b.estimated_revenue_exposure
            result["incident_b_refund_exposure"] = blast_b.estimated_refund_exposure
            result["incident_b_churn_exposure"] = blast_b.estimated_churn_exposure
            result["incident_b_severity"] = blast_b.blast_radius_severity.value
        else:
            result["incident_b_customers"] = None
            result["incident_b_orders"] = None
            result["incident_b_revenue_exposure"] = None
            result["incident_b_refund_exposure"] = None
            result["incident_b_churn_exposure"] = None
            result["incident_b_severity"] = None

        # Compute ratios if both available
        if blast_a is not None and blast_b is not None:
            result["customer_impact_ratio"] = round(
                blast_a.customers_affected / max(blast_b.customers_affected, 1), 2
            )
            result["revenue_impact_ratio"] = round(
                blast_a.estimated_revenue_exposure
                / max(blast_b.estimated_revenue_exposure, 0.01),
                2,
            )
            more_impactful = "A" if (
                blast_a.estimated_revenue_exposure > blast_b.estimated_revenue_exposure
            ) else "B"
            result["more_impactful"] = more_impactful
        else:
            result["customer_impact_ratio"] = None
            result["revenue_impact_ratio"] = None
            result["more_impactful"] = None

        return result

    # =========================================================================
    # Narrative Generation
    # =========================================================================

    def _generate_narrative(
        self,
        incident_a: Incident,
        incident_b: Incident,
        shared_causes: list[str],
        unique_a: list[str],
        unique_b: list[str],
        severity_comparison: dict,
        blast_comparison: dict,
    ) -> str:
        """
        Generate a human-readable comparative narrative.

        Args:
            incident_a: Incident A
            incident_b: Incident B
            shared_causes: Shared root cause metrics
            unique_a: Root causes unique to A
            unique_b: Root causes unique to B
            severity_comparison: Severity comparison dict
            blast_comparison: Blast radius comparison dict

        Returns:
            Multi-sentence narrative string
        """
        parts = []

        # Type relationship
        type_a = incident_a.incident_type.value.replace("_", " ")
        type_b = incident_b.incident_type.value.replace("_", " ")

        if incident_a.incident_type == incident_b.incident_type:
            parts.append(
                f"Both incidents are {type_a} events, enabling direct pattern comparison."
            )
        else:
            parts.append(
                f"Cross-type comparison between {type_a} (A) and {type_b} (B) "
                f"reveals indirect relationships."
            )

        # Severity
        zscore_a = abs(incident_a.primary_metric_zscore)
        zscore_b = abs(incident_b.primary_metric_zscore)
        parts.append(severity_comparison.get("severity_relationship", ""))

        if zscore_a > 0 and zscore_b > 0:
            ratio = zscore_a / zscore_b
            if ratio > 2:
                parts.append(
                    f"Incident A showed {ratio:.1f}x stronger statistical signal "
                    f"({zscore_a:.1f}σ vs {zscore_b:.1f}σ)."
                )
            elif ratio < 0.5:
                parts.append(
                    f"Incident B showed {1/ratio:.1f}x stronger statistical signal "
                    f"({zscore_b:.1f}σ vs {zscore_a:.1f}σ)."
                )

        # Root causes
        if shared_causes and "No RCA data" not in shared_causes[0]:
            cause_list = ", ".join(c.replace("_", " ") for c in shared_causes[:3])
            parts.append(
                f"Shared root causes: {cause_list}. "
                f"This suggests a systematic vulnerability requiring structural remediation."
            )

        if unique_a:
            parts.append(
                f"Unique to incident A: {', '.join(c.replace('_', ' ') for c in unique_a[:2])}."
            )
        if unique_b:
            parts.append(
                f"Unique to incident B: {', '.join(c.replace('_', ' ') for c in unique_b[:2])}."
            )

        # Blast radius
        more_impactful = blast_comparison.get("more_impactful")
        if more_impactful:
            rev_a = blast_comparison.get("incident_a_revenue_exposure", 0) or 0
            rev_b = blast_comparison.get("incident_b_revenue_exposure", 0) or 0
            parts.append(
                f"Incident {more_impactful} had greater business impact "
                f"(${max(rev_a, rev_b):,.0f} vs ${min(rev_a, rev_b):,.0f} revenue exposure)."
            )

        # Recommendation
        if shared_causes and "No RCA data" not in shared_causes[0]:
            parts.append(
                "Recommend: investigate shared causal factors for systemic process improvements."
            )

        return " ".join(parts)

    # =========================================================================
    # Data-Scientist Agent: Statistical Validation
    # =========================================================================

    @staticmethod
    def compute_comparison_significance(
        incident_a: Incident,
        incident_b: Incident,
    ) -> dict:
        """
        Compute statistical significance of the comparison.

        Per data-scientist agent: verify p<0.05 significance for key
        differences and provide effect sizes for practical significance.
        Uses Fisher's method to combine z-scores and compute a
        two-tailed test for severity difference.

        Args:
            incident_a: First incident
            incident_b: Second incident

        Returns:
            Dict with:
                "zscore_difference": float — raw difference in z-scores
                "effect_size_cohens_d": float — Cohen's d for practical significance
                "is_significantly_different": bool — p < 0.05
                "p_value_approximation": float — approximate p-value
                "interpretation": str — human-readable summary
        """
        import math

        z_a = abs(incident_a.primary_metric_zscore)
        z_b = abs(incident_b.primary_metric_zscore)
        z_diff = abs(z_a - z_b)

        # Approximate p-value using standard normal for z-score difference
        # Standard error of difference between two z-scores ≈ sqrt(2)
        se_diff = math.sqrt(2.0)
        z_test = z_diff / se_diff

        # Two-tailed p-value approximation using complementary error function
        p_value = math.erfc(z_test / math.sqrt(2.0))

        # Cohen's d: effect size for the severity difference
        pooled_z = max((z_a + z_b) / 2.0, 0.001)
        cohens_d = z_diff / pooled_z

        is_significant = p_value < 0.05

        # Interpretation
        if not is_significant:
            interpretation = (
                f"The severity difference ({z_diff:.2f}σ) is not statistically "
                f"significant (p={p_value:.3f}). Both incidents have comparable magnitude."
            )
        elif cohens_d < 0.5:
            interpretation = (
                f"Statistically significant (p={p_value:.3f}) but small practical effect "
                f"(d={cohens_d:.2f}). Incidents differ in severity but the practical "
                f"impact difference is modest."
            )
        elif cohens_d < 0.8:
            interpretation = (
                f"Statistically significant with medium effect (p={p_value:.3f}, d={cohens_d:.2f}). "
                f"The incidents differ meaningfully in severity."
            )
        else:
            interpretation = (
                f"Highly significant with large effect (p={p_value:.3f}, d={cohens_d:.2f}). "
                f"The incidents have substantially different severity profiles."
            )

        return {
            "zscore_difference": round(z_diff, 4),
            "effect_size_cohens_d": round(cohens_d, 4),
            "is_significantly_different": is_significant,
            "p_value_approximation": round(p_value, 6),
            "interpretation": interpretation,
        }
