"""
Incident Cascade Correlator (Component F)

Detects when multiple incidents are part of the same cascading failure
rather than independent events. Uses temporal proximity, entity overlap,
and causal plausibility scoring to identify incident propagation chains.

This module implements a graph-based approach to cascade detection, mapping
known business incident dependencies and scoring temporal relationships to
identify when incidents are causally related vs coincidental.

Cascade Detection Algorithm:
    1. Build incident dependency graph (predefined causal relationships)
    2. Score all incident pairs for cascade likelihood
    3. Group high-scoring pairs into cascade chains
    4. Identify root incidents and build cascade paths

Scoring Components:
    - Temporal Weight: Exponential decay based on time gap
    - Entity Overlap: Jaccard similarity of evidence entities
    - Causal Plausibility: Binary check against dependency graph

Example:
    >>> correlator = CascadeCorrelator()
    >>> incidents = [incident1, incident2, incident3]
    >>> cascades = correlator.correlate(incidents)
    >>> for cascade in cascades:
    ...     print(f"Root: {cascade.root_incident_id}")
    ...     print(f"Path: {cascade.cascade_path}")
"""

import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

import structlog

from api.models.enums import IncidentType
from api.models.incidents import Incident, IncidentCascade

logger = structlog.get_logger()


class CascadeCorrelator:
    """
    Detects and groups cascading incident chains.

    Analyzes temporal, structural, and causal relationships between incidents
    to identify propagation patterns. Uses a predefined dependency graph to
    validate causal plausibility and a union-find algorithm for grouping.

    Attributes:
        cascade_score_threshold: Minimum score to consider incidents related (default: 0.3)
        temporal_decay_factor: Decay factor for temporal weight (default: 3.0 days)
        incident_dependency_graph: Predefined causal relationships between incident types

    Example:
        >>> correlator = CascadeCorrelator(cascade_score_threshold=0.4)
        >>> cascades = correlator.correlate(incidents)
    """

    # Predefined incident dependency graph: A -> [B, C] means A can cause B and C
    INCIDENT_DEPENDENCY_GRAPH = {
        IncidentType.FULFILLMENT_SLA_DEGRADATION: [
            IncidentType.SUPPORT_LOAD_SURGE,
            IncidentType.CUSTOMER_SATISFACTION_REGRESSION,
        ],
        IncidentType.SUPPORT_LOAD_SURGE: [
            IncidentType.CUSTOMER_SATISFACTION_REGRESSION,
            IncidentType.CHURN_ACCELERATION,
        ],
        IncidentType.CUSTOMER_SATISFACTION_REGRESSION: [
            IncidentType.CHURN_ACCELERATION
        ],
        IncidentType.SUPPLIER_DEPENDENCY_FAILURE: [
            IncidentType.FULFILLMENT_SLA_DEGRADATION
        ],
        IncidentType.REFUND_SPIKE: [IncidentType.MARGIN_COMPRESSION],
        IncidentType.MARGIN_COMPRESSION: [IncidentType.LIQUIDITY_CRUNCH_RISK],
        IncidentType.CHURN_ACCELERATION: [IncidentType.MARGIN_COMPRESSION],
    }

    def __init__(
        self,
        cascade_score_threshold: float = 0.3,
        temporal_decay_factor: float = 3.0,
    ):
        """
        Initialize the cascade correlator.

        Args:
            cascade_score_threshold: Minimum score to consider incidents related (0.0-1.0)
            temporal_decay_factor: Decay factor for temporal weight in days
        """
        self.cascade_score_threshold = cascade_score_threshold
        self.temporal_decay_factor = temporal_decay_factor
        self.logger = structlog.get_logger()

    def correlate(self, incidents: list[Incident]) -> list[IncidentCascade]:
        """
        Detect cascading incident patterns from a list of incidents.

        Analyzes temporal, entity, and causal relationships to group incidents
        into cascading failure chains. Each cascade represents a propagation
        pattern where one incident triggers downstream incidents.

        Args:
            incidents: List of detected incidents to analyze

        Returns:
            List of IncidentCascade objects representing detected cascade patterns

        Example:
            >>> incidents = [
            ...     Incident(incident_type="refund_spike", detected_at=dt1, ...),
            ...     Incident(incident_type="margin_compression", detected_at=dt2, ...),
            ... ]
            >>> cascades = correlator.correlate(incidents)
            >>> print(f"Detected {len(cascades)} cascade patterns")
        """
        if len(incidents) < 2:
            self.logger.info(
                "insufficient_incidents_for_cascade_detection",
                incident_count=len(incidents),
            )
            return []

        self.logger.info(
            "starting_cascade_correlation",
            incident_count=len(incidents),
            threshold=self.cascade_score_threshold,
        )

        # Sort incidents by detection time
        sorted_incidents = sorted(incidents, key=lambda i: i.detected_at)

        # Score all incident pairs
        cascade_pairs = []
        for i in range(len(sorted_incidents)):
            for j in range(i + 1, len(sorted_incidents)):
                incident_a = sorted_incidents[i]
                incident_b = sorted_incidents[j]

                score = self._compute_cascade_score(incident_a, incident_b)

                if score >= self.cascade_score_threshold:
                    cascade_pairs.append({
                        "incident_a": incident_a,
                        "incident_b": incident_b,
                        "score": score,
                    })

                    self.logger.debug(
                        "cascade_pair_detected",
                        incident_a_id=incident_a.incident_id,
                        incident_a_type=incident_a.incident_type,
                        incident_b_id=incident_b.incident_id,
                        incident_b_type=incident_b.incident_type,
                        score=round(score, 4),
                    )

        # Group pairs into cascades
        cascades = self._group_into_cascades(cascade_pairs, sorted_incidents)

        self.logger.info(
            "cascade_correlation_complete",
            cascades_detected=len(cascades),
            total_incidents=len(incidents),
        )

        return cascades

    def _compute_cascade_score(
        self, incident_a: Incident, incident_b: Incident
    ) -> float:
        """
        Compute cascade likelihood score for an incident pair.

        Combines temporal proximity, entity overlap, and causal plausibility
        into a single score. Returns 0.0 if incidents cannot be causally related.

        Args:
            incident_a: Earlier incident (potential cause)
            incident_b: Later incident (potential effect)

        Returns:
            Cascade score between 0.0 and 1.0

        Example:
            >>> score = correlator._compute_cascade_score(incident1, incident2)
            >>> if score > 0.3:
            ...     print("Likely part of same cascade")
        """
        # Ensure temporal ordering
        if incident_a.detected_at >= incident_b.detected_at:
            return 0.0

        # Check causal plausibility first (fast rejection)
        if not self._is_causally_plausible(
            incident_a.incident_type, incident_b.incident_type
        ):
            return 0.0

        # Compute component scores
        temporal_weight = self._compute_temporal_weight(
            incident_a.detected_at, incident_b.detected_at
        )
        entity_overlap = self._compute_entity_overlap(
            incident_a.evidence_event_ids, incident_b.evidence_event_ids
        )

        # Causal plausibility is binary (1.0 if passes, 0.0 if fails)
        causal_plausibility = 1.0

        # Combined score (multiplicative)
        cascade_score = temporal_weight * entity_overlap * causal_plausibility

        return round(cascade_score, 4)

    def _compute_temporal_weight(
        self, time_a: datetime, time_b: datetime
    ) -> float:
        """
        Compute temporal weight based on time gap between incidents.

        Uses exponential decay: exp(-|days_between| / decay_factor)
        Peaks at 1.0 for incidents happening on the same day, decays
        to ~0.05 at 3x decay_factor days.

        Args:
            time_a: Detection time of earlier incident
            time_b: Detection time of later incident

        Returns:
            Temporal weight between 0.0 and 1.0

        Example:
            >>> weight = correlator._compute_temporal_weight(dt1, dt2)
            >>> # Weight close to 1.0 for incidents 1 day apart
            >>> # Weight close to 0.0 for incidents >10 days apart
        """
        time_diff = time_b - time_a
        days_between = abs(time_diff.total_seconds()) / 86400.0

        # Exponential decay: peaks at same day, decays over time
        weight = math.exp(-days_between / self.temporal_decay_factor)

        return round(weight, 4)

    def _compute_entity_overlap(
        self, evidence_a: list[str], evidence_b: list[str]
    ) -> float:
        """
        Compute entity overlap using Jaccard similarity.

        Measures the proportion of shared evidence entities between incidents.
        Jaccard = |A ∩ B| / |A ∪ B|

        Args:
            evidence_a: Evidence event IDs from first incident
            evidence_b: Evidence event IDs from second incident

        Returns:
            Jaccard similarity between 0.0 and 1.0

        Example:
            >>> overlap = correlator._compute_entity_overlap(
            ...     ["evt_1", "evt_2", "evt_3"],
            ...     ["evt_2", "evt_3", "evt_4"]
            ... )
            >>> # Returns 0.5 (2 shared / 4 total unique)
        """
        set_a = set(evidence_a)
        set_b = set(evidence_b)

        if not set_a or not set_b:
            # No evidence in one or both incidents
            # Use a baseline overlap for events with no direct entity sharing
            # This allows cascades based purely on temporal and causal factors
            return 0.1

        intersection = set_a & set_b
        union = set_a | set_b

        if not union:
            return 0.0

        jaccard = len(intersection) / len(union)

        return round(jaccard, 4)

    def _is_causally_plausible(
        self, type_a: IncidentType, type_b: IncidentType
    ) -> bool:
        """
        Check if incident type A can causally lead to incident type B.

        Uses the predefined dependency graph to validate causal relationships.

        Args:
            type_a: Type of potential cause incident
            type_b: Type of potential effect incident

        Returns:
            True if A -> B is a known causal relationship, False otherwise

        Example:
            >>> plausible = correlator._is_causally_plausible(
            ...     IncidentType.REFUND_SPIKE,
            ...     IncidentType.MARGIN_COMPRESSION
            ... )
            >>> # Returns True (refund spike can cause margin compression)
        """
        downstream_types = self.INCIDENT_DEPENDENCY_GRAPH.get(type_a, [])
        return type_b in downstream_types

    def _group_into_cascades(
        self, cascade_pairs: list[dict], all_incidents: list[Incident]
    ) -> list[IncidentCascade]:
        """
        Group incident pairs into cascade chains using union-find.

        Builds connected components from incident pairs, identifies root
        incidents (earliest in each chain), and constructs cascade paths.

        Args:
            cascade_pairs: List of incident pairs with scores
            all_incidents: Full list of incidents for reference

        Returns:
            List of IncidentCascade objects

        Example:
            >>> pairs = [
            ...     {"incident_a": inc1, "incident_b": inc2, "score": 0.8},
            ...     {"incident_a": inc2, "incident_b": inc3, "score": 0.7},
            ... ]
            >>> cascades = correlator._group_into_cascades(pairs, all_incidents)
            >>> # Returns single cascade with 3 incidents
        """
        if not cascade_pairs:
            return []

        # Build adjacency list
        graph = defaultdict(list)
        incident_map = {inc.incident_id: inc for inc in all_incidents}

        for pair in cascade_pairs:
            inc_a = pair["incident_a"]
            inc_b = pair["incident_b"]
            graph[inc_a.incident_id].append(inc_b.incident_id)

        # Find connected components using DFS
        visited = set()
        cascades = []

        for incident in all_incidents:
            if incident.incident_id in visited:
                continue

            # Find all incidents in this cascade
            component = self._find_component(
                incident.incident_id, graph, visited, incident_map
            )

            if len(component) >= 2:
                # Build cascade from component
                cascade = self._build_cascade_from_component(component, incident_map)
                cascades.append(cascade)

        return cascades

    def _find_component(
        self,
        start_id: str,
        graph: dict,
        visited: set,
        incident_map: dict,
    ) -> list[str]:
        """
        Find all incidents in the same cascade using DFS.

        Args:
            start_id: Starting incident ID
            graph: Adjacency list of incident relationships
            visited: Set of already visited incident IDs
            incident_map: Map of incident IDs to Incident objects

        Returns:
            List of incident IDs in the same component
        """
        component = []
        stack = [start_id]

        # Build reverse graph for bidirectional traversal
        reverse_graph = defaultdict(list)
        for source, targets in graph.items():
            for target in targets:
                reverse_graph[target].append(source)

        while stack:
            incident_id = stack.pop()

            if incident_id in visited:
                continue

            visited.add(incident_id)
            component.append(incident_id)

            # Add forward connections
            for neighbor in graph.get(incident_id, []):
                if neighbor not in visited:
                    stack.append(neighbor)

            # Add reverse connections
            for neighbor in reverse_graph.get(incident_id, []):
                if neighbor not in visited:
                    stack.append(neighbor)

        return component

    def _build_cascade_from_component(
        self, component_ids: list[str], incident_map: dict
    ) -> IncidentCascade:
        """
        Build an IncidentCascade object from a connected component.

        Identifies the root incident, builds the cascade path, and computes
        aggregated blast radius metrics.

        Args:
            component_ids: List of incident IDs in this cascade
            incident_map: Map of incident IDs to Incident objects

        Returns:
            IncidentCascade object

        Example:
            >>> cascade = correlator._build_cascade_from_component(
            ...     ["inc_1", "inc_2", "inc_3"],
            ...     incident_map
            ... )
        """
        # Get incidents sorted by detection time
        incidents = [incident_map[iid] for iid in component_ids]
        incidents.sort(key=lambda i: i.detected_at)

        # Root incident is the earliest
        root_incident = incidents[0]

        # Build cascade path
        cascade_path = self._build_cascade_path(incidents)

        # Compute total blast radius
        total_blast_radius = self._aggregate_blast_radius(incidents)

        # Compute cascade severity score (weighted by incident severity)
        cascade_score = self._compute_cascade_severity_score(incidents)

        cascade = IncidentCascade(
            root_incident_id=root_incident.incident_id,
            incident_ids=[inc.incident_id for inc in incidents],
            cascade_path=cascade_path,
            total_blast_radius=total_blast_radius,
            cascade_score=cascade_score,
        )

        return cascade

    def _build_cascade_path(self, incidents: list[Incident]) -> list[str]:
        """
        Build human-readable cascade path showing incident type propagation.

        Args:
            incidents: List of incidents in temporal order

        Returns:
            List of incident type strings showing propagation path

        Example:
            >>> path = correlator._build_cascade_path(incidents)
            >>> # Returns ["refund_spike", "margin_compression", "liquidity_crunch_risk"]
        """
        return [str(inc.incident_type) for inc in incidents]

    def _aggregate_blast_radius(self, incidents: list[Incident]) -> dict:
        """
        Aggregate blast radius metrics across all incidents in cascade.

        Combines impact metrics from all incidents to compute total cascade impact.

        Args:
            incidents: List of incidents in the cascade

        Returns:
            Dictionary of aggregated blast radius metrics

        Example:
            >>> radius = correlator._aggregate_blast_radius(incidents)
            >>> print(radius["total_incidents"])
            >>> print(radius["total_evidence_events"])
        """
        total_evidence_events = sum(inc.evidence_event_count for inc in incidents)

        # Extract unique entities from evidence
        all_evidence_ids = set()
        for inc in incidents:
            all_evidence_ids.update(inc.evidence_event_ids)

        # Aggregate severity counts (use .value for enum string: "low", "medium", "high", "critical")
        severity_distribution = defaultdict(int)
        for inc in incidents:
            sev = inc.severity.value if hasattr(inc.severity, "value") else str(inc.severity)
            severity_distribution[sev] += 1

        # Find highest severity in cascade
        severity_order = ["low", "medium", "high", "critical"]
        max_severity = "low"
        for inc in incidents:
            inc_severity = inc.severity.value if hasattr(inc.severity, "value") else str(inc.severity)
            if inc_severity in severity_order and severity_order.index(inc_severity) > severity_order.index(max_severity):
                max_severity = inc_severity

        return {
            "total_incidents": len(incidents),
            "total_evidence_events": total_evidence_events,
            "unique_evidence_entities": len(all_evidence_ids),
            "severity_distribution": dict(severity_distribution),
            "max_severity": max_severity,
            "incident_types": [str(inc.incident_type) for inc in incidents],
        }

    def _compute_cascade_severity_score(self, incidents: list[Incident]) -> float:
        """
        Compute overall severity score for the cascade.

        Weights incident severity and combines with cascade length to produce
        a score between 0.0 and 1.0.

        Args:
            incidents: List of incidents in the cascade

        Returns:
            Cascade severity score (0.0-1.0)

        Example:
            >>> score = correlator._compute_cascade_severity_score(incidents)
            >>> # High score for long cascades with severe incidents
        """
        if not incidents:
            return 0.0

        # Severity weights
        severity_weights = {
            "low": 0.25,
            "medium": 0.50,
            "high": 0.75,
            "critical": 1.0,
        }

        # Average weighted severity (use .value for enum)
        total_weight = sum(
            severity_weights.get(
                inc.severity.value if hasattr(inc.severity, "value") else str(inc.severity),
                0.5,
            )
            for inc in incidents
        )
        avg_severity = total_weight / len(incidents)

        # Cascade length factor (longer cascades are worse)
        # Normalize to [0, 1] where 1 incident = 0, 5+ incidents = 1
        length_factor = min(len(incidents) - 1, 4) / 4.0

        # Combined score: weighted average of severity and length
        # 70% severity, 30% length
        cascade_score = (0.7 * avg_severity) + (0.3 * length_factor)

        return round(min(cascade_score, 1.0), 4)
