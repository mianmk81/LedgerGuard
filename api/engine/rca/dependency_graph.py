"""
Business Dependency Graph for BRE-RCA Algorithm.

This module defines the static causal dependency graph representing known
business metric relationships. The graph is a directed acyclic graph (DAG)
where edges represent causal influence: A → B means "A causally influences B".

The dependency graph encodes domain knowledge about how business metrics
propagate through the organization:
- Supply chain delays → fulfillment delays → customer dissatisfaction
- Refunds → margin compression → cash flow issues
- Support load → resolution time → customer churn

This graph is used by the RCA engine to:
1. Identify upstream candidate root causes for an incident
2. Compute graph proximity scores (shortest path length)
3. Generate causal path narratives

Graph Structure:
- Nodes: Business health metric names (27 total)
- Edges: Directed causal relationships (21 total)
- Properties: DAG (no cycles), connected (all nodes reachable)

Version: dep_graph_v2
Last Updated: 2026-02-10
"""

from typing import Optional

import networkx as nx
import numpy as np
import structlog

logger = structlog.get_logger()

# Business dependency edges defining causal relationships
# Format: (upstream_metric, downstream_metric)
# Interpretation: upstream_metric causally influences downstream_metric
BUSINESS_DEPENDENCY_EDGES = [
    # Supply chain → fulfillment chain
    ("supplier_delay_rate", "delivery_delay_rate"),
    ("supplier_delay_rate", "fulfillment_backlog"),

    # Order volume → fulfillment pressure
    ("order_volume", "fulfillment_backlog"),

    # Fulfillment → delivery performance
    ("fulfillment_backlog", "delivery_delay_rate"),

    # Delivery performance → customer experience
    ("delivery_delay_rate", "ticket_volume"),
    ("delivery_delay_rate", "review_score_avg"),

    # Support load → support quality
    ("ticket_volume", "ticket_backlog"),
    ("ticket_backlog", "avg_resolution_time"),

    # Support quality → satisfaction
    ("avg_resolution_time", "review_score_avg"),

    # Satisfaction → retention
    ("review_score_avg", "churn_proxy"),
    ("ticket_volume", "churn_proxy"),

    # Retention → revenue
    ("churn_proxy", "daily_revenue"),

    # Refunds → margin and revenue
    ("refund_rate", "margin_proxy"),
    ("refund_rate", "daily_revenue"),

    # Revenue → margin
    ("daily_revenue", "margin_proxy"),

    # Expenses → cost ratios
    ("daily_expenses", "expense_ratio"),
    ("expense_ratio", "margin_proxy"),

    # Margin → cash
    ("margin_proxy", "net_cash_proxy"),

    # AR aging → cash
    ("dso_proxy", "ar_aging_amount"),
    ("ar_aging_amount", "net_cash_proxy"),
]

# Impact direction for causal graph visualization (profit=green, loss=red)
METRIC_IMPACT = {
    "daily_revenue": "positive",
    "margin_proxy": "positive",
    "net_cash_proxy": "positive",
    "review_score_avg": "positive",
    "order_volume": "positive",
    "refund_rate": "negative",
    "churn_proxy": "negative",
    "delivery_delay_rate": "negative",
    "ticket_volume": "negative",
    "ticket_backlog": "negative",
    "expense_ratio": "negative",
    "daily_expenses": "negative",
    "ar_aging_amount": "negative",
    "dso_proxy": "negative",
    "fulfillment_backlog": "negative",
    "avg_resolution_time": "negative",
    "supplier_delay_rate": "negative",
}

# Edge impact: when upstream increases, does downstream increase (pos) or decrease (neg)?
EDGE_IMPACT = {
    ("supplier_delay_rate", "delivery_delay_rate"): "negative",
    ("supplier_delay_rate", "fulfillment_backlog"): "negative",
    ("order_volume", "fulfillment_backlog"): "negative",
    ("fulfillment_backlog", "delivery_delay_rate"): "negative",
    ("delivery_delay_rate", "ticket_volume"): "negative",
    ("delivery_delay_rate", "review_score_avg"): "negative",
    ("ticket_volume", "ticket_backlog"): "negative",
    ("ticket_backlog", "avg_resolution_time"): "negative",
    ("avg_resolution_time", "review_score_avg"): "negative",
    ("review_score_avg", "churn_proxy"): "negative",
    ("ticket_volume", "churn_proxy"): "negative",
    ("churn_proxy", "daily_revenue"): "negative",
    ("refund_rate", "margin_proxy"): "negative",
    ("refund_rate", "daily_revenue"): "negative",
    ("daily_revenue", "margin_proxy"): "positive",
    ("daily_expenses", "expense_ratio"): "negative",
    ("expense_ratio", "margin_proxy"): "negative",
    ("margin_proxy", "net_cash_proxy"): "positive",
    ("dso_proxy", "ar_aging_amount"): "negative",
    ("ar_aging_amount", "net_cash_proxy"): "negative",
}

# Map incident types to their primary affected metric nodes
# This mapping enables the RCA engine to identify the starting point
# for upstream causal search in the dependency graph
INCIDENT_METRIC_MAP = {
    "REFUND_SPIKE": "refund_rate",
    "FULFILLMENT_SLA_DEGRADATION": "delivery_delay_rate",
    "SUPPORT_LOAD_SURGE": "ticket_volume",
    "CHURN_ACCELERATION": "churn_proxy",
    "MARGIN_COMPRESSION": "margin_proxy",
    "LIQUIDITY_CRUNCH_RISK": "net_cash_proxy",
    "SUPPLIER_DEPENDENCY_FAILURE": "supplier_delay_rate",
    "CUSTOMER_SATISFACTION_REGRESSION": "review_score_avg",
}


class BusinessDependencyGraph:
    """
    Static business dependency graph for causal analysis.

    Represents the known causal structure of business metrics as a
    directed acyclic graph (DAG). Provides graph algorithms for:
    - Finding upstream influencers (ancestors)
    - Computing causal path lengths
    - Enumerating all causal paths between metrics

    The graph is immutable once constructed and represents the current
    version of business domain knowledge. Graph evolution is tracked
    via version identifiers.

    Attributes:
        graph: NetworkX DiGraph containing the dependency structure
        version: Version identifier for this graph schema
        logger: Structured logger for observability

    Example:
        >>> dep_graph = BusinessDependencyGraph()
        >>> upstream = dep_graph.get_upstream_nodes("refund_rate")
        >>> print(f"Upstream causes: {upstream}")
        >>> path_length = dep_graph.get_shortest_path_length(
        ...     "supplier_delay_rate", "churn_proxy"
        ... )
        >>> print(f"Path length: {path_length}")
    """

    VERSION = "dep_graph_v2"
    VERSION_LEARNED = "dep_graph_v3"

    def __init__(self):
        """
        Initialize the business dependency graph.

        Constructs the directed graph from the canonical edge list,
        validates graph properties, and lazily loads Granger-learned
        edge strengths when the artifact is available.
        """
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(BUSINESS_DEPENDENCY_EDGES)
        self.logger = structlog.get_logger()

        # Validate graph properties
        self._validate_graph()

        # Try to load Granger-learned edge strengths (A+B integration)
        self._granger = None
        try:
            from api.engine.rca.granger_learner import get_granger_learner
            self._granger = get_granger_learner()
        except Exception as e:
            self.logger.warning("granger_learner_unavailable", error=str(e))

        self.version = self.VERSION_LEARNED if (self._granger and self._granger.is_loaded) else self.VERSION

        self.logger.info(
            "dependency_graph_initialized",
            version=self.version,
            node_count=self.graph.number_of_nodes(),
            edge_count=self.graph.number_of_edges(),
            is_dag=nx.is_directed_acyclic_graph(self.graph),
            granger_loaded=bool(self._granger and self._granger.is_loaded),
        )

    def get_upstream_nodes(self, metric_name: str) -> list[str]:
        """
        Get all upstream nodes that causally influence the given metric.

        Returns all ancestors in the dependency graph (nodes from which
        there exists a directed path to the target metric). These are
        potential root causes for anomalies in the target metric.

        Args:
            metric_name: Name of the target metric

        Returns:
            List of upstream metric names (ancestors in graph)
            Returns empty list if metric not in graph

        Example:
            >>> upstream = dep_graph.get_upstream_nodes("churn_proxy")
            >>> print(upstream)
            ['ticket_volume', 'review_score_avg', 'delivery_delay_rate', ...]
        """
        if metric_name not in self.graph:
            self.logger.warning(
                "metric_not_in_graph",
                metric_name=metric_name,
                available_nodes=list(self.graph.nodes())[:10],  # First 10 for brevity
            )
            return []

        # Get all ancestors (nodes with directed paths to target)
        ancestors = nx.ancestors(self.graph, metric_name)

        self.logger.debug(
            "upstream_nodes_computed",
            metric_name=metric_name,
            upstream_count=len(ancestors),
            upstream_nodes=list(ancestors)[:5],  # First 5 for brevity
        )

        return list(ancestors)

    def get_shortest_path_length(self, source: str, target: str) -> Optional[int]:
        """
        Compute shortest path length from source to target metric.

        Uses BFS to find the minimum number of edges in a directed path
        from source to target. Used to compute graph proximity scores
        in the RCA contribution formula.

        Args:
            source: Source metric name
            target: Target metric name

        Returns:
            Shortest path length (number of edges) or None if no path exists

        Example:
            >>> length = dep_graph.get_shortest_path_length(
            ...     "supplier_delay_rate", "churn_proxy"
            ... )
            >>> print(f"Path length: {length}")
            5
        """
        if source not in self.graph or target not in self.graph:
            self.logger.warning(
                "nodes_not_in_graph",
                source=source,
                target=target,
                source_exists=source in self.graph,
                target_exists=target in self.graph,
            )
            return None

        try:
            path_length = nx.shortest_path_length(
                self.graph, source=source, target=target
            )

            self.logger.debug(
                "shortest_path_computed",
                source=source,
                target=target,
                path_length=path_length,
            )

            return path_length
        except nx.NetworkXNoPath:
            self.logger.debug(
                "no_path_exists",
                source=source,
                target=target,
            )
            return None

    def get_all_paths(self, source: str, target: str) -> list[list[str]]:
        """
        Enumerate all simple paths from source to target.

        Returns all directed paths without cycles from source to target.
        Used for generating alternative causal explanations and path
        narratives. Limited to simple paths to avoid exponential blowup.

        Args:
            source: Source metric name
            target: Target metric name

        Returns:
            List of paths, where each path is a list of node names
            Returns empty list if no paths exist

        Example:
            >>> paths = dep_graph.get_all_paths(
            ...     "supplier_delay_rate", "churn_proxy"
            ... )
            >>> for path in paths:
            ...     print(" → ".join(path))
            supplier_delay_rate → delivery_delay_rate → ticket_volume → churn_proxy
            supplier_delay_rate → delivery_delay_rate → review_score_avg → churn_proxy
        """
        if source not in self.graph or target not in self.graph:
            self.logger.warning(
                "nodes_not_in_graph_for_paths",
                source=source,
                target=target,
            )
            return []

        try:
            # Use all_simple_paths with cutoff to limit explosion
            # Cutoff at 10 to avoid very long indirect paths
            paths = list(nx.all_simple_paths(
                self.graph, source=source, target=target, cutoff=10
            ))

            self.logger.debug(
                "all_paths_computed",
                source=source,
                target=target,
                path_count=len(paths),
                shortest_length=min(len(p) for p in paths) if paths else None,
                longest_length=max(len(p) for p in paths) if paths else None,
            )

            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.logger.debug(
                "no_paths_found",
                source=source,
                target=target,
            )
            return []

    def get_direct_predecessors(self, metric_name: str) -> list[str]:
        """
        Get immediate predecessors (direct causes) of a metric.

        Returns nodes with direct edges to the target metric (in-neighbors).
        These are the immediate causal influencers.

        Args:
            metric_name: Name of the target metric

        Returns:
            List of direct predecessor metric names

        Example:
            >>> predecessors = dep_graph.get_direct_predecessors("churn_proxy")
            >>> print(predecessors)
            ['review_score_avg', 'ticket_volume']
        """
        if metric_name not in self.graph:
            return []

        predecessors = list(self.graph.predecessors(metric_name))

        self.logger.debug(
            "direct_predecessors_computed",
            metric_name=metric_name,
            predecessor_count=len(predecessors),
            predecessors=predecessors,
        )

        return predecessors

    def get_direct_successors(self, metric_name: str) -> list[str]:
        """
        Get immediate successors (direct effects) of a metric.

        Returns nodes with edges from the source metric (out-neighbors).
        These are the metrics directly affected by changes in source.

        Args:
            metric_name: Name of the source metric

        Returns:
            List of direct successor metric names

        Example:
            >>> successors = dep_graph.get_direct_successors("delivery_delay_rate")
            >>> print(successors)
            ['ticket_volume', 'review_score_avg']
        """
        if metric_name not in self.graph:
            return []

        successors = list(self.graph.successors(metric_name))

        self.logger.debug(
            "direct_successors_computed",
            metric_name=metric_name,
            successor_count=len(successors),
            successors=successors,
        )

        return successors

    def get_metric_for_incident_type(self, incident_type: str) -> Optional[str]:
        """
        Map incident type to primary affected metric.

        Returns the metric node that represents the primary symptom
        of the given incident type, enabling RCA to start from the
        correct graph location.

        Args:
            incident_type: Incident type enum value (string form)

        Returns:
            Metric name or None if incident type not mapped

        Example:
            >>> metric = dep_graph.get_metric_for_incident_type("REFUND_SPIKE")
            >>> print(metric)
            'refund_rate'
        """
        metric = INCIDENT_METRIC_MAP.get(incident_type)

        if metric is None:
            self.logger.warning(
                "incident_type_not_mapped",
                incident_type=incident_type,
                available_types=list(INCIDENT_METRIC_MAP.keys()),
            )

        return metric

    def compute_graph_proximity(self, source: str, target: str) -> float:
        """
        Compute graph proximity score for RCA contribution formula.

        Proximity score is inversely proportional to path length:
        graph_proximity = 1.0 / (shortest_path_length + 1)

        This gives higher scores to direct causes (short paths) and
        lower scores to distant indirect causes.

        Args:
            source: Source metric name
            target: Target metric name

        Returns:
            Proximity score in range [0.0, 1.0]
            1.0 for same node, decreasing with distance, 0.0 for no path

        Example:
            >>> proximity = dep_graph.compute_graph_proximity(
            ...     "supplier_delay_rate", "churn_proxy"
            ... )
            >>> print(f"Proximity: {proximity:.4f}")
            0.1667  # 1 / (5 + 1)
        """
        # Same node has maximum proximity
        if source == target:
            return 1.0

        path_length = self.get_shortest_path_length(source, target)

        if path_length is None:
            # No path exists, zero proximity
            return 0.0

        # Base proximity: inverse of path length
        base_proximity = 1.0 / (path_length + 1)

        # B: augment with Granger-learned path strength (geometric mean of edge strengths)
        learned_proximity = base_proximity
        if self._granger and self._granger.is_loaded:
            try:
                path_nodes = nx.shortest_path(self.graph, source=source, target=target)
                strengths = [
                    self._granger.get_edge_strength(path_nodes[i], path_nodes[i + 1])
                    for i in range(len(path_nodes) - 1)
                ]
                # Geometric mean: length-normalized product of edge strengths
                path_strength = float(np.prod(strengths)) ** (1.0 / max(len(strengths), 1))
                learned_proximity = round(base_proximity * path_strength, 4)
            except Exception as e:
                self.logger.debug("granger_proximity_fallback", error=str(e))
                learned_proximity = base_proximity

        self.logger.debug(
            "graph_proximity_computed",
            source=source,
            target=target,
            path_length=path_length,
            base_proximity=round(base_proximity, 4),
            learned_proximity=round(learned_proximity, 4),
            granger_active=bool(self._granger and self._granger.is_loaded),
        )

        return learned_proximity

    def get_edge_strength(self, source: str, target: str) -> float:
        """
        Return the Granger-learned causal strength for a direct edge.

        Args:
            source: Upstream metric name
            target: Downstream metric name

        Returns:
            Strength in [0.1, 1.0]. Returns 1.0 when no artifact is loaded
            (neutral — identical to pre-Granger behavior).
        """
        if self._granger and self._granger.is_loaded:
            return self._granger.get_edge_strength(source, target)
        return 1.0

    def _validate_graph(self):
        """
        Validate graph properties.

        Ensures the dependency graph satisfies required properties:
        - Is a directed acyclic graph (DAG)
        - Contains all expected nodes
        - Has correct number of edges

        Raises:
            ValueError: If graph validation fails
        """
        # Check DAG property
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError(
                "Dependency graph contains cycles - must be a DAG. "
                "Check BUSINESS_DEPENDENCY_EDGES for circular dependencies."
            )

        # Verify minimum size
        if self.graph.number_of_nodes() < 10:
            raise ValueError(
                f"Dependency graph has too few nodes: {self.graph.number_of_nodes()}. "
                "Expected at least 10 business metrics."
            )

        if self.graph.number_of_edges() < 10:
            raise ValueError(
                f"Dependency graph has too few edges: {self.graph.number_of_edges()}. "
                "Expected at least 10 causal relationships."
            )

        # Check incident mappings reference valid nodes
        for incident_type, metric in INCIDENT_METRIC_MAP.items():
            if metric not in self.graph:
                raise ValueError(
                    f"Incident type {incident_type} maps to unknown metric {metric}. "
                    f"Available nodes: {list(self.graph.nodes())}"
                )

        self.logger.info(
            "dependency_graph_validated",
            is_dag=True,
            node_count=self.graph.number_of_nodes(),
            edge_count=self.graph.number_of_edges(),
        )
