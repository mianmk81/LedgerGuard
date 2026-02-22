"""
Forward Chain Builder — Causal path from degrading metric to incident.

Agent: data-scientist
Uses BusinessDependencyGraph to trace downstream paths from a
degrading metric to incident-mapped metrics. Builds prevention narrative.
"""

from collections import deque
from typing import Optional

import structlog

from api.engine.rca.dependency_graph import (
    BusinessDependencyGraph,
    INCIDENT_METRIC_MAP,
)

logger = structlog.get_logger()

# Reverse: metric → incident_type for forward chain end
METRIC_TO_INCIDENT = {v: k for k, v in INCIDENT_METRIC_MAP.items()}


class ForwardChainBuilder:
    """
    Builds forward causal chains from degrading metric to predicted incident.

    Traces downstream in the dependency graph to find paths ending at
    incident-mapped metrics. Generates prevention narrative.
    """

    def __init__(self, dep_graph: Optional[BusinessDependencyGraph] = None):
        self.dep_graph = dep_graph or BusinessDependencyGraph()
        self.logger = structlog.get_logger()
        self.max_path_length = 5

    def build_chain(self, start_metric: str) -> list[dict]:
        """
        Build forward chains from start_metric to incident types.

        Returns:
            List of dicts with path, incident_type, prevention_steps
        """
        if start_metric not in self.dep_graph.graph:
            self.logger.warning(
                "metric_not_in_graph",
                metric=start_metric,
            )
            return []

        # BFS to find paths to incident-mapped metrics
        results = []
        queue = deque([(start_metric, [start_metric])])
        visited_paths = set()

        while queue:
            current, path = queue.popleft()
            path_key = tuple(path)
            if path_key in visited_paths:
                continue
            visited_paths.add(path_key)

            if len(path) > self.max_path_length:
                continue

            # Check if we've reached an incident-mapped metric
            incident_type = METRIC_TO_INCIDENT.get(current)
            if incident_type:
                prevention = self._build_prevention_narrative(path, incident_type)
                results.append({
                    "path": path,
                    "incident_type": incident_type,
                    "prevention_steps": prevention,
                })
                continue

            # Expand to successors
            successors = self.dep_graph.get_direct_successors(current)
            for succ in successors:
                if succ not in path:  # Avoid cycles
                    queue.append((succ, path + [succ]))

        return results

    def _build_prevention_narrative(self, path: list[str], incident_type: str) -> str:
        """Generate prevention recommendation."""
        if len(path) < 2:
            metric_label = path[0].replace("_", " ")
            incident_label = incident_type.replace("_", " ").lower()
            return f"Keep a close eye on {metric_label} to prevent a {incident_label}."

        start = path[0].replace("_", " ")
        chain_labels = [s.replace("_", " ") for s in path[1:]]
        chain_str = " → ".join(chain_labels)
        incident_label = incident_type.replace("_", " ").lower()
        return (
            f"Intervene at {start} before it cascades through "
            f"{chain_str} and triggers a {incident_label}. "
            f"Address the root cause early to prevent downstream damage."
        )
