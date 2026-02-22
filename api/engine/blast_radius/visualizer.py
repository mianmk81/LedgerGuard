"""
Blast Radius Visualizer — Cytoscape.js JSON Generation.

This module generates Cytoscape.js-compatible JSON graph data for rendering
interactive blast radius visualizations in the frontend. The graph shows
entities as nodes and relationships as edges, with visual properties
encoded for severity, entity type, and impact magnitude.

Output format follows Cytoscape.js JSON specification:
{
    "elements": {
        "nodes": [...],
        "edges": [...]
    },
    "style": [...],
    "layout": {...}
}

Version: blast_vis_v1
"""

from datetime import datetime
from typing import Any, Optional

import structlog

from api.models.blast_radius import BlastRadius
from api.models.events import CanonicalEvent
from api.models.incidents import Incident

logger = structlog.get_logger()

# Visual style configuration
NODE_COLORS = {
    "customer": "#3B82F6",    # Blue
    "order": "#10B981",       # Green
    "invoice": "#F59E0B",     # Amber
    "payment": "#8B5CF6",     # Purple
    "expense": "#EF4444",     # Red
    "vendor": "#EC4899",      # Pink
    "product": "#06B6D4",     # Cyan
    "ticket": "#F97316",      # Orange
    "incident": "#DC2626",    # Red-600
}

NODE_SHAPES = {
    "customer": "ellipse",
    "order": "round-rectangle",
    "invoice": "rectangle",
    "payment": "diamond",
    "expense": "hexagon",
    "vendor": "triangle",
    "product": "barrel",
    "ticket": "star",
    "incident": "octagon",
}

SEVERITY_COLORS = {
    "contained": "#22C55E",      # Green
    "significant": "#F59E0B",    # Amber
    "severe": "#EF4444",         # Red
    "catastrophic": "#7F1D1D",   # Dark Red
}


class BlastRadiusVisualizer:
    """
    Generates Cytoscape.js JSON for blast radius visualization.

    Creates interactive graph data showing the entity impact network
    of an incident. Nodes represent entities, edges represent relationships,
    and visual properties encode entity type and impact severity.

    Attributes:
        logger: Structured logger

    Example:
        >>> visualizer = BlastRadiusVisualizer()
        >>> graph_json = visualizer.generate_graph(
        ...     blast_radius=blast_radius,
        ...     incident=incident,
        ...     events=events,
        ... )
        >>> # Send to frontend for Cytoscape.js rendering
    """

    def __init__(self):
        """Initialize the visualizer."""
        self.logger = structlog.get_logger()

    def generate_graph(
        self,
        blast_radius: BlastRadius,
        incident: Incident,
        events: list[CanonicalEvent],
        max_nodes: int = 200,
    ) -> dict[str, Any]:
        """
        Generate complete Cytoscape.js graph JSON.

        Args:
            blast_radius: Computed blast radius
            incident: The source incident
            events: Canonical events for relationship extraction
            max_nodes: Maximum nodes to include (for performance)

        Returns:
            Cytoscape.js JSON with elements, style, and layout

        Example:
            >>> graph = visualizer.generate_graph(blast, incident, events)
            >>> print(json.dumps(graph, indent=2))
        """
        nodes = []
        edges = []
        node_ids = set()

        # Add incident node (center of graph)
        incident_node = self._create_incident_node(incident, blast_radius)
        incident_node["data"]["impact"] = "negative"
        nodes.append(incident_node)
        node_ids.add(incident.incident_id)

        # Extract entity nodes and edges from events
        for event in events[:max_nodes]:
            # Add primary entity node
            entity_node = self._create_entity_node(event)
            if entity_node and entity_node["data"]["id"] not in node_ids:
                nodes.append(entity_node)
                node_ids.add(entity_node["data"]["id"])

                # Add edge from entity to incident
                edge = self._create_edge(
                    source=entity_node["data"]["id"],
                    target=incident.incident_id,
                    label="affected_by",
                )
                edges.append(edge)

            # Add related entity nodes and edges
            for rel_type, rel_id in event.related_entity_ids.items():
                if rel_id and rel_id not in node_ids and len(node_ids) < max_nodes:
                    rel_node = self._create_related_node(rel_type, rel_id)
                    nodes.append(rel_node)
                    node_ids.add(rel_id)

                    # Edge from related entity to primary entity
                    if event.entity_id:
                        edge = self._create_edge(
                            source=rel_id,
                            target=event.entity_id,
                            label=rel_type,
                        )
                        edges.append(edge)

        # Add downstream incident nodes
        for ds_id in blast_radius.downstream_incidents_triggered:
            if ds_id not in node_ids:
                ds_node = {
                    "data": {
                        "id": ds_id,
                        "label": f"Downstream: {ds_id[:12]}...",
                        "type": "incident",
                        "color": SEVERITY_COLORS.get("severe", "#EF4444"),
                        "shape": "octagon",
                        "size": 30,
                    },
                }
                nodes.append(ds_node)
                node_ids.add(ds_id)

                edge = self._create_edge(
                    source=incident.incident_id,
                    target=ds_id,
                    label="triggered",
                )
                edges.append(edge)

        # Build complete graph JSON
        graph = {
            "elements": {
                "nodes": nodes,
                "edges": edges,
            },
            "style": self._get_default_style(),
            "layout": self._get_layout_config(len(nodes)),
            "metadata": {
                "incident_id": incident.incident_id,
                "severity": blast_radius.blast_radius_severity.value,
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "generated_at": datetime.utcnow().isoformat(),
            },
        }

        self.logger.info(
            "blast_radius_graph_generated",
            incident_id=incident.incident_id,
            nodes=len(nodes),
            edges=len(edges),
        )

        return graph

    def generate_summary_graph(
        self,
        blast_radius: BlastRadius,
        incident: Incident,
    ) -> dict[str, Any]:
        """
        Generate a simplified summary graph showing entity type aggregates.

        Creates a small graph with one node per entity type showing counts,
        suitable for dashboard overview display.

        Args:
            blast_radius: Computed blast radius
            incident: The source incident

        Returns:
            Simplified Cytoscape.js JSON
        """
        nodes = []
        edges = []

        # Center incident node
        incident_node = self._create_incident_node(incident, blast_radius)
        nodes.append(incident_node)

        # Entity type aggregate nodes
        entity_counts = {
            "customer": blast_radius.customers_affected,
            "order": blast_radius.orders_affected,
            "product": blast_radius.products_affected,
            "vendor": blast_radius.vendors_involved,
        }

        for entity_type, count in entity_counts.items():
            if count > 0:
                node = {
                    "data": {
                        "id": f"agg_{entity_type}",
                        "label": f"{entity_type.title()}\n({count:,})",
                        "type": entity_type,
                        "color": NODE_COLORS.get(entity_type, "#6B7280"),
                        "shape": NODE_SHAPES.get(entity_type, "ellipse"),
                        "size": min(60, 20 + count * 0.05),
                        "count": count,
                    },
                }
                nodes.append(node)

                edge = self._create_edge(
                    source=f"agg_{entity_type}",
                    target=incident.incident_id,
                    label=f"{count:,} affected",
                )
                edges.append(edge)

        # Financial impact nodes
        if blast_radius.estimated_revenue_exposure > 0:
            nodes.append({
                "data": {
                    "id": "impact_revenue",
                    "label": f"Revenue\n${blast_radius.estimated_revenue_exposure:,.0f}",
                    "type": "metric",
                    "color": "#EF4444",
                    "shape": "rectangle",
                    "size": 40,
                },
            })
            edges.append(self._create_edge(
                "impact_revenue", incident.incident_id, "revenue at risk"
            ))

        if blast_radius.estimated_refund_exposure > 0:
            nodes.append({
                "data": {
                    "id": "impact_refund",
                    "label": f"Refunds\n${blast_radius.estimated_refund_exposure:,.0f}",
                    "type": "metric",
                    "color": "#F59E0B",
                    "shape": "rectangle",
                    "size": 35,
                },
            })
            edges.append(self._create_edge(
                "impact_refund", incident.incident_id, "refund exposure"
            ))

        return {
            "elements": {"nodes": nodes, "edges": edges},
            "style": self._get_default_style(),
            "layout": {"name": "concentric", "concentric": lambda n: 1 if n.data("type") == "incident" else 0},
            "metadata": {
                "incident_id": incident.incident_id,
                "severity": blast_radius.blast_radius_severity.value,
                "view": "summary",
                "generated_at": datetime.utcnow().isoformat(),
            },
        }

    # =========================================================================
    # Node/Edge Creation
    # =========================================================================

    def _create_incident_node(
        self, incident: Incident, blast_radius: BlastRadius
    ) -> dict:
        """Create the central incident node."""
        severity = blast_radius.blast_radius_severity.value
        return {
            "data": {
                "id": incident.incident_id,
                "label": incident.incident_type.value.replace("_", " ").title(),
                "type": "incident",
                "color": SEVERITY_COLORS.get(severity, "#DC2626"),
                "shape": "octagon",
                "size": 50,
                "severity": severity,
                "zscore": incident.primary_metric_zscore,
            },
        }

    def _create_entity_node(self, event: CanonicalEvent) -> Optional[dict]:
        """Create a node from a canonical event's primary entity."""
        entity_type = event.entity_type.value
        entity_id = event.entity_id

        if not entity_id:
            return None

        # Shorten ID for display
        short_id = entity_id.split(":")[-1][:12] if ":" in entity_id else entity_id[:12]

        # Impact: expense/refund=negative, order/invoice paid=positive
        impact = "neutral"
        if entity_type in ("expense", "vendor"):
            impact = "negative"
        elif entity_type in ("order", "payment"):
            impact = "positive"
        elif entity_type == "invoice":
            impact = "negative" if event.event_type.value == "invoice_overdue" else "positive"

        return {
            "data": {
                "id": entity_id,
                "label": f"{entity_type.title()}\n{short_id}",
                "type": entity_type,
                "color": NODE_COLORS.get(entity_type, "#6B7280"),
                "shape": NODE_SHAPES.get(entity_type, "ellipse"),
                "size": 25,
                "amount": event.amount,
                "impact": impact,
            },
        }

    def _create_related_node(self, rel_type: str, rel_id: str) -> dict:
        """Create a node from a related entity reference."""
        normalized_type = rel_type.lower()
        short_id = rel_id.split(":")[-1][:12] if ":" in rel_id else rel_id[:12]
        impact = "negative" if normalized_type in ("expense", "vendor") else (
            "positive" if normalized_type in ("order", "payment", "customer") else "neutral"
        )

        return {
            "data": {
                "id": rel_id,
                "label": f"{normalized_type.title()}\n{short_id}",
                "type": normalized_type,
                "color": NODE_COLORS.get(normalized_type, "#6B7280"),
                "shape": NODE_SHAPES.get(normalized_type, "ellipse"),
                "size": 20,
                "impact": impact,
            },
        }

    def _create_edge(self, source: str, target: str, label: str = "") -> dict:
        """Create an edge between two nodes."""
        return {
            "data": {
                "id": f"edge_{source}_{target}",
                "source": source,
                "target": target,
                "label": label or "related",
            },
        }

    # =========================================================================
    # Style & Layout
    # =========================================================================

    def _get_default_style(self) -> list[dict]:
        """Get default Cytoscape.js stylesheet."""
        return [
            {
                "selector": "node",
                "style": {
                    "label": "data(label)",
                    "background-color": "data(color)",
                    "shape": "data(shape)",
                    "width": "data(size)",
                    "height": "data(size)",
                    "font-size": "10px",
                    "text-wrap": "wrap",
                    "text-valign": "center",
                    "text-halign": "center",
                    "color": "#1F2937",
                    "border-width": 2,
                    "border-color": "#D1D5DB",
                },
            },
            {
                "selector": "node[type='incident']",
                "style": {
                    "font-size": "12px",
                    "font-weight": "bold",
                    "border-width": 3,
                    "border-color": "#DC2626",
                    "color": "#FFFFFF",
                    "text-outline-color": "#DC2626",
                    "text-outline-width": 2,
                },
            },
            {
                "selector": "edge",
                "style": {
                    "label": "data(label)",
                    "width": 1.5,
                    "line-color": "#9CA3AF",
                    "target-arrow-color": "#9CA3AF",
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier",
                    "font-size": "8px",
                    "text-rotation": "autorotate",
                    "color": "#6B7280",
                },
            },
        ]

    def _get_layout_config(self, node_count: int) -> dict:
        """
        Get layout configuration based on graph size.

        Uses different layouts depending on graph complexity:
        - Small (< 20 nodes): concentric layout (clean, centered)
        - Medium (20-100 nodes): dagre layout (hierarchical)
        - Large (100+ nodes): cose layout (force-directed, fast)

        Args:
            node_count: Number of nodes in the graph

        Returns:
            Cytoscape.js layout configuration
        """
        if node_count < 20:
            return {
                "name": "concentric",
                "minNodeSpacing": 50,
                "animate": True,
                "animationDuration": 500,
            }
        elif node_count < 100:
            return {
                "name": "dagre",
                "rankDir": "TB",
                "nodeSep": 50,
                "rankSep": 80,
                "animate": True,
                "animationDuration": 500,
            }
        else:
            return {
                "name": "cose",
                "nodeRepulsion": 8000,
                "idealEdgeLength": 100,
                "animate": False,
            }
