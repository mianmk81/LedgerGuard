"""
Causal Graph Builder — Reusable Cytoscape graph for RCA and dashboard.

Builds metric DAG with impact metadata for 3D visualization.
Used by cascades, credit_pulse, and dashboard routers.
"""

from api.engine.rca.dependency_graph import (
    BUSINESS_DEPENDENCY_EDGES,
    EDGE_IMPACT,
    METRIC_IMPACT,
    BusinessDependencyGraph,
)
from api.engine.rca.granger_learner import get_granger_learner as _get_granger

METRIC_LABELS = {
    "margin_proxy": "Margin",
    "refund_rate": "Refund rate",
    "net_cash_proxy": "Net cash",
    "delivery_delay_rate": "Delivery delay",
    "fulfillment_backlog": "Backlog",
    "supplier_delay_rate": "Supplier delay",
    "order_volume": "Order volume",
    "ticket_volume": "Tickets",
    "ticket_backlog": "Ticket backlog",
    "avg_resolution_time": "Resolution time",
    "review_score_avg": "Reviews",
    "churn_proxy": "Churn",
    "daily_revenue": "Revenue",
    "daily_expenses": "Expenses",
    "expense_ratio": "Expense ratio",
    "dso_proxy": "DSO",
    "ar_aging_amount": "AR aging",
}


# Map incident_type to dependency-graph metric when primary_metric doesn't match
INCIDENT_TYPE_TO_METRIC = {
    "churn_acceleration": "churn_proxy",
    "refund_spike": "refund_rate",
    "fulfillment_sla_degradation": "fulfillment_backlog",
    "support_load_surge": "ticket_volume",
    "margin_compression": "margin_proxy",
    "liquidity_crunch_risk": "net_cash_proxy",
    "supplier_dependency_failure": "supplier_delay_rate",
    "customer_satisfaction_regression": "review_score_avg",
}


def build_incident_causal_graph(incident, metric_statuses: dict | None = None) -> dict | None:
    """Build causal metric subgraph centered on incident's primary metric."""
    primary = getattr(incident, "primary_metric", None) or (
        incident.get("primary_metric") if isinstance(incident, dict) else None
    )
    incident_type_raw = getattr(incident, "incident_type", None) or (
        incident.get("incident_type") if isinstance(incident, dict) else None
    )
    incident_type = getattr(incident_type_raw, "value", incident_type_raw) if incident_type_raw else None

    dep_graph = BusinessDependencyGraph()
    if primary not in dep_graph.graph and incident_type:
        primary = INCIDENT_TYPE_TO_METRIC.get(str(incident_type), primary)
    if not primary or primary not in dep_graph.graph:
        return None

    nodes_set = {primary}
    ancestors = dep_graph.get_upstream_nodes(primary)
    for a in ancestors:
        nodes_set.add(a)
    for _ in range(2):
        new_added = set()
        for n in list(nodes_set):
            for s in dep_graph.get_direct_successors(n):
                new_added.add(s)
        nodes_set.update(new_added)

    edges_sub = [(u, v) for u, v in BUSINESS_DEPENDENCY_EDGES if u in nodes_set and v in nodes_set]

    nodes = [
        {
            "data": {
                "id": m,
                "label": METRIC_LABELS.get(m, m.replace("_", " ").title()),
                "size": 40 if m == primary else 22,
                "impact": _resolve_live_impact(m, metric_statuses),
                "primary": m == primary,
            },
        }
        for m in sorted(nodes_set)
    ]
    _granger = _get_granger()
    edges = [
        {
            "data": {
                "source": u,
                "target": v,
                "id": f"{u}→{v}",
                "label": EDGE_IMPACT.get((u, v), "influences"),
                "impact": EDGE_IMPACT.get((u, v), "neutral"),
                "weight": _granger.get_edge_strength(u, v),
            },
        }
        for u, v in edges_sub
    ]
    return {"elements": {"nodes": nodes, "edges": edges}}


def _resolve_live_impact(metric: str, metric_statuses: dict | None) -> str:
    """
    Determine the visual impact for a metric node, taking live health
    status into account.  A normally-positive metric (e.g. order_volume)
    that is currently critical/warning/degraded should appear red so the
    graph matches the contributing-factors list.
    """
    static_impact = METRIC_IMPACT.get(metric, "neutral")
    if not metric_statuses:
        return static_impact
    status = metric_statuses.get(metric)
    if status in ("critical", "warning", "degraded"):
        return "negative"
    return static_impact


def build_full_causal_graph(metric_statuses: dict | None = None) -> dict:
    """Build full causal metric DAG for 3D visualization.

    Args:
        metric_statuses: Optional mapping of metric_name → health status
            (e.g. "healthy", "degraded", "warning", "critical").
            When provided, nodes whose current status is bad are coloured
            red regardless of their static METRIC_IMPACT direction.
    """
    nodes_set = set()
    for u, v in BUSINESS_DEPENDENCY_EDGES:
        nodes_set.add(u)
        nodes_set.add(v)

    nodes = [
        {
            "data": {
                "id": m,
                "label": METRIC_LABELS.get(m, m.replace("_", " ").title()),
                "size": 25,
                "impact": _resolve_live_impact(m, metric_statuses),
            },
        }
        for m in sorted(nodes_set)
    ]
    _granger = _get_granger()
    edges = [
        {
            "data": {
                "source": u,
                "target": v,
                "id": f"{u}→{v}",
                "label": EDGE_IMPACT.get((u, v), "influences"),
                "impact": EDGE_IMPACT.get((u, v), "neutral"),
                "weight": _granger.get_edge_strength(u, v),
            },
        }
        for u, v in BUSINESS_DEPENDENCY_EDGES
    ]
    return {"elements": {"nodes": nodes, "edges": edges}}


def build_forward_chain_graph(chain: list[str], incident_type: str | None = None) -> dict:
    """Build graph from forward chain (metric1 → metric2 → ... → incident_type)."""
    if not chain:
        return {"elements": {"nodes": [], "edges": []}}

    nodes = []
    edges = []
    for i, m in enumerate(chain):
        is_incident = incident_type and m == incident_type
        label = m.replace("_", " ").title() if not is_incident else m
        nodes.append({
            "data": {
                "id": m,
                "label": label,
                "size": 35 if i == 0 else (50 if is_incident else 25),
                "impact": "negative" if is_incident else (METRIC_IMPACT.get(m, "neutral")),
                "primary": i == 0,
            },
        })
        if i > 0:
            edges.append({
                "data": {
                    "source": chain[i - 1],
                    "target": m,
                    "id": f"{chain[i-1]}→{m}",
                    "label": "leads to",
                    "impact": "negative",
                },
            })
    return {"elements": {"nodes": nodes, "edges": edges}}
