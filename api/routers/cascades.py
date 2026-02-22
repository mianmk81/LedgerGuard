"""
Cascade/blast radius analysis router.

Wired to:
- BlastRadiusMapper for impact assessment
- BlastRadiusVisualizer for Cytoscape graph generation
- StorageBackend for incident/cascade queries

Dual-layer graph: entity blast radius + causal metric DAG.
"""

from fastapi import APIRouter, Depends, HTTPException

from api.auth.dependencies import get_current_realm_id
from api.engine.rca.causal_graph_builder import build_incident_causal_graph
from api.storage import get_storage
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.get("/{incident_id}")
async def get_blast_radius(
    incident_id: str,
    realm_id: str = Depends(get_current_realm_id),
    max_depth: int = 3,
):
    """
    Get blast radius graph for incident.
    Returns Cytoscape-compatible graph structure for interactive visualization.
    """
    logger.info(
        "blast_radius_fetch",
        realm_id=realm_id,
        incident_id=incident_id,
        max_depth=max_depth,
    )

    storage = get_storage()

    # Try to load pre-computed blast radius
    blast_radius = storage.read_blast_radius(incident_id)

    if blast_radius is None:
        # Compute on-demand
        incidents = storage.read_incidents()
        incident = next(
            (i for i in incidents if i.incident_id == incident_id), None
        )
        if incident is None:
            raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

        from api.engine.blast_radius import BlastRadiusMapper

        mapper = BlastRadiusMapper(storage=storage)
        blast_radius = mapper.compute_blast_radius(incident)

    # Generate Cytoscape visualization
    from api.engine.blast_radius import BlastRadiusVisualizer

    visualizer = BlastRadiusVisualizer()

    # Load incident for visualizer
    incidents = storage.read_incidents()
    incident = next(
        (i for i in incidents if i.incident_id == incident_id), None
    )

    if incident:
        events = storage.read_canonical_events(limit=500)
        graph = visualizer.generate_graph(blast_radius, incident, events)
    else:
        # Incident was deleted after blast radius was computed
        graph = {"nodes": [], "edges": []}

    # Dual-layer: causal metric DAG centered on incident's primary metric
    causal_graph = build_incident_causal_graph(incident) if incident else None

    return {
        "success": True,
        "data": {
            "blast_radius": blast_radius.model_dump(mode="json"),
            "graph": graph,
            "causal_graph": causal_graph,
        },
    }


@router.get("/{incident_id}/impact")
async def get_cascade_impact(
    incident_id: str,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Get quantified cascade impact metrics.
    """
    logger.info("cascade_impact", realm_id=realm_id, incident_id=incident_id)

    storage = get_storage()

    blast_radius = storage.read_blast_radius(incident_id)
    if blast_radius is None:
        raise HTTPException(
            status_code=404,
            detail=f"No blast radius data found for incident {incident_id}",
        )

    # Also load cascades
    cascades = storage.read_cascades()
    related_cascade = next(
        (c for c in cascades if incident_id in c.incident_ids), None
    )

    return {
        "success": True,
        "data": {
            "incident_id": incident_id,
            "customers_affected": blast_radius.customers_affected,
            "orders_affected": blast_radius.orders_affected,
            "products_affected": blast_radius.products_affected,
            "vendors_involved": blast_radius.vendors_involved,
            "estimated_revenue_exposure": blast_radius.estimated_revenue_exposure,
            "estimated_refund_exposure": blast_radius.estimated_refund_exposure,
            "estimated_churn_exposure": blast_radius.estimated_churn_exposure,
            "blast_radius_severity": blast_radius.blast_radius_severity.value,
            "narrative": blast_radius.narrative,
            "cascade": related_cascade.model_dump(mode="json") if related_cascade else None,
        },
    }
