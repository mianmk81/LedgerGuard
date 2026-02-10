"""
Cascade/blast radius analysis router.
"""

from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth.dependencies import get_current_realm_id
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class EntityNode(BaseModel):
    """Entity graph node."""

    id: str
    type: str
    label: str
    impact_score: float
    depth: int


class EntityEdge(BaseModel):
    """Entity graph edge."""

    source: str
    target: str
    relationship: str
    strength: float


class BlastRadiusGraph(BaseModel):
    """Blast radius graph visualization data."""

    incident_id: str
    root_entity: EntityNode
    nodes: List[EntityNode]
    edges: List[EntityEdge]
    max_depth: int
    total_affected: int


@router.get("/{incident_id}", response_model=BlastRadiusGraph)
async def get_blast_radius(
    incident_id: str,
    realm_id: str = Depends(get_current_realm_id),
    max_depth: int = 3,
):
    """
    Get blast radius graph for incident.
    Returns Cytoscape-compatible graph structure.
    """
    logger.info(
        "blast_radius_fetch", realm_id=realm_id, incident_id=incident_id, max_depth=max_depth
    )

    # TODO: Generate actual graph from NetworkX
    root_node = EntityNode(
        id="INV-001", type="invoice", label="Invoice #001", impact_score=1.0, depth=0
    )

    return BlastRadiusGraph(
        incident_id=incident_id,
        root_entity=root_node,
        nodes=[
            root_node,
            EntityNode(
                id="CUST-123",
                type="customer",
                label="Customer ABC Corp",
                impact_score=0.8,
                depth=1,
            ),
            EntityNode(
                id="ACC-100", type="account", label="Accounts Receivable", impact_score=0.6, depth=2
            ),
        ],
        edges=[
            EntityEdge(source="INV-001", target="CUST-123", relationship="belongs_to", strength=1.0),
            EntityEdge(source="INV-001", target="ACC-100", relationship="affects", strength=0.8),
        ],
        max_depth=max_depth,
        total_affected=3,
    )


@router.get("/{incident_id}/impact")
async def get_cascade_impact(
    incident_id: str,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Get quantified cascade impact metrics.
    """
    logger.info("cascade_impact", realm_id=realm_id, incident_id=incident_id)

    return {
        "incident_id": incident_id,
        "total_entities_affected": 15,
        "financial_impact": 45000.0,
        "by_entity_type": {
            "invoice": {"count": 8, "impact": 30000.0},
            "customer": {"count": 5, "impact": 12000.0},
            "account": {"count": 2, "impact": 3000.0},
        },
        "cascade_depth": 3,
        "propagation_speed": "fast",
    }
