"""
Comparison and what-if simulation router.

Wired to:
- ComparatorEngine for incident comparison
- SimulationEngine for what-if scenarios
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth.dependencies import get_current_realm_id
from api.storage import get_storage
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class IncidentCompareRequest(BaseModel):
    """Request to compare two incidents."""

    incident_a_id: str
    incident_b_id: str


class ScenarioParameter(BaseModel):
    """Scenario parameter modification."""

    metric: str
    change: str  # e.g., "+50%", "-10%", "1500"


@router.post("/compare")
async def compare_incidents(
    request: IncidentCompareRequest,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Compare two incidents to identify shared patterns and divergences.
    Uses ComparatorEngine for systematic comparison.
    """
    logger.info(
        "comparison_start",
        realm_id=realm_id,
        incident_a=request.incident_a_id,
        incident_b=request.incident_b_id,
    )

    from api.engine.comparator import ComparatorEngine

    storage = get_storage()
    comparator = ComparatorEngine(storage=storage)

    try:
        comparison = comparator.compare(
            incident_a_id=request.incident_a_id,
            incident_b_id=request.incident_b_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {"success": True, "data": comparison.model_dump(mode="json")}


@router.post("/whatif")
async def whatif_simulation(
    parameters: List[ScenarioParameter],
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Run what-if simulation with modified parameters.
    Uses SimulationEngine for downstream impact prediction.
    """
    logger.info(
        "whatif_simulation",
        realm_id=realm_id,
        parameters_count=len(parameters),
    )

    from api.engine.simulation import WhatIfSimulator

    storage = get_storage()
    engine = WhatIfSimulator(storage=storage)

    perturbations = [
        {"metric": p.metric, "change": p.change}
        for p in parameters
    ]

    try:
        scenario = engine.simulate(perturbations=perturbations)
        return {"success": True, "data": scenario.model_dump(mode="json")}
    except Exception as e:
        logger.error("whatif_simulation_failed", error=str(e))
        raise HTTPException(status_code=422, detail=str(e))
