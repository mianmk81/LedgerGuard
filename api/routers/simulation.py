"""
Scenario simulation router.

Wired to:
- SimulationEngine for what-if analysis
- StorageBackend for scenario persistence
"""

from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth.dependencies import get_current_realm_id
from api.storage import get_storage
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class SimulationRequest(BaseModel):
    """Simulation request with metric perturbations."""

    perturbations: List[Dict]  # [{"metric": "order_volume", "change": "+50%"}]


@router.post("/run")
async def run_simulation(
    request: SimulationRequest,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Run scenario simulation with specified perturbations.
    Uses SimulationEngine for downstream impact prediction.
    """
    logger.info(
        "simulation_run",
        realm_id=realm_id,
        perturbation_count=len(request.perturbations),
    )

    from api.engine.simulation import WhatIfSimulator

    storage = get_storage()
    engine = WhatIfSimulator(storage=storage)

    try:
        scenario = engine.simulate(perturbations=request.perturbations)
        return {"success": True, "data": scenario.model_dump(mode="json")}
    except Exception as e:
        logger.error("simulation_failed", error=str(e))
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/history")
async def get_simulation_history(
    realm_id: str = Depends(get_current_realm_id),
    limit: int = 10,
):
    """
    Get recent simulation history.
    Note: WhatIfScenario retrieval is by ID; listing requires storage extension.
    """
    logger.info("simulation_history", realm_id=realm_id, limit=limit)

    # Currently storage has read_whatif_scenario(scenario_id) but no list method.
    # Return empty list with note about future extension.
    return {
        "success": True,
        "data": [],
        "message": "Simulation history listing will be available in a future release",
    }
