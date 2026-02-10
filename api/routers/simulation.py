"""
Scenario simulation router.
"""

from enum import Enum
from typing import Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth.dependencies import get_current_realm_id
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class SimulationType(str, Enum):
    """Simulation types."""

    MONTE_CARLO = "monte_carlo"
    STRESS_TEST = "stress_test"
    SCENARIO = "scenario"


class SimulationRequest(BaseModel):
    """Simulation request."""

    simulation_type: SimulationType
    iterations: int = 1000
    parameters: Dict[str, float]
    time_horizon_days: int = 90


class SimulationResult(BaseModel):
    """Simulation result."""

    simulation_id: str
    simulation_type: SimulationType
    iterations: int
    results: Dict[str, Dict[str, float]]  # metric -> {mean, std, p50, p95, p99}
    risk_metrics: Dict[str, float]


@router.post("/run", response_model=SimulationResult)
async def run_simulation(
    request: SimulationRequest,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Run scenario simulation.
    """
    import uuid

    simulation_id = str(uuid.uuid4())

    logger.info(
        "simulation_run",
        realm_id=realm_id,
        simulation_id=simulation_id,
        type=request.simulation_type.value,
        iterations=request.iterations,
    )

    # TODO: Run actual simulation
    return SimulationResult(
        simulation_id=simulation_id,
        simulation_type=request.simulation_type,
        iterations=request.iterations,
        results={
            "revenue": {
                "mean": 100000.0,
                "std": 5000.0,
                "p50": 99500.0,
                "p95": 108000.0,
                "p99": 112000.0,
            }
        },
        risk_metrics={"var_95": 8000.0, "cvar_95": 10000.0, "max_drawdown": 15000.0},
    )


@router.get("/history")
async def get_simulation_history(
    realm_id: str = Depends(get_current_realm_id),
    limit: int = 10,
):
    """
    Get simulation history.
    """
    logger.info("simulation_history", realm_id=realm_id, limit=limit)

    # TODO: Query from database
    return {"simulations": []}
