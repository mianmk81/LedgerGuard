"""
Comparison and what-if simulation router.
"""

from typing import Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth.dependencies import get_current_realm_id
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class ScenarioParameter(BaseModel):
    """Scenario parameter modification."""

    entity_type: str
    metric: str
    change_percent: float


class ComparisonRequest(BaseModel):
    """Comparison scenario request."""

    baseline_period: str
    comparison_period: str
    metrics: List[str]


class ComparisonResult(BaseModel):
    """Comparison result."""

    metric: str
    baseline_value: float
    comparison_value: float
    change_percent: float
    change_absolute: float
    significance: float


class ComparisonResponse(BaseModel):
    """Comparison response."""

    comparison_id: str
    baseline_period: str
    comparison_period: str
    results: List[ComparisonResult]


@router.post("/compare", response_model=ComparisonResponse)
async def compare_periods(
    request: ComparisonRequest,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Compare metrics between two time periods.
    """
    import uuid

    comparison_id = str(uuid.uuid4())

    logger.info(
        "comparison_start",
        realm_id=realm_id,
        comparison_id=comparison_id,
        baseline=request.baseline_period,
        comparison=request.comparison_period,
    )

    # TODO: Run actual comparison analysis
    return ComparisonResponse(
        comparison_id=comparison_id,
        baseline_period=request.baseline_period,
        comparison_period=request.comparison_period,
        results=[
            ComparisonResult(
                metric="total_revenue",
                baseline_value=100000.0,
                comparison_value=120000.0,
                change_percent=20.0,
                change_absolute=20000.0,
                significance=0.95,
            )
        ],
    )


@router.post("/whatif")
async def whatif_simulation(
    parameters: List[ScenarioParameter],
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Run what-if simulation with modified parameters.
    """
    import uuid

    simulation_id = str(uuid.uuid4())

    logger.info(
        "whatif_simulation",
        realm_id=realm_id,
        simulation_id=simulation_id,
        parameters_count=len(parameters),
    )

    # TODO: Run simulation
    return {
        "simulation_id": simulation_id,
        "parameters": [p.dict() for p in parameters],
        "predicted_impact": {
            "revenue_change": 15000.0,
            "risk_score": 0.3,
            "affected_entities": 25,
        },
        "confidence": 0.85,
    }
