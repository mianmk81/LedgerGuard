"""
On-demand analysis router - Trigger anomaly detection and RCA.
"""

from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth.dependencies import get_current_realm_id
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class AnalysisType(str, Enum):
    """Analysis type options."""

    ANOMALY_DETECTION = "anomaly_detection"
    ROOT_CAUSE = "root_cause"
    BLAST_RADIUS = "blast_radius"
    FULL = "full"


class AnalysisRequest(BaseModel):
    """Analysis request."""

    analysis_type: AnalysisType = AnalysisType.FULL
    entity_types: Optional[List[str]] = None
    time_range_days: int = 30
    sensitivity: float = 0.85


class AnomalyResult(BaseModel):
    """Anomaly detection result."""

    entity_id: str
    entity_type: str
    timestamp: str
    metric_name: str
    actual_value: float
    expected_value: float
    deviation: float
    confidence: float
    severity: str


class AnalysisResult(BaseModel):
    """Analysis result."""

    analysis_id: str
    analysis_type: AnalysisType
    started_at: str
    completed_at: Optional[str]
    anomalies_detected: int
    anomalies: List[AnomalyResult] = []


@router.post("/run", response_model=AnalysisResult)
async def run_analysis(
    request: AnalysisRequest,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Run on-demand analysis.
    """
    import uuid
    from datetime import datetime

    analysis_id = str(uuid.uuid4())

    logger.info(
        "analysis_started",
        realm_id=realm_id,
        analysis_id=analysis_id,
        analysis_type=request.analysis_type.value,
    )

    # TODO: Trigger analysis engine
    return AnalysisResult(
        analysis_id=analysis_id,
        analysis_type=request.analysis_type,
        started_at=datetime.utcnow().isoformat(),
        completed_at=None,
        anomalies_detected=0,
    )


@router.get("/result/{analysis_id}", response_model=AnalysisResult)
async def get_analysis_result(
    analysis_id: str,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Get analysis result.
    """
    from datetime import datetime

    logger.info("analysis_result_fetch", realm_id=realm_id, analysis_id=analysis_id)

    # TODO: Query results from database
    return AnalysisResult(
        analysis_id=analysis_id,
        analysis_type=AnalysisType.FULL,
        started_at=datetime.utcnow().isoformat(),
        completed_at=datetime.utcnow().isoformat(),
        anomalies_detected=3,
        anomalies=[
            AnomalyResult(
                entity_id="INV-001",
                entity_type="invoice",
                timestamp=datetime.utcnow().isoformat(),
                metric_name="amount",
                actual_value=15000.0,
                expected_value=5000.0,
                deviation=200.0,
                confidence=0.95,
                severity="high",
            )
        ],
    )
