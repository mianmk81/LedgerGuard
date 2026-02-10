"""
Incident management router - List, detail, and postmortem generation.
"""

from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth.dependencies import get_current_realm_id
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class IncidentSeverity(str, Enum):
    """Incident severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IncidentStatus(str, Enum):
    """Incident status."""

    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"


class Incident(BaseModel):
    """Incident summary."""

    incident_id: str
    title: str
    severity: IncidentSeverity
    status: IncidentStatus
    confidence: float
    detected_at: str
    resolved_at: Optional[str] = None
    affected_entities_count: int
    root_causes_count: int


class RootCause(BaseModel):
    """Root cause analysis result."""

    entity_id: str
    entity_type: str
    likelihood: float
    explanation: str


class IncidentDetail(BaseModel):
    """Detailed incident information."""

    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    confidence: float
    detected_at: str
    resolved_at: Optional[str] = None
    root_causes: List[RootCause]
    affected_entities: List[str]
    blast_radius_depth: int
    timeline: List[dict]


@router.get("/", response_model=List[Incident])
async def list_incidents(
    realm_id: str = Depends(get_current_realm_id),
    severity: Optional[IncidentSeverity] = None,
    status: Optional[IncidentStatus] = None,
    limit: int = 50,
):
    """
    List incidents with optional filtering.
    """
    logger.info("incidents_list", realm_id=realm_id, severity=severity, status=status, limit=limit)

    # TODO: Query from database
    from datetime import datetime

    return [
        Incident(
            incident_id="INC-001",
            title="Unusual invoice spike detected",
            severity=IncidentSeverity.HIGH,
            status=IncidentStatus.OPEN,
            confidence=0.92,
            detected_at=datetime.utcnow().isoformat(),
            affected_entities_count=15,
            root_causes_count=2,
        )
    ]


@router.get("/{incident_id}", response_model=IncidentDetail)
async def get_incident_detail(
    incident_id: str,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Get detailed incident information.
    """
    from datetime import datetime

    logger.info("incident_detail", realm_id=realm_id, incident_id=incident_id)

    # TODO: Query from database
    return IncidentDetail(
        incident_id=incident_id,
        title="Unusual invoice spike detected",
        description="Detected 3x increase in invoice amounts over expected baseline",
        severity=IncidentSeverity.HIGH,
        status=IncidentStatus.OPEN,
        confidence=0.92,
        detected_at=datetime.utcnow().isoformat(),
        root_causes=[
            RootCause(
                entity_id="CUST-123",
                entity_type="customer",
                likelihood=0.85,
                explanation="New large customer with atypical order pattern",
            )
        ],
        affected_entities=["INV-001", "INV-002", "INV-003"],
        blast_radius_depth=2,
        timeline=[
            {"timestamp": datetime.utcnow().isoformat(), "event": "Anomaly detected"},
            {"timestamp": datetime.utcnow().isoformat(), "event": "RCA initiated"},
        ],
    )


@router.get("/{incident_id}/postmortem")
async def get_incident_postmortem(
    incident_id: str,
    realm_id: str = Depends(get_current_realm_id),
    format: str = "json",
):
    """
    Generate incident postmortem report.
    Supports json, html, pdf formats.
    """
    logger.info("postmortem_generate", realm_id=realm_id, incident_id=incident_id, format=format)

    # TODO: Generate actual postmortem
    return {
        "incident_id": incident_id,
        "title": "Incident Postmortem: Unusual invoice spike",
        "summary": "A significant spike in invoice amounts was detected...",
        "root_causes": ["New large customer", "Seasonal demand increase"],
        "impact": "15 invoices affected, total value deviation: $45,000",
        "resolution": "Confirmed as legitimate business growth",
        "lessons_learned": ["Update baseline models quarterly", "Add customer segmentation"],
    }
