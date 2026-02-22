"""
Postmortem report models for the Business Reliability Engine.

This module defines comprehensive incident postmortem structures that
synthesize detection, root cause analysis, impact assessment, and
remediation recommendations into actionable executive reports.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .blast_radius import BlastRadius
from .enums import IncidentStatus, Severity
from .monitors import MonitorRule
from .rca import CausalChain


class TimelineEntry(BaseModel):
    """
    A significant event in the incident timeline.

    Represents a key moment in the incident lifecycle including detection,
    metric changes, and system responses. Timeline entries provide a
    chronological narrative of incident progression.

    Attributes:
        timestamp: When this timeline event occurred
        event_description: Human-readable description of what happened
        metric_name: Optional metric name if this entry relates to a metric
        metric_value: Optional metric value at this point in time
        evidence_event_ids: List of canonical event IDs supporting this entry
    """

    timestamp: datetime = Field(description="When this timeline event occurred")
    event_description: str = Field(
        description="Human-readable description of what happened"
    )
    metric_name: Optional[str] = Field(
        default=None, description="Optional metric name if entry relates to a metric"
    )
    metric_value: Optional[float] = Field(
        default=None, description="Optional metric value at this point in time"
    )
    evidence_event_ids: list[str] = Field(
        default_factory=list,
        description="List of canonical event IDs supporting this entry",
    )

    @field_validator("event_description")
    @classmethod
    def validate_description_not_empty(cls, v: str) -> str:
        """Ensure event description is meaningful."""
        if not v or len(v.strip()) < 5:
            raise ValueError("Event description must be at least 5 characters")
        return v.strip()


class Postmortem(BaseModel):
    """
    Comprehensive incident postmortem report.

    The definitive analytical artifact combining detection, root cause analysis,
    impact assessment, and recommendations. Postmortems serve as both operational
    documentation and machine learning training data for continuous improvement.

    This structure mirrors site reliability engineering postmortem practices
    adapted for business operations.

    Attributes:
        postmortem_id: Unique identifier for this postmortem
        incident_id: Primary incident being analyzed
        cascade_id: Optional cascade ID if incident was part of a cascade
        generated_at: When this postmortem was generated
        title: Executive summary title
        severity: Overall severity assessment
        duration: Incident duration in human-readable format
        status: Current incident status
        one_line_summary: Single sentence executive summary
        timeline: Chronological sequence of significant events
        causal_chain: Complete root cause analysis
        root_cause_summary: Human-readable root cause explanation
        blast_radius: Quantified business impact assessment
        contributing_factors: List of secondary contributing factors
        monitors: Recommended monitoring rules to prevent recurrence
        recommendations: Actionable remediation recommendations
        data_quality_score: Quality score of underlying data (0.0-1.0)
        confidence_note: Explanation of confidence level and limitations
        algorithm_version: Version of analysis algorithms used
        run_id: ID of the analysis run that generated this postmortem
    """

    postmortem_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this postmortem",
    )
    incident_id: str = Field(description="Primary incident being analyzed")
    cascade_id: Optional[str] = Field(
        default=None, description="Optional cascade ID if part of a cascade"
    )
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this postmortem was generated",
    )
    title: str = Field(description="Executive summary title")
    severity: Severity = Field(description="Overall severity assessment")
    duration: str = Field(description="Incident duration in human-readable format")
    status: IncidentStatus = Field(description="Current incident status")
    one_line_summary: str = Field(description="Single sentence executive summary")
    timeline: list[TimelineEntry] = Field(
        description="Chronological sequence of significant events"
    )
    causal_chain: CausalChain = Field(description="Complete root cause analysis")
    root_cause_summary: str = Field(description="Human-readable root cause explanation")
    blast_radius: BlastRadius = Field(description="Quantified business impact")
    contributing_factors: list[str] = Field(
        description="List of secondary contributing factors"
    )
    monitors: list[MonitorRule] = Field(
        description="Recommended monitoring rules to prevent recurrence"
    )
    recommendations: list[str] = Field(
        description="Actionable remediation recommendations"
    )
    data_quality_score: float = Field(
        description="Quality score of underlying data", ge=0.0, le=1.0
    )
    confidence_note: str = Field(
        description="Explanation of confidence level and limitations"
    )
    algorithm_version: str = Field(description="Version of analysis algorithms used")
    run_id: str = Field(
        description="ID of the analysis run that generated this postmortem"
    )

    @field_validator("data_quality_score")
    @classmethod
    def validate_quality_score(cls, v: float) -> float:
        """Ensure quality score is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Data quality score must be between 0.0 and 1.0")
        return round(v, 4)

    @field_validator("title", "one_line_summary", "root_cause_summary", "confidence_note")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Ensure text fields are not empty."""
        if not v or len(v.strip()) < 10:
            raise ValueError("Text field must be at least 10 characters")
        return v.strip()

    @field_validator("timeline")
    @classmethod
    def validate_timeline_not_empty(cls, v: list[TimelineEntry]) -> list[TimelineEntry]:
        """Ensure timeline has at least one entry."""
        if not v:
            raise ValueError("Timeline must contain at least one entry")
        return v

    @field_validator("timeline")
    @classmethod
    def validate_timeline_chronological(cls, v: list[TimelineEntry]) -> list[TimelineEntry]:
        """Ensure timeline entries are in chronological order."""
        for i in range(1, len(v)):
            if v[i].timestamp < v[i - 1].timestamp:
                raise ValueError("Timeline entries must be in chronological order")
        return v

    @field_validator("recommendations")
    @classmethod
    def validate_recommendations_not_empty(cls, v: list[str]) -> list[str]:
        """Ensure at least one recommendation is provided."""
        if not v:
            raise ValueError("At least one recommendation must be provided")
        return v

    @field_validator("monitors")
    @classmethod
    def validate_monitors_not_empty(cls, v: list[MonitorRule]) -> list[MonitorRule]:
        """Ensure at least one monitor is provided."""
        if not v:
            raise ValueError("At least one monitor rule must be provided")
        return v

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "postmortem_id": "pm_123e4567-e89b-12d3-a456-426614174000",
                "incident_id": "inc_123e4567",
                "cascade_id": None,
                "generated_at": "2026-02-10T16:00:00Z",
                "title": "Refund Spike from Product Quality Issue - SKU-945",
                "severity": "high",
                "duration": "2 hours",
                "status": "acknowledged",
                "one_line_summary": (
                    "8.5σ refund spike affecting 1,250 customers traced to "
                    "quality defect in Product SKU-945 shipped on Feb 8-9."
                ),
                "timeline": [
                    {
                        "timestamp": "2026-02-10T12:00:00Z",
                        "event_description": "First anomalous refund requests detected",
                        "evidence_event_ids": ["evt_001"],
                    }
                ],
                "causal_chain": {
                    "chain_id": "chain_123",
                    "incident_id": "inc_123e4567",
                    "paths": [],
                    "algorithm_version": "BRE-RCA-v1",
                    "causal_window": (
                        "2026-02-10T08:00:00Z",
                        "2026-02-10T14:00:00Z",
                    ),
                    "dependency_graph_version": "v2",
                    "run_id": "run_20260210",
                },
                "root_cause_summary": (
                    "Root cause identified as manufacturing defect in SKU-945 "
                    "batch shipped February 8-9, leading to elevated failure rate."
                ),
                "blast_radius": {
                    "incident_id": "inc_123e4567",
                    "customers_affected": 1250,
                    "orders_affected": 2100,
                    "products_affected": 1,
                    "vendors_involved": 1,
                    "estimated_revenue_exposure": 125000.00,
                    "estimated_refund_exposure": 45000.00,
                    "estimated_churn_exposure": 75,
                    "downstream_incidents_triggered": [],
                    "blast_radius_severity": "severe",
                    "narrative": "Severe impact with significant revenue exposure",
                },
                "contributing_factors": [
                    "Lack of quality inspection at receiving",
                    "No automated alerts on defect rate thresholds",
                ],
                "monitors": [],
                "recommendations": [
                    "Implement receiving quality inspection for Vendor-42",
                    "Set up real-time monitoring on product return rates",
                ],
                "data_quality_score": 0.98,
                "confidence_note": (
                    "Very high confidence (98% data quality). "
                    "Strong temporal correlation and clear causal path."
                ),
                "algorithm_version": "BRE-v1.0",
                "run_id": "run_20260210_160000",
            }
        }
