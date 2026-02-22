"""
Incident detection models for the Business Reliability Engine.

This module defines incident and cascade data structures representing
detected operational anomalies and their propagation patterns.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .enums import (
    Confidence,
    DetectionMethod,
    IncidentStatus,
    IncidentType,
    Severity,
)


class Incident(BaseModel):
    """
    A detected business reliability incident.

    Represents a statistically significant anomaly in business operations,
    including the detection methodology, affected metrics, and supporting
    evidence. Incidents are the primary detection output of the BRE system.

    Attributes:
        incident_id: Unique identifier for this incident
        incident_type: Classification of the incident pattern
        detected_at: When the incident was detected by the system
        incident_window_start: Beginning of the time window where anomaly occurred
        incident_window_end: End of the time window where anomaly occurred
        severity: Business impact severity level
        confidence: Statistical confidence in the detection
        detection_methods: List of algorithms that detected this incident
        primary_metric: Name of the primary metric showing anomalous behavior
        primary_metric_value: Actual value of the primary metric during incident
        primary_metric_baseline: Expected baseline value for comparison
        primary_metric_zscore: Z-score indicating deviation from baseline
        supporting_metrics: Additional metrics supporting the detection
        evidence_event_ids: List of event IDs that comprise the evidence
        evidence_event_count: Total count of evidence events
        data_quality_score: Quality score of underlying data (0.0-1.0)
        run_id: ID of the detection run that identified this incident
        cascade_id: Optional ID linking this to a cascade pattern
        status: Current lifecycle status of the incident
    """

    incident_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this incident",
    )
    incident_type: IncidentType = Field(
        description="Classification of the incident pattern"
    )
    detected_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the incident was detected by the system",
    )
    incident_window_start: datetime = Field(
        description="Beginning of the time window where anomaly occurred"
    )
    incident_window_end: datetime = Field(
        description="End of the time window where anomaly occurred"
    )
    severity: Severity = Field(description="Business impact severity level")
    confidence: Confidence = Field(
        description="Statistical confidence in the detection"
    )
    detection_methods: list[DetectionMethod] = Field(
        description="List of algorithms that detected this incident"
    )
    primary_metric: str = Field(
        description="Name of the primary metric showing anomalous behavior"
    )
    primary_metric_value: float = Field(
        description="Actual value of the primary metric during incident"
    )
    primary_metric_baseline: float = Field(
        description="Expected baseline value for comparison"
    )
    primary_metric_zscore: float = Field(
        description="Z-score indicating deviation from baseline"
    )
    supporting_metrics: list[dict] = Field(
        default_factory=list,
        description="Additional metrics supporting the detection",
    )
    evidence_event_ids: list[str] = Field(
        default_factory=list, description="List of event IDs that comprise the evidence"
    )
    evidence_event_count: int = Field(
        description="Total count of evidence events", ge=0
    )
    data_quality_score: float = Field(
        description="Quality score of underlying data", ge=0.0, le=1.0
    )
    run_id: str = Field(
        description="ID of the detection run that identified this incident"
    )
    cascade_id: Optional[str] = Field(
        default=None, description="Optional ID linking this to a cascade pattern"
    )
    status: IncidentStatus = Field(
        default=IncidentStatus.OPEN, description="Current lifecycle status"
    )

    @field_validator("incident_window_end")
    @classmethod
    def validate_time_window(cls, v: datetime, info) -> datetime:
        """Ensure incident window end is after start."""
        if "incident_window_start" in info.data:
            if v <= info.data["incident_window_start"]:
                raise ValueError("incident_window_end must be after incident_window_start")
        return v

    @field_validator("data_quality_score")
    @classmethod
    def validate_quality_score(cls, v: float) -> float:
        """Ensure quality score is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Data quality score must be between 0.0 and 1.0")
        return round(v, 4)

    @field_validator("primary_metric_zscore")
    @classmethod
    def validate_zscore(cls, v: float) -> float:
        """Validate z-score is reasonable."""
        if abs(v) > 100:
            raise ValueError("Z-score exceeds reasonable bounds")
        return round(v, 4)

    @field_validator("evidence_event_count")
    @classmethod
    def validate_event_count(cls, v: int) -> int:
        """Ensure event count is non-negative."""
        if v < 0:
            raise ValueError("Evidence event count must be non-negative")
        return v

    @field_validator("detection_methods")
    @classmethod
    def validate_detection_methods(cls, v: list[DetectionMethod]) -> list[DetectionMethod]:
        """Ensure at least one detection method is specified."""
        if not v:
            raise ValueError("At least one detection method must be specified")
        return v

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "incident_id": "inc_123e4567-e89b-12d3-a456-426614174000",
                "incident_type": "refund_spike",
                "detected_at": "2026-02-10T15:00:00Z",
                "incident_window_start": "2026-02-10T12:00:00Z",
                "incident_window_end": "2026-02-10T14:00:00Z",
                "severity": "high",
                "confidence": "very_high",
                "detection_methods": ["mad_zscore", "isolation_forest"],
                "primary_metric": "refund_rate_2h",
                "primary_metric_value": 0.15,
                "primary_metric_baseline": 0.03,
                "primary_metric_zscore": 8.5,
                "supporting_metrics": [
                    {"metric": "refund_count_2h", "value": 45, "baseline": 9}
                ],
                "evidence_event_ids": ["evt_001", "evt_002"],
                "evidence_event_count": 45,
                "data_quality_score": 0.98,
                "run_id": "run_20260210_150000",
                "cascade_id": None,
                "status": "open",
            }
        }


class IncidentCascade(BaseModel):
    """
    A chain of causally related incidents propagating through the system.

    Represents detected cascading failures where one incident triggers
    downstream incidents, similar to how site reliability incidents can
    propagate through microservices. Understanding cascades is critical
    for root cause isolation and targeted remediation.

    Attributes:
        cascade_id: Unique identifier for this cascade
        root_incident_id: ID of the initial incident that triggered the cascade
        incident_ids: Ordered list of all incidents in this cascade
        cascade_path: Human-readable path showing incident propagation
        total_blast_radius: Aggregated impact metrics across all incidents
        cascade_score: Severity score for the overall cascade (0.0-1.0)
        detected_at: When the cascade pattern was detected
    """

    cascade_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this cascade",
    )
    root_incident_id: str = Field(
        description="ID of the initial incident that triggered the cascade"
    )
    incident_ids: list[str] = Field(
        description="Ordered list of all incidents in this cascade"
    )
    cascade_path: list[str] = Field(
        description="Human-readable path showing incident propagation"
    )
    total_blast_radius: dict = Field(
        description="Aggregated impact metrics across all incidents"
    )
    cascade_score: float = Field(
        description="Severity score for the overall cascade", ge=0.0, le=1.0
    )
    detected_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the cascade pattern was detected",
    )

    @field_validator("cascade_score")
    @classmethod
    def validate_cascade_score(cls, v: float) -> float:
        """Ensure cascade score is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Cascade score must be between 0.0 and 1.0")
        return round(v, 4)

    @field_validator("incident_ids")
    @classmethod
    def validate_incident_ids(cls, v: list[str]) -> list[str]:
        """Ensure at least two incidents in a cascade."""
        if len(v) < 2:
            raise ValueError("A cascade must contain at least 2 incidents")
        return v

    @field_validator("cascade_path")
    @classmethod
    def validate_cascade_path(cls, v: list[str]) -> list[str]:
        """Ensure cascade path is not empty."""
        if not v:
            raise ValueError("Cascade path cannot be empty")
        return v

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "cascade_id": "casc_123e4567-e89b-12d3-a456-426614174000",
                "root_incident_id": "inc_root_001",
                "incident_ids": ["inc_root_001", "inc_downstream_002", "inc_downstream_003"],
                "cascade_path": [
                    "refund_spike",
                    "support_load_surge",
                    "customer_satisfaction_regression",
                ],
                "total_blast_radius": {
                    "customers_affected": 1250,
                    "revenue_exposure": 45000.00,
                },
                "cascade_score": 0.85,
                "detected_at": "2026-02-10T15:30:00Z",
            }
        }
