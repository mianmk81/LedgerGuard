"""
Event data models for the Business Reliability Engine.

This module defines the canonical event schema and data quality models
used throughout the ingestion and analysis pipeline.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .enums import EntityType, EventType


class CanonicalEvent(BaseModel):
    """
    Normalized event representation across all source systems.

    All incoming events from QBO, Shopify, and other systems are transformed
    into this canonical schema for consistent processing, analysis, and storage.
    This normalization enables cross-system correlation and unified anomaly detection.

    Attributes:
        event_id: Unique identifier for this event instance
        event_type: Standardized event type from EventType taxonomy
        event_time: When the business event actually occurred in the source system
        ingested_at: When this event was ingested into the BRE pipeline
        source: Source system identifier (e.g., "qbo", "shopify", "manual")
        source_entity_id: Original entity ID from the source system
        entity_type: Type of business entity this event relates to
        entity_id: Normalized entity ID in BRE's unified namespace
        related_entity_ids: Map of related entities (e.g., {"customer": "CUST-123"})
        amount: Monetary amount associated with the event, if applicable
        currency: ISO 4217 currency code
        attributes: Additional event-specific attributes as flexible JSON
        data_quality_flags: List of quality issues found during ingestion
        schema_version: Version of this canonical schema for evolution tracking
    """

    event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this event instance",
    )
    event_type: EventType = Field(
        description="Standardized event type from canonical taxonomy"
    )
    event_time: datetime = Field(
        description="Timestamp when the business event occurred in source system"
    )
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when event was ingested into BRE pipeline",
    )
    source: str = Field(
        description="Source system identifier (e.g., 'qbo', 'shopify', 'manual')"
    )
    source_entity_id: str = Field(
        description="Original entity ID from the source system"
    )
    entity_type: EntityType = Field(
        description="Type of business entity this event relates to"
    )
    entity_id: str = Field(
        description="Normalized entity ID in BRE's unified namespace"
    )
    related_entity_ids: dict[str, str] = Field(
        default_factory=dict,
        description="Map of related entities (e.g., {'customer': 'CUST-123'})",
    )
    amount: Optional[float] = Field(
        default=None, description="Monetary amount associated with event, if applicable"
    )
    currency: Optional[str] = Field(
        default="USD", description="ISO 4217 currency code"
    )
    attributes: dict = Field(
        default_factory=dict,
        description="Additional event-specific attributes as flexible JSON",
    )
    data_quality_flags: list[str] = Field(
        default_factory=list,
        description="List of quality issues identified during ingestion",
    )
    schema_version: str = Field(
        default="canonical_v1",
        description="Version of canonical schema for evolution tracking",
    )

    @field_validator("currency")
    @classmethod
    def validate_currency_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate currency is a 3-letter ISO code."""
        if v is not None and len(v) != 3:
            raise ValueError("Currency must be a 3-letter ISO 4217 code")
        return v.upper() if v else v

    @field_validator("amount")
    @classmethod
    def validate_amount_precision(cls, v: Optional[float]) -> Optional[float]:
        """Ensure monetary amounts have reasonable precision."""
        if v is not None and abs(v) > 1e12:
            raise ValueError("Amount exceeds maximum allowed value")
        return round(v, 2) if v is not None else None

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "event_id": "evt_123e4567-e89b-12d3-a456-426614174000",
                "event_type": "invoice_paid",
                "event_time": "2026-02-10T14:30:00Z",
                "ingested_at": "2026-02-10T14:32:15Z",
                "source": "qbo",
                "source_entity_id": "INV-1001",
                "entity_type": "invoice",
                "entity_id": "invoice:qbo:INV-1001",
                "related_entity_ids": {"customer": "customer:qbo:CUST-500"},
                "amount": 1250.00,
                "currency": "USD",
                "attributes": {"payment_method": "credit_card"},
                "data_quality_flags": [],
                "schema_version": "canonical_v1",
            }
        }


class QualityIssue(BaseModel):
    """
    Individual data quality issue identified during ingestion.

    Represents a specific quality problem found in a batch of events,
    including the affected field, issue type, and remediation guidance.

    Attributes:
        field: Field name where the issue was detected
        issue_type: Type of quality issue (e.g., "missing", "invalid_format")
        count: Number of records affected by this issue
        description: Human-readable description of the issue
    """

    field: str = Field(description="Field name where issue was detected")
    issue_type: str = Field(
        description="Type of quality issue (e.g., 'missing', 'invalid_format')"
    )
    count: int = Field(description="Number of records affected by this issue", ge=0)
    description: str = Field(description="Human-readable description of the issue")

    @field_validator("count")
    @classmethod
    def validate_count_positive(cls, v: int) -> int:
        """Ensure count is non-negative."""
        if v < 0:
            raise ValueError("Count must be non-negative")
        return v


class DataQualityReport(BaseModel):
    """
    Comprehensive data quality assessment for an ingestion batch.

    Tracks quality metrics across completeness, consistency, and timeliness
    dimensions to ensure reliable downstream analysis. Quality scores inform
    confidence levels in incident detection and causal analysis.

    Attributes:
        batch_id: Unique identifier for this ingestion batch
        source: Source system for this batch
        total_records: Total number of records in batch
        valid_records: Number of records that passed validation
        rejected_records: Number of records rejected due to quality issues
        completeness_score: Proportion of required fields populated (0.0-1.0)
        consistency_score: Proportion of records with consistent values (0.0-1.0)
        timeliness_score: Proportion of records ingested within SLA (0.0-1.0)
        overall_quality_score: Computed overall quality score (0.0-1.0)
        quality_issues: List of specific quality issues detected
        impact_advisory: Human-readable guidance on quality impact
    """

    batch_id: str = Field(description="Unique identifier for this ingestion batch")
    source: str = Field(description="Source system for this batch")
    total_records: int = Field(description="Total number of records in batch", ge=0)
    valid_records: int = Field(
        description="Number of records that passed validation", ge=0
    )
    rejected_records: int = Field(
        description="Number of records rejected due to quality issues", ge=0
    )
    completeness_score: float = Field(
        description="Proportion of required fields populated", ge=0.0, le=1.0
    )
    consistency_score: float = Field(
        description="Proportion of records with consistent values", ge=0.0, le=1.0
    )
    timeliness_score: float = Field(
        description="Proportion of records ingested within SLA", ge=0.0, le=1.0
    )
    overall_quality_score: float = Field(
        description="Computed overall quality score", ge=0.0, le=1.0
    )
    quality_issues: list[QualityIssue] = Field(
        default_factory=list, description="List of specific quality issues detected"
    )
    impact_advisory: str = Field(
        description="Human-readable guidance on quality impact"
    )

    @field_validator("overall_quality_score")
    @classmethod
    def compute_overall_score(cls, v: float, info) -> float:
        """Ensure overall quality score is reasonable."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Overall quality score must be between 0.0 and 1.0")
        return round(v, 4)

    @field_validator("valid_records", "rejected_records")
    @classmethod
    def validate_record_counts(cls, v: int, info) -> int:
        """Validate record counts are consistent."""
        if v < 0:
            raise ValueError("Record counts must be non-negative")
        return v

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "batch_id": "batch_20260210_143000",
                "source": "qbo",
                "total_records": 1000,
                "valid_records": 985,
                "rejected_records": 15,
                "completeness_score": 0.98,
                "consistency_score": 0.99,
                "timeliness_score": 0.95,
                "overall_quality_score": 0.9733,
                "quality_issues": [
                    {
                        "field": "amount",
                        "issue_type": "missing",
                        "count": 10,
                        "description": "Amount field missing in 10 invoice events",
                    }
                ],
                "impact_advisory": "High quality batch. Minor missing amount fields do not impact detection accuracy.",
            }
        }
