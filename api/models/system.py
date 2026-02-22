"""
System-level models for the Business Reliability Engine.

This module defines operational models for system health, run manifests,
and configuration tracking.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class RunManifest(BaseModel):
    """
    Execution manifest for a complete BRE analysis run.

    Tracks the complete lifecycle of a detection and analysis run including
    configuration, data quality, detected incidents, and generated artifacts.
    Run manifests provide traceability and reproducibility for all system outputs.

    Attributes:
        run_id: Unique identifier for this run
        started_at: When the run was initiated
        completed_at: When the run completed (None if still in progress)
        events_processed: Total number of canonical events processed
        config_version: Version identifier for configuration used
        schema_versions: Map of schema names to versions used in this run
        detection_methods_used: List of anomaly detection algorithms invoked
        models_invoked: List of analysis models called with metadata
        incidents_detected: Count of incidents detected in this run
        cascades_detected: Count of cascades detected in this run
        postmortems_generated: Count of postmortems generated in this run
        monitors_created: Count of new monitor rules created in this run
        data_quality_summary: Summary statistics of data quality across run
        status: Current status of the run (e.g., "running", "completed", "failed")
    """

    run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this run",
    )
    started_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the run was initiated"
    )
    completed_at: Optional[datetime] = Field(
        default=None, description="When the run completed (None if in progress)"
    )
    events_processed: int = Field(
        description="Total number of canonical events processed", ge=0
    )
    config_version: str = Field(
        description="Version identifier for configuration used"
    )
    schema_versions: dict[str, str] = Field(
        description="Map of schema names to versions used in this run"
    )
    detection_methods_used: list[str] = Field(
        description="List of anomaly detection algorithms invoked"
    )
    models_invoked: list[dict] = Field(
        description="List of analysis models called with metadata"
    )
    incidents_detected: int = Field(
        description="Count of incidents detected in this run", ge=0
    )
    cascades_detected: int = Field(
        description="Count of cascades detected in this run", ge=0
    )
    postmortems_generated: int = Field(
        description="Count of postmortems generated in this run", ge=0
    )
    monitors_created: int = Field(
        description="Count of new monitor rules created in this run", ge=0
    )
    data_quality_summary: dict = Field(
        description="Summary statistics of data quality across run"
    )
    status: str = Field(
        description="Current status of the run (e.g., 'running', 'completed', 'failed')"
    )

    @field_validator("completed_at")
    @classmethod
    def validate_completion_time(cls, v: Optional[datetime], info) -> Optional[datetime]:
        """Ensure completion time is after start time."""
        if v is not None and "started_at" in info.data:
            if v < info.data["started_at"]:
                raise ValueError("completed_at must be after started_at")
        return v

    @field_validator("events_processed", "incidents_detected", "cascades_detected", "postmortems_generated", "monitors_created")
    @classmethod
    def validate_counts_non_negative(cls, v: int) -> int:
        """Ensure count fields are non-negative."""
        if v < 0:
            raise ValueError("Count must be non-negative")
        return v

    @field_validator("config_version")
    @classmethod
    def validate_config_version_not_empty(cls, v: str) -> str:
        """Ensure config version is not empty."""
        if not v or len(v.strip()) < 1:
            raise ValueError("Config version cannot be empty")
        return v.strip()

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status is a known value."""
        valid_statuses = {"running", "completed", "failed", "cancelled"}
        if v.lower() not in valid_statuses:
            raise ValueError(
                f"Status must be one of: {', '.join(valid_statuses)}"
            )
        return v.lower()

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "run_id": "run_20260210_150000",
                "started_at": "2026-02-10T15:00:00Z",
                "completed_at": "2026-02-10T15:45:30Z",
                "events_processed": 15420,
                "config_version": "v1.2.5",
                "schema_versions": {
                    "canonical_event": "canonical_v1",
                    "incident": "incident_v1",
                    "postmortem": "postmortem_v1",
                },
                "detection_methods_used": ["mad_zscore", "isolation_forest", "changepoint"],
                "models_invoked": [
                    {"model": "gpt-4o", "purpose": "postmortem_generation", "tokens": 12500},
                    {"model": "gpt-4o", "purpose": "monitor_generation", "tokens": 3200},
                ],
                "incidents_detected": 3,
                "cascades_detected": 1,
                "postmortems_generated": 3,
                "monitors_created": 5,
                "data_quality_summary": {
                    "overall_quality_score": 0.96,
                    "completeness_score": 0.98,
                    "consistency_score": 0.97,
                    "timeliness_score": 0.93,
                },
                "status": "completed",
            }
        }


class HealthStatus(BaseModel):
    """
    Real-time health status of the BRE system.

    Provides operational health metrics for monitoring system availability,
    connectivity to external systems, and data freshness. Used for system
    health endpoints and operational dashboards.

    Attributes:
        api: Health status of the API service ("healthy", "degraded", "down")
        database: Health status of the database connection
        qbo_connection: Health status of QuickBooks Online connection
        last_scan_time: Timestamp of the most recent detection scan
        active_monitors: Count of currently active monitor rules
        data_freshness_hours: Hours since most recent event ingestion
    """

    api: str = Field(
        description="Health status of the API service ('healthy', 'degraded', 'down')"
    )
    database: str = Field(description="Health status of the database connection")
    qbo_connection: str = Field(
        description="Health status of QuickBooks Online connection"
    )
    last_scan_time: Optional[datetime] = Field(
        default=None, description="Timestamp of the most recent detection scan"
    )
    active_monitors: int = Field(
        description="Count of currently active monitor rules", ge=0
    )
    data_freshness_hours: Optional[float] = Field(
        default=None,
        description="Hours since most recent event ingestion",
        ge=0.0,
    )

    @field_validator("api", "database", "qbo_connection")
    @classmethod
    def validate_health_status(cls, v: str) -> str:
        """Validate health status values."""
        valid_statuses = {"healthy", "degraded", "down", "unknown"}
        if v.lower() not in valid_statuses:
            raise ValueError(
                f"Health status must be one of: {', '.join(valid_statuses)}"
            )
        return v.lower()

    @field_validator("active_monitors")
    @classmethod
    def validate_active_monitors(cls, v: int) -> int:
        """Ensure active monitors count is non-negative."""
        if v < 0:
            raise ValueError("Active monitors count must be non-negative")
        return v

    @field_validator("data_freshness_hours")
    @classmethod
    def validate_data_freshness(cls, v: Optional[float]) -> Optional[float]:
        """Ensure data freshness is reasonable."""
        if v is not None:
            if v < 0.0:
                raise ValueError("Data freshness must be non-negative")
            if v > 8760.0:  # 365 days
                raise ValueError("Data freshness exceeds reasonable bounds (>365 days)")
        return round(v, 2) if v is not None else None

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "api": "healthy",
                "database": "healthy",
                "qbo_connection": "healthy",
                "last_scan_time": "2026-02-10T15:45:00Z",
                "active_monitors": 12,
                "data_freshness_hours": 0.5,
            }
        }


class SystemConfig(BaseModel):
    """
    System configuration snapshot.

    Captures the complete configuration state including schema versions,
    detection thresholds, RCA parameters, and monitor defaults. Configuration
    snapshots are versioned and associated with run manifests for reproducibility.

    Attributes:
        schema_versions: Map of schema names to current versions
        detection_thresholds: Threshold configurations for anomaly detection
        rca_config: Configuration parameters for RCA algorithms
        monitor_defaults: Default settings for generated monitors
    """

    schema_versions: dict[str, str] = Field(
        description="Map of schema names to current versions"
    )
    detection_thresholds: dict = Field(
        description="Threshold configurations for anomaly detection"
    )
    rca_config: dict = Field(description="Configuration parameters for RCA algorithms")
    monitor_defaults: dict = Field(description="Default settings for generated monitors")

    @field_validator("schema_versions")
    @classmethod
    def validate_schema_versions_not_empty(cls, v: dict[str, str]) -> dict[str, str]:
        """Ensure schema versions are provided."""
        if not v:
            raise ValueError("Schema versions cannot be empty")
        return v

    @field_validator("detection_thresholds")
    @classmethod
    def validate_detection_thresholds_not_empty(cls, v: dict) -> dict:
        """Ensure detection thresholds are provided."""
        if not v:
            raise ValueError("Detection thresholds cannot be empty")
        return v

    @field_validator("rca_config")
    @classmethod
    def validate_rca_config_not_empty(cls, v: dict) -> dict:
        """Ensure RCA config is provided."""
        if not v:
            raise ValueError("RCA configuration cannot be empty")
        return v

    @field_validator("monitor_defaults")
    @classmethod
    def validate_monitor_defaults_not_empty(cls, v: dict) -> dict:
        """Ensure monitor defaults are provided."""
        if not v:
            raise ValueError("Monitor defaults cannot be empty")
        return v

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "schema_versions": {
                    "canonical_event": "canonical_v1",
                    "incident": "incident_v1",
                    "postmortem": "postmortem_v1",
                },
                "detection_thresholds": {
                    "mad_zscore_threshold": 3.5,
                    "isolation_forest_contamination": 0.1,
                    "changepoint_min_size": 30,
                    "min_evidence_events": 10,
                },
                "rca_config": {
                    "max_causal_paths": 5,
                    "min_contribution_score": 0.1,
                    "temporal_window_hours": 24,
                    "graph_proximity_weight": 0.3,
                },
                "monitor_defaults": {
                    "default_baseline_days": 30,
                    "default_check_frequency": "hourly",
                    "default_severity": "medium",
                },
            }
        }
