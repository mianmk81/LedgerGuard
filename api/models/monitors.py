"""
Monitoring and alerting models for the Business Reliability Engine.

This module defines monitoring rule and alert structures for proactive
detection and notification of potential incidents before they reach
critical severity.
"""

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .enums import AlertStatus, MonitorStatus, Severity


class MonitorRule(BaseModel):
    """
    A monitoring rule for proactive anomaly detection.

    Defines conditions to evaluate on business metrics at regular intervals
    to detect potential issues early. Monitor rules are typically generated
    from postmortem analysis to prevent recurrence of known incident patterns.

    Attributes:
        monitor_id: Unique identifier for this monitor
        name: Short descriptive name for the monitor
        description: Detailed description of what the monitor detects
        source_incident_id: ID of the incident that motivated this monitor
        metric_name: Name of the metric being monitored
        condition: Boolean condition expression (e.g., "value > baseline * 1.5")
        baseline_window_days: Number of days for baseline calculation
        check_frequency: How often to evaluate (e.g., "hourly", "daily")
        severity_if_triggered: Severity level to assign if condition is met
        enabled: Whether this monitor is currently active
        created_at: When this monitor was created
        alert_message_template: Template for alert messages with placeholders
    """

    monitor_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this monitor",
    )
    name: str = Field(description="Short descriptive name for the monitor")
    description: str = Field(description="Detailed description of what monitor detects")
    source_incident_id: str = Field(
        description="ID of the incident that motivated this monitor"
    )
    metric_name: str = Field(description="Name of the metric being monitored")
    condition: str = Field(
        description="Boolean condition expression (e.g., 'value > baseline * 1.5')"
    )
    baseline_window_days: int = Field(
        description="Number of days for baseline calculation", ge=1, le=365
    )
    check_frequency: str = Field(
        default="daily", description="How often to evaluate (e.g., 'hourly', 'daily')"
    )
    severity_if_triggered: Severity = Field(
        description="Severity level to assign if condition is met"
    )
    enabled: bool = Field(
        default=True, description="Whether this monitor is currently active"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When this monitor was created"
    )
    alert_message_template: str = Field(
        description="Template for alert messages with placeholders"
    )

    @field_validator("baseline_window_days")
    @classmethod
    def validate_baseline_window(cls, v: int) -> int:
        """Ensure baseline window is reasonable."""
        if not 1 <= v <= 365:
            raise ValueError("Baseline window must be between 1 and 365 days")
        return v

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        """Ensure monitor name is not empty."""
        if not v or len(v.strip()) < 3:
            raise ValueError("Monitor name must be at least 3 characters")
        return v.strip()

    @field_validator("description")
    @classmethod
    def validate_description_not_empty(cls, v: str) -> str:
        """Ensure description is not empty."""
        if not v or len(v.strip()) < 10:
            raise ValueError("Description must be at least 10 characters")
        return v.strip()

    @field_validator("condition")
    @classmethod
    def validate_condition_not_empty(cls, v: str) -> str:
        """Ensure condition is not empty."""
        if not v or len(v.strip()) < 3:
            raise ValueError("Condition must be at least 3 characters")
        return v.strip()

    @field_validator("metric_name")
    @classmethod
    def validate_metric_name_not_empty(cls, v: str) -> str:
        """Ensure metric name is not empty."""
        if not v or len(v.strip()) < 2:
            raise ValueError("Metric name must be at least 2 characters")
        return v.strip()

    @field_validator("alert_message_template")
    @classmethod
    def validate_alert_template_not_empty(cls, v: str) -> str:
        """Ensure alert template is not empty."""
        if not v or len(v.strip()) < 10:
            raise ValueError("Alert message template must be at least 10 characters")
        return v.strip()

    @field_validator("check_frequency")
    @classmethod
    def validate_check_frequency(cls, v: str) -> str:
        """Validate check frequency is a known value."""
        valid_frequencies = {
            "minutely",
            "5min",
            "15min",
            "30min",
            "hourly",
            "4hourly",
            "daily",
            "weekly",
        }
        if v.lower() not in valid_frequencies:
            raise ValueError(
                f"Check frequency must be one of: {', '.join(valid_frequencies)}"
            )
        return v.lower()

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "monitor_id": "mon_123e4567-e89b-12d3-a456-426614174000",
                "name": "Refund Rate Spike Detector",
                "description": (
                    "Monitors 2-hour rolling refund rate to detect spikes "
                    "similar to the Feb 10 incident. Triggers at 2x baseline."
                ),
                "source_incident_id": "inc_123e4567",
                "metric_name": "refund_rate_2h",
                "condition": "value > baseline * 2.0",
                "baseline_window_days": 30,
                "check_frequency": "hourly",
                "severity_if_triggered": "high",
                "enabled": True,
                "created_at": "2026-02-10T16:00:00Z",
                "alert_message_template": (
                    "ALERT: Refund rate is {value:.2%}, {multiplier:.1f}x "
                    "baseline of {baseline:.2%}. Check for product quality issues."
                ),
            }
        }


class MonitorAlert(BaseModel):
    """
    An alert generated by a monitor rule.

    Represents a specific instance where a monitor rule's condition was met,
    requiring human attention. Alerts track acknowledgment and resolution status
    to ensure proper operational response.

    Attributes:
        alert_id: Unique identifier for this alert
        monitor_id: ID of the monitor rule that generated this alert
        triggered_at: When the monitor condition was met
        metric_name: Name of the metric that triggered the alert
        current_value: Current value of the metric
        baseline_value: Baseline value for comparison
        threshold: Threshold expression that was exceeded
        severity: Severity level of this alert
        message: Human-readable alert message
        related_incident_id: ID of incident if one was created from this alert
        status: Current status of this alert
    """

    alert_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this alert",
    )
    monitor_id: str = Field(
        description="ID of the monitor rule that generated this alert"
    )
    triggered_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the monitor condition was met"
    )
    metric_name: str = Field(description="Name of the metric that triggered the alert")
    current_value: float = Field(description="Current value of the metric")
    baseline_value: float = Field(description="Baseline value for comparison")
    threshold: str = Field(description="Threshold expression that was exceeded")
    severity: Severity = Field(description="Severity level of this alert")
    message: str = Field(description="Human-readable alert message")
    related_incident_id: str = Field(
        description="ID of incident if one was created from this alert"
    )
    status: AlertStatus = Field(
        default=AlertStatus.ACTIVE, description="Current status of this alert"
    )

    @field_validator("metric_name")
    @classmethod
    def validate_metric_name_not_empty(cls, v: str) -> str:
        """Ensure metric name is not empty."""
        if not v or len(v.strip()) < 2:
            raise ValueError("Metric name must be at least 2 characters")
        return v.strip()

    @field_validator("message")
    @classmethod
    def validate_message_not_empty(cls, v: str) -> str:
        """Ensure alert message is not empty."""
        if not v or len(v.strip()) < 10:
            raise ValueError("Alert message must be at least 10 characters")
        return v.strip()

    @field_validator("threshold")
    @classmethod
    def validate_threshold_not_empty(cls, v: str) -> str:
        """Ensure threshold is not empty."""
        if not v or len(v.strip()) < 3:
            raise ValueError("Threshold must be at least 3 characters")
        return v.strip()

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "alert_id": "alert_123e4567-e89b-12d3-a456-426614174000",
                "monitor_id": "mon_123e4567",
                "triggered_at": "2026-02-11T14:00:00Z",
                "metric_name": "refund_rate_2h",
                "current_value": 0.06,
                "baseline_value": 0.03,
                "threshold": "value > baseline * 2.0",
                "severity": "high",
                "message": (
                    "ALERT: Refund rate is 6.00%, 2.0x baseline of 3.00%. "
                    "Check for product quality issues."
                ),
                "related_incident_id": "inc_new_001",
                "status": "active",
            }
        }
