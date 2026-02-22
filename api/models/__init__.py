"""
Pydantic v2 data models for the Business Reliability Engine.

This package contains all Pydantic models used throughout the LedgerGuard
system for type-safe data validation, serialization, and API contracts.

All models use Pydantic v2 with strict validation, comprehensive docstrings,
and field-level validators to ensure data quality and system reliability.

Model Organization:
    - enums: Enumeration types for consistent classification
    - events: Canonical event schema and data quality models
    - incidents: Incident detection and cascade models
    - rca: Root cause analysis and causal chain models
    - blast_radius: Business impact assessment models
    - postmortem: Comprehensive incident postmortem reports
    - monitors: Monitoring rules and alert models
    - simulation: What-if scenarios and incident comparisons
    - system: System health, configuration, and run manifests

Schema Versioning:
    All models include schema_version fields where applicable to support
    evolution and backward compatibility as the system matures.

Usage:
    >>> from api.models import CanonicalEvent, Incident, Postmortem
    >>> event = CanonicalEvent(
    ...     event_type=EventType.INVOICE_PAID,
    ...     event_time=datetime.utcnow(),
    ...     source="qbo",
    ...     source_entity_id="INV-1001",
    ...     entity_type=EntityType.INVOICE,
    ...     entity_id="invoice:qbo:INV-1001",
    ...     amount=1250.00
    ... )
"""

# Enumerations
from .enums import (
    AlertStatus,
    BlastRadiusSeverity,
    Confidence,
    DetectionMethod,
    EntityType,
    EventType,
    IncidentStatus,
    IncidentType,
    MonitorStatus,
    Severity,
)

# Event models
from .events import CanonicalEvent, DataQualityReport, QualityIssue

# Incident models
from .incidents import Incident, IncidentCascade

# Root cause analysis models
from .rca import CausalChain, CausalNode, CausalPath, EvidenceCluster

# Blast radius models
from .blast_radius import BlastRadius

# Postmortem models
from .postmortem import Postmortem, TimelineEntry

# Monitor models (primary definitions)
from .monitors import MonitorAlert, MonitorRule

# Simulation models
from .simulation import IncidentComparison, WhatIfScenario

# System models
from .system import HealthStatus, RunManifest, SystemConfig

# Export all models for easy importing
__all__ = [
    # Enumerations
    "AlertStatus",
    "BlastRadiusSeverity",
    "Confidence",
    "DetectionMethod",
    "EntityType",
    "EventType",
    "IncidentStatus",
    "IncidentType",
    "MonitorStatus",
    "Severity",
    # Event models
    "CanonicalEvent",
    "DataQualityReport",
    "QualityIssue",
    # Incident models
    "Incident",
    "IncidentCascade",
    # RCA models
    "CausalChain",
    "CausalNode",
    "CausalPath",
    "EvidenceCluster",
    # Blast radius models
    "BlastRadius",
    # Postmortem models
    "Postmortem",
    "TimelineEntry",
    # Monitor models
    "MonitorAlert",
    "MonitorRule",
    # Simulation models
    "IncidentComparison",
    "WhatIfScenario",
    # System models
    "HealthStatus",
    "RunManifest",
    "SystemConfig",
]

# Schema version registry for tracking evolution
SCHEMA_VERSIONS = {
    "canonical_event": "canonical_v1",
    "incident": "incident_v1",
    "incident_cascade": "cascade_v1",
    "causal_chain": "rca_v1",
    "blast_radius": "blast_v1",
    "postmortem": "postmortem_v1",
    "monitor_rule": "monitor_v1",
    "monitor_alert": "alert_v1",
    "what_if_scenario": "scenario_v1",
    "incident_comparison": "comparison_v1",
    "run_manifest": "manifest_v1",
    "health_status": "health_v1",
    "system_config": "config_v1",
}

# Model registry mapping names to classes
MODEL_REGISTRY = {
    "canonical_event": CanonicalEvent,
    "incident": Incident,
    "incident_cascade": IncidentCascade,
    "causal_chain": CausalChain,
    "blast_radius": BlastRadius,
    "postmortem": Postmortem,
    "monitor_rule": MonitorRule,
    "monitor_alert": MonitorAlert,
    "what_if_scenario": WhatIfScenario,
    "incident_comparison": IncidentComparison,
    "run_manifest": RunManifest,
    "health_status": HealthStatus,
    "system_config": SystemConfig,
}


def get_schema_version(model_name: str) -> str:
    """
    Get the current schema version for a model.

    Args:
        model_name: Name of the model (e.g., "canonical_event", "incident")

    Returns:
        Schema version string (e.g., "canonical_v1")

    Raises:
        KeyError: If model_name is not recognized
    """
    return SCHEMA_VERSIONS[model_name]


def get_model_class(model_name: str):
    """
    Get the Pydantic model class by name.

    Args:
        model_name: Name of the model (e.g., "canonical_event", "incident")

    Returns:
        Pydantic model class

    Raises:
        KeyError: If model_name is not recognized
    """
    return MODEL_REGISTRY[model_name]
